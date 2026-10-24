from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.agent.native_adapter import normalize_native, tools_as_openai
from app.agent.prompt_builder import (
    build_system_prompt,
    read_agent_md,
)
from app.agent.react_parser import ParsedOutput, parse_react
from app.agent.tool_executor import ToolExecutionResult, ToolExecutor
from app.agent.uncertainty import (
    FORCED_SEARCH_EMPTY_TEXT,
    forced_search_query,
    is_refusal_answer,
    should_force_search,
)
from app.config import Settings
from app.context.manager import ContextBundle, ContextManager, estimate_messages, estimate_tokens
from app.context.state_card import StateCardManager
from app.db import session_scope
from app.errors import (
    AgentError,
    BudgetExceededError,
    ErrorClass,
    ServiceUnavailableError,
    TaskTurnLimitError,
)
from app.mcp.client import MCPClient
from app.mcp.tools.artifacts import ArtifactError, read_artifact
from app.models.db_models import Artifact, Message, Session, utcnow
from app.models.gateway import ModelGateway
from app.models.providers import ChatChunk, ModelResult
from app.observability.tracing import (
    Tracer,
    bind_context,
    current_trace_id,
    get_tracer,
    set_trace_id,
)
from app.sessions.repository import SessionRepository

Emit = Callable[[str, dict[str, Any]], Awaitable[None]]

FINAL_ANSWER_RE = __import__("re").compile(r"(?im)Final\s+Answer\s*[:：]")


@dataclass
class AgentRunResult:
    trace_id: str
    final_answer: str
    finish_reason: str
    task_turn_count: int
    mode: str


@dataclass
class TurnState:
    trace_id: str
    session: Session
    turn_no: int
    user_input: str
    conversation: list[dict[str, Any]]
    bundle: ContextBundle | None = None
    model_calls: int = 0
    tool_iterations: int = 0
    task_counted: bool = False
    self_corrects: int = 0
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recoverable_failures: int = 0
    forced_search_used: bool = False
    message_chars: int = 0
    thought_chars: int = 0
    state_manager: StateCardManager | None = None
    repo: SessionRepository | None = None


@dataclass
class ForcedSearchOutcome:
    executed: bool
    status: str
    reason: str | None = None
    tool_call_id: str | None = None
    result: ToolExecutionResult | None = None


class StreamSplitter:
    """Classifies streaming text into thought vs final-answer deltas for ReAct mode.

    Final answers are buffered until the turn decides whether a forced web search
    is required, so a refusal never reaches the user before the search runs.
    """

    def __init__(
        self,
        mode: str,
        emit: Emit,
        *,
        buffer_final: bool = True,
        buffer_limit: int = 20_000,
    ) -> None:
        self.mode = mode
        self.emit = emit
        self.buffer_final = buffer_final
        self.buffer_limit = max(1, buffer_limit)
        self.pending = ""
        self.final_started = False
        self.final_buffer = ""
        self.buffered_chars = 0
        self.buffer_limit_hit = False
        self.message_chars = 0
        self.thought_chars = 0

    async def _final(self, text: str) -> None:
        if not text:
            return
        if not self.buffer_final or self.buffer_limit_hit:
            self.message_chars += len(text)
            await self.emit("message_delta", {"delta": text})
            return
        if self.buffered_chars + len(text) > self.buffer_limit:
            # 超限立即放行，后续正文不再缓冲，也不再做不确定判定。
            self.buffer_limit_hit = True
            released = self.final_buffer + text
            self.final_buffer = ""
            self.buffered_chars = 0
            self.message_chars += len(released)
            await self.emit("message_delta", {"delta": released})
            return
        self.final_buffer += text
        self.buffered_chars += len(text)

    async def feed(self, chunk: ChatChunk) -> None:
        if chunk.reasoning:
            self.thought_chars += len(chunk.reasoning)
            await self.emit("thought_delta", {"delta": chunk.reasoning})
        if not chunk.content:
            return
        if self.mode == "native" or self.final_started:
            await self._final(chunk.content)
            return
        self.pending += chunk.content
        match = FINAL_ANSWER_RE.search(self.pending)
        if match:
            before = self.pending[: match.start()]
            after = self.pending[match.end() :]
            if before.strip():
                self.thought_chars += len(before)
                await self.emit("thought_delta", {"delta": before})
            self.final_started = True
            self.pending = ""
            if after:
                await self._final(after)
            return
        keep = 24
        if len(self.pending) > keep:
            safe = self.pending[:-keep]
            self.pending = self.pending[-keep:]
            if safe.strip():
                self.thought_chars += len(safe)
                await self.emit("thought_delta", {"delta": safe})

    def take_final(self) -> str:
        text, self.final_buffer = self.final_buffer, ""
        self.buffered_chars = 0
        return text

    async def release_buffered(self) -> None:
        """Flush buffered body when the model turns out not to answer yet."""
        if not self.final_buffer:
            return
        text, self.final_buffer = self.final_buffer, ""
        self.buffered_chars = 0
        self.message_chars += len(text)
        await self.emit("message_delta", {"delta": text})

    async def flush(self) -> None:
        if not self.pending:
            return
        text, self.pending = self.pending, ""
        if self.final_started:
            await self._final(text)
        else:
            self.thought_chars += len(text)
            await self.emit("thought_delta", {"delta": text})

    async def reset(self) -> None:
        self.pending = ""
        self.final_started = False
        self.final_buffer = ""
        self.buffered_chars = 0
        self.buffer_limit_hit = False
        self.message_chars = 0
        self.thought_chars = 0
        await self.emit("context_update", {"reset": True})


class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        gateway: ModelGateway,
        mcp: MCPClient,
        *,
        user_id: str = "dev-user",
        tracer: Tracer | None = None,
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.mcp = mcp
        self.user_id = user_id
        self.tracer = tracer or get_tracer(settings)
        self.executor = ToolExecutor(
            mcp, settings, tracer=self.tracer, user_id=user_id
        )
        self._registry: dict[str, dict[str, Any]] | None = None

    async def registry(self, refresh: bool = False) -> dict[str, dict[str, Any]]:
        if self._registry is None or refresh:
            self._registry = await self.executor.fetch_registry()
        return self._registry

    @staticmethod
    def _with_turn_ttft(emit: Emit, turn_span: Any, started_at: float) -> Emit:
        """Record the delay from request entry to the first visible delta."""

        seen = False

        async def wrapped(kind: str, payload: dict[str, Any]) -> None:
            nonlocal seen
            if not seen and kind in {"thought_delta", "message_delta"}:
                seen = True
                turn_span.set(
                    first_token_ms=int((time.monotonic() - started_at) * 1000)
                )
            await emit(kind, payload)

        return wrapped

    async def run(
        self,
        session_id: UUID,
        user_text: str,
        emit: Emit,
        trace_id: str | None = None,
        client_message_id: str | None = None,
    ) -> AgentRunResult:
        if trace_id is not None:
            set_trace_id(trace_id)
        trace_id = trace_id or self.tracer.new_trace()
        bind_context(str(session_id))
        try:
            async with self.tracer.span("request", session_id=str(session_id)) as request_span:
                async with self.tracer.span("turn", session_id=str(session_id)) as turn_span:
                    return await self._run(
                        session_id,
                        user_text,
                        self._with_turn_ttft(
                            emit, turn_span, request_span.started_at
                        ),
                        client_message_id,
                    )
        except AgentError as exc:
            await emit(
                "error",
                {
                    "code": exc.code,
                    "message": exc.message,
                    "class": str(exc.error_class),
                    "retryable": exc.retryable,
                },
            )
            await emit("done", {"finish_reason": "error", "trace_id": trace_id})
            return AgentRunResult(trace_id, "", "error", 0, "react")
        except Exception as exc:
            await emit(
                "error",
                {
                    "code": "INTERNAL_ERROR",
                    "message": f"服务内部错误: {exc}",
                    "class": str(ErrorClass.STATE),
                    "retryable": False,
                },
            )
            await emit("done", {"finish_reason": "error", "trace_id": trace_id})
            return AgentRunResult(trace_id, "", "error", 0, "react")

    async def _run(
        self,
        session_id: UUID,
        user_text: str,
        emit: Emit,
        client_message_id: str | None = None,
    ) -> AgentRunResult:
        async with session_scope(self.settings) as db:
            repo = SessionRepository(db, self.user_id)
            session = await repo.get(session_id)
            if session is None:
                raise AgentError("会话不存在或无权访问", code="SESSION_NOT_FOUND")
            turn_no = await repo.next_turn_no(session_id)
            bind_context(str(session_id), turn_no)
            session.last_trace_id = current_trace_id()
            session.updated_at = utcnow()
            user_message = await repo.add_message(
                session_id=session_id,
                turn_no=turn_no,
                role="user",
                content=user_text,
                token_est=estimate_tokens(user_text),
                meta=(
                    {
                        "client_message_id": client_message_id,
                        "trace_id": current_trace_id(),
                    }
                    if client_message_id
                    else {"trace_id": current_trace_id()}
                ),
            )
            assert user_message is not None
            if session.title == "新会话":
                compact = " ".join(user_text.split())
                session.title = compact[:24] + ("…" if len(compact) > 24 else "")
            await db.commit()

            await emit(
                "meta",
                {
                    "session_id": str(session_id),
                    "turn_no": turn_no,
                    "trace_id": current_trace_id(),
                    "mode": session.mode,
                    "task_turn_count": session.task_turn_count,
                },
            )

            try:
                registry = await self.registry()
            except Exception as exc:
                await emit(
                    "error",
                    {
                        "code": "MCP_UNAVAILABLE",
                        "message": f"工具服务不可用: {exc}",
                        "class": str(ErrorClass.STATE),
                        "retryable": True,
                    },
                )
                await emit(
                    "done", {"finish_reason": "error", "trace_id": current_trace_id()}
                )
                return AgentRunResult(
                    current_trace_id(), "", "error", session.task_turn_count, session.mode
                )

            agent_md = read_agent_md(self.settings.agent_md_file)
            state_manager = StateCardManager(db)
            context_manager = ContextManager(
                db, self.settings, user_id=self.user_id, tracer=self.tracer
            )
            card = await state_manager.get(session_id)
            assert card is not None
            base_system = build_system_prompt(registry, session.mode)
            try:
                async with self.tracer.span("context_build") as span:
                    bundle = await context_manager.build(
                        session_id,
                        turn_no=turn_no,
                        user_input=user_text,
                        base_system=base_system,
                        agent_md=agent_md,
                    )
                    span.set(
                        token_est=bundle.token_est,
                        threshold=bundle.threshold,
                        compressed=bundle.compressed,
                    )
            except BudgetExceededError as exc:
                message = (
                    "当前会话上下文已超出预算，压缩后仍无法容纳本轮任务。"
                    "本轮任务已收束，请新建会话继续或清理历史。"
                )
                await repo.add_message(
                    session_id=session_id,
                    turn_no=turn_no,
                    role="assistant",
                    content=message,
                    token_est=estimate_tokens(message),
                    meta={"kind": "budget_exceeded"},
                )
                await StateCardManager(db).update_after_turn(
                    session_id,
                    turn_no=turn_no,
                    user_input=user_text,
                    final_answer="",
                    tool_events=[],
                    errors=[exc.message],
                )
                session.updated_at = utcnow()
                await db.commit()
                await emit("message_delta", {"delta": message})
                await emit(
                    "error",
                    {
                        "code": exc.code,
                        "message": exc.message,
                        "class": str(ErrorClass.BUDGET),
                        "retryable": False,
                    },
                )
                await emit(
                    "done",
                    {"finish_reason": "budget_exceeded", "trace_id": current_trace_id()},
                )
                return AgentRunResult(
                    current_trace_id(), "", "budget_exceeded", session.task_turn_count, session.mode
                )

            if bundle.compressed:
                await self._emit_context_update(emit, bundle)
            state = TurnState(
                trace_id=current_trace_id(),
                session=session,
                turn_no=turn_no,
                user_input=user_text,
                conversation=list(bundle.messages),
                bundle=bundle,
                state_manager=state_manager,
                repo=repo,
            )
            return await self._loop(db, state, registry, emit)

    async def _loop(
        self,
        db: Any,
        state: TurnState,
        registry: dict[str, dict[str, Any]],
        emit: Emit,
    ) -> AgentRunResult:
        session = state.session
        repo = state.repo
        state_manager = state.state_manager
        if repo is None or state_manager is None:
            raise RuntimeError("turn state not initialized")

        while True:
            if state.model_calls >= self.settings.max_model_calls_per_turn:
                return await self._wrap_up(
                    db, state, repo, state_manager, emit, "max_model_calls", registry
                )
            if self._over_budget(state):
                compacted = await self._compact_conversation(state, emit)
                if not compacted:
                    return await self._wrap_up(
                        db, state, repo, state_manager, emit, "budget_exceeded", registry
                    )

            splitter = StreamSplitter(
                session.mode,
                emit,
                buffer_limit=self.settings.final_answer_buffer_max,
            )
            try:
                result = await self.gateway.complete(
                    state.conversation,
                    tools=tools_as_openai(registry) if session.mode == "native" else None,
                    on_delta=splitter.feed,
                    on_reset=splitter.reset,
                    max_tokens=self.settings.max_output_tokens,
                )
            except ServiceUnavailableError as exc:
                state.errors.append(exc.message)
                fallback_message = "服务暂不可用，请稍后重试。"
                await self._save_assistant(
                    repo,
                    state,
                    content=fallback_message,
                    meta={"kind": "error", "error_code": exc.code, "reason": exc.message},
                )
                await state_manager.update_after_turn(
                    state.session.id,
                    turn_no=state.turn_no,
                    user_input=state.user_input,
                    final_answer="",
                    tool_events=state.tool_events,
                    errors=state.errors,
                )
                await emit("message_delta", {"delta": fallback_message})
                await emit(
                    "error",
                    {
                        "code": exc.code,
                        "message": exc.message,
                        "class": str(exc.error_class),
                        "retryable": True,
                    },
                )
                await emit(
                    "done",
                    {
                        "finish_reason": "model_unavailable",
                        "trace_id": state.trace_id,
                    },
                )
                session.updated_at = utcnow()
                await db.commit()
                return AgentRunResult(
                    state.trace_id, "", "model_unavailable", session.task_turn_count, session.mode
                )
            state.model_calls += 1
            await splitter.flush()
            parsed = self._parse(session.mode, result, registry)
            if parsed.kind != "final":
                # 非最终回答时缓冲正文只是模型的过程性输出，按原行为放行；
                # 但 native 的正文若疑似拒绝收尾，则不提前展示给用户。
                buffered_body = splitter.final_buffer.strip()
                if not (
                    session.mode == "native"
                    and is_refusal_answer(
                        buffered_body,
                        match_window=self.settings.refusal_match_window,
                        max_answer_len=self.settings.refusal_max_answer,
                    )
                ):
                    await splitter.release_buffered()
            state.message_chars = splitter.message_chars

            if parsed.kind == "invalid":
                state.errors.append(parsed.reason)
                if (
                    session.mode in {"react", "native"}
                    and state.self_corrects < 1
                    and state.model_calls < self.settings.max_model_calls_per_turn
                ):
                    # 需求 4.2 规定同一轮只回喂 1 次；
                    # MAX_SELF_CORRECT 用于限制 6.7 的可恢复工具错误。
                    state.self_corrects += 1
                    state.conversation.append(
                        {
                            "role": "assistant",
                            "content": self._model_text_for_context(
                                result.text or parsed.raw
                            ),
                        }
                    )
                    state.conversation.append(
                        {
                            "role": "user",
                            "content": (
                                f"上一次工具调用不符合要求：{parsed.reason}。"
                                "请使用原生 tool_calls，一次只调用一个工具，"
                                "或直接给出最终答案。"
                                if session.mode == "native"
                                else (
                                    "你上一条输出不符合 ReAct 协议："
                                    f"{parsed.reason}。请修正后重新作答，"
                                    "严格使用 Thought/Action/Action Input 或 Thought/Final Answer。"
                                )
                            ),
                        }
                    )
                    await self.tracer.emit_event(
                        "parse_self_correct",
                        {"reason": parsed.reason, "turn_no": state.turn_no},
                    )
                    continue
                session.parse_failures = int(session.parse_failures) + 1
                switched = False
                if session.mode == "react" and session.parse_failures >= 2:
                    session.mode = "native"
                    session.parse_failures = 0
                    switched = True
                    await self.tracer.emit_event(
                        "mode_switch", {"to": "native", "reason": "consecutive_parse_failures"}
                    )
                    await emit(
                        "context_update",
                        {"mode": "native", "reason": "consecutive_parse_failures"},
                    )
                await self._save_assistant(
                    repo,
                    state,
                    content="",
                    meta={
                        "kind": "parse_error",
                        "thought": self._model_text_for_context(result.text or "")[
                            :1000
                        ],
                        "reason": parsed.reason,
                        "provider": result.provider,
                        "usage": result.usage,
                    },
                )
                await state_manager.update_after_turn(
                    state.session.id,
                    turn_no=state.turn_no,
                    user_input=state.user_input,
                    final_answer="",
                    tool_events=state.tool_events,
                    errors=state.errors,
                )
                session.updated_at = utcnow()
                await db.commit()
                await emit(
                    "error",
                    {
                        "code": "PARSE_ERROR",
                        "message": (
                            f"模型输出不符合协议：{parsed.reason}"
                            + ("，本会话已切换为原生工具调用模式" if switched else "")
                        ),
                        "class": str(ErrorClass.RECOVERABLE),
                        "retryable": True,
                    },
                )
                await emit("done", {"finish_reason": "parse_error", "trace_id": state.trace_id})
                return AgentRunResult(
                    state.trace_id, "", "parse_error", session.task_turn_count, session.mode
                )

            if parsed.kind == "final":
                if session.mode == "react":
                    session.parse_failures = 0
                buffered = splitter.take_final().strip()
                answer = buffered or parsed.answer.strip()
                replace_reason: str | None = None
                has_ok_search = self._has_ok_search(state)
                forced_available = (
                    state.forced_search_used is False
                    and self.settings.forced_search_max_per_turn >= 1
                )
                force = (
                    not splitter.buffer_limit_hit
                    and should_force_search(
                        answer=answer,
                        has_ok_search=has_ok_search,
                        already_forced=not forced_available,
                        enabled=self.settings.forced_search_enabled,
                        finish_kind="final",
                        match_window=self.settings.refusal_match_window,
                        max_answer_len=self.settings.refusal_max_answer,
                    )
                )
                # 6.1.4：总开关关闭或本轮强制次数上限为 0 时不发检索请求，
                # 但仍拦截拒绝回答并回固定文案。
                disabled_intercept = (
                    not force
                    and (
                        not self.settings.forced_search_enabled
                        or self.settings.forced_search_max_per_turn < 1
                    )
                    and not state.forced_search_used
                    and not splitter.buffer_limit_hit
                    and not has_ok_search
                    and is_refusal_answer(
                        answer,
                        match_window=self.settings.refusal_match_window,
                        max_answer_len=self.settings.refusal_max_answer,
                    )
                )
                if force or disabled_intercept:
                    state.forced_search_used = True
                    outcome = await self._run_forced_search(
                        db, repo, state, registry, emit
                    )
                    if outcome.status == "ok":
                        if state.model_calls >= self.settings.max_model_calls_per_turn:
                            return await self._wrap_up(
                                db, state, repo, state_manager, emit,
                                "max_model_calls", registry,
                            )
                        if self._over_budget(state):
                            compacted = await self._compact_conversation(state, emit)
                            if not compacted:
                                return await self._wrap_up(
                                    db, state, repo, state_manager, emit,
                                    "budget_exceeded", registry,
                                )
                        self._append_forced_observation(state, outcome)
                        continue
                    # 检索不可用/为空等：固定文案作为 final 落库，不再二次强制。
                    answer = FORCED_SEARCH_EMPTY_TEXT
                    replace_reason = (
                        f"search_{outcome.status}"
                        if outcome.executed
                        else f"search_skipped:{outcome.reason}"
                    )
                elif (
                    state.forced_search_used
                    and not splitter.buffer_limit_hit
                    and is_refusal_answer(
                        answer,
                        match_window=self.settings.refusal_match_window,
                        max_answer_len=self.settings.refusal_max_answer,
                    )
                ):
                    await self.tracer.emit_event(
                        "forced_answer_replaced",
                        {
                            "turn_no": state.turn_no,
                            "reason": "refusal_after_search",
                            "answer_chars": len(answer),
                        },
                    )
                    answer = FORCED_SEARCH_EMPTY_TEXT
                    replace_reason = "refusal_after_search"

                if state.message_chars == 0 and answer:
                    await emit("message_delta", {"delta": answer})
                await self._save_assistant(
                    repo,
                    state,
                    content=answer,
                    meta={
                        "kind": "final",
                        "thought": parsed.thought[:2000],
                        "provider": result.provider,
                        "layer": result.layer,
                        "usage": result.usage,
                        "forced_search": state.forced_search_used,
                        "replace_reason": replace_reason,
                    },
                )
                await state_manager.update_after_turn(
                    state.session.id,
                    turn_no=state.turn_no,
                    user_input=state.user_input,
                    final_answer=answer,
                    tool_events=state.tool_events,
                    errors=state.errors,
                )
                session.updated_at = utcnow()
                await db.commit()
                await emit(
                    "done",
                    {
                        "finish_reason": "final_answer",
                        "trace_id": state.trace_id,
                        "task_turn_count": session.task_turn_count,
                    },
                )
                return AgentRunResult(
                    state.trace_id, answer, "final_answer", session.task_turn_count, session.mode
                )

            # action branch
            if state.tool_iterations >= self.settings.max_tool_iterations:
                return await self._wrap_up(
                    db, state, repo, state_manager, emit, "max_tool_iterations", registry
                )
            if not state.task_counted and session.task_turn_count >= self.settings.max_task_turns:
                session.status = "wrapped"
                await db.commit()
                await emit(
                    "error",
                    {
                        "code": TaskTurnLimitError.code,
                        "message": (
                            f"已达到 {self.settings.max_task_turns} 个任务型轮次上限，"
                            "该任务链已收束，请新建会话继续。"
                        ),
                        "class": str(ErrorClass.PERMANENT),
                        "retryable": False,
                    },
                )
                return await self._wrap_up(
                    db,
                    state,
                    repo,
                    state_manager,
                    emit,
                    "max_task_turns",
                    registry,
                )
            if parsed.name in (session.blocked_tools or []):
                observation = (
                    f"工具 {parsed.name} 因连续失败已被禁用，请改用其他工具或直接给出最终答案。"
                )
                blocked_call_id = _tool_call_id(parsed)
                state.errors.append(observation)
                await emit(
                    "tool_start",
                    {
                        "tool_call_id": blocked_call_id,
                        "name": parsed.name,
                        "arguments": parsed.arguments,
                        "turn_no": state.turn_no,
                    },
                )
                await emit(
                    "tool_result",
                    {
                        "tool_call_id": blocked_call_id,
                        "name": parsed.name,
                        "status": "blocked",
                        "error": observation,
                        "latency_ms": 0,
                    },
                )
                blocked_row = await repo.add_tool_call(
                    session_id=state.session.id,
                    turn_no=state.turn_no,
                    tool_name=parsed.name,
                    arguments=parsed.arguments,
                    status="blocked",
                    result_summary=None,
                    result_ref=None,
                    error=observation,
                    latency_ms=0,
                    idempotency_key=parsed.arguments.get("client_op_id"),
                    call_id=uuid.UUID(hex=blocked_call_id),
                )
                blocked_row.result_meta = {}
                if not state.task_counted:
                    session.task_turn_count = int(session.task_turn_count) + 1
                    state.task_counted = True
                state.tool_events.append(
                    {
                        "tool_call_id": blocked_call_id,
                        "name": parsed.name,
                        "status": "blocked",
                        "summary": None,
                        "error": observation,
                        "latency_ms": 0,
                        "result_ref": None,
                        "attempts": 0,
                        "turn_no": state.turn_no,
                    }
                )
                await db.commit()
                self._append_action_observation(
                    state, result, parsed, blocked_call_id, observation
                )
                state.tool_iterations += 1
                continue

            if session.mode == "react":
                session.parse_failures = 0
            state.tool_iterations += 1
            if not state.task_counted:
                session.task_turn_count = int(session.task_turn_count) + 1
                state.task_counted = True
                await db.commit()
            tool_call_id = _tool_call_id(parsed)
            await emit(
                "tool_start",
                {
                    "tool_call_id": tool_call_id,
                    "name": parsed.name,
                    "arguments": parsed.arguments,
                    "turn_no": state.turn_no,
                },
            )
            action_message = await self._save_assistant(
                repo,
                state,
                content="",
                meta={
                    "kind": "action",
                    "thought": parsed.thought[:2000],
                    "plan": (
                        _plan_text(parsed.thought)
                        if session.mode == "react"
                        else f"调用工具 {parsed.name}"
                    ),
                    "tool_name": parsed.name,
                    "arguments": parsed.arguments,
                    "provider": result.provider,
                },
            )
            started = time.monotonic()
            tool_call_row = await repo.add_tool_call(
                session_id=state.session.id,
                turn_no=state.turn_no,
                tool_name=parsed.name,
                arguments=parsed.arguments,
                status="running",
                result_summary=None,
                result_ref=None,
                error=None,
                latency_ms=0,
                idempotency_key=parsed.arguments.get("client_op_id"),
                message_id=action_message.id,
                call_id=uuid.UUID(hex=tool_call_id),
            )
            await db.commit()
            tool_result = await self.executor.execute(
                parsed.name,
                parsed.arguments,
                session_id=state.session.id,
                turn_no=state.turn_no,
                tool_call_id=tool_call_id,
            )
            latency = int((time.monotonic() - started) * 1000)
            await self._persist_tool_call(
                repo,
                state,
                parsed,
                tool_result,
                tool_call_id,
                latency,
                row=tool_call_row,
            )
            if tool_result.artifact_id:
                await self._save_artifact_row(db, state, tool_result, tool_call_id)
            await self._save_tool_message(repo, state, tool_result, tool_call_id)
            event: dict[str, Any] = {
                "tool_call_id": tool_call_id,
                "name": parsed.name,
                "status": "ok" if tool_result.ok else "error",
                "summary": tool_result.summary if tool_result.ok else None,
                "error": tool_result.error,
                "latency_ms": tool_result.latency_ms,
                "result_ref": tool_result.result_ref,
                "attempts": tool_result.attempts,
            }
            if parsed.name == "search":
                event["citations"] = tool_result.citations
            state.tool_events.append(
                {
                    **event,
                    "summary": tool_result.summary,
                    "result_ref": tool_result.result_ref,
                    "turn_no": state.turn_no,
                }
            )
            await emit("tool_result", event)
            if tool_result.ok:
                state.recoverable_failures = 0
                streaks = dict(session.tool_failure_streaks or {})
                streaks[parsed.name] = 0
                session.tool_failure_streaks = streaks
            else:
                state.errors.append(tool_result.error or "工具失败")
                state.recoverable_failures += 1
                streaks = dict(session.tool_failure_streaks or {})
                streak = int(streaks.get(parsed.name, 0)) + 1
                streaks[parsed.name] = streak
                session.tool_failure_streaks = streaks
                if streak >= 3 and parsed.name not in (session.blocked_tools or []):
                    session.blocked_tools = [*(session.blocked_tools or []), parsed.name]
                    await self.tracer.emit_event(
                        "tool_disabled",
                        {"tool_name": parsed.name, "reason": "consecutive_failures"},
                    )
                    await db.commit()
            observation = self._observation_text(tool_result)
            self._append_action_observation(state, result, parsed, tool_call_id, observation)
            await db.commit()
            if state.tool_iterations >= self.settings.max_tool_iterations:
                return await self._wrap_up(
                    db, state, repo, state_manager, emit, "max_tool_iterations", registry
                )
            if (
                state.recoverable_failures
                >= max(1, self.settings.max_self_correct)
                and not tool_result.ok
            ):
                return await self._wrap_up(
                    db, state, repo, state_manager, emit, "recoverable_limit", registry
                )

    def _append_action_observation(
        self,
        state: TurnState,
        result: ModelResult,
        parsed: ParsedOutput,
        tool_call_id: str,
        observation: str,
    ) -> None:
        if state.session.mode == "native":
            content = self._model_text_for_context(result.text or "")
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": parsed.name,
                            "arguments": json.dumps(parsed.arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
            state.conversation.append(assistant_message)
            state.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": observation,
                }
            )
            return
        state.conversation.append(
            {
                "role": "assistant",
                "content": "Action: "
                + parsed.name
                + "\nAction Input: "
                + json.dumps(parsed.arguments, ensure_ascii=False),
            }
        )
        state.conversation.append({"role": "user", "content": f"Observation: {observation}"})

    def _model_text_for_context(self, text: str) -> str:
        """Drop refusal-like model text before it reaches context or storage."""
        stripped = (text or "").strip()
        if stripped and is_refusal_answer(
            stripped,
            match_window=self.settings.refusal_match_window,
            max_answer_len=self.settings.refusal_max_answer,
        ):
            return ""
        return text or ""

    def _has_ok_search(self, state: TurnState) -> bool:
        return any(
            event.get("name") == "search" and event.get("status") == "ok"
            for event in state.tool_events
        )

    def _forced_search_skip_reason(
        self, state: TurnState, registry: dict[str, dict[str, Any]]
    ) -> str | None:
        if not self.settings.forced_search_enabled:
            return "disabled"
        if self.settings.forced_search_max_per_turn < 1:
            return "max_per_turn_reached"
        if "search" in (state.session.blocked_tools or []):
            return "tool_disabled"
        if "search" not in registry:
            return "tool_missing"
        if not getattr(self.mcp, "started", True):
            return "mcp_offline"
        if state.tool_iterations >= self.settings.max_tool_iterations:
            return "max_tool_iterations"
        if (
            not state.task_counted
            and state.session.task_turn_count >= self.settings.max_task_turns
        ):
            return "max_task_turns"
        return None

    async def _run_forced_search(
        self,
        db: Any,
        repo: SessionRepository,
        state: TurnState,
        registry: dict[str, dict[str, Any]],
        emit: Emit,
    ) -> ForcedSearchOutcome:
        query = forced_search_query(
            state.user_input, self.settings.forced_search_query_max
        )
        skip_reason = self._forced_search_skip_reason(state, registry)
        if skip_reason is not None:
            await self.tracer.emit_event(
                "search_forced_skipped",
                {
                    "trigger": "uncertainty",
                    "reason": skip_reason,
                    "query_chars": len(query),
                },
                session_id=str(state.session.id),
                turn_no=state.turn_no,
            )
            return ForcedSearchOutcome(
                executed=False, status="skipped", reason=skip_reason
            )

        arguments = {"query": query, "max_results": 5}
        parsed = ParsedOutput(kind="action", name="search", arguments=arguments)
        tool_call_id = uuid.uuid4().hex
        await emit(
            "tool_start",
            {
                "tool_call_id": tool_call_id,
                "name": "search",
                "arguments": arguments,
                "turn_no": state.turn_no,
            },
        )
        action_message = await self._save_assistant(
            repo,
            state,
            content="",
            meta={
                "kind": "action",
                "plan": "回答缺少可靠信息，后端强制联网检索",
                "tool_name": "search",
                "arguments": arguments,
                "forced_search": True,
            },
        )
        started = time.monotonic()
        row = await repo.add_tool_call(
            session_id=state.session.id,
            turn_no=state.turn_no,
            tool_name="search",
            arguments=arguments,
            status="running",
            result_summary=None,
            result_ref=None,
            error=None,
            latency_ms=0,
            idempotency_key=None,
            message_id=action_message.id,
            call_id=uuid.UUID(hex=tool_call_id),
        )
        await db.commit()
        tool_result = await self.executor.execute(
            "search",
            arguments,
            session_id=state.session.id,
            turn_no=state.turn_no,
            tool_call_id=tool_call_id,
        )
        latency = int((time.monotonic() - started) * 1000)
        await self._persist_tool_call(
            repo,
            state,
            parsed,
            tool_result,
            tool_call_id,
            latency,
            row=row,
        )
        if tool_result.artifact_id:
            await self._save_artifact_row(db, state, tool_result, tool_call_id)
        await self._save_tool_message(repo, state, tool_result, tool_call_id)
        await emit(
            "tool_result",
            {
                "tool_call_id": tool_call_id,
                "name": "search",
                "status": "ok" if tool_result.ok else "error",
                "summary": tool_result.summary if tool_result.ok else None,
                "error": tool_result.error,
                "latency_ms": tool_result.latency_ms,
                "result_ref": tool_result.result_ref,
                "attempts": tool_result.attempts,
                "citations": tool_result.citations,
            },
        )
        state.tool_events.append(
            {
                "tool_call_id": tool_call_id,
                "name": "search",
                "status": "ok" if tool_result.ok else "error",
                "summary": tool_result.summary,
                "error": tool_result.error,
                "latency_ms": tool_result.latency_ms,
                "result_ref": tool_result.result_ref,
                "attempts": tool_result.attempts,
                "citations": tool_result.citations,
                "turn_no": state.turn_no,
            }
        )
        state.tool_iterations += 1
        if not state.task_counted:
            state.session.task_turn_count = int(state.session.task_turn_count) + 1
            state.task_counted = True
        await self._record_tool_streak(state, "search", tool_result.ok)
        await db.commit()

        result_count = self._search_result_count(tool_result)
        reachable_count = self._search_reachable_count(tool_result)
        if not tool_result.ok:
            state.errors.append(tool_result.error or "强制检索失败")
            status = "error"
        elif result_count == 0:
            status = "empty"
        else:
            status = "ok"
        await self.tracer.emit_event(
            "search_forced",
            {
                "trigger": "uncertainty",
                "query_chars": len(query),
                "status": status,
                "result_count": result_count,
                "reachable_count": reachable_count,
                "latency_ms": tool_result.latency_ms,
                "error": tool_result.error if not tool_result.ok else None,
            },
            session_id=str(state.session.id),
            turn_no=state.turn_no,
        )
        return ForcedSearchOutcome(
            executed=True,
            status=status,
            tool_call_id=tool_call_id,
            result=tool_result,
        )

    async def _record_tool_streak(self, state: TurnState, name: str, ok: bool) -> None:
        session = state.session
        streaks = dict(session.tool_failure_streaks or {})
        if ok:
            streaks[name] = 0
            session.tool_failure_streaks = streaks
            return
        streak = int(streaks.get(name, 0)) + 1
        streaks[name] = streak
        session.tool_failure_streaks = streaks
        if streak >= 3 and name not in (session.blocked_tools or []):
            session.blocked_tools = [*(session.blocked_tools or []), name]
            await self.tracer.emit_event(
                "tool_disabled",
                {"tool_name": name, "reason": "consecutive_failures"},
            )

    @staticmethod
    def _search_result_count(result: ToolExecutionResult) -> int:
        if not result.ok:
            return 0
        if isinstance(result.data, dict):
            items = result.data.get("results")
            if isinstance(items, list):
                return len(items)
            count = result.data.get("count")
            if isinstance(count, (int, float)):
                return int(count)
        return len(result.citations)

    @staticmethod
    def _search_reachable_count(result: ToolExecutionResult) -> int:
        return sum(1 for item in result.citations if item.get("reachable"))

    def _append_forced_observation(
        self, state: TurnState, outcome: ForcedSearchOutcome
    ) -> None:
        if outcome.result is None or outcome.tool_call_id is None:
            return
        result = outcome.result
        arguments = result.arguments or {
            "query": forced_search_query(
                state.user_input, self.settings.forced_search_query_max
            ),
            "max_results": 5,
        }
        observation = self._observation_text(result)
        if state.session.mode == "native":
            state.conversation.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": outcome.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            state.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": outcome.tool_call_id,
                    "content": observation,
                }
            )
            return
        state.conversation.append(
            {
                "role": "assistant",
                "content": "Action: search\nAction Input: "
                + json.dumps(arguments, ensure_ascii=False),
            }
        )
        state.conversation.append(
            {"role": "user", "content": f"Observation: {observation}"}
        )

    def _parse(
        self, mode: str, result: ModelResult, registry: dict[str, dict[str, Any]]
    ) -> ParsedOutput:
        if mode == "native":
            return normalize_native(result.native_response(), registry)
        return parse_react(result.text, registry)

    def _over_budget(self, state: TurnState) -> bool:
        if state.bundle is None:
            return False
        tokens = estimate_messages(state.conversation)
        return tokens > state.bundle.threshold

    async def _compact_conversation(self, state: TurnState, emit: Emit) -> bool:
        threshold = state.bundle.threshold if state.bundle else self.settings.threshold
        before = estimate_messages(state.conversation)
        limit = max(2000, self.settings.tool_result_context_limit)
        for index, message in enumerate(state.conversation):
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system" or len(content) <= limit:
                continue
            is_tool_result = role == "tool" or (
                role == "user" and content.startswith("Observation:")
            )
            if not is_tool_result:
                continue
            refs = re.findall(r"artifact://[A-Za-z0-9_-]+", content)
            summary = content[:1000] + "\n…(已压缩)"
            if refs:
                summary += f"\n完整结果见 {refs[0]}"
            replacement: dict[str, Any] = {"role": role, "content": summary}
            if message.get("tool_calls"):
                replacement["tool_calls"] = message["tool_calls"]
            if role == "tool" and message.get("tool_call_id"):
                replacement["tool_call_id"] = message["tool_call_id"]
            state.conversation[index] = replacement
        after = estimate_messages(state.conversation)
        await self.tracer.emit_event(
            "context_compress",
            {
                "tokens_before": before,
                "tokens_after": after,
                "threshold": threshold,
                "stage": "fold_tool_results",
                "session_id": str(state.session.id),
                "turn_no": state.turn_no,
            },
        )
        await emit(
            "context_update",
            {
                "compressed": True,
                "tokens_before": before,
                "tokens_after": after,
                "threshold": threshold,
                "stage": "fold_tool_results",
            },
        )
        return after <= threshold

    async def _emit_context_update(self, emit: Emit, bundle: ContextBundle) -> None:
        await emit(
            "context_update",
            {
                "compressed": True,
                "tokens_before": bundle.tokens_before,
                "tokens_after": bundle.tokens_after,
                "threshold": bundle.threshold,
            },
        )

    def _observation_text(self, result: ToolExecutionResult) -> str:
        if not result.ok:
            return f"工具执行失败：{result.error}"
        content = result.content
        limit = self.settings.tool_result_context_limit
        if len(content) > limit:
            ref = f"\n完整结果见 {result.result_ref}" if result.result_ref else ""
            return f"{result.summary}{ref}"
        return content

    async def _save_assistant(
        self,
        repo: SessionRepository,
        state: TurnState,
        *,
        content: str,
        meta: dict[str, Any],
    ) -> Message:
        payload = {"trace_id": current_trace_id(), **meta}
        return await repo.add_message(
            session_id=state.session.id,
            turn_no=state.turn_no,
            role="assistant",
            content=content,
            token_est=estimate_tokens(content),
            meta=payload,
        )

    async def _save_tool_message(
        self,
        repo: SessionRepository,
        state: TurnState,
        result: ToolExecutionResult,
        tool_call_id: str,
    ) -> None:
        content = result.content if result.ok else (result.error or "工具执行失败")
        await repo.add_message(
            session_id=state.session.id,
            turn_no=state.turn_no,
            role="tool",
            content=content,
            token_est=estimate_tokens(content),
            meta={
                "type": "tool_result",
                "trace_id": current_trace_id(),
                "tool_call_id": tool_call_id,
                "tool_name": result.name,
                "status": "ok" if result.ok else "error",
                "summary": result.summary,
                "result_ref": result.result_ref,
                "arguments": result.arguments,
                "latency_ms": result.latency_ms,
                "attempts": result.attempts,
            },
        )

    async def _persist_tool_call(
        self,
        repo: SessionRepository,
        state: TurnState,
        parsed: ParsedOutput,
        result: ToolExecutionResult,
        tool_call_id: str,
        latency: int,
        *,
        row: Any = None,
    ) -> None:
        if row is None:
            row = await repo.add_tool_call(
                session_id=state.session.id,
                turn_no=state.turn_no,
                tool_name=parsed.name,
                arguments=result.arguments or parsed.arguments,
                status="ok" if result.ok else "error",
                result_summary=result.summary if result.ok else None,
                result_ref=result.result_ref,
                error=result.error,
                latency_ms=latency,
                idempotency_key=(result.arguments or {}).get("client_op_id"),
                retry_count=max(0, result.attempts - 1),
                call_id=uuid.UUID(hex=tool_call_id),
            )
            row.result_meta = self._result_meta(result)
            return
        row.tool_name = parsed.name
        row.arguments = result.arguments or parsed.arguments
        row.status = "ok" if result.ok else "error"
        row.result_summary = result.summary if result.ok else None
        row.result_ref = result.result_ref
        row.error = result.error
        row.latency_ms = latency
        row.idempotency_key = (result.arguments or {}).get("client_op_id")
        row.retry_count = max(0, result.attempts - 1)
        row.result_meta = self._result_meta(result)
        await repo.db.flush()

    @staticmethod
    def _result_meta(result: ToolExecutionResult) -> dict[str, Any]:
        if result.name != "search":
            return {}
        return {
            "citations": result.citations,
            "reachable_count": sum(1 for item in result.citations if item.get("reachable")),
            "checked_at": utcnow().isoformat(),
        }

    async def _save_artifact_row(
        self,
        db: Any,
        state: TurnState,
        result: ToolExecutionResult,
        tool_call_id: str,
    ) -> None:
        if not result.artifact_id:
            return
        content = result.content
        if result.result_ref and result.result_ref.startswith("artifact://"):
            try:
                content = read_artifact(
                    result.result_ref,
                    session_id=str(state.session.id),
                    base_dir=self.settings.artifacts_dir,
                )
            except ArtifactError:
                content = result.content
        db.add(
            Artifact(
                id=result.artifact_id,
                session_id=state.session.id,
                tool_call_id=tool_call_id,
                size=len(content),
                content=content,
            )
        )
        await db.flush()

    async def _wrap_up(
        self,
        db: Any,
        state: TurnState,
        repo: SessionRepository,
        state_manager: StateCardManager,
        emit: Emit,
        reason: str,
        registry: dict[str, dict[str, Any]],
    ) -> AgentRunResult:
        session = state.session
        forced_search = state.forced_search_used
        answer = ""
        replace_reason: str | None = None
        last_result: ModelResult | None = None
        splitter = StreamSplitter(
            session.mode,
            emit,
            buffer_final=forced_search,
            buffer_limit=self.settings.final_answer_buffer_max,
        )
        over_budget = bool(
            state.bundle
            and estimate_messages(state.conversation) > state.bundle.threshold
        )
        if (
            state.model_calls < self.settings.max_model_calls_per_turn
            and not over_budget
        ):
            instruction = (
                "本轮已达到执行上限。请基于已有 Observation 直接输出最终进展总结，"
                "不要再调用任何工具，必须使用 Final Answer 字段。"
            )
            state.conversation.append({"role": "user", "content": instruction})
            try:
                result = await self.gateway.complete(
                    state.conversation,
                    on_delta=splitter.feed,
                    on_reset=splitter.reset,
                    max_tokens=self.settings.max_output_tokens,
                )
                last_result = result
                state.model_calls += 1
                parsed = self._parse(session.mode, result, registry)
                if parsed.kind == "final":
                    answer = parsed.answer.strip()
            except AgentError:
                answer = ""
        await splitter.flush()
        buffered = splitter.take_final().strip()
        if buffered:
            answer = buffered
        if forced_search and not splitter.buffer_limit_hit and is_refusal_answer(
            answer,
            match_window=self.settings.refusal_match_window,
            max_answer_len=self.settings.refusal_max_answer,
        ):
            await self.tracer.emit_event(
                "forced_answer_replaced",
                {
                    "turn_no": state.turn_no,
                    "reason": "refusal_after_search",
                    "answer_chars": len(answer),
                },
            )
            answer = FORCED_SEARCH_EMPTY_TEXT
            replace_reason = "refusal_after_search"
        if not answer:
            answer = self._local_summary(state)
            await emit("message_delta", {"delta": answer})
        elif splitter.message_chars == 0:
            await emit("message_delta", {"delta": answer})
        meta: dict[str, Any] = {"kind": "wrap_up", "reason": reason}
        if forced_search:
            meta["forced_search"] = True
            meta["replace_reason"] = replace_reason
        if last_result is not None:
            meta.update(
                {
                    "provider": last_result.provider,
                    "layer": last_result.layer,
                    "usage": last_result.usage,
                }
            )
        await self._save_assistant(
            repo,
            state,
            content=answer,
            meta=meta,
        )
        try:
            await state_manager.update_after_turn(
                session.id,
                turn_no=state.turn_no,
                user_input=state.user_input,
                final_answer=answer,
                tool_events=state.tool_events,
                errors=state.errors,
            )
        except Exception:
            pass
        session.updated_at = utcnow()
        await db.commit()
        await emit(
            "done",
            {
                "finish_reason": reason,
                "trace_id": state.trace_id,
                "task_turn_count": session.task_turn_count,
            },
        )
        return AgentRunResult(
            state.trace_id, answer, reason, session.task_turn_count, session.mode
        )

    def _local_summary(self, state: TurnState) -> str:
        facts = []
        for event in state.tool_events[-5:]:
            if event.get("status") == "ok" and event.get("summary"):
                facts.append(f"- {event.get('name')}: {event.get('summary')}")
        lines = ["任务已达到本轮执行上限，以下是当前进展总结："]
        if facts:
            lines.append("已完成的工具调用：")
            lines.extend(facts)
        if state.errors:
            lines.append(f"未解决的问题：{state.errors[-1]}")
        lines.append("可以继续追问细节，或新建会话继续该任务。")
        return "\n".join(lines)


def _message_only(emit: Emit) -> Callable[[ChatChunk], Awaitable[None]]:
    async def _handler(chunk: ChatChunk) -> None:
        if chunk.content:
            await emit("message_delta", {"delta": chunk.content})
        elif chunk.reasoning:
            await emit("thought_delta", {"delta": chunk.reasoning})

    return _handler


def _plan_text(thought: str) -> str:
    first_line = next((line.strip() for line in thought.splitlines() if line.strip()), "")
    cleaned = re.sub(r"^(Thought|思考)\s*[:：]\s*", "", first_line, flags=re.IGNORECASE)
    return cleaned[:160]


def _tool_call_id(parsed: ParsedOutput) -> str:
    # 始终生成全局唯一 ID：模型给的 call_id 可能跨轮重复，会覆盖 artifact 或与主键冲突。
    return uuid.uuid4().hex


__all__ = ["AgentLoop", "AgentRunResult", "StreamSplitter"]
