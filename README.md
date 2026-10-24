# Minimal Agent

一个从零实现的最小可用 Agent Web 应用：自研 ReAct 主循环、手写 MCP JSON-RPC 协议、5 个工具、三层模型路由与熔断、上下文压缩、全链路追踪，以及 Vue 3 流式聊天界面。

不依赖 LangGraph、OpenHands、OpenClaw 或任何 Agent 编排框架，也不使用官方 MCP SDK。

- 作者：Audreator
- 后端：Python 3.11+ / FastAPI / SQLAlchemy 2.x async / PostgreSQL / httpx
- 前端：Vue 3 / Vite / TypeScript / Pinia
- 模型：DeepSeek（L1）→ Qwen（L2）→ 本地模型（L3），配置驱动，缺失层自动跳过
- 协议：文本 ReAct，失败后自动切换 native tool calls

---

## 1. 功能概览

- 单轮内可连续执行多次工具调用，直到模型给出 `Final Answer`
- 5 个手写 MCP 工具：`calculator`、`todo`、`search`、`weather`、`read_docs`
- 会话隔离：所有查询强制带 `session_id + user_id`
- 上下文预算：默认 `context_window × 0.7`，超限自动压缩；测试阈值 2000
- 状态卡 + 增量摘要 + 最近 5 轮原文 + artifact 引用的分层记忆
- 三层模型降级、指数退避 + 抖动、30 秒熔断与半开探测
- 异常六分类：瞬时、可恢复、永久、工具执行、预算、状态
- 全链路 trace：`trace_id / span_id / parent_span_id`，同时写 JSON 日志与 `events` 表
- SSE 流式输出：`fetch + ReadableStream`，支持 `last_event_id` 断线续传
- 30 个任务型轮次上限；纯聊天不计入，也不受上限限制
- 最终回答命中“不知道 / 没有相关信息”时先隐藏，后端强制联网检索一次，仍无结果则统一回复固定文案
- 管理后台提供首 Token P95：总览、趋势与模型分层

---

## 2. 系统设计

### 2.1 请求链路

```text
Browser (Vue 3)
  │  POST /api/sessions/{id}/messages  (SSE)
  ▼
FastAPI
  │  会话校验 / 并发锁 / 流缓冲
  ▼
AgentLoop
  │  1. ContextManager 组装上下文
  │  2. ModelGateway 调模型（L1 → L2 → L3）
  │  3. ReactParser / NativeAdapter 解析
  │  4. ToolExecutor → MCP Client → MCP Server 子进程
  │  5. Observation 回填，继续循环
  ▼
Final Answer / 上限收束
```

### 2.2 ReAct 主循环

固定字段：

```text
Thought: ...
Action: tool_name
Action Input: { ...JSON... }
Observation: 由系统回填
Final Answer: ...
```

- 解析器容忍：多余空行、中文冒号、Markdown 代码块、JSON 尾逗号、前后缀说明文字
- 质量校验：工具必须存在于注册表，参数必须符合该工具 `inputSchema`
- 同一轮格式失败先原样回喂 1 次自纠（需求 4.2）
- 连续 2 轮失败后，该会话切换为 native tool calls
- native 模式思考过程取 `reasoning_content`；切换成功后固化，不再回退

### 2.3 轮次与上限

- 1 轮 = 1 条用户消息 + 其引发的全部工具迭代
- 只有发生真实工具调用的轮次才计入 30 轮任务上限
- 纯聊天不计数，也不受 30 轮限制
- 单轮工具迭代上限 8，单轮模型调用上限 12，可配置
- 第 31 个任务型轮次在工具执行前被拒绝，并输出进展总结
- 同一工具连续失败 3 次后在当前会话禁用
- 可恢复工具错误最多 5 轮，之后收束并提示用户

### 2.4 MCP 协议

- stdio 传输，换行分隔的 UTF-8 JSON，一行一条，不使用 Content-Length
- 完整实现 `initialize → notifications/initialized → tools/list → tools/call`
- 请求 id 单调递增，按 id 匹配响应
- 支持服务端通知与 `{code, message, data}` 错误对象
- 服务端 stdout 只允许 JSON-RPC 报文；日志全部写 stderr
- 启动时做 stdout 纯净性自检，并给 stdout 加协议守卫
- 客户端处理子进程启动失败、非 JSON 行、超时、进程异常退出、单行过大
- 大结果通过 `artifact://{tool_call_id}` 回读

### 2.5 模型路由与熔断

```text
L1 DeepSeek → L2 Qwen → L3 本地模型
```

- 缺失 API Key / Base URL 的层自动跳过
- 单 provider 失败：指数退避 + 抖动，最多 3 次，遵守 `Retry-After`
- 401/403/404/参数错误等 4xx 不重试，立即降级
- 连续 3 次失败熔断 30 秒，期间直接走下一层
- 30 秒后进入半开，只放行 1 个探测请求
- 探测成功恢复健康；失败重新熔断并继续降级
- 已产生流式输出后发生中断：先发送 `context_update reset` 清除前端半截内容，再重试或切层，避免重复显示

### 2.6 异常分类

| 类型 | 典型场景 | 处理 |
| --- | --- | --- |
| 瞬时 | 429、5xx、超时 | 退避重试 3 次，再熔断/降级 |
| 可恢复 | 工具失败、参数非法、JSON 错误 | 回喂 Observation / 自纠，最多 5 轮 |
| 永久 | 401、403、404 | 立即终止或降级 |
| 工具执行 | 工具报错、`isError` | 重试 3 次后标记错误结果注入上下文 |
| 预算 | 压缩后仍超限 | 本地总结并收束，不再调用模型 |
| 状态 | 子进程启动失败、写库/恢复异常 | 终止当前轮次并落库错误状态 |

---

## 3. Memory 设计：召回时机与放置方式

本项目中的 Memory 不是单一向量库，而是分层组合：

| 层 | 内容 | 生命周期 |
| --- | --- | --- |
| 状态卡 | goal、confirmed_facts、decisions、todos、artifacts、open_questions、confidence | 每轮更新，随会话持久化 |
| 历史摘要 | 最近 5 轮之外的增量摘要 | 触发压缩时生成并累加 |
| 最近 5 轮 | 用户/助手/工具结果原文 | 每轮保留 |
| 工具结果引用 | `artifact://{tool_call_id}` | 大结果落盘，按需回读 |
| 长期记忆 | `vector_documents` 表与 `VectorRetriever` 接口 | P1 预留，本期不启用 |

### 3.1 召回时机

1. **每轮开始时**：读取会话消息、状态卡、历史摘要，组装本轮首次模型输入。
2. **每次模型调用前**：检查 token 预算；超限先压缩，再继续调用，而不是直接终止。
3. **工具迭代中**：每个 Observation 立即追加到当前轮上下文；若膨胀超预算，先折叠工具结果再继续。
4. **追问时**：
   - 纯对话：状态卡 + 最近 5 轮原文；
   - 带工具追问：状态卡中的 artifacts、历史摘要中的工具摘要，以及 `artifact://` 引用；需要细节时由 `read_docs` 回读命中片段。
5. **收束时**：达到 30 轮、工具/模型调用上限、预算上限或连续失败阈值时，用已有状态卡与工具摘要生成本地进展总结。

### 3.2 放置顺序

每次调用模型的上下文顺序固定为：

```text
1. 系统规则 + 工具清单（含 JSON Schema）
2. Agent.md（存在则只读注入，限制长度）
3. 状态卡（结构化 JSON）
4. 历史摘要（最近 5 轮之前的增量摘要）
5. 最近 5 轮原文（用户 / 助手 / 工具结果）
6. 当前用户输入
```

### 3.3 写入与压缩

- 用户消息：进入循环时立即落库
- 助手行动/最终答案：每一步落库
- 工具调用：执行前先写 `running` 记录，执行后更新状态、耗时、结果摘要与引用
- 状态卡：每轮结束更新；解析失败、预算收束、模型不可用也会记录待确认问题
- 工具结果：
  - ≤ 2000 字符：保留关键输出；
  - > 2000 字符：上下文只放摘要 + `artifact://` 引用，完整内容落 artifact；
  - `search` 结果超过 8K 时截断并落 artifact
- 压缩阶梯：
  1. 折叠超长工具结果
  2. 最近 5 轮以外压增量摘要
  3. 状态卡瘦身
  4. 摘要二次压缩
  5. 仍超限则收束任务并提示用户
- 原始 CoT 不进入上下文：只保留计划、决策、假设；native 模式的 `reasoning_content` 也不写入历史摘要
- 低置信信息只以“待确认”形式进入 `open_questions`，不扩散成事实

### 3.4 隔离

所有消息、状态卡、摘要、工具调用、artifact 都以 `session_id + user_id` 查询；artifact 读取额外校验所属会话，禁止跨会话复用。

---

## 4. 目录结构

```text
backend/
  app/
    main.py                 # FastAPI 入口、生命周期、MCP 启动
    config.py               # pydantic-settings 配置
    db.py                   # async engine、迁移入口
    api/                    # routes + SSE
    agent/                  # loop / react_parser / native_adapter / prompt_builder / tool_executor
    context/                # 状态卡、摘要、压缩、P1 向量接口占位
    models/                 # 网关、熔断、providers、DeepSeek 搜索
    mcp/                    # client / server / schema / tools
    sessions/               # 仓储、服务、Schema
    observability/          # trace / span / JSON 日志 / events
    errors/                 # 异常分类
  alembic/                  # 数据库迁移
  tests/
    unit/
    integration/
    e2e/
  scripts/probe_search.py   # DeepSeek 原生搜索探针
frontend/
  src/
    views/ChatView.vue
    components/SessionList.vue
    components/MessageStream.vue
    components/ToolCallCard.vue
    components/ThoughtPanel.vue
    api/sse.ts
    stores/chat.ts
tasks/v0.1/                 # 需求与示例文档
```

---

## 5. 快速开始

### 5.1 前置依赖

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+（推荐 Docker）

```bash
docker run -d --name agent-pg \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agent \
  -p 55432:5432 postgres:16

docker run -d --name agent-pg-test \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agent_test \
  -p 55433:5432 postgres:16
```

### 5.2 后端

```bash
cd backend
python -m pip install -r requirements-dev.txt

copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux

python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

后端启动时会：

- 通过 Alembic 执行数据库迁移；
- 确保 `dev-user` 存在；
- 拉起 MCP Server 子进程；
- 输出可用模型层与工具清单到健康检查。

### 5.3 前端

```bash
cd frontend
npm ci
npm run dev
```

访问 `http://127.0.0.1:5173`。Vite 已将 `/api` 代理到 `http://127.0.0.1:8000`。

### 5.4 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/agent` | PostgreSQL 连接 |
| `AUTO_MIGRATE` | `true` | 启动时执行 Alembic |
| `DEV_USER_ID` | `dev-user` | 固定开发用户 |
| `CONTEXT_WINDOW` | `1000000` | 上下文窗口 |
| `CONTEXT_BUDGET_RATIO` | `0.7` | 预算比例 |
| `CONTEXT_COMPRESS_THRESHOLD` | 空 | 为空时使用窗口 × 比例；测试设为 2000 |
| `KEEP_RECENT_TURNS` | `5` | 保留原文的最近轮数 |
| `MAX_TASK_TURNS` | `30` | 任务型轮次上限 |
| `MAX_TOOL_ITERATIONS` | `8` | 单轮工具迭代上限 |
| `MAX_MODEL_CALLS_PER_TURN` | `12` | 单轮模型调用上限 |
| `MAX_SELF_CORRECT` | `5` | 可恢复工具错误自纠上限 |
| `TOOL_RETRY` | `3` | 单次工具调用重试次数 |
| `LLM_RETRY` | `3` | 单层模型重试次数 |
| `BREAKER_OPEN_SECONDS` | `30` | 熔断时长 |
| `MODEL_LAYERS` | `deepseek,qwen,local` | 层顺序 |
| `MODEL_L1_BASE_URL` / `MODEL_L1_API_KEY` / `MODEL_L1_NAME` | 空 / 空 / `deepseek-chat` | L1 |
| `MODEL_L2_*` | 空 / `qwen-plus` | L2 |
| `MODEL_L3_*` | 空 / `local-model` | L3 |
| `DEEPSEEK_SEARCH_ENABLED` | `true` | 是否允许联网搜索 |
| `DEEPSEEK_SEARCH_BASE_URL` | `https://api.deepseek.com` | 搜索服务地址 |
| `DEEPSEEK_SEARCH_API_KEY` | 空 | 未配置时回退 `MODEL_L1_API_KEY` |
| `DEEPSEEK_SEARCH_MODEL` | `deepseek-v4-pro` | 支持 `web_search` 的模型 |
| `DOCS_ROOT` | `backend/docs` | `read_docs` 根目录 |
| `ARTIFACTS_DIR` | `backend/artifacts` | 大结果 artifact 目录 |
| `FORCED_SEARCH_ENABLED` | `true` | 关闭时不发检索请求，但仍拦截拒绝回答并回固定文案 |
| `FORCED_SEARCH_MAX_PER_TURN` | `1` | 每轮强制检索上限，仅支持 0/1；0 不检索但仍回固定文案 |
| `FORCED_SEARCH_QUERY_MAX` | `500` | 强制检索词最大字符 |
| `FINAL_ANSWER_BUFFER_MAX` | `20000` | 最终回答缓冲上限，超限立即放行 |
| `REFUSAL_MATCH_WINDOW` | `200` | 拒绝表述匹配窗口 |
| `REFUSAL_MAX_ANSWER` | `800` | 超过该长度不判定为拒绝 |

> `search` 固定使用 DeepSeek 原生联网搜索：请求 `POST /responses`，工具为 `{"type":"web_search"}`，只接受服务端返回的 `url_citation`，不解析模型自造链接。无 Key 或未返回 citations 时会明确失败。

---

## 6. 测试与校验

```bash
cd backend
python -m ruff check .
python -m mypy app
python -m pytest -q

python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest -m e2e -q        # 需要真实 Key
```

前端：

```bash
cd frontend
npm run typecheck
npm run build
```

真实联网 e2e 需要：

```powershell
$env:MODEL_L1_API_KEY="..."
$env:DEEPSEEK_SEARCH_API_KEY="..."
python -m pytest -m e2e -q
```

搜索探针：

```bash
cd backend
python scripts/probe_search.py "杭州今天天气"
```

---

## 7. API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 模型层、MCP 子进程与工具状态 |
| `GET` | `/api/sessions` | 会话列表 |
| `POST` | `/api/sessions` | 创建会话 |
| `DELETE` | `/api/sessions/{id}` | 删除会话（运行中返回 409） |
| `GET` | `/api/sessions/{id}/messages` | 历史消息 |
| `POST` | `/api/sessions/{id}/messages` | 发送消息，返回 SSE |
| `GET` | `/api/tools` | 5 个工具及 inputSchema |
| `GET` | `/api/traces/{trace_id}` | 完整 span 链 |

### SSE 事件

`meta`、`thought_delta`、`message_delta`、`tool_start`、`tool_result`、`context_update`、`done`、`error`，全部带递增 `seq`。

- 断线重连：`POST` 时携带 `last_event_id`，从缺失位置回放并继续跟随活动流。
- 请求幂等：`client_message_id` 同时存于内存缓冲与数据库，避免首包丢失重试或进程重启后重复执行。
- 模型流中途失败：服务端发送 `context_update {reset: true}`，前端清除本次模型调用的半截内容与思考，再接收重试层的输出；已完成的工具卡片保留。
- 最终回答改为整段到达；强制检索以现有工具卡片与引用列表呈现，不新增 SSE 事件名。

---

## 8. 数据库表

- `users`：开发用户
- `sessions`：模式、任务轮次、失败计数、禁用工具、最近 trace
- `messages`：用户/助手/工具消息与元数据
- `tool_calls`：参数、状态、耗时、结果摘要、artifact 引用、幂等键
- `state_cards`：结构化工作记忆
- `summaries`：增量历史摘要及覆盖轮次
- `todos`：按用户归属的待办；`(user_id, client_op_id)` 唯一
- `todo_operations`：写操作幂等记录，按 user + action 校验
- `events`：trace/span 事件
- `artifacts`：大结果元数据
- `vector_documents` / `snapshots`：P1 预留

---

## 9. 可观测性

- 每个请求：`request → turn → context_build / model_call / tool_call`
- 自动注入 `trace_id`、`span_id`、`parent_span_id`
- 记录：`session_id`、`turn_no`、耗时、token 用量、错误码、工具名、参数摘要、结果大小
- 首 Token：`model_call` / `model_success` / `turn` 事件携带 `first_token_ms`，后台按 P95 聚合
- 强制检索：`search_forced`、`search_forced_skipped`、`forced_answer_replaced` 事件
- 双写：按天 JSON 日志文件 + `events` 表
- 按 trace 检索：`GET /api/traces/{trace_id}`

---

## 10. 已知边界

- 本期不做向量检索与完整快照重放，仅保留表结构与 `VectorRetriever` 接口占位。
- 登录固定为 `dev-user`，不实现鉴权体系。
- `search` 依赖 DeepSeek 侧支持 `web_search` 的模型；无 Key 时 e2e 自动跳过。
- 30 轮上限按会话累计；需要开启新任务链时建议新建会话。

---

## 11. 文档

- 需求概括：`tasks/v0.1/T26102101_需求概括.md`
- 开发提示词与示例代码：`tasks/v0.1/T26102101_提示词.md`、`tasks/v0.1/T26102101_示例代码.md`

Author: **Audreator**
