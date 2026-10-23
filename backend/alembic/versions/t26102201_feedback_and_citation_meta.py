"""add feedback and tool result meta

Revision ID: t26102201_feedback
Revises: 7931940255e4
Create Date: 2026-10-22 12:00:00.000000
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "t26102201_feedback"
down_revision: str | None = "7931940255e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("rating", sa.String(length=8), nullable=False),
        sa.Column("reason_code", sa.String(length=32), nullable=True),
        sa.Column("reason_text", sa.String(length=200), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id", "user_id", name="uq_feedback_message_user"),
    )
    op.create_index(
        "ix_feedback_session_created", "message_feedback", ["session_id", "created_at"]
    )
    op.create_index(
        "ix_feedback_rating_created", "message_feedback", ["rating", "created_at"]
    )
    op.create_index("ix_feedback_reason", "message_feedback", ["reason_code"])

    op.add_column(
        "tool_calls",
        sa.Column(
            "result_meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("ix_tool_calls_created", "tool_calls", ["created_at"])
    op.create_index("ix_events_kind_ts", "events", ["kind", "ts"])


def downgrade() -> None:
    op.drop_index("ix_events_kind_ts", table_name="events")
    op.drop_index("ix_tool_calls_created", table_name="tool_calls")
    op.drop_column("tool_calls", "result_meta")
    op.drop_index("ix_feedback_reason", table_name="message_feedback")
    op.drop_index("ix_feedback_rating_created", table_name="message_feedback")
    op.drop_index("ix_feedback_session_created", table_name="message_feedback")
    op.drop_table("message_feedback")
