"""Tests for bound_runner.run_bound_cron_job — especially subagent waiting."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cron.bound_runner import run_bound_cron_job
from nanobot.cron.types import CronJob, CronPayload, CronSchedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(session_key: str = "cron:test-job") -> CronJob:
    return CronJob(
        id="test-job",
        name="Test Cron Job",
        schedule=CronSchedule(kind="cron", expr="0 * * * *"),
        enabled=True,
        payload=CronPayload(
            message="Do something",
            channel="cli",
            to="direct",
            session_key=session_key,
            origin_channel="cli",
            origin_chat_id="direct",
        ),
    )


class FakeSubagentManager:
    """Minimal subagent manager mock that tracks await_by_session calls."""

    def __init__(self):
        self.awaited_sessions: list[str] = []

    async def await_by_session(self, session_key: str) -> None:
        self.awaited_sessions.append(session_key)


class FakeAgent:
    """Minimal agent satisfying BoundCronAgent protocol."""

    def __init__(self):
        self.tools: dict[str, Any] = {}
        self.subagents = FakeSubagentManager()
        self._response = OutboundMessage(
            channel="cli", chat_id="direct", content="done",
        )

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage:
        return self._response


class FakeCronRecorder:
    def __init__(self):
        self.records: dict[str, dict] = {}

    def write_run_record(self, run_id: str, record: dict[str, Any]) -> None:
        self.records[run_id] = record


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunBoundCronJob:
    @pytest.mark.asyncio
    async def test_waits_for_subagents_before_returning(self):
        """run_bound_cron_job must call await_by_session before returning."""
        agent = FakeAgent()
        cron = FakeCronRecorder()
        job = _make_job(session_key="cron:my-job")

        result = await run_bound_cron_job(job, agent=agent, cron=cron)

        assert result == "done"
        assert agent.subagents.awaited_sessions == ["cron:my-job"]

    @pytest.mark.asyncio
    async def test_await_called_before_ok_record(self):
        """The 'ok' run record must be written after subagent await."""
        call_order: list[str] = []
        agent = FakeAgent()
        cron = FakeCronRecorder()

        original_await = agent.subagents.await_by_session

        async def tracking_await(session_key: str) -> None:
            call_order.append("await_by_session")
            await original_await(session_key)

        agent.subagents.await_by_session = tracking_await

        original_write = cron.write_run_record

        def tracking_write(run_id: str, record: dict) -> None:
            if record.get("status") == "ok":
                call_order.append("write_ok_record")
            original_write(run_id, record)

        cron.write_run_record = tracking_write

        job = _make_job()
        await run_bound_cron_job(job, agent=agent, cron=cron)

        assert call_order == ["await_by_session", "write_ok_record"]

    @pytest.mark.asyncio
    async def test_error_does_not_await(self):
        """If submit_cron_turn raises, await_by_session should NOT be called."""
        agent = FakeAgent()
        agent.submit_cron_turn = AsyncMock(side_effect=RuntimeError("boom"))
        cron = FakeCronRecorder()
        job = _make_job()

        with pytest.raises(RuntimeError, match="boom"):
            await run_bound_cron_job(job, agent=agent, cron=cron)

        assert agent.subagents.awaited_sessions == []

    @pytest.mark.asyncio
    async def test_missing_session_key_raises(self):
        agent = FakeAgent()
        cron = FakeCronRecorder()
        job = _make_job()
        job.payload.session_key = None

        with pytest.raises(ValueError, match="missing payload.session_key"):
            await run_bound_cron_job(job, agent=agent, cron=cron)
