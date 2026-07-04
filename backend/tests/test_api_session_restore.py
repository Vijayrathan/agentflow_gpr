"""Key-free tests for the refresh/reconnect mechanics in backend.api:
transcript recording, dead-socket resilience, and current-socket routing."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend import api


class FakeWS:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send_json(self, payload):
        if self.fail:
            raise RuntimeError("client gone")
        self.sent.append(payload)


def _chat(ws=None):
    chat = api._new_chat_session("test-session")
    chat.ws = ws
    return chat


def test_send_records_chat_visible_events_and_delivers():
    ws = FakeWS()
    chat = _chat(ws)
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "hi"}))
    asyncio.run(api._send(chat, {"type": "stage_change", "stage_name": "Layers"}))
    assert [e["type"] for e in chat.transcript] == ["agent_message", "stage_change"]
    assert ws.sent == chat.transcript


def test_send_does_not_record_transient_events():
    ws = FakeWS()
    chat = _chat(ws)
    asyncio.run(api._send(chat, {"type": "pipeline_busy", "busy": True}))
    asyncio.run(api._send(chat, {"type": "model_update", "scene": {}}))
    assert chat.transcript == []
    assert len(ws.sent) == 2


def test_send_survives_no_socket_and_dead_socket():
    # No socket at all: event still lands in the transcript.
    chat = _chat(ws=None)
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "kept"}))
    # Dead socket: send failure is swallowed, transcript still gets the event.
    chat.ws = FakeWS(fail=True)
    asyncio.run(api._send(chat, {"type": "progress", "content": "done"}))
    assert [e["type"] for e in chat.transcript] == ["agent_message", "progress"]


def test_send_targets_current_socket_after_reconnect():
    old, new = FakeWS(), FakeWS()
    chat = _chat(old)
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "before"}))
    chat.ws = new  # page refresh swapped the socket mid-turn
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "after"}))
    assert [m["content"] for m in old.sent] == ["before"]
    assert [m["content"] for m in new.sent] == ["after"]
    assert [m["content"] for m in chat.transcript] == ["before", "after"]
