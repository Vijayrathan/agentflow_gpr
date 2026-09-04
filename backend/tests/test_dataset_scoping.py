"""Key-free tests for per-user/per-chat dataset directory scoping:
API sessions author ./dataset/<user>/<basename>__<sid8>; the CLI keeps the
legacy ./dataset/<basename> path; user ids can never path-traverse."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import agentflow_single_agent as sap


def _sid8(session_id: str) -> str:
    return hashlib.sha1(session_id.encode()).hexdigest()[:8]


def _save_cfg(session, basename="demo"):
    payload = json.dumps({"num_samples": 2, "model_basename": basename})
    session.save_section.invoke({"section": "dataset_config", "payload": payload})
    return session.store["dataset_config"]["output_dir"]


def test_scoped_session_authors_user_chat_dir():
    s = sap.SingleAgentSession(user_id="vijay", session_id="abc")
    out = _save_cfg(s)
    assert out == f"{sap.DATASET_ROOT}/vijay/demo__{_sid8('abc')}"


def test_scoped_dir_stable_across_resaves():
    s = sap.SingleAgentSession(user_id="vijay", session_id="abc")
    assert _save_cfg(s) == _save_cfg(s)  # deterministic — re-save never moves it


def test_same_basename_different_chats_never_collide():
    a = _save_cfg(sap.SingleAgentSession(user_id="vijay", session_id="chat-1"))
    b = _save_cfg(sap.SingleAgentSession(user_id="vijay", session_id="chat-2"))
    assert a != b


def test_same_basename_different_users_never_collide():
    a = _save_cfg(sap.SingleAgentSession(user_id="alice", session_id="s"))
    b = _save_cfg(sap.SingleAgentSession(user_id="bob", session_id="s"))
    assert a != b


def test_traversal_user_id_is_sanitized():
    s = sap.SingleAgentSession(user_id="../../etc", session_id="abc")
    out = _save_cfg(s)
    # Separators/dots collapse: the dir stays INSIDE the dataset root as a
    # plain "etc" segment — no ".." component ever survives.
    assert ".." not in Path(out).parts
    assert out == f"{sap.DATASET_ROOT}/etc/demo__{_sid8('abc')}"


def test_unscoped_session_keeps_legacy_path():
    s = sap.SingleAgentSession()  # CLI shape: no user/session scope
    assert _save_cfg(s) == f"{sap.DATASET_ROOT}/demo"


def test_scoped_output_dir_helper_matches_tool():
    s = sap.SingleAgentSession(user_id="vijay", session_id="abc")
    assert _save_cfg(s, "wet sand") == sap._scoped_output_dir(
        "vijay", "abc", "wet sand"
    )
