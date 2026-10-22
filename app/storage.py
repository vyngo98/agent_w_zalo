import json
import os
import uuid
from datetime import datetime, timezone

from app.config import DOCUMENTS_PATH, SESSIONS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def _read_session(session_id: str):
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def get_session(session_id: str):
    return _read_session(session_id)

def _write_session(session: dict) -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    with open(_session_path(session["id"]), "w") as f:
        json.dump(session, f, indent=2)

def create_session(title: str = None) -> dict:
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": _now(),
        "messages": [],
    }
    _write_session(session)
    return session

def append_message(session_id: str, role: str, content: str, sources=None, images=None):
    """Append a message to a session and return the stored message dict.

    `sources` (optional): list of retrieval citations, e.g.
    [{"page": 12, "content_type": "table"}], stored on the message so the
    frontend can display them and they survive a reload.

    `images` (optional): list of image-search results, e.g.
    [{"title": .., "image": url, "thumbnail": url, "url": page}], stored on the
    message so the frontend can render a gallery (and a future Zalo integration
    can send them as photos); survives a reload.
    """
    session = _read_session(session_id)
    if session is None:
        return None

    message = {"role": role, "content": content, "ts": _now()}
    if sources:
        message["sources"] = sources
    if images:
        message["images"] = images
    session["messages"].append(message)

    # Use the first user message as a friendlier session title.
    if role == "user" and len([m for m in session["messages"] if m["role"] == "user"]) == 1:
        session["title"] = content[:60] + ("…" if len(content) > 60 else "")

    _write_session(session)
    return message