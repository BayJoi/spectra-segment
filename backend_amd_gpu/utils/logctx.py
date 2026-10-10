from __future__ import annotations

import contextvars
import uuid

_rid = contextvars.ContextVar("rid", default="")
_sid = contextvars.ContextVar("sid", default="")


def new_request() -> str:
    r = uuid.uuid4().hex[:8]
    _rid.set(r)
    return r


def set_session(sid: str) -> None:
    _sid.set(sid)


def clear_request() -> None:
    _rid.set("")
    _sid.set("")


def rid() -> str:
    return _rid.get() or "-"


def sid() -> str:
    return _sid.get() or "-"
