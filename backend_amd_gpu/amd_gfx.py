from __future__ import annotations


def gfx_override_value(gfx: str | None) -> str | None:
    if not gfx:
        return None
    if gfx.startswith("gfx101"):
        return "10.1.0"
    if gfx.startswith("gfx103"):
        return "10.3.0"
    if gfx.startswith("gfx110"):
        return "11.0.0"
    return None
