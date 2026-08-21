def format_duration(seconds: float) -> str:
    """Human-friendly duration: 240ms / 2.4s / 38s / 1m 05s."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m {s:02d}s"
