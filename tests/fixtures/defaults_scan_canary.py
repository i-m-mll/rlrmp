"""Negative canary fixture for the default-fallback AST scanner."""

CANARY_PAYLOAD = {}
CANARY_VALUE = CANARY_PAYLOAD.get("default_scan_canary", 12345)
