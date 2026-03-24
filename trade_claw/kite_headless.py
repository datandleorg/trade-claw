"""KiteConnect for Celery workers: no Streamlit; token from env or `.kite_session.json`."""

from __future__ import annotations

import json
import os
from pathlib import Path

from kiteconnect import KiteConnect

from trade_claw.constants import SESSION_FILE


def get_kite_headless() -> KiteConnect:
    """
    API key from `KITE_API_KEY`; access token from `KITE_ACCESS_TOKEN` or Streamlit-persisted session file.
    Session file shape: `{"api_key": "...", "access_token": "..."}` (same as kite_session.save_session_to_file).
    """
    api_key = (os.environ.get("KITE_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("KITE_API_KEY is not set")

    token = (os.environ.get("KITE_ACCESS_TOKEN") or "").strip()
    path = Path(SESSION_FILE)
    if not token and path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if (data.get("api_key") or "").strip() == api_key:
            token = (data.get("access_token") or "").strip()

    if not token:
        raise ValueError(
            "No Kite access token: log in via Streamlit once (writes .kite_session.json) or set KITE_ACCESS_TOKEN"
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(token)
    return kite
