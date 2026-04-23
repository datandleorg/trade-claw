"""1-minute index candlestick chart → Claude Haiku 4.5 vision trend (NIFTY, BANKNIFTY); API every N minutes (default 3)."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from zoneinfo import ZoneInfo

from trade_claw.constants import FO_INDEX_UNDERLYING_LABELS, MOCK_ENGINE_BANK_SYMBOLS
from trade_claw.mock_market_signal import load_index_session_interval_df

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger(__name__)

VISION_INDEX_KEYS: tuple[str, ...] = ("NIFTY", "BANKNIFTY")
TREND_LABELS = frozenset({"BULLISH", "BEARISH", "NEUTRAL"})
VISION_CANDLE_INTERVAL = "minute"


def reference_vision_index_for_mock_engine(u: str) -> str:
    """
    Which index vision series applies for this scan underlying.
    Bank equities → BANKNIFTY; NIFTY/BANKNIFTY indices → self; else → NIFTY (broad).
    """
    ux = (u or "").strip().upper()
    if ux == "NIFTY":
        return "NIFTY"
    if ux == "BANKNIFTY":
        return "BANKNIFTY"
    if ux in MOCK_ENGINE_BANK_SYMBOLS:
        return "BANKNIFTY"
    return "NIFTY"


def vision_enabled() -> bool:
    return os.environ.get("MOCK_INDEX_VISION_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "",
    )


def vision_interval_minutes() -> float:
    try:
        return float(os.environ.get("MOCK_INDEX_VISION_INTERVAL_MIN", "3"))
    except ValueError:
        return 3.0


def vision_model_name() -> str:
    return (
        os.environ.get("MOCK_INDEX_VISION_MODEL") or "claude-haiku-4-5-20251001"
    ).strip()


def should_refresh_vision(prev_as_of_ist: str | None) -> bool:
    if not prev_as_of_ist:
        return True
    try:
        raw = str(prev_as_of_ist).strip()
        if raw.endswith("Z"):
            prev = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        else:
            prev = datetime.fromisoformat(raw[:26])
        if prev.tzinfo is None:
            prev = prev.replace(tzinfo=IST)
        now = datetime.now(IST)
        delta = (now - prev.astimezone(IST)).total_seconds()
        return delta >= vision_interval_minutes() * 60
    except Exception:  # noqa: BLE001
        return True


def build_index_vision_candlestick_figure(df, index_key: str) -> go.Figure:
    """Session 1-minute OHLC (same data sent to Claude as PNG)."""
    label = FO_INDEX_UNDERLYING_LABELS.get(index_key, index_key)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=label,
            )
        ]
    )
    fig.update_layout(
        title=f"{label} — 1 minute candles (session)",
        template="plotly_dark",
        height=420,
        width=900,
        xaxis_rangeslider_visible=False,
        margin=dict(l=48, r=24, t=48, b=40),
    )
    return fig


# Backward compatibility for imports
def build_index_3m_candlestick_figure(df, index_key: str) -> go.Figure:
    """Deprecated name; use ``build_index_vision_candlestick_figure``."""
    return build_index_vision_candlestick_figure(df, index_key)


def ohlc_session_to_png_bytes(df: pd.DataFrame, index_key: str) -> bytes:
    """
    Raster chart for Claude vision — **matplotlib + mplfinance** (non-interactive Agg backend).
    Avoids Kaleido/Chrome required by Plotly ``to_image``.
    """
    import io
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    label = FO_INDEX_UNDERLYING_LABELS.get(index_key, index_key)
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.set_index("date")
    work = work.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    )
    ohlc = work[["Open", "High", "Low", "Close"]].astype(float)
    buf = io.BytesIO()
    mpf.plot(
        ohlc,
        type="candle",
        style="charles",
        title=f"{label} — 1 minute candles (session)",
        volume=False,
        figsize=(9, 4.2),
        savefig=dict(
            fname=buf,
            format="png",
            dpi=110,
            bbox_inches="tight",
            pad_inches=0.2,
        ),
    )
    plt.close("all")
    buf.seek(0)
    return buf.getvalue()


def _parse_trend_payload(text: str) -> tuple[str | None, str | None]:
    """
    Parse JSON with trend and optional rationale from model output.
    Returns (trend, rationale_or_none).
    """
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            t = str(obj.get("trend", "")).upper().strip()
            if t in TREND_LABELS:
                r = obj.get("rationale")
                rat = str(r).strip() if r is not None else None
                return t, rat or None
        except json.JSONDecodeError:
            pass
    u = text.upper()
    for w in ("BULLISH", "BEARISH", "NEUTRAL"):
        if w in u:
            return w, None
    return None, None


def classify_trend_from_png_claude(
    png_bytes: bytes, index_key: str
) -> tuple[str | None, str | None, str | None]:
    """
    Claude Haiku 4.5 vision: returns (trend, rationale, error).
    trend in BULLISH|BEARISH|NEUTRAL, or None on failure.
    """
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        return None, None, "ANTHROPIC_API_KEY missing"
    model = vision_model_name()
    label = FO_INDEX_UNDERLYING_LABELS.get(index_key, index_key)
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
    except Exception as e:  # noqa: BLE001
        return None, None, f"Anthropic client: {e}"

    user_text = (
        f"Index: {label} ({index_key}). The image is a session intraday chart with **1-minute** candlesticks.\n"
        "Classify the **overall** session trend for this index as BULLISH, BEARISH, or NEUTRAL.\n"
        "Reply with **JSON only**, no markdown:\n"
        '{"trend":"BULLISH","rationale":"one or two concise sentences explaining visible structure and why"}\n'
        "BULLISH = clear upward / higher lows; BEARISH = clear downward / lower highs; "
        "NEUTRAL = choppy, range, or unclear direction."
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Claude vision API error for %s: %s", index_key, e)
        return None, None, str(e)

    parts = getattr(resp, "content", None) or []
    raw_text = ""
    for b in parts:
        if getattr(b, "type", None) == "text":
            raw_text += getattr(b, "text", "") or ""

    trend, rationale = _parse_trend_payload(raw_text)
    if trend is None:
        return None, None, f"unparseable vision response: {(raw_text or '')[:300]}"
    return trend, rationale, None


def _refresh_trends(
    kite,
    nse_instruments: list,
    session_d: date,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    as_of = datetime.now(IST).isoformat()
    for key in VISION_INDEX_KEYS:
        df, err = load_index_session_interval_df(
            kite, nse_instruments, session_d, key, VISION_CANDLE_INTERVAL
        )
        if df is None or err or df.empty:
            out[key] = {
                "trend": None,
                "error": err or "no 1m data",
                "rationale": None,
            }
            continue
        try:
            png = ohlc_session_to_png_bytes(df, key)
        except Exception as e:  # noqa: BLE001
            logger.warning("chart/png failed for %s: %s", key, e)
            out[key] = {"trend": None, "error": str(e), "rationale": None}
            continue
        trend, rationale, verr = classify_trend_from_png_claude(png, key)
        out[key] = {
            "trend": trend,
            "error": verr,
            "rationale": rationale,
        }
    return {
        "index_trends_3m": out,
        "index_vision_as_of_ist": as_of,
        "index_vision_model": vision_model_name(),
        "index_vision_refreshed": True,
        "index_vision_disabled": False,
    }


def resolve_index_trends_for_tick(
    kite,
    nse_instruments: list,
    session_d: date,
    prev_snapshot: dict | None,
) -> dict[str, Any]:
    """
    Throttled vision refresh (default every 3 min wall clock); 1m candles + Claude each refresh.
    Merged into run_scan ``out`` as ``index_trends_3m`` (key name kept for telemetry compatibility).
    """
    prev_scan = (prev_snapshot or {}).get("last_scan") or {}
    prev_trends = prev_scan.get("index_trends_3m")
    prev_as_of = prev_scan.get("index_vision_as_of_ist")

    if not vision_enabled():
        return {
            "index_trends_3m": None,
            "index_vision_disabled": True,
            "index_vision_as_of_ist": prev_as_of,
            "index_vision_model": vision_model_name(),
            "index_vision_refreshed": False,
        }

    if (
        not should_refresh_vision(prev_as_of if isinstance(prev_as_of, str) else None)
        and isinstance(prev_trends, dict)
        and prev_trends
    ):
        return {
            "index_trends_3m": prev_trends,
            "index_vision_as_of_ist": prev_as_of,
            "index_vision_model": prev_scan.get("index_vision_model") or vision_model_name(),
            "index_vision_refreshed": False,
            "index_vision_disabled": False,
        }

    return _refresh_trends(kite, nse_instruments, session_d)


def envelope_direction_allowed_by_trend(
    direction: str,
    ref_index: str,
    index_trends_3m: dict[str, Any] | None,
    *,
    vision_disabled: bool,
) -> tuple[bool, str]:
    """
    Strict: BULLISH signal needs BULLISH trend; BEARISH needs BEARISH.
    NEUTRAL/missing/failed vision → disallow.
    """
    if vision_disabled:
        return False, "index_trend_blocked: MOCK_INDEX_VISION_ENABLED=0"
    if not index_trends_3m or not isinstance(index_trends_3m, dict):
        return False, "index_trend_blocked: no vision data"
    block = index_trends_3m.get(ref_index)
    if not isinstance(block, dict):
        return False, f"index_trend_blocked: no entry for {ref_index}"
    trend = block.get("trend")
    err = block.get("error")
    if err and trend is None:
        return False, f"index_trend_blocked: {ref_index} vision error: {err}"
    if trend not in TREND_LABELS:
        return False, f"index_trend_blocked: {ref_index} trend missing"
    if trend == "NEUTRAL":
        return False, f"index_trend_blocked: {ref_index}=NEUTRAL"
    d = (direction or "").upper().strip()
    if d == "BULLISH" and trend != "BULLISH":
        return False, f"index_trend_blocked: signal BULLISH vs {ref_index}={trend}"
    if d == "BEARISH" and trend != "BEARISH":
        return False, f"index_trend_blocked: signal BEARISH vs {ref_index}={trend}"
    return True, ""
