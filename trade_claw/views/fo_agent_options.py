"""F&O Agent: deterministic strategy signal + OpenAI tool loop for mock option choice (no live orders)."""
import calendar
from datetime import date
import html
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    ENVELOPE_EMA_PERIOD,
    FO_BROKERAGE_PER_LOT_RT_DEFAULT,
    FO_CLOSED_AT_REALISED,
    FO_ENVELOPE_BANDWIDTH_MAX_PCT,
    FO_ENVELOPE_BANDWIDTH_MIN_PCT,
    FO_ENVELOPE_BANDWIDTH_STEP,
    FO_INDEX_UNDERLYING_LABELS,
    FO_STRATEGY_ENVELOPE,
    FO_STRATEGY_MA_CROSS,
    FO_STRATEGY_OPTIONS,
    FO_TAXES_PER_LOT_RT_DEFAULT,
    FO_UNDERLYING_OPTIONS,
    MA_EMA_FAST,
    MA_EMA_SLOW,
    NFO_EXCHANGE,
    NSE_EXCHANGE,
)
from trade_claw.env_trading_params import option_stop_premium_fraction, option_target_premium_fraction
from trade_claw.fo_openai_agent import configure_fo_agent_logging, get_openai_api_key, run_fo_agent_pipeline
from trade_claw.mock_market_signal import (
    fo_options_default_envelope_bandwidth_pct,
    mock_agent_envelope_pct,
)
from trade_claw.kite_session import get_kite_credentials
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)
from trade_claw.pl_style import pl_markdown, pl_title_color
from trade_claw.strategies import add_ma_ema_line_traces, add_ma_envelope_line_traces

_default_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


def _foa_underlying_select_label(u: str) -> str:
    if u in FO_INDEX_UNDERLYING_LABELS:
        return f"{FO_INDEX_UNDERLYING_LABELS[u]} — `{u}`"
    return u


def _abbrev_rupee(x: float) -> str:
    if x == 0:
        return "₹0"
    sgn = "-" if x < 0 else ""
    v = abs(float(x))
    if v >= 1e7:
        return f"{sgn}₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sgn}₹{v / 1e5:.2f} L"
    if v >= 1e3:
        return f"{sgn}₹{v / 1e3:.1f} K"
    return f"{sgn}₹{v:,.0f}"


def _abbrev_pl_html(x: float) -> str:
    full = html.escape(f"₹{x:+,.2f}")
    short = html.escape(_abbrev_rupee(x))
    if x > 0:
        col = "#15803d"
    elif x < 0:
        col = "#b91c1c"
    else:
        col = "#64748b"
    return f'<span title="{full}" style="color:{col};font-weight:600">{short}</span>'


def _resolve_openai_key() -> str | None:
    k = get_openai_api_key()
    if k:
        return k.strip()
    try:
        s = st.secrets.get("OPENAI_API_KEY") if getattr(st, "secrets", None) else None
        if s:
            return str(s).strip()
    except Exception:
        pass
    return None


def render_fo_agent_options(kite):
    configure_fo_agent_logging()

    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load NSE instruments: {e}")
                st.stop()
    if st.session_state.get("nfo_instruments") is None:
        with st.spinner("Loading NFO instruments (options master)..."):
            try:
                st.session_state.nfo_instruments = kite.instruments(NFO_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load NFO instruments: {e}")
                st.stop()

    nse = st.session_state.nse_instruments
    nfo = st.session_state.nfo_instruments
    symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in nse}

    st.title("F&O Agent (OpenAI + mock trades only)")
    today = date.today()
    _, _last_dom = calendar.monthrange(today.year, today.month)
    month_start = date(today.year, today.month, 1)
    month_end = date(today.year, today.month, _last_dom)
    session_upper = min(today, month_end)

    st.info(
        "**Mock backtest only.** The model only sees **`search_instruments`*