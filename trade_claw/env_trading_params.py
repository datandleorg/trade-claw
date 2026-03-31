"""Runtime trading parameters from the environment (read at call time; see .env.example)."""

from __future__ import annotations

import os

from trade_claw.constants import FO_ENVELOPE_BANDWIDTH_MAX_PCT, FO_ENVELOPE_BANDWIDTH_MIN_PCT

_DEFAULT_MOCK_AGENT_ENVELOPE_DECIMAL = 0.25
_DEFAULT_INTRADAY_ENVELOPE_DECIMAL = 0.0030
# Min fractional distance past envelope band for mock signal (vs band price); 0 = touch OK.
_DEFAULT_MOCK_ENGINE_BREAKOUT_CLEAR_PCT = 0.0002


def fno_envelope_decimal_per_side() -> float:
    """
    F&O / mock / agent: EMA envelope half-width as decimal each side
    (e.g. 0.25 = 25% above/below EMA; 0.003 = 0.3%).
    """
    raw = (os.environ.get("MOCK_AGENT_ENVELOPE_PCT") or "").strip()
    if raw:
        return float(raw)
    return float(_DEFAULT_MOCK_AGENT_ENVELOPE_DECIMAL)


def fo_options_default_envelope_bandwidth_pct() -> float:
    """F&O envelope slider default (% each side); clamped to UI min/max."""
    bw = round(100.0 * fno_envelope_decimal_per_side(), 6)
    return float(min(FO_ENVELOPE_BANDWIDTH_MAX_PCT, max(FO_ENVELOPE_BANDWIDTH_MIN_PCT, bw)))


def intraday_envelope_decimal() -> float:
    """Strategies / All-10 / trade_engine: EMA envelope half-width (default 0.003 = 0.3% each side)."""
    raw = (os.environ.get("INTRADAY_ENVELOPE_DECIMAL") or "").strip()
    if raw:
        return float(raw)
    return float(_DEFAULT_INTRADAY_ENVELOPE_DECIMAL)


def option_target_premium_fraction() -> float:
    """
    Long premium target as fraction above entry (0.25 = +25%).
    ``MOCK_ENGINE_OPTION_TARGET_PCT`` when set is whole-number percent (25 → 0.25).
    Else ``FO_OPTION_TARGET_PCT`` as decimal in env (default 0.25).
    """
    raw_mock = (os.environ.get("MOCK_ENGINE_OPTION_TARGET_PCT") or "").strip()
    if raw_mock:
        try:
            pct = float(raw_mock)
            pct = max(0.1, min(500.0, pct))
            return pct / 100.0
        except ValueError:
            pass
    raw_fo = (os.environ.get("FO_OPTION_TARGET_PCT") or "0.25").strip()
    try:
        return float(raw_fo)
    except ValueError:
        return 0.25


def option_stop_premium_fraction() -> float:
    """
    Long premium stop as fraction below entry (0.10 = −10%; 0 disables stop for exit rules that check > 0).
    ``MOCK_ENGINE_OPTION_STOP_PCT`` or ``MOCK_ENGINE_STOP_LOSS_CLAMP_PCT`` when set: whole-number percent.
    Else ``FO_OPTION_STOP_LOSS_PCT`` as decimal (default 0.10).
    """
    raw_opt = (os.environ.get("MOCK_ENGINE_OPTION_STOP_PCT") or "").strip()
    raw = raw_opt if raw_opt else (os.environ.get("MOCK_ENGINE_STOP_LOSS_CLAMP_PCT") or "").strip()
    if raw:
        try:
            pct = float(raw)
            pct = max(5.0, min(90.0, pct))
            return pct / 100.0
        except ValueError:
            pass
    raw_fo = (os.environ.get("FO_OPTION_STOP_LOSS_PCT") or "0.10").strip()
    try:
        return float(raw_fo)
    except ValueError:
        return 0.10


def fo_options_default_option_target_pct_ui() -> float:
    """F&O target slider default (% above entry)."""
    raw = (os.environ.get("MOCK_ENGINE_OPTION_TARGET_PCT") or "").strip()
    if raw:
        try:
            v = float(raw)
            return float(min(200.0, max(0.5, v)))
        except ValueError:
            pass
    return float(min(200.0, max(0.5, round(100.0 * option_target_premium_fraction(), 2))))


def fo_options_default_option_stop_loss_pct_ui() -> float:
    """F&O stop slider default (% below entry)."""
    raw_opt = (os.environ.get("MOCK_ENGINE_OPTION_STOP_PCT") or "").strip()
    raw = raw_opt if raw_opt else (os.environ.get("MOCK_ENGINE_STOP_LOSS_CLAMP_PCT") or "").strip()
    if raw:
        try:
            v = float(raw)
            return float(min(50.0, max(0.0, v)))
        except ValueError:
            pass
    return float(min(50.0, max(0.0, round(100.0 * option_stop_premium_fraction(), 2))))


def mock_engine_option_stop_multiplier() -> float:
    """Long premium: stored stop = entry × multiplier (percent below entry)."""
    return max(0.05, 1.0 - option_stop_premium_fraction())


def mock_engine_option_target_multiplier() -> float:
    """Long premium: stored target = entry × multiplier (percent above entry)."""
    return 1.0 + option_target_premium_fraction()


def mock_engine_stop_loss_floor_multiplier() -> float:
    """Alias for :func:`mock_engine_option_stop_multiplier`."""
    return mock_engine_option_stop_multiplier()


def mock_engine_breakout_clear_pct() -> float:
    """
    Mock envelope signal: require close to exceed the band by at least this fraction of band level
    (e.g. 0.0002 = 0.02% above upper for bullish). ``0`` disables (fresh touch/cross only).
    """
    raw = (os.environ.get("MOCK_ENGINE_BREAKOUT_CLEAR_PCT") or "").strip()
    if raw:
        try:
            v = float(raw)
            return float(max(0.0, min(0.05, v)))
        except ValueError:
            pass
    return float(_DEFAULT_MOCK_ENGINE_BREAKOUT_CLEAR_PCT)
