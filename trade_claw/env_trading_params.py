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


def _env_int(name: str, default: int = 0) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


# --- Strict clean breakout (0 / unset = disabled unless noted) ---


def mock_engine_breakout_min_body_frac() -> float:
    """Min |close-open| / (high-low) on breakout bar; ``0`` = disabled."""
    v = _env_float("MOCK_ENGINE_BREAKOUT_MIN_BODY_FRAC", 0.0)
    return float(max(0.0, min(1.0, v)))


def mock_engine_breakout_max_lower_wick_frac() -> float:
    """Max (min(open,close)-low) / range on breakout bar; ``0`` = skip check."""
    v = _env_float("MOCK_ENGINE_BREAKOUT_MAX_LOWER_WICK_FRAC", 0.0)
    return float(max(0.0, min(1.0, v)))


def mock_engine_breakout_max_upper_wick_frac() -> float:
    """Max (high-max(open,close)) / range; ``0`` = skip check."""
    v = _env_float("MOCK_ENGINE_BREAKOUT_MAX_UPPER_WICK_FRAC", 0.0)
    return float(max(0.0, min(1.0, v)))


def mock_engine_breakout_range_expand_lookback() -> int:
    """Require range[i_b] > mean(range of prior N bars); ``0`` = off."""
    return max(0, _env_int("MOCK_ENGINE_BREAKOUT_RANGE_EXPAND_LOOKBACK", 0))


def mock_engine_breakout_volume_lookback() -> int:
    """Require volume[i_b] > mean(prior M); ``0`` = off. Skip if all vol zero."""
    return max(0, _env_int("MOCK_ENGINE_BREAKOUT_VOLUME_LOOKBACK", 0))


def mock_engine_breakout_require_directional_body() -> bool:
    """Bull requires close > open; bear requires close < open."""
    return _env_truthy("MOCK_ENGINE_BREAKOUT_REQUIRE_DIRECTIONAL_BODY")


def mock_engine_breakout_require_confirm_bar() -> bool:
    """Breakout on bar len-2, confirmation on bar len-1."""
    return _env_truthy("MOCK_ENGINE_BREAKOUT_REQUIRE_CONFIRM_BAR")


# --- LLM stop / target (option premium rupees; bounds as whole-number %) ---


def mock_llm_risk_instruction() -> str:
    """Free-text appended to LLM prompt for risk style."""
    return (os.environ.get("MOCK_LLM_RISK_INSTRUCTION") or "").strip()


def mock_llm_sltp_target_pct_bounds() -> tuple[float, float]:
    """
    Min/max target **above** entry as decimal fraction (e.g. 0.10 = +10%).
    Env: whole-number percents MOCK_LLM_SLTP_TARGET_PCT_MIN / _MAX.
    If unset, derived from option_target_premium_fraction with slack.
    """
    raw_lo = (os.environ.get("MOCK_LLM_SLTP_TARGET_PCT_MIN") or "").strip()
    raw_hi = (os.environ.get("MOCK_LLM_SLTP_TARGET_PCT_MAX") or "").strip()
    base = option_target_premium_fraction()
    if raw_lo and raw_hi:
        try:
            lo = max(0.01, float(raw_lo) / 100.0)
            hi = max(lo + 0.001, min(5.0, float(raw_hi) / 100.0))
            return lo, hi
        except ValueError:
            pass
    lo = max(0.005, base * 0.25)
    hi = max(lo + 0.01, min(4.0, base * 3.0))
    return lo, hi


def mock_llm_sltp_stop_pct_bounds() -> tuple[float, float]:
    """
    Min/max stop **below** entry as decimal fraction of entry (e.g. 0.10 = 10% below).
    Env: MOCK_LLM_SLTP_STOP_PCT_MIN / _MAX whole-number percents (distance below entry).
    If unset, derived from option_stop_premium_fraction with slack.
    """
    raw_lo = (os.environ.get("MOCK_LLM_SLTP_STOP_PCT_MIN") or "").strip()
    raw_hi = (os.environ.get("MOCK_LLM_SLTP_STOP_PCT_MAX") or "").strip()
    base = option_stop_premium_fraction()
    if raw_lo and raw_hi:
        try:
            lo = max(0.001, float(raw_lo) / 100.0)
            hi = max(lo + 0.001, min(0.95, float(raw_hi) / 100.0))
            return lo, hi
        except ValueError:
            pass
    lo = max(0.005, base * 0.25)
    hi = max(lo + 0.01, min(0.90, base * 2.5))
    return lo, hi


def mock_llm_sltp_fallback_to_env() -> bool:
    """If true, invalid LLM stop/target use env multipliers instead of aborting."""
    return _env_truthy("MOCK_LLM_SLTP_FALLBACK")
