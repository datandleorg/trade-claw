"""Browse saved F&O Options JSON snapshots: filters, pagination, detailed trade + charts."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import ENVELOPE_EMA_PERIOD
from trade_claw.fo_options_persist import list_fo_options_snapshot_paths, load_fo_options_snapshot
from trade_claw.pl_style import pl_title_color
from trade_claw.strategies import add_ma_ema_line_traces, add_ma_envelope_line_traces


def _parse_session_date(params: dict[str, Any]) -> date | None:
    raw = params.get("session_date")
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError):
        return None


def _parse_saved_at(saved_at: str | None) -> datetime | None:
    if not saved_at:
        return None
    try:
        return datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
    except ValueError:
        return None


def _flatten_snapshots() -> list[dict[str, Any]]:
    """One record per (file × run index)."""
    records: list[dict[str, Any]] = []
    for p in list_fo_options_snapshot_paths():
        try:
            data = load_fo_options_snapshot(p)
        except (OSError, json.JSONDecodeError, TypeError):
            continue

        params = data.get("params") or {}
        metrics = data.get("metrics") or {}
        runs = data.get("runs")
        if not isinstance(runs, list) or not runs:
            continue
        for i, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            records.append(
                {
                    "source_path": str(p),
                    "source_name": p.name,
                    "saved_at_utc": data.get("saved_at_utc"),
                    "params_fingerprint": data.get("params_fingerprint"),
                    "schema_version": data.get("schema_version"),
                    "params": params,
                    "metrics": metrics,
                    "run": run,
                    "run_index": i,
                }
            )
    return records


def _records_to_df(records: list[dict] | None) -> pd.DataFrame | None:
    if not records:
        return None
    df = pd.DataFrame(records)
    if df.empty:
        return None
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df


def _run_display_dict(run: dict[str, Any]) -> dict[str, Any]:
    """Run metadata without embedded OHLC lists (for JSON / table)."""
    skip = {"df_u", "df_o"}
    out = {k: v for k, v in run.items() if k not in skip}
    return out


def _trade_detail(run: dict[str, Any]) -> dict[str, Any]:
    t = run.get("trade")
    if isinstance(t, dict):
        return dict(t)
    keys = [
        "Strategy",
        "Signal",
        "Leg",
        "Option",
        "Strike",
        "Strike pick",
        "Lots",
        "Lot size",
        "Qty",
        "Entry",
        "Target prem.",
        "Stop prem.",
        "Closed at",
        "Exit",
        "Txn cost",
        "P/L gross",
        "P/L",
        "Value",
        "entry_bar_idx",
        "exit_bar_idx",
        "opt_entry_idx",
        "why",
        "chain_note",
        "agent_rationale",
    ]
    return {k: run[k] for k in keys if k in run}


def _f_money(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        v = float(x)
        return f"₹{v:,.2f}"
    except (TypeError, ValueError):
        return str(x) if x is not None else "—"


def _f_strike(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "—"
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return str(x) if x is not None else "—"


def _f_int(x: Any) -> str:
    try:
        if x is None:
            return "—"
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x) if x is not None else "—"


def _td_get(td: dict[str, Any], run: dict[str, Any], key: str) -> Any:
    """Prefer nested `trade` fields, fall back to flat run row."""
    if key in td and td.get(key) is not None:
        return td.get(key)
    return run.get(key)


def _trade_has_position(td: dict[str, Any], run: dict[str, Any]) -> bool:
    try:
        if int(_td_get(td, run, "Lots") or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    opt = str(_td_get(td, run, "Option") or "").strip()
    return bool(opt and opt not in ("—", ""))


def _render_params_summary(p: dict[str, Any]) -> None:
    st.markdown("##### Run configuration")
    is_env = bool(p.get("strategy_is_envelope"))
    rows = [
        {"Parameter": "Session date", "Value": str(p.get("session_date", "—"))},
        {"Parameter": "Interval", "Value": str(p.get("chosen_interval", "—"))},
        {"Parameter": "Strategy", "Value": str(p.get("strategy_choice", "—"))},
    ]
    if is_env:
        rows.append(
            {"Parameter": "Envelope % (each side)", "Value": f"{100 * float(p.get('envelope_pct') or 0):.3f}%"}
        )
        rows.append({"Parameter": "EMA period (envelope)", "Value": str(p.get("envelope_ema_period", "—"))})
    else:
        rows.append({"Parameter": "MA EMA fast / slow", "Value": f"{p.get('ma_ema_fast', '—')} / {p.get('ma_ema_slow', '—')}"})
    rows.extend(
        [
            {"Parameter": "Target % (premium)", "Value": f"{100 * float(p.get('option_target_pct') or 0):.2f}%"},
            {"Parameter": "Stop loss % (premium)", "Value": f"{100 * float(p.get('option_stop_loss_pct') or 0):.2f}%"},
            {"Parameter": "Strike policy", "Value": str(p.get("strike_policy_label", "—"))},
            {"Parameter": "Manual strike", "Value": str(p.get("manual_strike_val") or "—")},
            {"Parameter": "Brokerage / lot (RT)", "Value": _f_money(p.get("brokerage_per_lot_rt"))},
            {"Parameter": "Taxes / lot (RT)", "Value": _f_money(p.get("taxes_per_lot_rt"))},
        ]
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Full params JSON"):
        st.json(p)


def _render_trade_detail_view(td: dict[str, Any], run: dict[str, Any]) -> None:
    """Readable trade layout: status, metrics, instrument table, narrative, optional raw JSON."""
    has_pos = _trade_has_position(td, run)
    note = str(run.get("Note") or "—")
    closed = str(_td_get(td, run, "Closed at") or "—")

    if has_pos:
        try:
            pl_net = float(_td_get(td, run, "P/L") or 0.0)
        except (TypeError, ValueError):
            pl_net = 0.0
        if pl_net > 0:
            st.success(f"**Mock position closed** · {closed} · net P/L **{_f_money(pl_net)}**")
        elif pl_net < 0:
            st.error(f"**Mock position closed** · {closed} · net P/L **{_f_money(pl_net)}**")
        else:
            st.info(f"**Mock position closed** · {closed} · net P/L **{_f_money(pl_net)}**")
    else:
        st.warning("**No option trade executed** for this session (see note below).")

    st.markdown("##### Position & instrument")
    inst_rows = [
        {"Field": "Underlying run", "Value": str(run.get("Underlying", "—"))},
        {"Field": "Strategy", "Value": str(_td_get(td, run, "Strategy") or "—")},
        {"Field": "Signal (spot)", "Value": str(_td_get(td, run, "Signal") or "—")},
        {"Field": "Leg", "Value": str(_td_get(td, run, "Leg") or "—")},
        {"Field": "Option symbol", "Value": str(_td_get(td, run, "Option") or "—")},
        {"Field": "Strike", "Value": _f_strike(_td_get(td, run, "Strike"))},
        {"Field": "Strike pick", "Value": str(_td_get(td, run, "Strike pick") or "—")},
        {"Field": "Lots", "Value": _f_int(_td_get(td, run, "Lots"))},
        {"Field": "Lot size", "Value": _f_int(_td_get(td, run, "Lot size"))},
        {"Field": "Qty", "Value": _f_int(_td_get(td, run, "Qty"))},
    ]
    st.dataframe(pd.DataFrame(inst_rows), use_container_width=True, hide_index=True)

    st.markdown("##### Premiums, exit & P/L")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Entry premium", _f_money(_td_get(td, run, "Entry")))
        st.metric("Exit", _f_money(_td_get(td, run, "Exit")))
    with m2:
        st.metric("Target premium", _f_money(_td_get(td, run, "Target prem.")))
        st.metric("Stop premium", _f_money(_td_get(td, run, "Stop prem.")))
    with m3:
        st.metric("P/L (net)", _f_money(_td_get(td, run, "P/L")))
        st.metric("P/L (gross)", _f_money(_td_get(td, run, "P/L gross")))
    with m4:
        st.metric("Txn cost", _f_money(_td_get(td, run, "Txn cost")))
        st.metric("Notional (entry × qty)", _f_money(_td_get(td, run, "Value")))

    st.markdown("##### Bar indices (underlying vs option series)")
    idx_rows = [
        {
            "entry_bar_idx (underlying)": _td_get(td, run, "entry_bar_idx") if _td_get(td, run, "entry_bar_idx") is not None else "—",
            "opt_entry_idx (option)": _td_get(td, run, "opt_entry_idx") if _td_get(td, run, "opt_entry_idx") is not None else "—",
            "exit_bar_idx (option)": _td_get(td, run, "exit_bar_idx") if _td_get(td, run, "exit_bar_idx") is not None else "—",
        }
    ]
    st.dataframe(pd.DataFrame(idx_rows), use_container_width=True, hide_index=True)

    st.markdown("##### Session note")
    st.markdown(f"> {note}" if note != "—" else "*No row note.*")

    why = _td_get(td, run, "why")
    if why:
        st.markdown("##### Strategy text (`why`)")
        st.markdown(str(why)[:8000] + ("…" if len(str(why)) > 8000 else ""))

    chain = _td_get(td, run, "chain_note")
    if chain:
        st.markdown("##### Chain / data note")
        st.caption(str(chain))

    ar = _td_get(td, run, "agent_rationale")
    if ar:
        st.markdown("##### Agent rationale")
        st.markdown(str(ar)[:6000] + ("…" if len(str(ar)) > 6000 else ""))

    with st.expander("Raw trade + row JSON (debug)"):
        st.json({"trade_detail": td, "run_row_no_ohlc": _run_display_dict(run)})


def render_fo_options_snapshots_report(kite) -> None:
    _ = kite  # login required by app shell; snapshots are local files only

    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("F&O Options", key="snap_nav_fo"):
            st.session_state.view = "fo_options"
            st.session_state.selected_symbol = None
            st.rerun()
    with n2:
        if st.button("Intraday home", key="snap_nav_all10"):
            st.session_state.view = "all10"
            st.session_state.selected_symbol = None
            st.rerun()
    with n3:
        if st.button("Stock list", key="snap_nav_dash"):
            st.session_state.view = "dashboard"
            st.session_state.selected_symbol = None
            st.rerun()

    st.title("F&O options — saved snapshots report")
    st.caption(
        "Reads JSON files from **`data/fo_options_runs/`** (or **`FO_OPTIONS_RUNS_DIR`**). "
        "Use filters and pagination to review each saved run in detail."
    )

    all_records = _flatten_snapshots()
    if not all_records:
        st.warning("No snapshot JSON files found. Generate some from **F&O Options** (change parameters to save).")
        return

    underlyings = sorted(
        {str(r["params"].get("underlying", "")).strip() for r in all_records if r["params"].get("underlying")}
    )
    dates_sess = [_parse_session_date(r["params"]) for r in all_records]
    dates_sess = [d for d in dates_sess if d is not None]
    d_min = min(dates_sess) if dates_sess else date.today()
    d_max = max(dates_sess) if dates_sess else date.today()
    cal_min = date(2015, 1, 1)
    cal_max = date.today()

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
    with f1:
        date_from = st.date_input(
            "Session date from",
            value=d_min,
            min_value=cal_min,
            max_value=cal_max,
            key="fo_snap_d0",
        )
    with f2:
        date_to = st.date_input(
            "Session date to",
            value=d_max,
            min_value=cal_min,
            max_value=cal_max,
            key="fo_snap_d1",
        )
    with f3:
        script = st.selectbox(
            "Script (underlying)",
            options=["All"] + underlyings,
            index=0,
            key="fo_snap_script",
        )
    with f4:
        page_size = st.selectbox("Per page", options=[3, 5, 10, 20], index=1, key="fo_snap_ps")

    if date_from > date_to:
        st.error("**From** date must be ≤ **To** date.")
        return

    filtered: list[dict[str, Any]] = []
    for r in all_records:
        p = r["params"]
        u = str(p.get("underlying", "")).strip()
        if script != "All" and u != script:
            continue
        sd = _parse_session_date(p)
        if sd is None:
            continue
        if not (date_from <= sd <= date_to):
            continue
        filtered.append(r)

    def _sort_key(x: dict[str, Any]) -> datetime:
        sa = _parse_saved_at(x.get("saved_at_utc"))
        if sa is not None and sa.tzinfo is None:
            sa = sa.replace(tzinfo=timezone.utc)
        return sa or datetime.min.replace(tzinfo=timezone.utc)

    filtered.sort(key=_sort_key, reverse=True)

    filter_key = (str(date_from), str(date_to), script, int(page_size))
    if st.session_state.get("fo_snap_filter_key") != filter_key:
        st.session_state.fo_snap_filter_key = filter_key
        st.session_state.fo_snap_page = 1

    n = len(filtered)
    total_pages = max(1, (n + page_size - 1) // page_size)
    if st.session_state.get("fo_snap_page", 1) > total_pages:
        st.session_state.fo_snap_page = total_pages
    if st.session_state.get("fo_snap_page", 1) < 1:
        st.session_state.fo_snap_page = 1
    page = int(st.session_state.fo_snap_page)

    st.markdown(f"**{n}** snapshot run(s) match · **{total_pages}** page(s)")

    pg1, pg2, pg3, pg4 = st.columns([1, 1, 1, 4])
    with pg1:
        if st.button("⏮ First", disabled=page <= 1, key="fo_snap_first"):
            st.session_state.fo_snap_page = 1
            st.rerun()
    with pg2:
        if st.button("◀ Prev", disabled=page <= 1, key="fo_snap_prev"):
            st.session_state.fo_snap_page = page - 1
            st.rerun()
    with pg3:
        if st.button("Next ▶", disabled=page >= total_pages, key="fo_snap_next"):
            st.session_state.fo_snap_page = page + 1
            st.rerun()
    with pg4:
        if st.button("⏭ Last", disabled=page >= total_pages, key="fo_snap_last"):
            st.session_state.fo_snap_page = total_pages
            st.rerun()

    start = (page - 1) * page_size
    page_items = filtered[start : start + page_size]

    for idx, rec in enumerate(page_items):
        p = rec["params"]
        m = rec["metrics"]
        run = rec["run"]
        sd = _parse_session_date(p)
        sd_s = sd.isoformat() if sd else "—"
        title = f"`{rec['source_name']}` · **{p.get('underlying', '—')}** · session **{sd_s}**"
        with st.expander(title, expanded=(len(page_items) == 1)):
            st.caption(f"Saved UTC: `{rec.get('saved_at_utc', '—')}` · fingerprint `{rec.get('params_fingerprint', '—')}`")

            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            with mc1:
                st.metric("Trades", m.get("total_trades", "—"))
            with mc2:
                st.metric("Target exits", m.get("target_exits", "—"))
            with mc3:
                st.metric("Stop exits", m.get("stop_exits", "—"))
            with mc4:
                st.metric("EOD exits", m.get("eod_exits", "—"))
            with mc5:
                st.metric("Total P/L (net)", f"₹{float(m.get('total_pl_net', 0) or 0):+,.0f}")
            with mc6:
                st.metric("Txn cost", f"₹{float(m.get('total_txn_cost', 0) or 0):,.0f}")

            td = _trade_detail(run)
            ctx = {k: v for k, v in run.items() if k not in ("df_u", "df_o", "trade")}
            ctx.update(td)

            t1, t2, t3 = st.tabs(["Parameters", "Trade detail", "Charts"])
            with t1:
                _render_params_summary(p)
                with st.expander("Raw metrics JSON"):
                    st.json(m)
            with t2:
                _render_trade_detail_view(td, run)
            with t3:
                df_u = _records_to_df(run.get("df_u") if isinstance(run.get("df_u"), list) else None)
                df_o = _records_to_df(run.get("df_o") if isinstance(run.get("df_o"), list) else None)
                env = bool(p.get("strategy_is_envelope"))
                epct = float(p.get("envelope_pct") or 0.0)

                cleft, cright = st.columns(2)
                with cleft:
                    if df_u is not None and not df_u.empty:
                        st.markdown("**Underlying**")
                        fig_u = go.Figure()
                        fig_u.add_trace(
                            go.Candlestick(
                                x=df_u["date"],
                                open=df_u["open"],
                                high=df_u["high"],
                                low=df_u["low"],
                                close=df_u["close"],
                                name="Underlying",
                            )
                        )
                        if env:
                            add_ma_envelope_line_traces(fig_u, df_u, ema_period=ENVELOPE_EMA_PERIOD, pct=epct)
                            utitle = "Underlying + envelope"
                        else:
                            add_ma_ema_line_traces(fig_u, df_u)
                            utitle = f"Underlying (EMA {p.get('ma_ema_fast')}/{p.get('ma_ema_slow')})"
                        ei = ctx.get("entry_bar_idx")
                        if ei is not None and str(ctx.get("Signal", "")).strip() in ("BUY", "SELL"):
                            try:
                                ei = int(ei)
                                sig_m = str(ctx.get("Signal"))
                                du = df_u["date"]
                                if 0 <= ei < len(du):
                                    fig_u.add_trace(
                                        go.Scatter(
                                            x=[du.iloc[ei]],
                                            y=[float(df_u.iloc[ei]["close"])],
                                            mode="markers",
                                            marker=dict(
                                                symbol="triangle-up" if sig_m == "BUY" else "triangle-down",
                                                size=12,
                                                color="lime" if sig_m == "BUY" else "tomato",
                                                line=dict(width=1, color="black"),
                                            ),
                                            name=sig_m,
                                        )
                                    )
                            except (TypeError, ValueError, IndexError):
                                pass
                        fig_u.update_layout(title=utitle, height=360, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig_u, use_container_width=True)
                    else:
                        st.info("No underlying OHLC in snapshot.")

                with cright:
                    if df_o is not None and not df_o.empty:
                        st.markdown("**Option premium**")
                        opt_name = str(ctx.get("Option") or "Option")
                        fig_o = go.Figure()
                        fig_o.add_trace(
                            go.Candlestick(
                                x=df_o["date"],
                                open=df_o["open"],
                                high=df_o["high"],
                                low=df_o["low"],
                                close=df_o["close"],
                                name=opt_name[:40],
                            )
                        )
                        oei = ctx.get("opt_entry_idx")
                        oxi = ctx.get("exit_bar_idx")
                        ddt = df_o["date"]
                        ent = ctx.get("Entry")
                        exv = ctx.get("Exit")
                        if oei is not None and ent is not None:
                            try:
                                oei = int(oei)
                                if 0 <= oei < len(ddt):
                                    fig_o.add_trace(
                                        go.Scatter(
                                            x=[ddt.iloc[oei]],
                                            y=[float(ent)],
                                            mode="markers",
                                            marker=dict(symbol="triangle-up", size=11, color="cyan", line=dict(width=1, color="black")),
                                            name="Entry",
                                        )
                                    )
                            except (TypeError, ValueError, IndexError):
                                pass
                        if oxi is not None and exv is not None:
                            try:
                                oxi = int(oxi)
                                if 0 <= oxi < len(ddt):
                                    fig_o.add_trace(
                                        go.Scatter(
                                            x=[ddt.iloc[oxi]],
                                            y=[float(exv)],
                                            mode="markers",
                                            marker=dict(symbol="diamond", size=10, color="gold", line=dict(width=1, color="orange")),
                                            name="Exit",
                                        )
                                    )
                            except (TypeError, ValueError, IndexError):
                                pass
                        tp = ctx.get("Target prem.")
                        sp = ctx.get("Stop prem.")
                        if tp is not None:
                            try:
                                fig_o.add_hline(
                                    y=float(tp),
                                    line_dash="dash",
                                    line_color="rgba(34,197,94,0.85)",
                                    annotation_text="Target",
                                    annotation_position="right",
                                )
                            except (TypeError, ValueError):
                                pass
                        if sp is not None:
                            try:
                                fig_o.add_hline(
                                    y=float(sp),
                                    line_dash="dash",
                                    line_color="rgba(239,68,68,0.85)",
                                    annotation_text="Stop",
                                    annotation_position="right",
                                )
                            except (TypeError, ValueError):
                                pass
                        try:
                            net = float(ctx.get("P/L") or 0.0)
                        except (TypeError, ValueError):
                            net = 0.0
                        fig_o.update_layout(
                            title=dict(
                                text=f"{opt_name[:48]} · Net ₹{net:+,.2f} · {ctx.get('Closed at', '—')}",
                                font=dict(color=pl_title_color(net), size=12),
                            ),
                            height=360,
                            xaxis_rangeslider_visible=False,
                        )
                        st.plotly_chart(fig_o, use_container_width=True)
                    else:
                        st.info("No option OHLC in snapshot (no trade / no option series).")

        if idx < len(page_items) - 1:
            st.divider()
