"""Nifty 50 single-day view: multiselect stocks, strategies, charts, mock trades."""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    ALL10_STRATEGY_OPTIONS,
    ALLOCATED_AMOUNT,
    CLOSED_AT_REALISED,
    DEFAULT_INTERVALS,
    DEFAULT_INTRADAY_INTERVAL,
    ENVELOPE_EMA_PERIOD,
    NIFTY50_DEFAULT_SELECTED,
    NIFTY50_SYMBOLS,
    NSE_EXCHANGE,
)
from trade_claw.env_trading_params import intraday_envelope_decimal
from trade_claw.market_data import candles_to_dataframe, get_instrument_token
from trade_claw.plotly_ohlc import (
    add_candlestick_trace,
    add_volume_bar_trace,
    create_ohlc_volume_figure,
    finalize_ohlc_volume_figure,
)
from trade_claw.pl_style import pl_markdown, pl_title_color, style_pl_dataframe
from trade_claw.strategies import (
    add_ma_ema_line_traces,
    add_ma_envelope_line_traces,
    filter_analyses_by_strategy_choice,
    get_applicable_strategies,
    get_strategy_analyses,
    simulate_envelope_trade_close,
    simulate_trade_close,
)
from trade_claw.trade_engine import build_trade_rows_from_analyses
from trade_claw.views.reports import seed_reports_from_home


def render_all_ten(kite):
    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load instruments: {e}")
                st.stop()
    instruments = st.session_state.nse_instruments
    symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in instruments}

    st.title("Nifty 50 – single day")
    trade_view = st.radio(
        "View trades",
        options=["All trades", "Long only", "Short only"],
        index=1,  # default: Long only
        horizontal=True,
        key="all10_trade_view",
    )
    strategy_for_scrip = st.selectbox(
        "Strategy (each selected stock)",
        options=ALL10_STRATEGY_OPTIONS,
        index=ALL10_STRATEGY_OPTIONS.index("MA (EMA crossover)"),
        key="all10_strategy",
        help="MA = 9/20 EMA cross. Envelope = EMA20 ±0.5% bands: enter on breakout, exit on opposite band cross.",
    )
    selected_stocks = st.multiselect(
        "Stocks to run",
        options=NIFTY50_SYMBOLS,
        default=NIFTY50_DEFAULT_SELECTED,
        key="all10_stocks",
        help="Select stocks to run strategy and trades.",
    )
    intraday_intervals = [i for i in DEFAULT_INTERVALS if i != "day"]
    chosen_date = st.date_input(
        "Date",
        value=datetime(2026, 3, 16).date(),
        key="all10_date",
    )
    _int_idx = (
        intraday_intervals.index(DEFAULT_INTRADAY_INTERVAL)
        if DEFAULT_INTRADAY_INTERVAL in intraday_intervals
        else 0
    )
    chosen_interval = st.selectbox(
        "Minute interval",
        options=intraday_intervals,
        index=_int_idx,
        key="all10_interval",
    )
    st.caption("Changing date, interval or stocks reloads data and reruns strategy analysis.")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("Go to stock list", key="all10_back"):
            st.session_state.view = "dashboard"
            st.session_state.selected_symbol = None
            st.rerun()
    with b2:
        if st.button("Reports (use these settings)", key="all10_reports"):
            seed_reports_from_home(
                chosen_date,
                selected_stocks,
                strategy_for_scrip,
                chosen_interval,
                trade_view,
            )
            st.rerun()
    with b3:
        if st.button("Index ETFs", key="all10_etf"):
            st.session_state.view = "index_etfs"
            st.session_state.selected_symbol = None
            st.rerun()
    with b4:
        if st.button("F&O Options", key="all10_fo"):
            st.session_state.view = "fo_options"
            st.session_state.selected_symbol = None
            st.rerun()
    with b5:
        if st.button("F&O Agent", key="all10_fo_agent"):
            st.session_state.view = "fo_agent"
            st.session_state.selected_symbol = None
            st.rerun()

    from_dt = datetime(chosen_date.year, chosen_date.month, chosen_date.day, 9, 15, 0)
    to_dt = datetime(chosen_date.year, chosen_date.month, chosen_date.day, 15, 30, 0)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

    symbol_results = []
    symbols_to_load = selected_stocks if selected_stocks else NIFTY50_DEFAULT_SELECTED
    with st.spinner(f"Loading data for {len(symbols_to_load)} stock(s)..."):
        for symbol in symbols_to_load:
            name = symbol_to_name.get(symbol, symbol)
            token = get_instrument_token(symbol, instruments)
            if token is None:
                continue
            try:
                candles = kite.historical_data(token, from_str, to_str, interval=chosen_interval)
            except Exception:
                continue
            df = candles_to_dataframe(candles)
            if df.empty:
                continue
            last_close = float(df["close"].iloc[-1])
            strategies = get_applicable_strategies(df, chosen_interval)
            analyses = get_strategy_analyses(df, chosen_interval)
            analyses, strategies = filter_analyses_by_strategy_choice(
                analyses, strategies, strategy_for_scrip
            )
            strategy_label = ", ".join(strategies) if strategies else "—"
            trade_rows = []
            seen_strategy = set()
            if analyses:
                for sname, text, sig in analyses:
                    if sname in seen_strategy:
                        continue
                    entry_bar_idx = sig.get("entry_bar_idx")
                    if sig.get("envelope"):
                        if not sig.get("signal") or entry_bar_idx is None:
                            continue
                        if not (0 <= entry_bar_idx < len(df)):
                            continue
                        entry = float(df.iloc[entry_bar_idx]["close"])
                        if entry <= 0:
                            continue
                        qty = int(ALLOCATED_AMOUNT / entry)
                        if qty < 1:
                            continue
                        closed_at, exit_price, pl, exit_bar_idx = simulate_envelope_trade_close(
                            df,
                            entry_bar_idx,
                            entry,
                            sig["signal"],
                            qty,
                            sig.get("ema_period", ENVELOPE_EMA_PERIOD),
                            sig.get("pct", intraday_envelope_decimal()),
                        )
                    elif sig.get("signal") and sig.get("target") is not None and sig.get("stop") is not None:
                        if entry_bar_idx is not None and 0 <= entry_bar_idx < len(df):
                            entry = float(df.iloc[entry_bar_idx]["close"])
                        else:
                            entry = last_close
                        if entry <= 0:
                            continue
                        qty = int(ALLOCATED_AMOUNT / entry)
                        if qty < 1:
                            continue
                        target = sig["target"]
                        stop = sig["stop"]
                        closed_at, exit_price, pl, exit_bar_idx = simulate_trade_close(
                            df, entry_bar_idx, entry, target, stop, sig["signal"], qty
                        )
                    else:
                        continue
                    value = entry * qty
                    trade_rows.append({
                        "Strategy": sname,
                        "Signal": sig["signal"],
                        "Qty": qty,
                        "Entry": entry,
                        "entry_bar_idx": entry_bar_idx,
                        "exit_bar_idx": exit_bar_idx,
                        "Closed at": closed_at,
                        "Exit": exit_price,
                        "P/L": pl,
                        "Value": value,
                    })
                    seen_strategy.add(sname)
            symbol_results.append({
                "symbol": symbol,
                "name": name,
                "df": df,
                "strategies": strategies,
                "strategy_label": strategy_label,
                "analyses": analyses,
                "trade_rows": trade_rows,
                "last_close": last_close,
                "day_high": df["high"].max(),
                "day_low": df["low"].min(),
                "day_vol": df["volume"].sum(),
            })

    def _matches_view(t: dict) -> bool:
        s = t.get("Signal")
        if trade_view == "Long only":
            return s == "BUY"
        if trade_view == "Short only":
            return s == "SELL"
        return True

    def _analysis_matches_view(sig: dict) -> bool:
        s = sig.get("signal")
        if trade_view == "Long only":
            return s == "BUY"
        if trade_view == "Short only":
            return s == "SELL"
        return True

    filtered_results = [
        {**r, "trade_rows": [t for t in r["trade_rows"] if _matches_view(t)]}
        for r in symbol_results
    ]
    total_trades = sum(len(r["trade_rows"]) for r in filtered_results)
    total_finished = sum(
        1 for r in filtered_results for t in r["trade_rows"]
        if t.get("Closed at") in CLOSED_AT_REALISED
    )
    realised_pl = sum(
        t["P/L"] for r in filtered_results for t in r["trade_rows"]
        if t.get("Closed at") in CLOSED_AT_REALISED
    )
    unrealised_pl = sum(
        t["P/L"] for r in filtered_results for t in r["trade_rows"]
        if t.get("Closed at") == "EOD"
    )
    total_pl = realised_pl + unrealised_pl
    total_traded_value = sum(t["Value"] for r in filtered_results for t in r["trade_rows"])

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total trades", total_trades)
    m2.metric("Total finished trades", total_finished)
    with m3:
        st.markdown(f"**Realised P/L**  \n{pl_markdown(realised_pl)}")
    with m4:
        st.markdown(f"**Unrealised P/L**  \n{pl_markdown(unrealised_pl)}")
    with m5:
        st.markdown(f"**Total P/L**  \n{pl_markdown(total_pl)}")
    m6.metric("Total traded value", f"₹{total_traded_value:,.2f}")
    st.caption(
        "Up to ₹10,000 per trade · Total traded value = sum of actual value (entry × qty). "
        "Strategies: ORB, VWAP, RSI, Flag, MA (9/20 EMA), Envelope (EMA20 ±0.5%)."
    )
    st.divider()

    for r in filtered_results:
        symbol = r["symbol"]
        name = r["name"]
        df = r["df"]
        strategy_label = r["strategy_label"]
        analyses = r["analyses"]
        trade_rows = r["trade_rows"]
        last_close = r["last_close"]
        with st.expander(f"**{symbol}** — {name} · {strategy_label}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Close", f"₹{last_close:,.2f}")
            c2.metric("High", f"₹{r['day_high']:,.2f}")
            c3.metric("Low", f"₹{r['day_low']:,.2f}")
            c4.metric("Volume", f"{r['day_vol']:,.0f}")
            col_left, col_right = st.columns([2, 1])
            with col_left:
                if trade_rows:
                    dates = df["date"]
                    for _idx, t in enumerate(trade_rows):
                        fig, pr, vr = create_ohlc_volume_figure(df)
                        add_candlestick_trace(fig, df, name="OHLC", price_row=pr, volume_row=vr)
                        if t.get("Strategy") == "MA":
                            add_ma_ema_line_traces(
                                fig,
                                df,
                                row=pr if vr is not None else None,
                                col=1 if vr is not None else None,
                            )
                        elif t.get("Strategy") == "Envelope":
                            add_ma_envelope_line_traces(
                                fig,
                                df,
                                row=pr if vr is not None else None,
                                col=1 if vr is not None else None,
                            )
                        if vr is not None:
                            add_volume_bar_trace(fig, df, volume_row=vr)
                        ei = t.get("entry_bar_idx")
                        exi = t.get("exit_bar_idx")
                        if ei is not None and 0 <= ei < len(dates):
                            sym = "triangle-up" if t.get("Signal") == "BUY" else "triangle-down"
                            color = "lime" if t.get("Signal") == "BUY" else "tomato"
                            _em = dict(
                                x=[dates.iloc[ei]],
                                y=[t["Entry"]],
                                mode="markers",
                                marker=dict(symbol=sym, size=14, color=color, line=dict(width=1, color="black")),
                                name=t["Signal"],
                            )
                            if vr is not None:
                                fig.add_trace(go.Scatter(**_em), row=pr, col=1)
                            else:
                                fig.add_trace(go.Scatter(**_em))
                        if exi is not None and 0 <= exi < len(dates):
                            _xm = dict(
                                x=[dates.iloc[exi]],
                                y=[t["Exit"]],
                                mode="markers",
                                marker=dict(symbol="diamond", size=10, color="gold", line=dict(width=1, color="orange")),
                                name="Exit",
                            )
                            if vr is not None:
                                fig.add_trace(go.Scatter(**_xm), row=pr, col=1)
                            else:
                                fig.add_trace(go.Scatter(**_xm))
                        _pl = float(t["P/L"])
                        finalize_ohlc_volume_figure(fig, height=360)
                        fig.update_layout(
                            title=dict(
                                text=f"{t['Strategy']} – {t['Signal']} · P/L ₹{_pl:+,.2f}",
                                font=dict(color=pl_title_color(_pl), size=14),
                            ),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("No trades match the current view for this stock.")
            with col_right:
                st.markdown("**Why (numbers)**")
                if analyses:
                    for sname, text, sig in analyses:
                        if not _analysis_matches_view(sig):
                            continue
                        st.markdown(f"**{sname}**")
                        st.caption(text)
                        if sig.get("signal") and sig.get("envelope"):
                            entry_bar_idx = sig.get("entry_bar_idx")
                            if entry_bar_idx is not None and 0 <= entry_bar_idx < len(df):
                                entry = float(df.iloc[entry_bar_idx]["close"])
                            else:
                                entry = last_close
                            qty = int(ALLOCATED_AMOUNT / entry) if entry > 0 else 0
                            closed_at, exit_price, pl, _ = simulate_envelope_trade_close(
                                df,
                                entry_bar_idx,
                                entry,
                                sig["signal"],
                                qty,
                                sig.get("ema_period", ENVELOPE_EMA_PERIOD),
                                sig.get("pct", intraday_envelope_decimal()),
                            )
                            ue = sig.get("upper_at_entry", 0)
                            le = sig.get("lower_at_entry", 0)
                            if sig["signal"] == "SELL":
                                st.error(
                                    f"**SELL** · Qty {qty:,.0f} · Break below lower band · "
                                    f"Upper ₹{ue:,.2f} · Lower ₹{le:,.2f}"
                                )
                            else:
                                st.success(
                                    f"**BUY** · Qty {qty:,.0f} · Break above upper band · "
                                    f"Upper ₹{ue:,.2f} · Lower ₹{le:,.2f}"
                                )
                            st.markdown(
                                f"Exit on **opposite** band cross · Closed **{closed_at}** · "
                                f"Exit ₹{exit_price:,.2f} · P/L: {pl_markdown(pl)}"
                            )
                        elif sig.get("signal") and sig.get("target") is not None and sig.get("stop") is not None:
                            entry_bar_idx = sig.get("entry_bar_idx")
                            if entry_bar_idx is not None and 0 <= entry_bar_idx < len(df):
                                entry = float(df.iloc[entry_bar_idx]["close"])
                            else:
                                entry = last_close
                            qty = int(ALLOCATED_AMOUNT / entry) if entry > 0 else 0
                            target = sig["target"]
                            stop = sig["stop"]
                            closed_at, exit_price, pl, _ = simulate_trade_close(
                                df, entry_bar_idx, entry, target, stop, sig["signal"], qty
                            )
                            if sig["signal"] == "SELL":
                                st.error(f"**SELL** · Qty {qty:,.0f} · Target ₹{target:,.2f} · Stop ₹{stop:,.2f}")
                            else:
                                st.success(f"**BUY** · Qty {qty:,.0f} · Target ₹{target:,.2f} · Stop ₹{stop:,.2f}")
                            st.markdown(
                                f"Trade closed at **{closed_at}** · Exit ₹{exit_price:,.2f} · P/L: {pl_markdown(pl)}"
                            )
                else:
                    st.caption("No strategy met the criteria for this data.")

            if trade_rows:
                st.markdown("**Mock trade record**")
                display_cols = ["Strategy", "Signal", "Qty", "Entry", "Closed at", "Exit", "P/L", "Value"]
                trade_df = pd.DataFrame(trade_rows)[display_cols]
                _styled = style_pl_dataframe(
                    trade_df.style.format({
                        "Qty": "{:,.0f}",
                        "Entry": "₹{:,.2f}", "Exit": "₹{:,.2f}", "P/L": "₹{:+,.2f}", "Value": "₹{:,.2f}",
                    }, na_rep="—"),
                    "P/L",
                )
                st.dataframe(_styled, use_container_width=True, hide_index=True)
                _tval = sum(t["Value"] for t in trade_rows)
                _tpl = sum(t["P/L"] for t in trade_rows)
                st.markdown(
                    f"**Total value:** ₹{_tval:,.2f} · **Total P/L:** {pl_markdown(_tpl)} "
                    "(target/stop, opposite envelope, or EOD)."
                )
