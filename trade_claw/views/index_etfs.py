"""NSE index ETFs — buy-only Institutional floor (50/200 SMA) dashboard."""
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_claw.constants import (
    INSTITUTIONAL_AGGRESSIVE_BUY,
    INSTITUTIONAL_EXTENDED_ABOVE_PCT,
    INSTITUTIONAL_STANDARD_BUY,
    NSE_EXCHANGE,
    NSE_INDEX_ETF_SYMBOLS,
)
from trade_claw.institutional_floor import analyze_institutional_floor
from trade_claw.market_data import candles_to_dataframe, get_instrument_token


def render_index_etfs(kite):
    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load instruments: {e}")
                st.stop()
    instruments = st.session_state.nse_instruments
    symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in instruments}

    st.title("Index ETFs (NSE) — Institutional floor")
    st.markdown(
        """
**Buy-only sizing** using **200-day SMA** (long-term trend / “floor”) and **50-day SMA** (golden-cross context).

| Condition | Suggested size | Logic |
| :--- | :--- | :--- |
| Price **at or below** 200-day SMA | **₹{:,}** (aggressive) | Statistically cheaper vs ~1 year of history |
| Price **above** 200-day SMA | **₹{:,}** (standard) | Regular accumulation |
| Price **≥ {:.0f}% above** 200-day SMA | Still standard | Avoid chasing; keep plan |
| **50 > 200** | Structure bullish | Ongoing “golden cross” regime |
| **50 < 200** | Cautious for lump sums | Prefer DCA until 50 crosses above 200 |

*Not financial advice. Indices/ETFs only; verify symbols exist in your broker.*
        """.format(
            INSTITUTIONAL_AGGRESSIVE_BUY,
            INSTITUTIONAL_STANDARD_BUY,
            INSTITUTIONAL_EXTENDED_ABOVE_PCT,
        )
    )

    n1, n2, n3, n4, n5 = st.columns(5)
    with n1:
        if st.button("Home (intraday)", key="etf_nav_all10"):
            st.session_state.view = "all10"
            st.rerun()
    with n2:
        if st.button("Reports", key="etf_nav_rep"):
            st.session_state.view = "reports"
            st.session_state.selected_symbol = None
            st.rerun()
    with n3:
        if st.button("Stock list", key="etf_nav_dash"):
            st.session_state.view = "dashboard"
            st.session_state.selected_symbol = None
            st.rerun()
    with n4:
        if st.button("F&O Options", key="etf_nav_fo"):
            st.session_state.view = "fo_options"
            st.session_state.selected_symbol = None
            st.rerun()
    with n5:
        if st.button("F&O Agent", key="etf_nav_fo_agent"):
            st.session_state.view = "fo_agent"
            st.session_state.selected_symbol = None
            st.rerun()

    lookback_days = st.slider("Daily history (calendar days)", min_value=300, max_value=800, value=400, step=50)
    selected = st.multiselect(
        "ETFs to analyse",
        options=NSE_INDEX_ETF_SYMBOLS,
        default=NSE_INDEX_ETF_SYMBOLS,
        key="etf_symbols",
    )
    symbols = selected if selected else NSE_INDEX_ETF_SYMBOLS

    to_d = datetime.now().date()
    from_d = to_d - timedelta(days=lookback_days)
    from_dt = datetime(from_d.year, from_d.month, from_d.day, 0, 0, 0)
    to_dt = datetime(to_d.year, to_d.month, to_d.day, 23, 59, 59)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:%S")

    run = st.button("Load & analyse", type="primary", key="etf_run")
    if not run and "etf_cache_rows" not in st.session_state:
        st.info("Pick ETFs and click **Load & analyse** (needs ~200+ trading days for SMA200).")
        return

    if run:
        rows = []
        charts = {}
        with st.spinner("Fetching daily data…"):
            for symbol in symbols:
                token = get_instrument_token(symbol, instruments)
                if token is None:
                    rows.append({
                        "Symbol": symbol,
                        "Name": symbol_to_name.get(symbol, symbol),
                        "Status": "Not found on NSE",
                        "Close": None,
                        "% vs SMA200": None,
                        "SMA50": None,
                        "SMA200": None,
                        "Suggestion": "—",
                        "Amount": None,
                        "Golden 50>200": None,
                        "Cross (20d)": None,
                    })
                    continue
                try:
                    candles = kite.historical_data(token, from_str, to_str, interval="day")
                except Exception as e:
                    rows.append({
                        "Symbol": symbol,
                        "Name": symbol_to_name.get(symbol, symbol),
                        "Status": str(e)[:80],
                        "Close": None,
                        "% vs SMA200": None,
                        "SMA50": None,
                        "SMA200": None,
                        "Suggestion": "—",
                        "Amount": None,
                        "Golden 50>200": None,
                        "Cross (20d)": None,
                    })
                    continue
                df = candles_to_dataframe(candles)
                if df.empty:
                    rows.append({
                        "Symbol": symbol,
                        "Name": symbol_to_name.get(symbol, symbol),
                        "Status": "No data",
                        "Close": None,
                        "% vs SMA200": None,
                        "SMA50": None,
                        "SMA200": None,
                        "Suggestion": "—",
                        "Amount": None,
                        "Golden 50>200": None,
                        "Cross (20d)": None,
                    })
                    continue
                df = df.sort_values("date").reset_index(drop=True)
                res = analyze_institutional_floor(df)
                name = symbol_to_name.get(symbol, symbol)
                if not res["ok"]:
                    rows.append({
                        "Symbol": symbol,
                        "Name": name,
                        "Status": res.get("error") or "Error",
                        "Close": None,
                        "% vs SMA200": None,
                        "SMA50": None,
                        "SMA200": None,
                        "Suggestion": "—",
                        "Amount": None,
                        "Golden 50>200": None,
                        "Cross (20d)": None,
                    })
                    continue
                rows.append({
                    "Symbol": symbol,
                    "Name": name,
                    "Status": "OK",
                    "Close": res["close"],
                    "% vs SMA200": res["pct_vs_200"],
                    "SMA50": res["sma50"],
                    "SMA200": res["sma200"],
                    "Suggestion": res["recommendation"],
                    "Amount": res["amount"],
                    "Golden 50>200": "Yes" if res["golden"] else "No",
                    "Cross (20d)": "Yes" if res["recent_golden_cross"] else "No",
                })
                charts[symbol] = {"df": df, "res": res}

        st.session_state.etf_cache_rows = rows
        st.session_state.etf_cache_charts = charts
        st.rerun()

    rows = st.session_state.get("etf_cache_rows") or []
    charts = st.session_state.get("etf_cache_charts") or {}

    if not rows:
        return

    summary = pd.DataFrame(rows)
    ok_df = summary[summary["Status"] == "OK"]
    n_ok = len(ok_df)
    n_agg = int((ok_df["Suggestion"] == "Aggressive buy").sum()) if n_ok else 0
    st.metric("ETFs with full SMA data", n_ok)
    st.caption(f"In **aggressive** (value) zone now: **{n_agg}** of {n_ok} (price at/below 200-day SMA).")

    show = summary.drop(columns=["Status"], errors="ignore") if "Status" in summary.columns else summary
    st.dataframe(
        show.style.format({
            "Close": "₹{:,.2f}",
            "% vs SMA200": "{:+.1f}%",
            "SMA50": "₹{:,.2f}",
            "SMA200": "₹{:,.2f}",
            "Amount": "₹{:,.0f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Charts & notes")
    for symbol in symbols:
        if symbol not in charts:
            continue
        pack = charts[symbol]
        df = pack["df"]
        res = pack["res"]
        if not res.get("ok"):
            continue
        name = symbol_to_name.get(symbol, symbol)
        with st.expander(f"**{symbol}** — {name}", expanded=False):
            st.caption(res["detail"])
            st.caption(res["structure_note"])
            if res.get("recent_golden_cross") and res.get("golden"):
                st.success("50 crossed above 200 recently (within ~20 sessions) and 50 still above 200.")
            fig = go.Figure()
            x = df["date"]
            fig.add_trace(go.Scatter(x=x, y=df["close"], name="Close", line=dict(color="#1e293b", width=1)))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=res["sma50_series"],
                    name="SMA 50",
                    line=dict(color="#2563eb", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=res["sma200_series"],
                    name="SMA 200",
                    line=dict(color="#ea580c", width=2),
                )
            )
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"**Last close:** ₹{res['close']:,.2f} · **SMA50:** ₹{res['sma50']:,.2f} · "
                f"**SMA200:** ₹{res['sma200']:,.2f} · **vs 200:** {res['pct_vs_200']:+.1f}%"
            )
            st.markdown(
                f"**Suggested contribution (buy-only):** ₹{res['amount']:,.0f} — *{res['recommendation']}*"
            )
