"""Nifty 50 top-10 list with links to stock detail."""
import streamlit as st

from trade_claw.constants import NIFTY50_TOP10, NSE_EXCHANGE


def render_dashboard(kite):
    if st.session_state.nse_instruments is None:
        with st.spinner("Loading NSE instruments..."):
            try:
                st.session_state.nse_instruments = kite.instruments(NSE_EXCHANGE)
            except Exception as e:
                st.error(f"Failed to load instruments: {e}")
                st.stop()
    instruments = st.session_state.nse_instruments
    symbol_to_name = {i["tradingsymbol"]: i.get("name", i["tradingsymbol"]) for i in instruments}
    st.title("Nifty 50 – Top 10")
    st.caption("Click a stock to view historical prices and chart.")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        if st.button("Home (single day)", key="dash_home_all10"):
            st.session_state.view = "all10"
            st.rerun()
    with h2:
        if st.button("Reports (date range)", key="dash_reports"):
            st.session_state.view = "reports"
            st.session_state.selected_symbol = None
            st.session_state.rep_cache_detail = None
            st.rerun()
    with h3:
        if st.button("Index ETFs", key="dash_index_etf"):
            st.session_state.view = "index_etfs"
            st.session_state.selected_symbol = None
            st.rerun()
    with h4:
        if st.button("F&O Options", key="dash_fo"):
            st.session_state.view = "fo_options"
            st.session_state.selected_symbol = None
            st.rerun()
    r2a, r2b = st.columns(2)
    with r2a:
        if st.button("F&O Agent (OpenAI)", key="dash_fo_agent"):
            st.session_state.view = "fo_agent"
            st.session_state.selected_symbol = None
            st.rerun()
    for symbol in NIFTY50_TOP10:
        name = symbol_to_name.get(symbol, symbol)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{symbol}** — {name}")
        with col2:
            if st.button("View", key=f"btn_{symbol}"):
                st.session_state.selected_symbol = symbol
                st.rerun()
