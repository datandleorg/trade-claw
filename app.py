"""
Streamlit app: Trade Claw — home hub. Use sidebar pages for F&O Options and LLM Mock Engine.
Run: uv run streamlit run app.py
"""

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from trade_claw.streamlit_kite_shell import ensure_kite_session, page_config


def main() -> None:
    page_config()
    ensure_kite_session()
    st.title("Trade Claw")
    st.caption("Kite session is active. Open a module from the links below (same entries appear in the sidebar).")
    st.page_link("pages/1_F_Options.py", label="F&O Options", icon="📈")
    st.page_link("pages/2_LLM_Mock_Engine.py", label="LLM Mock Engine (BANKNIFTY)", icon="🤖")


if __name__ == "__main__":
    main()
