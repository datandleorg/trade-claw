"""
Streamlit app: Trade Claw — Mock AI engine (home). F&O Options: sidebar → F_Options page.
Run: uv run streamlit run app.py
"""
from dotenv import load_dotenv

load_dotenv()

from trade_claw.streamlit_kite_shell import ensure_kite_session, page_config
from trade_claw.views.mock_engine import render_mock_engine


def main():
    page_config()
    kite = ensure_kite_session()
    render_mock_engine(kite)


if __name__ == "__main__":
    main()
