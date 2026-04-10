"""Dedicated multipage route for LLM-only BANKNIFTY mock engine."""

from dotenv import load_dotenv

load_dotenv()

from trade_claw.streamlit_kite_shell import ensure_kite_session, page_config
from trade_claw.views.llm_mock_engine import render_llm_mock_engine

page_config()
kite = ensure_kite_session()
render_llm_mock_engine(kite)
