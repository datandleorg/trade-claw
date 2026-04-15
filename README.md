# Trade Claw

Streamlit app for **Kite Connect** (Zerodha): view top 10 Nifty 50 stocks and historical OHLC charts. Uses [uv](https://docs.astral.sh/uv/) for dependencies and running.

## What it does

This UI is intentionally slim: **`app.py`** is a **home hub** after Kite login, with sidebar links to two product pages.

- **Login**: Kite redirect or manual token; session persists (same **`ensure_kite_session`** / **`page_config`** behaviour as the rest of the project).
- **F&O Options** (`pages/1_F_Options.py`): pick an **underlying** and **session date**, strike policy, mock exits (target / stop / EOD), and optional JSON run snapshots under `data/fo_options_runs/` (see **`.env.example`**).
- **LLM Mock Engine** (`pages/2_LLM_Mock_Engine.py`): live BANKNIFTY mock book, run observability, and **Agent chat**. Chat is stored in **`LLM_MOCK_AGENT_MEMORY_PATH`** (default `data/llm_mock_agent_memory.json`); the **minute Celery supervisor** reads the same transcript each tick. Set **`OPENAI_API_KEY`**; optional **`LLM_MOCK_AGENT_CHAT_MODEL`**, **`LLM_MOCK_SUPERVISOR_MODEL`**, **`ANTHROPIC_API_KEY`** for vision. Other Streamlit routes and workers may still exist in the repo for local development; the sample **Docker Compose** stack below runs only what this deployment needs.

## Prerequisites

- **Python 3.12+**
- **uv** – install: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or see [uv docs](https://docs.astral.sh/uv/getting-started/installation/))
- **Kite Connect API** key and secret from [kite.trade](https://kite.trade)

## Setup

1. **Clone and enter the project**
   ```bash
   cd trade-claw
   ```

2. **Create `.env` from the example**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set your Kite credentials (and for **F&O Agent**, OpenAI):
   ```
   KITE_API_KEY=your_actual_api_key
   KITE_API_SECRET=your_actual_api_secret
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=gpt-5.4-mini
   ```

3. **Install dependencies with uv**
   ```bash
   uv sync
   ```

4. **Kite redirect URL**  
   In the [Kite developer dashboard](https://kite.trade/dashboard), set the app’s **Redirect URL** to the URL where this app is served, for example:
   - Local: `http://localhost:8501/` or `http://localhost:8501/callback`
   - With ngrok: `https://your-subdomain.ngrok-free.app/` or `https://your-subdomain.ngrok-free.app/callback`

## Run the app

```bash
uv run streamlit run app.py
```

- Open the URL shown in the terminal (e.g. `http://localhost:8501`).
- If you use **ngrok**, run either command below and use the ngrok URL as the Kite redirect URL:
  ```bash
  ngrok http 8501
  ```
  or to use your reserved domain:
  ```bash
  ngrok http --url=joesph-nonalliterative-nelida.ngrok-free.dev 8501
  ```

## Docker Compose (sample)

This sample stack runs the slim deployment together:
- `streamlit` (UI on port `8501`)
- `celery-worker-llm-mock` (queue `llm_mock` — **`scan_llm_mock_banknifty`**)
- `celery-beat` (IST schedule for that task only)
- `redis` (broker/result backend)

1. Build and start:
   ```bash
   docker compose up --build
   ```
2. Open Streamlit at `http://localhost:8501`.
3. Stop:
   ```bash
   docker compose down
   ```

If you want to remove Redis data volumes/images used by the stack:
```bash
docker compose down -v --rmi local
```

## Production (VPS + domain + TLS)

For deployment on a server such as a **DigitalOcean droplet** with your **own domain**, **Nginx** reverse proxy, and **Certbot** (Let’s Encrypt), use **`docker-compose.prod.yml`** and follow **[deploy/README.md](deploy/README.md)**. Set **`DOMAINS`** (comma-separated) or **`DOMAIN`**, plus **`CERTBOT_EMAIL`** for the first certificate, in `.env`; see `.env.example`.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit entrypoint: Kite session + **home** with `st.page_link` to **F&O Options** and **LLM Mock Engine** |
| `trade_claw/constants.py` | Symbols, intervals; `ENVELOPE_PCT` = 0.0030 mirrors default intraday envelope when `INTRADAY_ENVELOPE_DECIMAL` is unset |
| `trade_claw/env_trading_params.py` | Single place for **`.env`** trading knobs: intraday envelope, F&O/mock envelope (`MOCK_AGENT_ENVELOPE_PCT`), option target/stop (`MOCK_ENGINE_OPTION_*` / `FO_OPTION_*`) |
| `trade_claw/kite_session.py` | Credentials, session state, `.kite_session.json` |
| `trade_claw/market_data.py` | `candles_to_dataframe`, `get_instrument_token` |
| `trade_claw/strategies.py` | ORB, VWAP, RSI, Flag, MA, envelope + simulation |
| `trade_claw/views/all_ten.py` | Single-day multi-stock strategies & mock trades |
| `trade_claw/views/dashboard.py` | Top-10 list |
| `trade_claw/views/stock_detail.py` | Single-stock historical chart |
| `trade_claw/views/reports.py` | Date-range P/L report + pagination |
| `trade_claw/trade_engine.py` | Shared trade row builder + split-by-day |
| `trade_claw/institutional_floor.py` | SMA 50/200 buy-only sizing for ETFs |
| `trade_claw/views/index_etfs.py` | Index ETF dashboard |
| `trade_claw/mock_engine_run.py` | Celery tick: SL/target exits, 15:20 square-off, LangGraph entry |
| `trade_claw/mock_trading_graph.py` | LangGraph: signal → candidates → LLM → mock DB |
| `trade_claw/mock_llm_banknifty_run.py` | Autonomous minute runner for LLM-only BANKNIFTY engine |
| `trade_claw/mock_llm_banknifty_graph.py` | ReAct-style supervisor + optional vision tool + mock execution |
| `trade_claw/mock_trade_store.py` | SQLite `mock_trades` (WAL) |
| `trade_claw/kite_headless.py` | Kite client for workers (env + `.kite_session.json`) |
| `trade_claw/views/mock_engine.py` | Streamlit HUD for mock engine |

## Notes

- **Access token** from Kite is valid until **7:30 AM IST** the next day; you may need to log in again after that.
- Session is stored in `.kite_session.json` (gitignored). Use **Clear session** in the sidebar to log out and remove it.
- Historical data may require an appropriate Kite Connect plan.
