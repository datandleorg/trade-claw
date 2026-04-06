# Trade Claw

Streamlit app for **Kite Connect** (Zerodha): view top 10 Nifty 50 stocks and historical OHLC charts. Uses [uv](https://docs.astral.sh/uv/) for dependencies and running.

## What it does

- **Login** with Kite (redirect or manual token). Session is saved so a browser refresh keeps you logged in. After login, the app opens on **F&O Options** by default (sidebar: **Intraday home** for the Nifty 50 top-10 flow).
- **Dashboard**: list of 10 Nifty 50 stocks (RELIANCE, TCS, HDFCBANK, INFY, etc.). Click **View** to open a stock. **Reports (date range)** opens the reports page.
- **Home (single day)**: intraday strategies and mock trades for selected stocks. **Reports (use these settings)** copies date, stocks, strategy, interval, and long/short filter into Reports.
- **Reports**: pick **from/to dates**, **stocks**, **strategy**, **interval**, and **trade direction**; **Generate report** loads history, runs one session per calendar day per symbol, and shows **totals in cards**, **P/L by stock**, and a **paginated** trade table.
- **Index ETFs (NSE)**: buy-only **Institutional floor** — daily **SMA 50 / SMA 200**, suggested **₹10k standard** vs **₹20k aggressive** when price is at/below the 200-day SMA; golden-cross context and charts per ETF. Sidebar **Navigate → Index ETFs**.
- **F&O Options**: pick **one underlying** and **session date**. **Index options** on the dropdown: **NIFTY**, **BANKNIFTY**, **FINNIFTY**, **MIDCPNIFTY**, **NIFTYNXT50** (Nifty Next 50), plus **Nifty 50 stocks**. **Strike policy** (ATM default = nearest listed strike to spot, plus ITM/OTM steps) and optional **manual strike** override. **Always 1 lot**. Exits: **target**, **stop loss** (sliders; defaults from `.env` via **`trade_claw/env_trading_params.py`** — `MOCK_ENGINE_OPTION_*` first, then `FO_OPTION_*`), or **EOD**. **Net P/L** after ₹/lot costs. Changing any parameter **saves a JSON snapshot** under `data/fo_options_runs/` (`FO_OPTIONS_RUNS_DIR` to override). Sidebar **Navigate → F&O Options**.
- **F&O snapshots report**: browse saved JSON runs with **session date range**, **script (underlying)** filter, **pagination**, and per-run **tabs** (parameters, trade detail, charts). Sidebar **Navigate → F&O snapshots report**.
- **F&O Agent (OpenAI)**: **Current calendar month** session date (not after today). **Deterministic** underlying signal (envelope or EMA), then an **OpenAI** ReAct loop with **only** Zerodha-style read tools **`search_instruments`**, **`get_historical_data`**, plus app-only **`submit_mock_trade_choice`** (no `place_order` / GTT / etc.). **No live orders**—mock P/L is computed in app code. Set **`OPENAI_API_KEY`** and optional **`OPENAI_MODEL`** (default `gpt-5.4-mini`) in `.env` or Streamlit secrets. With **`KITE_MCP_ENABLED=1`** (or **`KITE_MCP_STREAMABLE_URL`** / **`KITE_MCP_COMMAND`**), those read tools hit the **Zerodha Kite MCP server**; otherwise **KiteConnect** runs the same parameters in-process. See `.env.example` for **`KITE_MCP_*`**. Set **`KITE_MCP_TOOL_OUTPUT_FILE`** (e.g. `./logs/kite_mcp_tools.jsonl`) to append each MCP tool response as one JSON line. **Logs**: loggers `trade_claw.fo_openai_agent` and `trade_claw.kite_mcp_client`; set **`FO_AGENT_LOG_LEVEL=DEBUG`** for more detail. Sidebar **Navigate → F&O Agent**.
- **Mock AI engine (autonomous)**: **Celery Beat** (IST, weekdays) triggers **`scan_mock_market`**, which runs **SL/target exits** then a **LangGraph** flow **per configured underlying** (default list = **`FO_UNDERLYING_OPTIONS`**, same as F&O Options; optional **`MOCK_ENGINE_UNDERLYINGS`** env subset) that has no open leg: spot **20-EMA ± bandwidth** breakout on the latest bar → long **CE** or **PE** only → top-five strikes → **OpenAI** structured pick → **mock** insert into SQLite (**`MOCK_TRADES_DB_PATH`**, WAL). **At most one OPEN row per underlying**; several underlyings can be open at once. **15:20 IST** square-off of every open row; before that, **target/stop** on option LTP each tick. The worker needs **Kite** access: log in once via Streamlit so **`.kite_session.json`** exists on the same host, or set **`KITE_ACCESS_TOKEN`**. Sidebar **Navigate → Mock AI engine**: **Live** tab (telemetry + live LTP/charts) and **Analytics** tab (multi-month stats from `mock_trades`, CSV export, optional snapshot replay via **`MOCK_ENGINE_SNAPSHOT_BARS`**). See **`.env.example`** for **`MOCK_AGENT_*`**. Stack: Redis + `uv run celery -A trade_claw.celery_app worker --loglevel=info` + `uv run celery -A trade_claw.celery_app beat --loglevel=info`. **Flow & architecture:** [docs/MOCK_ENGINE.md](docs/MOCK_ENGINE.md).
- **Stock page**: date range, interval, OHLC candlestick chart (Plotly), and summary metrics.

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

This sample stack runs all app components together:
- `streamlit` (UI on port `8501`)
- `celery-worker`
- `celery-beat`
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

To expose Streamlit publicly with the included `ngrok` service:
- add `NGROK_AUTHTOKEN=...` in `.env`
- run `docker compose up -d`
- this stack uses your reserved domain: `https://joesph-nonalliterative-nelida.ngrok-free.dev`
- optional ngrok inspector is available at `http://localhost:4040`

## Production (VPS + domain + TLS)

For deployment on a server such as a **DigitalOcean droplet** with your **own domain**, **Nginx** reverse proxy, and **Certbot** (Let’s Encrypt), use **`docker-compose.prod.yml`** and follow **[deploy/README.md](deploy/README.md)**. Set **`DOMAINS`** (comma-separated) or **`DOMAIN`**, plus **`CERTBOT_EMAIL`** for the first certificate, in `.env`; see `.env.example`.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit entrypoint: auth, routing |
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
| `trade_claw/mock_trade_store.py` | SQLite `mock_trades` (WAL) |
| `trade_claw/kite_headless.py` | Kite client for workers (env + `.kite_session.json`) |
| `trade_claw/views/mock_engine.py` | Streamlit HUD for mock engine |

## Notes

- **Access token** from Kite is valid until **7:30 AM IST** the next day; you may need to log in again after that.
- Session is stored in `.kite_session.json` (gitignored). Use **Clear session** in the sidebar to log out and remove it.
- Historical data may require an appropriate Kite Connect plan.
