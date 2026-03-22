# Trade Claw

Streamlit app for **Kite Connect** (Zerodha): view top 10 Nifty 50 stocks and historical OHLC charts. Uses [uv](https://docs.astral.sh/uv/) for dependencies and running.

## What it does

- **Login** with Kite (redirect or manual token). Session is saved so a browser refresh keeps you logged in.
- **Dashboard**: list of 10 Nifty 50 stocks (RELIANCE, TCS, HDFCBANK, INFY, etc.). Click **View** to open a stock. **Reports (date range)** opens the reports page.
- **Home (single day)**: intraday strategies and mock trades for selected stocks. **Reports (use these settings)** copies date, stocks, strategy, interval, and long/short filter into Reports.
- **Reports**: pick **from/to dates**, **stocks**, **strategy**, **interval**, and **trade direction**; **Generate report** loads history, runs one session per calendar day per symbol, and shows **totals in cards**, **P/L by stock**, and a **paginated** trade table.
- **Index ETFs (NSE)**: buy-only **Institutional floor** — daily **SMA 50 / SMA 200**, suggested **₹10k standard** vs **₹20k aggressive** when price is at/below the 200-day SMA; golden-cross context and charts per ETF. Sidebar **Navigate → Index ETFs**.
- **F&O Options**: pick **one underlying** at a time; **strike policy** (ATM default = nearest listed strike to spot, plus ITM/OTM steps) and optional **single manual strike** override. **Always 1 lot**. **Net P/L** after ₹/lot costs. Option series are resolved from Kite’s **NFO instruments** list; if the first row returns **no minute history**, the app **walks duplicate symbols and further expiries** (same strike policy per expiry) until candles load. A **month table** at the bottom replays the same mock rules on **each weekday** in the session month (weekends skipped). Sidebar **Navigate → F&O Options**.
- **F&O Agent (OpenAI)**: **Current calendar month** session date (not after today). **Deterministic** underlying signal (envelope or EMA), then an **OpenAI** ReAct loop with **allowlisted** tools only (`search_nfo_options`, `get_ohlc_bars`, `submit_mock_trade_choice`). **No Kite order APIs**—mock P/L is computed in app code. Set **`OPENAI_API_KEY`** and optional **`OPENAI_MODEL`** (default `gpt-5.4-mini`) in `.env` or Streamlit secrets. Tools use the **Kite Python SDK** in-process (not Cursor IDE MCP). **Logs**: logger `trade_claw.fo_openai_agent` prints **LLM** rounds (model, usage, tool_call names) and **Kite tool** lines (`[kite-tool/MCP-style]`) to stderr; set **`FO_AGENT_LOG_LEVEL=DEBUG`** for more detail. Sidebar **Navigate → F&O Agent**.
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

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit entrypoint: auth, routing |
| `trade_claw/constants.py` | Symbols, intervals, strategy params (`ENVELOPE_PCT` = 0.5%) |
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

## Notes

- **Access token** from Kite is valid until **7:30 AM IST** the next day; you may need to log in again after that.
- Session is stored in `.kite_session.json` (gitignored). Use **Clear session** in the sidebar to log out and remove it.
- Historical data may require an appropriate Kite Connect plan.
