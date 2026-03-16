# Trade Claw

Streamlit app for **Kite Connect** (Zerodha): view top 10 Nifty 50 stocks and historical OHLC charts. Uses [uv](https://docs.astral.sh/uv/) for dependencies and running.

## What it does

- **Login** with Kite (redirect or manual token). Session is saved so a browser refresh keeps you logged in.
- **Dashboard**: list of 10 Nifty 50 stocks (RELIANCE, TCS, HDFCBANK, INFY, etc.). Click **View** to open a stock.
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
   Edit `.env` and set your Kite credentials:
   ```
   KITE_API_KEY=your_actual_api_key
   KITE_API_SECRET=your_actual_api_secret
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
- If you use **ngrok**, run `ngrok http 8501` and use the ngrok URL as the Kite redirect URL.

## Notes

- **Access token** from Kite is valid until **7:30 AM IST** the next day; you may need to log in again after that.
- Session is stored in `.kite_session.json` (gitignored). Use **Clear session** in the sidebar to log out and remove it.
- Historical data may require an appropriate Kite Connect plan.
