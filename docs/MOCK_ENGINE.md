# Autonomous mock trading engine — flow and architecture

This document describes how the **Nifty index options mock engine** works in trade-claw: what runs when, which modules participate, and how data moves from Kite → Celery → LangGraph → SQLite → Streamlit.

It reflects the implementation in `trade_claw/mock_engine_run.py`, `trade_claw/mock_trading_graph.py`, `trade_claw/mock_market_signal.py`, `trade_claw/mock_trade_store.py`, and `trade_claw/kite_headless.py`.

---

## 1. Purpose and boundaries

- **Mock only**: no `place_order` or live broker execution. Rows in `mock_trades` represent simulated long-premium trades.
- **Underlying**: Nifty spot (NSE index **NIFTY 50** via `fetch_underlying_intraday` with underlying key `NIFTY`).
- **Options**: NFO Nifty index contracts; **long-only** — bullish signals trade **CE** only, bearish **PE** only (the opposite leg is never shown to the LLM).
- **Concurrency model**: Celery Beat triggers a scan **about once per minute** on weekdays during configured hours; Streamlit only **reads** the SQLite database and does not drive the loop.

---

## 2. High-level architecture

```mermaid
flowchart TB
  subgraph sched [Scheduler]
    Beat[Celery Beat IST]
    Redis[(Redis broker)]
  end
  subgraph worker [Worker process]
    Task[scan_mock_market]
    Run[mock_engine_run.run_scan]
    Graph[LangGraph compiled graph]
    Kite[KiteConnect headless]
  end
  subgraph storage [Persistence]
    WAL[(SQLite WAL\nmock_engine.db)]
  end
  subgraph ui [Dashboard]
    ST[Streamlit Mock AI engine page]
  end
  Beat --> Redis
  Redis --> Task
  Task --> Run
  Run --> Kite
  Run --> Graph
  Graph --> Kite
  Run --> WAL
  Graph --> WAL
  ST --> WAL
```

| Layer | Role |
| :--- | :--- |
| **Celery Beat** | Fires `trade_claw.scan_mock_market` on a crontab (IST, weekdays, hours 9–15). |
| **Celery worker** | Executes `run_scan_safe()` → `run_scan()`. |
| **Kite headless** | Builds `KiteConnect` from `KITE_API_KEY` + token (`.kite_session.json` or `KITE_ACCESS_TOKEN`). |
| **LangGraph** | Multi-step flow: signal → candidates → LLM → execute (insert open row). |
| **SQLite** | `mock_trades` table + LangGraph checkpoint tables in the same file (`MOCK_TRADES_DB_PATH`), WAL mode. |
| **Streamlit** | Fragment auto-refresh; reads `mock_trades` for table and cumulative PnL chart. |

---

## 3. End-to-end lifecycle of one scan (`run_scan`)

Each invocation of `run_scan()` (from `trade_claw/mock_engine_run.py`) follows this **strict order**.

### Step 0 — Initialise storage

- `mock_trade_store.init_db()` ensures the database file exists, applies `PRAGMA journal_mode=WAL`, and creates `mock_trades` (and indexes) if needed.

### Step 1 — Resolve “now” and session date

- Wall clock in **Asia/Kolkata** (`now_ist()`).
- **Session calendar date** for instruments and DTE is `session_date_ist(dt)` (the IST calendar date of that moment).

### Step 2 — Build Kite client

- `get_kite_headless()` (`trade_claw/kite_headless.py`).
- Failure (missing key/token) → return a result dict with `skipped` set; **no** graph run.

### Step 3 — Weekend short-circuit

- If `weekday() >= 5` → `skipped: "weekend"` and exit.

### Step 4 — Manage open positions (always before new entries)

**4a — Stop / target on option LTP** (`process_stop_target_exits`)

- For each row with `status = OPEN`, fetch option **LTP** via `kite.ltp(["NFO:<tradingsymbol>"])`.
- **Long premium** logic:
  - If `target > entry` and `ltp >= target` → close (take profit).
  - Else if `stop < entry` and `stop > 0` and `ltp <= stop` → close (stop loss).
- **Exit price**: `max(0.01, ltp - slippage)` where slippage is a random draw in `[MOCK_AGENT_SLIPPAGE_LO, MOCK_AGENT_SLIPPAGE_HI]` (same family as entry).
- **Realised PnL**: `(exit_price - entry_price) * quantity`, then row updated to `CLOSED` with `exit_time`, `exit_price`, `realized_pnl`.

**4b — 15:20 IST square-off** (`should_force_square_off` + `force_square_off_all`)

- If IST time is **≥ 15:20** on a weekday:
  - Every remaining `OPEN` row is closed at synthetic LTP (again `ltp - slippage`).
  - The function returns immediately with `skipped: "after_square_off_window"` so **no new LangGraph entry** runs in that tick (and subsequent ticks the same day after 15:20 also hit this branch first).

### Step 5 — Gate new entries

If **not** in the force square-off branch, the scan continues:

- **`in_entry_window`**: new mock entries are allowed only on weekdays when IST time is **≥ 09:15** and **< 15:20** (aligned with “no new risk” after the square-off window).
- **`has_open_trade()`**: if any `OPEN` row exists, skip the graph (`skipped: "open_position"`) — **at most one open mock position** at a time.

### Step 6 — Load instrument masters

- `kite.instruments("NSE")` and `kite.instruments("NFO")` for the graph’s signal and option filtering.

### Step 7 — Run LangGraph with SQLite checkpointer

- Open `SqliteSaver.from_conn_string(<absolute path to MOCK_TRADES_DB_PATH>)` (plain path string, not a `sqlite://` URL — required by LangGraph’s saver).
- `invoke_mock_graph(...)` compiles the graph with that checkpointer and invokes it with a **fresh `thread_id` per scan** so each beat tick is isolated; checkpoint tables still live in the same WAL file for durability and tooling compatibility.

The return value is merged into the Celery result under `graph`: `error`, `trade_id`, `signal_text` / `notes`.

---

## 4. LangGraph internals (`mock_trading_graph.py`)

The graph is built per invocation with **closure** over `kite`, `session_d`, `nse_instruments`, and `nfo_instruments` (so checkpointed state stays JSON-serialisable).

```mermaid
flowchart LR
  START --> signal[signal_node]
  signal -->|direction set| candidates[candidates_node]
  signal -->|no breakout| END
  candidates -->|ok| llm[llm_node]
  candidates -->|error| END
  llm -->|pick ok| execute[execute_node]
  llm -->|error| END
  execute --> END
```

### Node: `signal`

1. Loads **full session** Nifty spot **1-minute** candles for `session_d` from **09:15 to 15:30** (`load_nifty_session_minute_df` → `fetch_underlying_intraday`).
2. Runs `envelope_breakout_on_last_bar` with **20-period EMA** and bandwidth **`MOCK_AGENT_ENVELOPE_PCT`** (decimal, e.g. `0.005` = 0.5% each side of the EMA — same geometry as `strategies._envelope_series`).
3. **Trigger** (on the **latest** completed bar only):
   - Close crosses **above** upper band → **BULLISH**, leg **CE**.
   - Close crosses **below** lower band → **BEARISH**, leg **PE**.
4. If no cross → state gets `signal_text` / `notes`; routing sends flow to **END** (no LLM).

### Node: `candidates`

1. Filters NFO instruments to **NIFTY** index options and the chosen leg only (`top_five_option_instruments`).
2. Uses **nearest expiry on or after** `session_d`.
3. Builds **five** strikes around ATM (ladder steps −2 … +2, deduplicated, padded if the chain is thin).
4. Enriches each with **LTP** (batch `kite.ltp`), **DTE**, `lot_size`, etc., for the LLM prompt.

### Node: `llm`

1. **Structured output** (`LLMPick`): `tradingsymbol`, `stop_loss`, `target`, `rationale` (premium prices, not index points).
2. Validates `tradingsymbol` is **exactly** one of the five candidates.

### Node: `execute`

1. Re-reads LTP for the chosen symbol.
2. **Entry (mock BUY)**: `entry_price = max(0.01, ltp - slippage)` with slippage in the configured rupee range.
3. Clamps **target** above entry and **stop** below entry if the model returns inconsistent premium levels.
4. Sizes with `fo_lot_qty_for_allocation` and `ALLOCATED_AMOUNT` → whole lots → `quantity` = units for PnL.
5. **`insert_open_trade`** → new `OPEN` row (`mock_trade_store`).

There is **no** explicit `transaction_type` column in SQLite; behaviour is **BUY** on insert and **SELL** on close in the sense that exits are always closing the long (synthetic sell at LTP minus slippage).

---

## 5. What Celery Beat actually schedules

Configured in `trade_claw/celery_app.py`:

- **Timezone**: `Asia/Kolkata`, `enable_utc=False`.
- **Schedule**: every minute, hours **9–15**, **Monday–Friday** (`day_of_week='1-5'`).
- **Task name**: `trade_claw.scan_mock_market` → `run_scan_safe()`.

**Important:** Beat still fires at 09:00–09:14 and after 15:30 on weekdays. **In-task guards** (`in_entry_window`, `should_force_square_off`, weekend check) restrict **meaningful** behaviour to the intended NSE-style windows. After **15:20**, every tick first attempts square-off (idempotent if already flat) then exits without opening new trades.

---

## 6. SQLite schema (`mock_trade_store.py`)

Core columns match the product blueprint; two extra columns support realistic PnL:

| Column | Meaning |
| :--- | :--- |
| `trade_id` | Primary key |
| `entry_time` / `exit_time` | Timestamps (UTC string convention in code) |
| `instrument` | NFO `tradingsymbol` |
| `direction` | `BULLISH` / `BEARISH` |
| `entry_price` | Simulated **buy** fill (LTP minus slippage) |
| `stop_loss` / `target` | Premium levels from LLM (possibly clamped) |
| `llm_rationale` | Model explanation |
| `status` | `OPEN` / `CLOSED` |
| `exit_price` / `realized_pnl` | Set on close |
| `lot_size` | Exchange lot size used for sizing |
| `quantity` | Total units (lots × lot size) for PnL |
| `entry_bars_json` / `exit_bars_json` | Optional JSON arrays of **option** minute OHLC (see `MOCK_ENGINE_SNAPSHOT_BARS`) |
| `entry_underlying_bars_json` / `exit_underlying_bars_json` | Optional JSON arrays of **NIFTY 50** minute OHLC (same bar cap; added by migration — only **new** trades after this feature populate them unless you backfill) |

WAL allows concurrent **writer** (Celery) and **reader** (Streamlit).

### Long-horizon analysis (no Kite backfill)

For **months** of history, analytics are driven only from **`mock_trades`**. Kite does **not** reliably provide **minute** history for **old / expired** option contracts, so do not depend on re-fetching option intraday charts for past trades.

Streamlit **Mock AI engine → Analytics** tab uses [`mock_trade_analytics.py`](trade_claw/mock_trade_analytics.py): IST calendar date range → UTC SQL bounds on `entry_time`, KPIs, monthly PnL, equity curve by `exit_time`, direction breakdown, CSV export, and optional **replay** of stored entry/exit snapshots. Replay draws **NIFTY** candles plus **EMA envelope** (same helper as Live; band width uses **current** `MOCK_AGENT_ENVELOPE_PCT`) when underlying JSON exists, and **option** candles with horizontal **entry / target / stop / exit** lines from the trade row (same rules as the Live option chart).

### Telemetry (`mock_engine_telemetry.py`)

Table **`mock_engine_telemetry`** (single row `id = 1`) stores JSON:

- **`last_scan`**: written on **every** worker tick — IST timestamp, `skipped` reason, exit counts, live `nifty_ltp` / `open_option_ltp` when Kite is available, `open_trades_detail`, `agent_envelope_pct`, `agent_ema_period`, and summary `graph` fields.
- **`last_graph`**: full LangGraph `invoke` state after a successful graph run (signal, `candidates`, LLM fields, errors). **Preserved** when a later tick skips (e.g. `open_position`) so the Streamlit HUD still shows the last candidates/pick.

---

## 7. Daily operations (operator view)

1. **Morning**: Start **Redis**, **Celery worker**, **Celery beat** (see `README.md`).
2. **Kite session**: Log in via Streamlit once so `.kite_session.json` is written on the **same machine** as the worker, or export `KITE_ACCESS_TOKEN`.
3. **OpenAI**: Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`) for the LLM node.
4. **Monitoring**: Streamlit → **Mock AI engine** — **Live** tab: auto-refreshing quotes/charts + telemetry; **Analytics** tab: multi-month stats from `mock_trades`, CSV export, snapshot replay (if enabled).
5. **End of day**: After **15:20 IST**, open rows are closed; Beat may still enqueue tasks until hour 15 ends, but logic stays idempotent.

---

## 8. Key environment variables

| Variable | Purpose |
| :--- | :--- |
| `KITE_API_KEY` | Kite app key |
| `KITE_ACCESS_TOKEN` | Optional override if not using `.kite_session.json` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | LLM node |
| `MOCK_TRADES_DB_PATH` | SQLite file path (default under `data/`) |
| `MOCK_AGENT_ENVELOPE_PCT` | EMA bandwidth (decimal) |
| `MOCK_AGENT_SLIPPAGE_LO` / `MOCK_AGENT_SLIPPAGE_HI` | Slippage range (rupees) |
| `MOCK_ENGINE_SNAPSHOT_BARS` | Max 1m bars stored **per leg** for option **and** NIFTY snapshots at entry/exit (`0` = off, default `60`) |
| `REDIS_URL` / `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | Celery |

See `.env.example` for commented templates.

---

## 9. Source file map

| File | Responsibility |
| :--- | :--- |
| `trade_claw/celery_app.py` | Celery app, IST, beat schedule |
| `trade_claw/worker_tasks.py` | `scan_mock_market` task |
| `trade_claw/mock_engine_run.py` | Per-tick orchestration, exits, graph invoke |
| `trade_claw/mock_trading_graph.py` | LangGraph definition and `invoke_mock_graph` |
| `trade_claw/mock_market_signal.py` | IST helpers, envelope breakout, top-five strikes |
| `trade_claw/mock_trade_store.py` | CRUD + WAL connection |
| `trade_claw/mock_trade_analytics.py` | Date-range queries + pandas summaries for Analytics tab |
| `trade_claw/mock_trade_snapshot.py` | Optional minute-bar JSON at entry/exit |
| `trade_claw/kite_headless.py` | Worker-safe Kite auth |
| `trade_claw/mock_engine_telemetry.py` | SQLite snapshot for Streamlit (`last_scan` / `last_graph`) |
| `trade_claw/views/mock_engine.py` | Streamlit HUD (telemetry + live Kite charts) |
| `app.py` | Route to mock engine view |

---

## 10. Known simplifications vs a full production stack

- **Instrument load every tick**: Full NSE/NFO masters are downloaded each time the graph runs; acceptable for a mock demo, could be cached.
- **Single open position**: No pyramiding or multiple concurrent mocks.
- **Checkpoint thread IDs**: New UUID suffix per scan limits cross-tick LangGraph “memory”; ledger truth is **`mock_trades`**, not the checkpoint.
- **Beat granularity**: Crontab is coarser than “09:15–15:30 only”; correctness relies on **Python time checks** inside `run_scan`.

For behaviour changes, start with `mock_engine_run.run_scan` ordering, then `mock_market_signal` gates, then graph nodes in `mock_trading_graph.py`.
