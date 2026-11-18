"""SQLite persistence layer for the Indian Stock Research & Swing Analyzer.

Schema requirement: database migration version 2 (PRAGMA user_version = 2).
This module keeps analytical market history separate from as-traded execution
prices. It does not download market data or calculate indicators.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "indian_stock_research.db"
REQUIRED_SCHEMA_VERSION = 2


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection():
    """Open a connection and commit or roll back one transaction safely."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _check_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != REQUIRED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {REQUIRED_SCHEMA_VERSION} is required; "
            f"found version {version}. Run the approved migration first."
        )


def _text(value: Any, field: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required.")
        return None
    result = str(value).strip()
    if not result and required:
        raise ValueError(f"{field} is required.")
    return result or None


def _plain(value: Any) -> Any:
    """Convert common NumPy/Pandas scalar values to standard Python values."""
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    return value


def _date_string(value: Any) -> str:
    value = _plain(value)
    if value is None:
        raise ValueError("Price/analysis date is required.")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        if isinstance(converted, datetime):
            return converted.date().isoformat()
    raw = str(value).strip()
    try:
        parsed = datetime.strptime(raw[:10], "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("Date must be a valid YYYY-MM-DD date.") from error
    if len(raw) > 10 and raw[10] not in ("T", " "):
        raise ValueError("Date must use YYYY-MM-DD or an ISO datetime.")
    return parsed.date().isoformat()


def _finite_number(value: Any, field: str, *, allow_missing: bool = False) -> float | None:
    value = _plain(value)
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_missing:
            return None
        raise ValueError(f"{field} is required and must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        if allow_missing:
            return None
        raise ValueError(f"{field} must be numeric.") from error
    if not math.isfinite(number):
        # A missing volume may be represented by NaN, but infinity is never
        # a valid market-data value.
        if allow_missing and math.isnan(number):
            return None
        raise ValueError(f"{field} must be finite (not NaN or infinity).")
    return number


def _volume_number(value: Any) -> int | None:
    number = _finite_number(value, "Volume", allow_missing=True)
    if number is None:
        return None
    if number < 0 or not number.is_integer():
        raise ValueError("Volume must be a non-negative whole number.")
    return int(number)


def _validate_ohlcv(open_price: Any, high_price: Any, low_price: Any,
                    close_price: Any, volume: Any) -> tuple[float, float, float, float, int | None]:
    opening = _finite_number(open_price, "Open")
    high = _finite_number(high_price, "High")
    low = _finite_number(low_price, "Low")
    close = _finite_number(close_price, "Close")
    qty = _volume_number(volume)
    assert opening is not None and high is not None and low is not None and close is not None
    if min(opening, high, low, close) <= 0:
        raise ValueError("OHLC prices must be positive.")
    if high < max(opening, low, close):
        raise ValueError("High price is inconsistent with OHLC values.")
    if low > min(opening, high, close):
        raise ValueError("Low price is inconsistent with OHLC values.")
    return opening, high, low, close, qty


def _canonical_symbol(value: str) -> tuple[str, str, str]:
    """Return (canonical symbol, exchange, provider symbol) for a ticker."""
    ticker = _text(value, "Stock symbol").upper()
    if ticker.startswith("^"):
        raise ValueError("Benchmark symbols must be saved through benchmark functions.")
    if ticker.endswith(".NS"):
        return ticker[:-3], "NSE", ticker
    if ticker.endswith(".BO"):
        return ticker[:-3], "BSE", ticker
    return ticker, "NSE", f"{ticker}.NS"


def _stock_identity(yahoo_symbol: str, metadata: Mapping[str, Any] | None = None,
                    company_name: str | None = None, industry: str | None = None,
                    nse_symbol: str | None = None, exchange: str | None = None) -> dict[str, str | None]:
    canonical, inferred_exchange, inferred_yahoo = _canonical_symbol(yahoo_symbol)
    meta = dict(metadata or {})
    canonical = _text(nse_symbol or meta.get("NSE_Symbol") or canonical, "NSE symbol").upper()
    return {
        "symbol": canonical,
        "yahoo_symbol": _text(meta.get("Yahoo_Symbol") or inferred_yahoo, "Yahoo symbol").upper(),
        "company_name": _text(company_name or meta.get("Company_Name") or canonical, "Company name"),
        "exchange": _text(exchange or meta.get("Exchange") or inferred_exchange, "Exchange").upper(),
        "industry": _text(industry or meta.get("NSE_Industry"), "Industry", required=False),
    }


def _ensure_stock(connection: sqlite3.Connection, identity: Mapping[str, Any]) -> int:
    connection.execute(
        """INSERT INTO stocks (symbol, company_name, exchange, industry, yahoo_symbol)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(symbol) DO UPDATE SET
             company_name = excluded.company_name,
             exchange = excluded.exchange,
             industry = COALESCE(excluded.industry, stocks.industry),
             yahoo_symbol = COALESCE(excluded.yahoo_symbol, stocks.yahoo_symbol)""",
        (identity["symbol"], identity["company_name"], identity["exchange"],
         identity["industry"], identity["yahoo_symbol"]),
    )
    row = connection.execute(
        "SELECT stock_id FROM stocks WHERE symbol = ?", (identity["symbol"],)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Could not retrieve stock_id for {identity['symbol']}.")
    return int(row[0])


def _source_fields(source_info: Mapping[str, Any] | None,
                   data_source: str | None = None,
                   price_basis: str | None = None,
                   retrieved_at_utc: str | None = None) -> tuple[str, str, str]:
    info = dict(source_info or {})
    source = _text(data_source or info.get("source") or "unknown", "Data source")
    basis = _text(price_basis or info.get("price_basis") or "unspecified", "Price basis")
    retrieved = _text(
        retrieved_at_utc or info.get("downloaded_at_utc") or _now_utc(),
        "Retrieval timestamp",
    )
    return source, basis, retrieved


def _frame_rows(frame: Any) -> list[tuple[str, float, float, float, float, int | None]]:
    if frame is None or getattr(frame, "empty", True):
        raise ValueError("Market data frame is empty.")
    required = ("Open", "High", "Low", "Close")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Market data is missing required columns: " + ", ".join(missing))
    prepared = []
    for index, row in frame.iterrows():
        trading_date = _date_string(index)
        volume = row["Volume"] if "Volume" in frame.columns else None
        values = _validate_ohlcv(
            row["Open"], row["High"], row["Low"], row["Close"], volume
        )
        prepared.append((trading_date, *values))
    if len({row[0] for row in prepared}) != len(prepared):
        raise ValueError("Market data contains duplicate trading dates.")
    return prepared


def test_connection() -> None:
    """Check SQLite integrity, schema version and expected tables."""
    with get_connection() as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        print("Database connection successful!")
        print("Database integrity:", integrity)
        print("Database schema version:", version)
        if integrity != "ok":
            raise RuntimeError(f"Database integrity check failed: {integrity}")
        _check_schema(connection)


def get_all_stocks() -> list[tuple[Any, ...]]:
    with get_connection() as connection:
        _check_schema(connection)
        return connection.execute(
            """SELECT stock_id, symbol, company_name, exchange, industry, yahoo_symbol
               FROM stocks ORDER BY symbol"""
        ).fetchall()


def add_stock(symbol: str, company_name: str, exchange: str = "NSE",
              industry: str | None = None, yahoo_symbol: str | None = None) -> int:
    """Add/update a stock. `symbol` is the canonical NSE/BSE symbol."""
    canonical = _text(symbol, "Stock symbol").upper()
    company = _text(company_name, "Company name")
    exch = _text(exchange or "NSE", "Exchange").upper()
    provider = _text(yahoo_symbol, "Yahoo symbol", required=False)
    with get_connection() as connection:
        _check_schema(connection)
        return _ensure_stock(connection, {
            "symbol": canonical, "company_name": company, "exchange": exch,
            "industry": _text(industry, "Industry", required=False),
            "yahoo_symbol": provider.upper() if provider else None,
        })


def get_stock_prices(symbol: str, include_provenance: bool = False) -> list[tuple[Any, ...]]:
    """Retrieve stored analytical history by NSE symbol or Yahoo ticker."""
    requested = _text(symbol, "Stock symbol").upper()
    columns = "h.price_basis, h.data_source, h.retrieved_at_utc" if include_provenance else ""
    comma = ", " if columns else ""
    with get_connection() as connection:
        _check_schema(connection)
        return connection.execute(
            f"""SELECT s.symbol, h.price_date, h.open_price, h.high_price,
                       h.low_price, h.close_price, h.volume{comma}{columns}
                FROM stocks s JOIN historical_prices h ON s.stock_id = h.stock_id
                WHERE UPPER(s.symbol) = ? OR UPPER(s.yahoo_symbol) = ?
                ORDER BY h.price_date DESC""",
            (requested, requested),
        ).fetchall()


def save_historical_price(symbol: str, price_date: Any, open_price: Any,
                          high_price: Any, low_price: Any, close_price: Any,
                          volume: Any = 0, price_basis: str = "unspecified",
                          data_source: str = "manual",
                          retrieved_at_utc: str | None = None) -> bool:
    """Insert/update one analytical daily price for an existing stock."""
    ticker = _text(symbol, "Stock symbol").upper()
    trading_date = _date_string(price_date)
    opening, high, low, close, qty = _validate_ohlcv(
        open_price, high_price, low_price, close_price, volume
    )
    source, basis, retrieved = _source_fields(
        None, data_source, price_basis, retrieved_at_utc
    )
    with get_connection() as connection:
        _check_schema(connection)
        row = connection.execute(
            """SELECT stock_id FROM stocks
               WHERE UPPER(symbol) = ? OR UPPER(yahoo_symbol) = ?""",
            (ticker, ticker),
        ).fetchone()
        if row is None:
            raise ValueError(f"{ticker} does not exist. Add the stock first.")
        connection.execute(
            """INSERT INTO historical_prices
               (stock_id, price_date, open_price, high_price, low_price,
                close_price, volume, price_basis, data_source, retrieved_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(stock_id, price_date) DO UPDATE SET
                 open_price=excluded.open_price, high_price=excluded.high_price,
                 low_price=excluded.low_price, close_price=excluded.close_price,
                 volume=excluded.volume, price_basis=excluded.price_basis,
                 data_source=excluded.data_source,
                 retrieved_at_utc=excluded.retrieved_at_utc""",
            (row[0], trading_date, opening, high, low, close, qty, basis, source, retrieved),
        )
    return True


def save_stock_history(yahoo_symbol: str, frame: Any,
                       metadata: Mapping[str, Any] | None = None,
                       source_info: Mapping[str, Any] | None = None) -> int:
    """Bulk upsert a validated OHLCV DataFrame into historical_prices."""
    identity = _stock_identity(yahoo_symbol, metadata)
    rows = _frame_rows(frame)  # Validate the full batch before opening a transaction.
    source, basis, retrieved = _source_fields(source_info)
    with get_connection() as connection:
        _check_schema(connection)
        stock_id = _ensure_stock(connection, identity)
        connection.executemany(
            """INSERT INTO historical_prices
               (stock_id, price_date, open_price, high_price, low_price,
                close_price, volume, price_basis, data_source, retrieved_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(stock_id, price_date) DO UPDATE SET
                 open_price=excluded.open_price, high_price=excluded.high_price,
                 low_price=excluded.low_price, close_price=excluded.close_price,
                 volume=excluded.volume, price_basis=excluded.price_basis,
                 data_source=excluded.data_source,
                 retrieved_at_utc=excluded.retrieved_at_utc""",
            [(stock_id, *row, basis, source, retrieved) for row in rows],
        )
    return len(rows)


def save_benchmark_history(frame: Any, benchmark_symbol: str = "NIFTY50",
                           benchmark_name: str = "NIFTY 50",
                           provider_symbol: str = "^NSEI",
                           source_info: Mapping[str, Any] | None = None) -> int:
    """Bulk upsert benchmark history into benchmark_prices, not stocks."""
    rows = _frame_rows(frame)
    source, basis, retrieved = _source_fields(source_info)
    code = _text(benchmark_symbol, "Benchmark symbol").upper()
    name = _text(benchmark_name, "Benchmark name")
    provider = _text(provider_symbol, "Benchmark provider symbol").upper()
    with get_connection() as connection:
        _check_schema(connection)
        connection.execute(
            """INSERT INTO benchmarks
               (benchmark_symbol, benchmark_name, exchange, provider_symbol)
               VALUES (?, ?, 'NSE', ?)
               ON CONFLICT(benchmark_symbol) DO UPDATE SET
                 benchmark_name=excluded.benchmark_name,
                 provider_symbol=excluded.provider_symbol""",
            (code, name, provider),
        )
        benchmark_id = connection.execute(
            "SELECT benchmark_id FROM benchmarks WHERE benchmark_symbol = ?", (code,)
        ).fetchone()[0]
        connection.executemany(
            """INSERT INTO benchmark_prices
               (benchmark_id, price_date, open_price, high_price, low_price,
                close_price, volume, price_basis, data_source, retrieved_at_utc)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(benchmark_id, price_date) DO UPDATE SET
                 open_price=excluded.open_price, high_price=excluded.high_price,
                 low_price=excluded.low_price, close_price=excluded.close_price,
                 volume=excluded.volume, price_basis=excluded.price_basis,
                 data_source=excluded.data_source,
                 retrieved_at_utc=excluded.retrieved_at_utc""",
            [(benchmark_id, *row, basis, source, retrieved) for row in rows],
        )
    return len(rows)


def _record_value(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record:
            value = _plain(record[key])
            if value is not None:
                return value
    return default


def _optional_float(record: Mapping[str, Any], *keys: str) -> float | None:
    return _finite_number(_record_value(record, *keys), keys[0], allow_missing=True)


def _json_safe(value: Any) -> Any:
    """Recursively convert Pandas/NumPy values and NaN to JSON-safe values."""
    value = _plain(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _json_default(value: Any) -> Any:
    return _json_safe(value)


def save_research_result(record: Mapping[str, Any],
                         source_info: Mapping[str, Any] | None = None,
                         metadata: Mapping[str, Any] | None = None) -> int:
    """Append one stock research snapshot and return its research_id."""
    ticker = _text(_record_value(record, "Symbol"), "Research Symbol")
    identity = _stock_identity(ticker, metadata)
    analysis_date = _date_string(_record_value(record, "Latest_Date", "analysis_date", default=date.today()))
    source, basis, _ = _source_fields(
        source_info,
        data_source=_record_value(record, "Data_Source"),
        price_basis=_record_value(record, "Price_Basis"),
    )
    values = (
        _optional_float(record, "Close", "close_price"),
        _optional_float(record, "RSI_14", "rsi"),
        _optional_float(record, "SMA_20", "sma_20"),
        _optional_float(record, "SMA_50", "sma_50"),
        _optional_float(record, "SMA_200", "sma_200"),
        _optional_float(record, "Relative_Strength_20D_Pct", "relative_strength"),
        _optional_float(record, "Setup_Score", "research_score"),
        _record_value(record, "Trend", "trend"),
        _record_value(record, "Assessment", "Recommendation", "recommendation"),
    )
    with get_connection() as connection:
        _check_schema(connection)
        stock_id = _ensure_stock(connection, identity)
        cursor = connection.execute(
            """INSERT INTO research_results
               (stock_id, analysis_date, close_price, rsi, sma_20, sma_50,
                sma_200, relative_strength, research_score, trend,
                recommendation, data_source, price_basis)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (stock_id, analysis_date, *values, source, basis),
        )
        return int(cursor.lastrowid)


def save_scan_results(records: Iterable[Mapping[str, Any]],
                      scan_label: str = "Stock scan",
                      source_info: Mapping[str, Any] | None = None) -> int:
    """Append scan rows; repeated scans remain separate historical events."""
    prepared = []
    for record in records:
        ticker = _text(_record_value(record, "Symbol"), "Scan Symbol")
        identity = _stock_identity(
            ticker,
            {
                "Company_Name": _record_value(record, "Company_Name"),
                "NSE_Industry": _record_value(record, "NSE_Industry"),
            },
        )
        scan_date = _date_string(_record_value(record, "Latest_Date", "scan_date", default=date.today()))
        source, basis, _ = _source_fields(
            source_info,
            data_source=_record_value(record, "Data_Source"),
            price_basis=_record_value(record, "Price_Basis"),
        )
        notes = {
            "scan_label": scan_label,
            "trend": _record_value(record, "Trend"),
            "assessment": _record_value(record, "Assessment"),
            "relative_strength_20d_pct": _record_value(record, "Relative_Strength_20D_Pct"),
            "nse_industry": identity["industry"],
        }
        prepared.append((identity, scan_date,
                         _record_value(record, "Assessment", "signal_type"),
                         _optional_float(record, "Setup_Score", "signal_strength"),
                         _optional_float(record, "Close", "close_price"),
                         int(bool(_record_value(record, "Breakout_Confirmed", default=False))),
                         int(bool(_record_value(record, "High_Volume_Session", default=False))),
                         json.dumps(_json_safe(notes), default=_json_default, allow_nan=False),
                         source, basis))
    if not prepared:
        return 0
    with get_connection() as connection:
        _check_schema(connection)
        inserted = 0
        for identity, scan_date, signal, strength, close, breakout, volume_ok, notes, source, basis in prepared:
            stock_id = _ensure_stock(connection, identity)
            connection.execute(
                """INSERT INTO scan_results
                   (stock_id, scan_date, signal_type, signal_strength, close_price,
                    breakout_detected, volume_confirmation, notes, data_source, price_basis)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (stock_id, scan_date, signal, strength, close, breakout, volume_ok,
                 notes, source, basis),
            )
            inserted += 1
    return inserted


def save_sector_results(records: Iterable[Mapping[str, Any]],
                        analysis_date: Any,
                        source_info: Mapping[str, Any] | None = None) -> int:
    """Upsert industry-strength rows into sector_data."""
    date_value = _date_string(analysis_date)
    source, basis, _ = _source_fields(source_info)
    prepared = []
    for record in records:
        sector = _text(_record_value(record, "NSE_Industry", "sector_name"), "Industry")
        scanned = _record_value(record, "Scanned_Constituents", "stocks_count", default=0)
        try:
            scanned_count = int(scanned or 0)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid scanned constituent count for {sector}.")
        if scanned_count < 0:
            raise ValueError("Scanned constituent count cannot be negative.")
        notes = {
            "expected_constituents": _record_value(record, "Expected_Constituents"),
            "coverage_pct": _record_value(record, "Coverage_Pct"),
            "coverage_status": _record_value(record, "Coverage_Status"),
            "advancers_1d": _record_value(record, "Advancers_1D_Count"),
            "positive_return_20d": _record_value(record, "Positive_Return_20D_Count"),
            "outperforming_nifty_20d": _record_value(record, "Outperforming_NIFTY_20D_Count"),
            "sector_rank_20d": _record_value(record, "Sector_Rank_20D"),
        }
        prepared.append((sector, date_value,
                         _optional_float(record, "Sector_Return_20D_Pct", "average_return"),
                         _optional_float(record, "Sector_Relative_Strength_20D_Pct", "relative_strength"),
                         _optional_float(record, "Strength_Score", "strength_score"),
                         _record_value(record, "Strength_Status", "Trend", "trend"),
                         scanned_count,
                         json.dumps(_json_safe(notes), default=_json_default, allow_nan=False),
                         source, basis))
    if not prepared:
        return 0
    with get_connection() as connection:
        _check_schema(connection)
        connection.executemany(
            """INSERT INTO sector_data
               (sector_name, analysis_date, average_return, relative_strength,
                strength_score, trend, stocks_count, notes, data_source, price_basis)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(sector_name, analysis_date) DO UPDATE SET
                 average_return=excluded.average_return,
                 relative_strength=excluded.relative_strength,
                 strength_score=excluded.strength_score,
                 trend=excluded.trend, stocks_count=excluded.stocks_count,
                 notes=excluded.notes, data_source=excluded.data_source,
                 price_basis=excluded.price_basis""",
            prepared,
        )
    return len(prepared)


if __name__ == "__main__":
    test_connection()
    print("Stocks:", get_all_stocks())
