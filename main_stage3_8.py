"""Stage 3.8 launcher: Stage 3.7 research workflow plus SQLite persistence.

This is an additive launcher. It imports the tested Stage 3.7/3.6/3.5
application, then wraps a few runtime functions to persist validated market
history and completed research outputs. It does not edit any earlier stage.

Database requirement: indian_stock_research.db with PRAGMA user_version = 2.
Analytical market history is stored in historical_prices / benchmark_prices.
The separate execution_prices table is intentionally not populated here.
"""
from __future__ import annotations

from pathlib import Path
import threading
import traceback
from typing import Any

import pandas as pd

import main_stage3_7 as stage37  # Applies the already-tested Stage 3.7 hooks.
import main_stage3_5 as app
import db_manager_v2 as db


DATABASE_PRICE_BASIS = "adjusted_analytical_ohlc"
BENCHMARK_PROVIDER_SYMBOL = "^NSEI"

_STATS_LOCK = threading.Lock()
_SOURCE_LOCK = threading.Lock()
_IN_INDIVIDUAL_ANALYSIS = False
_SOURCE_INFO_BY_SYMBOL: dict[str, dict[str, Any]] = {}
PERSISTENCE_STATS = {
    "stock_price_rows": 0,
    "benchmark_price_rows": 0,
    "research_rows": 0,
    "scan_rows": 0,
    "sector_rows": 0,
    "write_errors": 0,
}


# Keep references to the Stage 3.7/3.6-wrapped functions before adding Stage 3.8.
_stage37_download_stock_data = app.download_stock_data
_stage37_build_scan_record = app.build_scan_record
_stage37_save_outputs = app.save_outputs
_stage37_scan_multiple_stocks = app.scan_multiple_stocks
_stage37_analyze_stock = app.analyze_stock


def _add_stat(key: str, amount: int = 1) -> None:
    with _STATS_LOCK:
        PERSISTENCE_STATS[key] = PERSISTENCE_STATS.get(key, 0) + amount


def _source_info_for_frame(data: pd.DataFrame, symbol: str | None = None) -> dict[str, Any]:
    """Return stable provider and price-basis provenance for a downloaded frame."""
    info = {}
    if isinstance(data, pd.DataFrame):
        info.update(data.attrs.get("market_data", {}) or {})

    if symbol:
        with _SOURCE_LOCK:
            saved = _SOURCE_INFO_BY_SYMBOL.get(str(symbol).upper(), {})
        for key, value in saved.items():
            if info.get(key) in (None, "", "unknown"):
                info[key] = value

    source = info.get("source") or "market_data_manager"
    retrieved = info.get("downloaded_at_utc")
    return {
        "source": str(source),
        # The Stage 3.1 manager produces adjusted analytical OHLC. It is not
        # an exchange-verified as-traded execution-price series.
        "price_basis": DATABASE_PRICE_BASIS,
        "downloaded_at_utc": retrieved,
    }


def _metadata_for_symbol(yahoo_symbol: str) -> dict[str, Any]:
    """Use the dated NIFTY 50 snapshot when available; other tickers stay valid."""
    metadata = getattr(app, "NIFTY50_METADATA", {}).get(
        str(yahoo_symbol).upper(), {}
    )
    return dict(metadata or {})


def persist_downloaded_data(yahoo_symbol: str, data: pd.DataFrame) -> int:
    """Persist a raw validated download to the stock or benchmark price table."""
    ticker = str(yahoo_symbol).strip().upper()
    source_info = _source_info_for_frame(data, ticker)

    if ticker == BENCHMARK_PROVIDER_SYMBOL:
        rows = db.save_benchmark_history(
            data,
            benchmark_symbol="NIFTY50",
            benchmark_name="NIFTY 50",
            provider_symbol=ticker,
            source_info=source_info,
        )
        _add_stat("benchmark_price_rows", rows)
        return rows

    rows = db.save_stock_history(
        yahoo_symbol=ticker,
        frame=data,
        metadata=_metadata_for_symbol(ticker),
        source_info=source_info,
    )
    _add_stat("stock_price_rows", rows)
    return rows


def download_stock_data_stage38(*args, **kwargs):
    """Preserve Stage 3.5 download/cache behavior and persist its returned bars."""
    data = _stage37_download_stock_data(*args, **kwargs)
    symbol = str(args[0] if args else kwargs.get("symbol", "")).strip().upper()
    if not symbol:
        return data

    info = _source_info_for_frame(data, symbol)
    with _SOURCE_LOCK:
        _SOURCE_INFO_BY_SYMBOL[symbol] = info

    try:
        rows = persist_downloaded_data(symbol, data)
        print(f"SQLite: saved {rows} daily row(s) for {symbol}.")
    except Exception as error:
        _add_stat("write_errors")
        print(
            f"WARNING: {symbol} analysis can continue, but its market history "
            f"was not saved to SQLite: {error}"
        )
    return data


def build_scan_record_stage38(symbol: str, data: pd.DataFrame) -> dict[str, Any]:
    """Retain Stage 3.6 explanations and append data provenance fields."""
    record = _stage37_build_scan_record(symbol, data)
    info = _source_info_for_frame(data, symbol)
    record["Data_Source"] = info["source"]
    record["Price_Basis"] = info["price_basis"]
    if info.get("downloaded_at_utc"):
        record["Data_Retrieved_At_UTC"] = info["downloaded_at_utc"]
    return record


def _research_source_info(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": record.get("Data_Source") or "market_data_manager",
        "price_basis": record.get("Price_Basis") or DATABASE_PRICE_BASIS,
        "downloaded_at_utc": record.get("Data_Retrieved_At_UTC"),
    }


def _save_individual_research(symbol: str, data: pd.DataFrame) -> None:
    record = app.build_scan_record(symbol, data)
    research_id = db.save_research_result(
        record,
        source_info=_research_source_info(record),
        metadata=_metadata_for_symbol(symbol),
    )
    _add_stat("research_rows")
    print(f"SQLite: saved individual research record #{research_id} for {symbol}.")


def save_outputs_stage38(symbol, data, create_chart=True, open_chart=False):
    """Keep Stage 3.7 output/report behavior; persist individual analysis once."""
    result = _stage37_save_outputs(
        symbol,
        data,
        create_chart=create_chart,
        open_chart=open_chart,
    )
    if _IN_INDIVIDUAL_ANALYSIS and isinstance(data, pd.DataFrame) and not data.empty:
        try:
            _save_individual_research(symbol, data)
        except Exception as error:
            _add_stat("write_errors")
            print(f"WARNING: individual research result was not saved: {error}")
            traceback.print_exc()
    return result


def analyze_stock_stage38(*args, **kwargs):
    """Mark the single-stock flow so its final processed snapshot is persisted."""
    global _IN_INDIVIDUAL_ANALYSIS
    previous = _IN_INDIVIDUAL_ANALYSIS
    _IN_INDIVIDUAL_ANALYSIS = True
    try:
        return _stage37_analyze_stock(*args, **kwargs)
    finally:
        _IN_INDIVIDUAL_ANALYSIS = previous


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _resolve_sector_analysis_date(
    scanner_results: pd.DataFrame,
    explicit_date: Any = None,
) -> str:
    """Use an explicit date or the latest market date represented in scan rows.

    Never substitute the computer's current date: sector results describe the
    market-data snapshot, not the time at which the scan happened to run.
    """
    if explicit_date is not None and str(explicit_date).strip():
        parsed = pd.to_datetime(explicit_date, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid sector analysis date: {explicit_date!r}")
        return parsed.date().isoformat()

    if "Latest_Date" not in scanner_results.columns:
        raise ValueError(
            "Cannot date sector results: scanner results do not contain Latest_Date."
        )

    dates = pd.to_datetime(scanner_results["Latest_Date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(
            "Cannot date sector results: no valid Latest_Date values were found."
        )

    return dates.max().date().isoformat()


def persist_scan_outputs(
    scanner_results: pd.DataFrame,
    sector_results: pd.DataFrame | None = None,
    scan_label: str = "Stock scan",
    sector_analysis_date: Any = None,
) -> dict[str, int]:
    """Persist completed scanner rows, research snapshots and optional sectors."""
    if scanner_results is None or scanner_results.empty:
        return {"scan_rows": 0, "research_rows": 0, "sector_rows": 0}

    resolved_sector_date = None
    if sector_results is not None and not sector_results.empty:
        resolved_sector_date = _resolve_sector_analysis_date(
            scanner_results, sector_analysis_date
        )

    records = scanner_results.to_dict(orient="records")
    scan_count = db.save_scan_results(records, scan_label=scan_label)
    _add_stat("scan_rows", scan_count)

    research_count = 0
    for record in records:
        symbol = str(record.get("Symbol", "")).upper()
        try:
            db.save_research_result(
                record,
                source_info=_research_source_info(record),
                metadata=_metadata_for_symbol(symbol),
            )
            research_count += 1
        except Exception as error:
            _add_stat("write_errors")
            print(f"WARNING: research row for {symbol} was not saved: {error}")

    _add_stat("research_rows", research_count)

    sector_count = 0
    if sector_results is not None and not sector_results.empty:
        sector_count = db.save_sector_results(
            sector_results.to_dict(orient="records"),
            analysis_date=resolved_sector_date,
            source_info={
                "source": "stage3.5_market_data_manager",
                "price_basis": DATABASE_PRICE_BASIS,
            },
        )
        _add_stat("sector_rows", sector_count)

    return {
        "scan_rows": scan_count,
        "research_rows": research_count,
        "sector_rows": sector_count,
    }


def scan_multiple_stocks_stage38(*args, **kwargs):
    """Run all Stage 3.7 scan/report hooks, then persist only fresh CSV outputs."""
    scanner_path = app.DATA_FOLDER / "scanner_results_latest.csv"
    sector_path = app.DATA_FOLDER / "sector_strength_latest.csv"
    previous_scanner = _file_fingerprint(scanner_path)
    previous_sector = _file_fingerprint(sector_path)

    result = _stage37_scan_multiple_stocks(*args, **kwargs)

    current_scanner = _file_fingerprint(scanner_path)
    if current_scanner is None or current_scanner == previous_scanner:
        return result

    try:
        scanner_frame = pd.read_csv(scanner_path)
        if scanner_frame.empty:
            print("SQLite: scan export is empty; no scan records saved.")
            return result

        current_sector = _file_fingerprint(sector_path)
        sector_frame = None
        if current_sector is not None and current_sector != previous_sector:
            candidate = pd.read_csv(sector_path)
            if not candidate.empty:
                sector_frame = candidate

        scan_label = kwargs.get("scan_label", "Custom watchlist scan")
        saved = persist_scan_outputs(
            scanner_results=scanner_frame,
            sector_results=sector_frame,
            scan_label=scan_label,
        )
        print(
            "SQLite persistence: "
            f"{saved['scan_rows']} scan row(s), "
            f"{saved['research_rows']} research row(s), "
            f"{saved['sector_rows']} sector row(s)."
        )
    except Exception as error:
        _add_stat("write_errors")
        print(f"WARNING: scan completed, but SQLite scan persistence failed: {error}")
        traceback.print_exc()

    return result


def print_database_status() -> None:
    """Print current SQLite record counts and write-error count."""
    with db.get_connection() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        print("\n--- SQLITE DATABASE STATUS ---")
        print(f"Schema version: {version}")
        for table in (
            "stocks", "historical_prices", "benchmarks", "benchmark_prices",
            "research_results", "scan_results", "sector_data", "execution_prices",
        ):
            count = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            print(f"{table}: {count}")
    print(f"SQLite write errors this session: {PERSISTENCE_STATS['write_errors']}")


def main() -> None:
    """Stage 3.8 menu: original research features plus SQLite persistence."""
    db.test_connection()
    app.archive_existing_reports()

    while True:
        print("\n" + "=" * 68)
        print("INDIAN STOCK RESEARCH & SWING ANALYZER — STAGE 3.8")
        print("SQLite persistence is enabled; Stage 3.7 research/report logic is retained.")
        print("=" * 68)
        print("1. Analyze one stock (HTML report + SQLite research record)")
        print("2. Scan custom watchlist (HTML dashboard + SQLite records)")
        print("3. Scan NIFTY 50 (sector dashboard + SQLite records)")
        print("4. Backtest one stock (historical simulation)")
        print("5. View SQLite database status")
        print("6. Exit")

        choice = input("\nChoose an option (1-6): ").strip()
        if choice == "1":
            app.analyze_stock()
        elif choice == "2":
            app.scan_multiple_stocks()
        elif choice == "3":
            app.scan_nifty50_universe()
        elif choice == "4":
            app.backtest_one_stock()
        elif choice == "5":
            print_database_status()
        elif choice == "6":
            print("\nProgram closed.")
            break
        else:
            print("\nInvalid choice. Enter 1, 2, 3, 4, 5, or 6.")


# Apply Stage 3.8 runtime wrappers only to the already-imported Stage 3.5 module.
app.download_stock_data = download_stock_data_stage38
app.build_scan_record = build_scan_record_stage38
app.save_outputs = save_outputs_stage38
app.analyze_stock = analyze_stock_stage38
app.scan_multiple_stocks = scan_multiple_stocks_stage38


if __name__ == "__main__":
    main()
