"""Stage 3.6 launcher. Keeps Stage 3.5 code and main.py unchanged."""
from __future__ import annotations

from datetime import datetime
import time

import pandas as pd

import main_stage3_5 as app
from research_explainability import (
    add_peer_comparisons,
    build_research_report_columns,
    enrich_scan_record,
)


# Keep references to the fully tested Stage 3.5 functions.
_stage35_build_scan_record = app.build_scan_record
_stage35_show_summary = app.show_summary
_stage35_build_sector_strength_report = app.build_sector_strength_report
_stage35_scan_multiple_stocks = app.scan_multiple_stocks


def build_scan_record_stage36(symbol, data):
    """Keep the original score; append transparent research explanations."""
    record = _stage35_build_scan_record(symbol, data)
    return enrich_scan_record(record)


def show_summary_stage36(symbol, data):
    """Preserve the existing analysis output and append plain-language context."""
    _stage35_show_summary(symbol, data)
    record = build_scan_record_stage36(symbol, data)
    print("\n--- STAGE 3.6: EXPLAINABLE RESEARCH ---")
    print(f"Trend evidence:          {record['Research_Trend_State']}")
    print(f"Momentum evidence:      {record['Research_Momentum_State']}")
    print(f"Benchmark comparison:   {record['Research_Benchmark_State']}")
    print(f"Volume context:         {record['Research_Volume_State']}")
    print(f"Price-action evidence:  {record['Research_Price_Action_State']}")
    print(f"Core score explanation: {record['Core_Score_Explanation']}")
    print(f"Research conclusion:    {record['Research_Conclusion']}")
    print(f"Positive evidence:      {record['Research_Positive_Evidence']}")
    print(f"Cautions:               {record['Research_Caution_Evidence']}")
    print(f"Missing data:           {record['Research_Missing_Evidence']}")
    print(record["Research_Use_Limitation"])


def build_sector_strength_stage36(stock_results, universe_metadata):
    """Use the Stage 3.5 sector summary, but compare stocks with other peers."""
    stocks, sectors = _stage35_build_sector_strength_report(
        stock_results, universe_metadata
    )
    return add_peer_comparisons(stocks, sectors)


def scan_multiple_stocks_stage36(*args, **kwargs):
    """Run Stage 3.5 scan, then save a compact explainability report."""
    started = time.perf_counter()
    latest_path = app.DATA_FOLDER / "scanner_results_latest.csv"
    previous_mtime = (
        latest_path.stat().st_mtime_ns if latest_path.exists() else None
    )
    _stage35_scan_multiple_stocks(*args, **kwargs)

    if not latest_path.exists():
        return
    current_mtime = latest_path.stat().st_mtime_ns
    if previous_mtime is not None and current_mtime <= previous_mtime:
        # The Stage 3.5 scanner did not produce a fresh report (for example,
        # the user cancelled input). Do not re-publish an old scan as current.
        return

    try:
        results = pd.read_csv(latest_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        print(f"\nCould not read scanner output for Stage 3.6: {error}")
        return

    if results.empty:
        return

    research = build_research_report_columns(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = (
        app.REPORT_ARCHIVE_FOLDER
        / f"research_explainability_{timestamp}.csv"
    )
    latest_research_path = app.DATA_FOLDER / "research_explainability_latest.csv"

    try:
        research.to_csv(archive_path, index=False)
        research.to_csv(latest_research_path, index=False)
    except OSError as error:
        print(f"\nCould not save Stage 3.6 research report: {error}")
        return

    print("\n" + "=" * 110)
    print("STAGE 3.6 — EXPLAINABLE RESEARCH SUMMARY")
    print("=" * 110)
    display = [
        column for column in [
            "Symbol", "Setup_Score", "Core_Rules_Passed",
            "Core_Rule_Completeness_Pct", "Research_Trend_State",
            "Research_Momentum_State", "Research_Benchmark_State",
            "Research_Conclusion", "Stock_vs_Sector_20D_Pct",
            "Peer_Comparison_Quality",
        ] if column in research.columns
    ]
    print(research[display].round(2).to_string(index=False))
    print(f"\nExplainability report: {archive_path.resolve()}")
    print(f"Latest explainability report: {latest_research_path.resolve()}")
    print(
        "Note: Stage 3.6 adds context and peer comparisons; it does not "
        "change the original setup score or establish a profitable strategy."
    )
    print(f"Stage 3.6 report preparation: {time.perf_counter() - started:.2f}s")


# Patch the imported Stage 3.5 module in memory only. No source file is edited.
app.build_scan_record = build_scan_record_stage36
app.show_summary = show_summary_stage36
app.build_sector_strength_report = build_sector_strength_stage36
app.scan_multiple_stocks = scan_multiple_stocks_stage36


def main():
    """Stage 3.6 menu using the existing Stage 3.5 features."""
    app.archive_existing_reports()
    while True:
        print("\n" + "=" * 58)
        print("INDIAN STOCK RESEARCH & SWING ANALYZER — STAGE 3.6")
        print("=" * 58)
        print("1. Analyze one stock")
        print("2. Scan custom watchlist")
        print("3. Scan NIFTY 50")
        print("4. Backtest one stock (historical simulation)")
        print("5. Exit")

        choice = input("\nChoose an option (1-5): ").strip()
        if choice == "1":
            app.analyze_stock()
        elif choice == "2":
            app.scan_multiple_stocks()
        elif choice == "3":
            app.scan_nifty50_universe()
        elif choice == "4":
            app.backtest_one_stock()
        elif choice == "5":
            print("\nProgram closed.")
            break
        else:
            print("\nInvalid choice. Enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
