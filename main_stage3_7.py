"""Stage 3.7 launcher: research reports and charts.

Builds on the tested Stage 3.6 launcher. The original main.py and earlier
stage files are not edited. Stage 3.7 only adds report-generation hooks.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import time
import traceback

import pandas as pd

import main_stage3_6 as stage36  # Applies the already-tested Stage 3.6 patches.
import main_stage3_5 as app
from research_reports import generate_scan_report, generate_stock_report


REPORT_FOLDER = app.DATA_FOLDER / "research_reports"
REPORT_CHART_FOLDER = app.CHART_FOLDER / "research_reports"
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_CHART_FOLDER.mkdir(parents=True, exist_ok=True)

# Preserve the Stage 3.6-wrapped functions before adding Stage 3.7 hooks.
_stage36_save_outputs = app.save_outputs
_stage36_scan_multiple_stocks = app.scan_multiple_stocks


def save_outputs_stage37(symbol, data, create_chart=True, open_chart=False):
    """Keep the existing CSV/chart output and additionally create an HTML report."""
    result = _stage36_save_outputs(
        symbol,
        data,
        create_chart=create_chart,
        open_chart=open_chart,
    )
    if create_chart and isinstance(data, pd.DataFrame) and not data.empty:
        try:
            record = app.build_scan_record(symbol, data)
            paths = generate_stock_report(
                symbol=symbol,
                data=data,
                record=record,
                report_dir=REPORT_FOLDER,
                chart_dir=REPORT_CHART_FOLDER,
            )
            print("\nStage 3.7 stock research report:")
            print(f"Latest HTML: {paths['latest_html'].resolve()}")
            print(f"Archived HTML: {paths['archive_html'].resolve()}")
            print(f"Research chart: {paths['latest_chart'].resolve()}")
        except Exception as error:
            # Do not turn a successful stock analysis into a failed analysis
            # merely because optional report rendering failed.
            print(f"\nWarning: stock analysis completed, but Stage 3.7 report failed: {error}")
            traceback.print_exc()
    return result


def _file_mtime(path: Path):
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def scan_multiple_stocks_stage37(*args, **kwargs):
    """Run the Stage 3.6 scan, then publish a scan-level HTML dashboard."""
    scanner_path = app.DATA_FOLDER / "scanner_results_latest.csv"
    research_path = app.DATA_FOLDER / "research_explainability_latest.csv"
    sector_path = app.DATA_FOLDER / "sector_strength_latest.csv"

    previous_scanner_mtime = _file_mtime(scanner_path)
    previous_research_mtime = _file_mtime(research_path)
    previous_sector_mtime = _file_mtime(sector_path)

    started = time.perf_counter()
    _stage36_scan_multiple_stocks(*args, **kwargs)

    # A cancelled scan or failed scan should not republish an old report.
    if _file_mtime(scanner_path) is None:
        return
    if previous_scanner_mtime is not None and _file_mtime(scanner_path) <= previous_scanner_mtime:
        return
    if _file_mtime(research_path) is None:
        print("Stage 3.7 skipped: Stage 3.6 explainability CSV was not found.")
        return
    if previous_research_mtime is not None and _file_mtime(research_path) <= previous_research_mtime:
        print("Stage 3.7 skipped: no fresh Stage 3.6 explainability report was produced.")
        return

    try:
        results = pd.read_csv(scanner_path)
        research = pd.read_csv(research_path)
        if results.empty:
            print("Stage 3.7 skipped: scanner report contains no rows.")
            return

        # Add Stage 3.6 research columns without duplicating base scanner fields.
        if "Symbol" in results.columns and "Symbol" in research.columns:
            add_columns = [
                column for column in research.columns
                if column == "Symbol" or column not in results.columns
            ]
            results = results.merge(
                research[add_columns], on="Symbol", how="left", validate="one_to_one"
            )

        # Only classify the report as a NIFTY 50 dashboard when the sector
        # summary was freshly created during this scan.
        sector_frame = None
        current_sector_mtime = _file_mtime(sector_path)
        sector_is_fresh = (
            current_sector_mtime is not None
            and (
                previous_sector_mtime is None
                or current_sector_mtime > previous_sector_mtime
            )
        )
        if sector_is_fresh:
            candidate = pd.read_csv(sector_path)
            if not candidate.empty:
                sector_frame = candidate

        scan_label = kwargs.get("scan_label", "Custom watchlist scan")
        if sector_frame is not None:
            scan_label = str(scan_label or "NIFTY 50 universe scan")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = generate_scan_report(
            frame=results,
            sector_frame=sector_frame,
            report_dir=REPORT_FOLDER,
            chart_dir=REPORT_CHART_FOLDER,
            scan_label=scan_label,
            timestamp=timestamp,
        )

        print("\n" + "=" * 78)
        print("STAGE 3.7 — RESEARCH REPORT AND CHARTS")
        print("=" * 78)
        print(f"Latest HTML dashboard: {paths['latest_html'].resolve()}")
        print(f"Archived HTML dashboard: {paths['archive_html'].resolve()}")
        for key, path in paths.items():
            if key.startswith("chart_"):
                print(f"{key.replace('_', ' ').title()}: {path.resolve()}")
        print(
            "The dashboard is a presentation of the Stage 3.5/3.6 research output. "
            "It does not change scores, backtests, or trading rules."
        )
        print(f"Report preparation: {time.perf_counter() - started:.2f}s")
    except Exception as error:
        print(f"\nThe scan completed, but Stage 3.7 dashboard generation failed: {error}")
        traceback.print_exc()


# Patch only the in-memory Stage 3.5 module used by this launcher.
app.save_outputs = save_outputs_stage37
app.scan_multiple_stocks = scan_multiple_stocks_stage37


def main():
    """Stage 3.7 menu; all prior functionality remains available."""
    app.archive_existing_reports()
    while True:
        print("\n" + "=" * 62)
        print("INDIAN STOCK RESEARCH & SWING ANALYZER — STAGE 3.7")
        print("=" * 62)
        print("1. Analyze one stock (HTML report + chart)")
        print("2. Scan custom watchlist (HTML dashboard)")
        print("3. Scan NIFTY 50 (sector + relative-strength dashboard)")
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
