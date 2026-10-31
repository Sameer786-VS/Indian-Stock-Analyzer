#!/usr/bin/env python3
"""Reconcile Yahoo historical OHLCV snapshots against official NSE bhavcopy.

This script is deliberately read-only with respect to the project's existing
inputs and backtest outputs. It downloads selected NSE daily equity bhavcopies,
compares their as-traded OHLCV with Yahoo's share-count-adjusted OHLCV after
applying the known future bonus/split factors, and writes an audit folder.

Run from the project root:
    python nse_bhavcopy_reconciliation.py

Optional:
    python nse_bhavcopy_reconciliation.py --input-dir "path\\to\\input_bars"

Dependencies: pandas, requests (both commonly already installed for this project).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIRS = [
    PROJECT_ROOT / "data" / "processed" / "backtest_engine_v2" / "20261030_225252" / "input_bars",
    PROJECT_ROOT / "data" / "processed" / "backtest_engine_v21" / "20261031_095213" / "raw_input_bars",
    PROJECT_ROOT / "data" / "processed" / "backtest_engine_v21" / "20261031_095213" / "input_bars",
]

SYMBOL_FILES = {
    "TCS": "TCS_NS_input_bars.csv",
    "INFY": "INFY_NS_input_bars.csv",
    "HDFCBANK": "HDFCBANK_NS_input_bars.csv",
    "ICICIBANK": "ICICIBANK_NS_input_bars.csv",
    "RELIANCE": "RELIANCE_NS_input_bars.csv",
}

# Share-count factors: e.g. 1:1 bonus = 2 new shares for each old share.
# These are used only to translate historical NSE as-traded prices to the
# current share-count basis used by Yahoo's historical OHLC snapshots.
ACTIONS = [
    {"symbol": "TCS", "ex_date": "2018-05-31", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "INFY", "ex_date": "2018-09-04", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "HDFCBANK", "ex_date": "2019-09-19", "factor": 2.0, "action": "STOCK_SPLIT_2_FOR_1"},
    {"symbol": "HDFCBANK", "ex_date": "2025-08-26", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "ICICIBANK", "ex_date": "2017-06-20", "factor": 1.1, "action": "BONUS_1_FOR_10"},
    {"symbol": "RELIANCE", "ex_date": "2017-09-07", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "RELIANCE", "ex_date": "2024-10-28", "factor": 2.0, "action": "BONUS_1_FOR_1"},
]

# The 18-Mar-2025 session was a valid NSE session but the Yahoo rows were
# flat/zero-volume. 15-Jan-2026 was an NSE cash-equity holiday; it is not
# expected to have a bhavcopy.
SPECIAL_DATES = {
    "2025-03-18": "VALID_NSE_SESSION_YAHOO_STOCK_ROWS_INVALID",
    "2026-01-15": "OFFICIAL_NSE_CASH_EQUITY_HOLIDAY_NO_BHAVCOPY_EXPECTED",
}

NSE_CUTOVER = date(2024, 7, 8)
PRICE_TOLERANCE_PCT = 0.25


def find_input_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input directory not found: {p}")
        return p
    for p in DEFAULT_INPUT_DIRS:
        if p.exists() and any(p.glob("*_input_bars.csv")):
            return p
    raise FileNotFoundError(
        "Could not find saved V2 input_bars folder. Run from your project root, "
        "or pass --input-dir with the folder containing *_input_bars.csv files."
    )


def clean_col(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def parse_num(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s in {"", "-", "nan", "None", "null"}:
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_yahoo_snapshots(input_dir: Path) -> Dict[str, pd.DataFrame]:
    result = {}
    for symbol, filename in SYMBOL_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required input snapshot: {path}")
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            # Accommodate a date stored as the first unnamed index column.
            first = df.columns[0]
            if str(first).startswith("Unnamed"):
                df = df.rename(columns={first: "Date"})
            else:
                raise ValueError(f"{filename} has no Date column")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                raise ValueError(f"{filename} missing required column {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
        result[symbol] = df.set_index("Date", drop=False)
    return result


def sample_dates(yahoo: Dict[str, pd.DataFrame]) -> List[date]:
    wanted = set()
    targets = [(a["symbol"], date.fromisoformat(a["ex_date"])) for a in ACTIONS]
    targets += [(symbol, date.fromisoformat(ds)) for ds in SPECIAL_DATES for symbol in SYMBOL_FILES]
    for symbol, target in targets:
        idx = yahoo[symbol].index
        # Always include target date itself, then nearest available source row
        # before and after. This checks event transitions with adjacent sessions.
        wanted.add(target)
        before = [d for d in idx if d < target]
        after = [d for d in idx if d > target]
        if before:
            wanted.add(max(before))
        if after:
            wanted.add(min(after))
    # A holiday is known not to have a cash-market bhavcopy; keep it in the
    # log, but do not spend a download request on it.
    wanted.discard(date(2026, 1, 15))
    return sorted(wanted)


def build_urls(day: date) -> List[str]:
    if day < NSE_CUTOVER:
        mon = day.strftime("%b").upper()
        filename = f"cm{day.day:02d}{mon}{day.year}bhav.csv.zip"
        rel = f"content/historical/EQUITIES/{day.year}/{mon}/{filename}"
        return [f"https://archives.nseindia.com/{rel}", f"https://nsearchives.nseindia.com/{rel}"]
    filename = f"BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
    rel = f"content/cm/{filename}"
    return [f"https://nsearchives.nseindia.com/{rel}", f"https://archives.nseindia.com/{rel}"]


def fetch_zip(session: requests.Session, day: date, cache_dir: Path) -> Tuple[Optional[bytes], str, str]:
    cache_path = cache_dir / f"nse_cm_bhavcopy_{day:%Y%m%d}.zip"
    if cache_path.exists() and zipfile.is_zipfile(cache_path):
        return cache_path.read_bytes(), "CACHE", str(cache_path)

    last_error = ""
    urls = build_urls(day)
    for url in urls:
        try:
            response = session.get(url, timeout=30)
            content_type = response.headers.get("Content-Type", "")
            if response.status_code == 200 and zipfile.is_zipfile(io.BytesIO(response.content)):
                cache_path.write_bytes(response.content)
                return response.content, "DOWNLOADED", url
            snippet = response.text[:120].replace("\n", " ") if not zipfile.is_zipfile(io.BytesIO(response.content)) else ""
            last_error = f"HTTP {response.status_code}; content_type={content_type}; response={snippet}"
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.35)
    return None, "DOWNLOAD_FAILED", " | ".join(urls) + " | " + last_error


def pick_column(columns: Iterable[object], aliases: Iterable[str]) -> Optional[object]:
    lookup = {clean_col(c): c for c in columns}
    for alias in aliases:
        if clean_col(alias) in lookup:
            return lookup[clean_col(alias)]
    return None


def parse_bhavcopy(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV found inside zip: {zf.namelist()}")
        # CM final bhavcopy normally has one primary CSV. Ignore auxiliary files.
        member = max(members, key=lambda n: zf.getinfo(n).file_size)
        raw = pd.read_csv(zf.open(member), low_memory=False)

    symbol_col = pick_column(raw.columns, ["SYMBOL", "TckrSymb", "TickerSymbol"])
    series_col = pick_column(raw.columns, ["SERIES", "SctySrs", "SecuritySeries"])
    open_col = pick_column(raw.columns, ["OPEN", "OpnPric", "OpenPrice"])
    high_col = pick_column(raw.columns, ["HIGH", "HghPric", "HighPrice"])
    low_col = pick_column(raw.columns, ["LOW", "LwPric", "LowPrice"])
    close_col = pick_column(raw.columns, ["CLOSE", "ClsPric", "ClosePrice"])
    volume_col = pick_column(raw.columns, ["TOTTRDQTY", "TtlTradgVol", "TotalTradedQuantity", "TtlTradgVol"])
    if not all([symbol_col, open_col, high_col, low_col, close_col, volume_col]):
        raise ValueError(f"Unrecognised NSE bhavcopy schema. Columns: {list(raw.columns)}")

    out = pd.DataFrame({
        "NSE_Symbol": raw[symbol_col].astype(str).str.strip().str.upper(),
        "NSE_Series": raw[series_col].astype(str).str.strip().str.upper() if series_col else "",
        "NSE_Open": pd.to_numeric(raw[open_col], errors="coerce"),
        "NSE_High": pd.to_numeric(raw[high_col], errors="coerce"),
        "NSE_Low": pd.to_numeric(raw[low_col], errors="coerce"),
        "NSE_Close": pd.to_numeric(raw[close_col], errors="coerce"),
        "NSE_Volume": pd.to_numeric(raw[volume_col], errors="coerce"),
    })
    # The stock universe here is ordinary NSE equity series. Prefer EQ to
    # exclude BE/BL/other special series when the field is available.
    out = out[out["NSE_Symbol"].isin(SYMBOL_FILES)]
    if (out["NSE_Series"] == "EQ").any():
        out = out[out["NSE_Series"] == "EQ"]
    return out.drop_duplicates("NSE_Symbol", keep="last")


def future_share_factor(symbol: str, day: date) -> float:
    factor = 1.0
    for action in ACTIONS:
        if action["symbol"] == symbol and date.fromisoformat(action["ex_date"]) > day:
            factor *= float(action["factor"])
    return factor


def pct_diff(actual: Optional[float], expected: Optional[float]) -> Optional[float]:
    if actual is None or expected is None or expected == 0:
        return None
    return (actual / expected - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Yahoo stock OHLCV against official NSE bhavcopy.")
    parser.add_argument("--input-dir", default=None, help="Folder containing five *_input_bars.csv snapshots.")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between NSE requests (default 0.5).")
    args = parser.parse_args()

    input_dir = find_input_dir(args.input_dir)
    yahoo = load_yahoo_snapshots(input_dir)
    dates = sample_dates(yahoo)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "data" / "processed" / "nse_bhavcopy_reconciliation" / run_id
    cache_dir = out_dir / "bhavcopy_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("INDIAN STOCK ANALYZER — NSE BHAVCOPY RECONCILIATION")
    print(f"Input snapshots: {input_dir}")
    print(f"Output folder:   {out_dir}")
    print(f"Unique NSE dates to retrieve: {len(dates)}")
    print("This script does not change main.py, V2, or V2.1 files.")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    })
    try:
        session.get("https://www.nseindia.com/", timeout=20)
    except requests.RequestException:
        pass

    bhav_by_date: Dict[date, pd.DataFrame] = {}
    download_log = []
    # Log the known 15-Jan-2026 exchange holiday explicitly; no bhavcopy is expected.
    download_log.append({
        "Date": "2026-01-15",
        "Status": "SKIPPED_OFFICIAL_CASH_EQUITY_HOLIDAY",
        "Details": SPECIAL_DATES["2026-01-15"],
        "Matched_Stock_Rows": 0,
    })

    for i, day in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] NSE bhavcopy {day} ...", flush=True)
        blob, status, detail = fetch_zip(session, day, cache_dir)
        row = {"Date": day.isoformat(), "Status": status, "Details": detail}
        if blob is not None:
            try:
                bhav_by_date[day] = parse_bhavcopy(blob)
                row["Matched_Stock_Rows"] = int(len(bhav_by_date[day]))
                row["Status"] = "DOWNLOADED_AND_PARSED" if status != "CACHE" else "CACHE_AND_PARSED"
            except Exception as exc:
                row["Status"] = "PARSE_FAILED"
                row["Details"] = f"{detail} | {type(exc).__name__}: {exc}"
        else:
            row["Matched_Stock_Rows"] = 0
        download_log.append(row)
        time.sleep(max(0.0, args.delay))

    compare_rows = []
    for day in dates:
        nse_df = bhav_by_date.get(day)
        nse_map = {} if nse_df is None else {r["NSE_Symbol"]: r for _, r in nse_df.iterrows()}
        for symbol, yahoo_df in yahoo.items():
            if day not in yahoo_df.index:
                continue
            y = yahoo_df.loc[day]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[-1]
            n = nse_map.get(symbol)
            factor = future_share_factor(symbol, day)
            row = {
                "Date": day.isoformat(), "Symbol": symbol,
                "Yahoo_Open": y.get("Open"), "Yahoo_High": y.get("High"),
                "Yahoo_Low": y.get("Low"), "Yahoo_Close": y.get("Close"),
                "Yahoo_Volume": y.get("Volume"),
                "Future_Share_Factor": factor,
                "Expected_Yahoo_Price_From_NSE": None,
                "Observed_NSE_Close_Divided_By_Yahoo_Close": None,
                "Max_Absolute_OHLC_Difference_Pct": None,
                "Volume_Difference_Pct": None,
                "NSE_Series": None,
                "NSE_Open": None, "NSE_High": None, "NSE_Low": None,
                "NSE_Close": None, "NSE_Volume": None,
                "Result": ("NSE_BHAVCOPY_UNAVAILABLE_OR_PARSE_FAILED" if nse_df is None else "SYMBOL_ROW_NOT_FOUND_IN_NSE_BHAVCOPY"),
            }
            if n is not None:
                raw_prices = [parse_num(n.get(f"NSE_{k}")) for k in ["Open", "High", "Low", "Close"]]
                yahoo_prices = [parse_num(y.get(k)) for k in ["Open", "High", "Low", "Close"]]
                expected_prices = [v / factor if v is not None else None for v in raw_prices]
                diffs = [abs(pct_diff(yp, ep)) for yp, ep in zip(yahoo_prices, expected_prices) if yp is not None and ep is not None]
                raw_close = parse_num(n.get("NSE_Close"))
                yahoo_close = parse_num(y.get("Close"))
                raw_volume = parse_num(n.get("NSE_Volume"))
                yahoo_volume = parse_num(y.get("Volume"))
                expected_volume = raw_volume * factor if raw_volume is not None else None
                vdiff = pct_diff(yahoo_volume, expected_volume)
                max_diff = max(diffs) if diffs else None
                result = "PRICE_BASIS_MATCH" if max_diff is not None and max_diff <= PRICE_TOLERANCE_PCT else "PRICE_BASIS_REVIEW"
                if raw_close is not None and yahoo_close:
                    observed_factor = raw_close / yahoo_close
                else:
                    observed_factor = None
                row.update({
                    "Expected_Yahoo_Price_From_NSE": expected_prices[3],
                    "Observed_NSE_Close_Divided_By_Yahoo_Close": observed_factor,
                    "Max_Absolute_OHLC_Difference_Pct": max_diff,
                    "Volume_Difference_Pct": vdiff,
                    "NSE_Series": n.get("NSE_Series"),
                    "NSE_Open": raw_prices[0], "NSE_High": raw_prices[1],
                    "NSE_Low": raw_prices[2], "NSE_Close": raw_prices[3],
                    "NSE_Volume": raw_volume,
                    "Result": result,
                })
            elif day.isoformat() == "2025-03-18" and nse_df is not None:
                row["Result"] = "OFFICIAL_NSE_FILE_AVAILABLE_BUT_SYMBOL_ROW_NOT_FOUND"
            compare_rows.append(row)

    pd.DataFrame(download_log).to_csv(out_dir / "nse_download_log.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(out_dir / "yahoo_vs_nse_reconciliation.csv", index=False)

    summary = []
    cmp_df = pd.DataFrame(compare_rows)
    for symbol in SYMBOL_FILES:
        subset = cmp_df[cmp_df["Symbol"] == symbol] if not cmp_df.empty else pd.DataFrame()
        matched = subset[subset["Result"].isin(["PRICE_BASIS_MATCH", "PRICE_BASIS_REVIEW"])] if not subset.empty else pd.DataFrame()
        summary.append({
            "Symbol": symbol,
            "Yahoo_Sample_Rows": int(len(subset)),
            "NSE_Matched_Rows": int(len(matched)),
            "Price_Basis_Matches": int((matched["Result"] == "PRICE_BASIS_MATCH").sum()) if not matched.empty else 0,
            "Price_Basis_Reviews": int((matched["Result"] == "PRICE_BASIS_REVIEW").sum()) if not matched.empty else 0,
            "Missing_NSE_Rows": int(subset["Result"].isin(["NSE_BHAVCOPY_UNAVAILABLE_OR_PARSE_FAILED", "SYMBOL_ROW_NOT_FOUND_IN_NSE_BHAVCOPY", "OFFICIAL_NSE_FILE_AVAILABLE_BUT_SYMBOL_ROW_NOT_FOUND"]).sum()) if not subset.empty else 0,
        })
    pd.DataFrame(summary).to_csv(out_dir / "reconciliation_summary.csv", index=False)

    notes = f"""NSE BHAVCOPY RECONCILIATION\nRun: {run_id}\nInput: {input_dir}\n\nPurpose\nCompare the saved Yahoo OHLCV sample with NSE official daily equity bhavcopy prices.\nThe expected Yahoo share-basis value is calculated as NSE as-traded OHLC divided\nby the product of known future bonus/split share factors after each sample date.\nHistorical Yahoo volume is expected to be NSE volume multiplied by that same factor.\n\nImportant limits\n- This is a sample audit around known corporate-action dates and flagged dates, not\n  a complete bar-by-bar certification of the entire 2016-2026 dataset.\n- An NSE download failure is not proof of a holiday; inspect nse_download_log.csv.\n- 15-Jan-2026 is treated as a known cash-equity holiday and is not downloaded.\n- A match within {PRICE_TOLERANCE_PCT:.2f}% is a price-basis consistency check, not a guarantee\n  that every historical bar is correct.\n- No project input or existing backtest output is overwritten.\n\nFiles\n- nse_download_log.csv: URL/download/parse status for each date.\n- yahoo_vs_nse_reconciliation.csv: row-by-row NSE vs Yahoo comparison.\n- reconciliation_summary.csv: per-stock match counts.\n- bhavcopy_cache/: downloaded official NSE ZIP files.\n\nIf NSE blocks downloads, use the official NSE All Reports page, select the\nCM-UDiFF Common Bhavcopy Final (zip) for dates on/after 08-Jul-2024, or the\nlegacy CM Bhavcopy archive for earlier dates, save ZIPs in bhavcopy_cache using\nnse_cm_bhavcopy_YYYYMMDD.zip naming, then rerun the script.\n"""
    (out_dir / "README_AUDIT.txt").write_text(notes, encoding="utf-8")

    print("\nSaved reconciliation reports:")
    print(f"  {out_dir / 'nse_download_log.csv'}")
    print(f"  {out_dir / 'yahoo_vs_nse_reconciliation.csv'}")
    print(f"  {out_dir / 'reconciliation_summary.csv'}")
    print(f"  {out_dir / 'README_AUDIT.txt'}")
    print("\nUpload the entire output folder as a ZIP for review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
