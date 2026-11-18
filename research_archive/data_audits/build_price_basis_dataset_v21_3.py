#!/usr/bin/env python3
"""Build a dual-basis historical OHLCV dataset for Indian Stock Analyzer.

V2.1.3 is an offline, non-destructive data preparation step. It preserves the
Yahoo quote series as received, reconstructs an estimated as-traded series by
reversing verified share-count and empirically observed non-share adjustments,
and replaces the known invalid 2025-03-18 stock bars with official NSE bhavcopy
OHLCV. It does NOT run a backtest and does NOT overwrite main.py, V2, or V2.1.

Run from the project root:
    python build_price_basis_dataset_v21_3.py

Optional paths:
    python build_price_basis_dataset_v21_3.py --input-dir "...\\raw_input_bars"
    python build_price_basis_dataset_v21_3.py --audit-dir "...\\20261031_110958"

Dependencies: pandas, numpy (normally already installed for the project).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

STOCK_FILES = {
    "TCS": "TCS_NS_input_bars.csv",
    "INFY": "INFY_NS_input_bars.csv",
    "HDFCBANK": "HDFCBANK_NS_input_bars.csv",
    "ICICIBANK": "ICICIBANK_NS_input_bars.csv",
    "RELIANCE": "RELIANCE_NS_input_bars.csv",
}
INDEX_FILES = {"NSEI": "NSEI_input_bars.csv"}
ALL_FILES = {**STOCK_FILES, **INDEX_FILES}

# Product of future share-count events after the bar date. A 1:1 bonus/split
# doubles the number of shares and therefore halves the adjusted price.
SHARE_ACTIONS = [
    {"symbol": "TCS", "ex_date": "2018-05-31", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "INFY", "ex_date": "2018-09-04", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "HDFCBANK", "ex_date": "2019-09-19", "factor": 2.0, "action": "STOCK_SPLIT_2_FOR_1"},
    {"symbol": "HDFCBANK", "ex_date": "2025-08-26", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "ICICIBANK", "ex_date": "2017-06-20", "factor": 1.1, "action": "BONUS_1_FOR_10"},
    {"symbol": "RELIANCE", "ex_date": "2017-09-07", "factor": 2.0, "action": "BONUS_1_FOR_1"},
    {"symbol": "RELIANCE", "ex_date": "2024-10-28", "factor": 2.0, "action": "BONUS_1_FOR_1"},
]

# Reliance 2020 rights issue: 1 new share for every 15 held at Rs 1,257.
# Theoretical ex-rights adjustment factor = TERP / previous close.
RIGHTS_EX_DATE = date(2020, 5, 13)
RIGHTS_PREVIOUS_CLOSE = 1479.25  # NSE close on 2020-05-12
RIGHTS_ISSUE_PRICE = 1257.00
RIGHTS_RATIO_EXISTING = 15
RIGHTS_TERP = (RIGHTS_RATIO_EXISTING * RIGHTS_PREVIOUS_CLOSE + RIGHTS_ISSUE_PRICE) / (RIGHTS_RATIO_EXISTING + 1)
RIGHTS_PRICE_FACTOR = RIGHTS_TERP / RIGHTS_PREVIOUS_CLOSE

# Reliance / Jio Financial demerger. The factor below is the stable observed
# Yahoo-vs-NSE historical OHLC adjustment inferred from the 2023-07-19/20/21
# event window and then checked across all other sampled NSE dates. It is a
# PROVIDER BASIS FACTOR, not a share-count multiplier or an entitlement ratio.
DEMERGER_EX_DATE = date(2023, 7, 20)
DEMERGER_PROVIDER_PRICE_FACTOR = 0.9229996751041144
JFS_ENTITLEMENT_PER_RIL_SHARE = 1.0
JFS_DISCOVERED_VALUE_ON_DEMERGER = 261.85  # Rs per JFS share; mark/reference only

HOLIDAY_DATES = {date(2026, 1, 15): "OFFICIAL_NSE_CASH_EQUITY_HOLIDAY"}
REPAIR_DATE = date(2025, 3, 18)
PRICE_TOLERANCE_PCT = 0.25
VOLUME_TOLERANCE_PCT = 0.50


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def newest_dir(paths):
    existing = [p for p in paths if p.exists()]
    return sorted(existing, key=lambda p: p.name, reverse=True)[0] if existing else None


def find_input_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input directory does not exist: {p}")
        return p
    v21_root = PROJECT_ROOT / "data" / "processed" / "backtest_engine_v21"
    raw_dirs = sorted(v21_root.glob("*/raw_input_bars"), key=lambda p: p.parent.name, reverse=True)
    for p in raw_dirs:
        if all((p / f).exists() for f in ALL_FILES.values()):
            return p
    v2_root = PROJECT_ROOT / "data" / "processed" / "backtest_engine_v2"
    v2_dirs = sorted(v2_root.glob("*/input_bars"), key=lambda p: p.parent.name, reverse=True)
    for p in v2_dirs:
        if all((p / f).exists() for f in ALL_FILES.values()):
            return p
    raise FileNotFoundError(
        "Could not find the six input snapshots. Expected V2.1 raw_input_bars or V2 input_bars. "
        "Pass --input-dir with the folder containing the six *_input_bars.csv files."
    )


def find_audit_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Audit directory does not exist: {p}")
        return p
    root = PROJECT_ROOT / "data" / "processed" / "nse_corporate_action_event_audit"
    candidates = [p for p in root.glob("*") if (p / "bhavcopy_cache").exists()]
    selected = newest_dir(candidates)
    if selected is None:
        raise FileNotFoundError(
            f"Could not find an NSE audit folder with bhavcopy_cache under {root}. "
            "Pass --audit-dir with your event-window audit output folder."
        )
    required = ["2025-03-18_replacement_candidates.csv", "yahoo_vs_nse_reconciliation.csv"]
    missing = [n for n in required if not (selected / n).exists()]
    if missing:
        raise FileNotFoundError(f"NSE audit folder is missing {missing}: {selected}")
    return selected


def clean_col(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def pick_col(columns, aliases):
    lookup = {clean_col(c): c for c in columns}
    for alias in aliases:
        if clean_col(alias) in lookup:
            return lookup[clean_col(alias)]
    return None


def parse_number(v):
    if v is None or pd.isna(v):
        return np.nan
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return np.nan


def parse_bhavcopy(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No CSV inside bhavcopy ZIP: {zf.namelist()}")
        member = max(csvs, key=lambda n: zf.getinfo(n).file_size)
        raw = pd.read_csv(zf.open(member), low_memory=False)
    cols = {
        "symbol": pick_col(raw.columns, ["SYMBOL", "TckrSymb", "TickerSymbol"]),
        "series": pick_col(raw.columns, ["SERIES", "SctySrs", "SecuritySeries"]),
        "open": pick_col(raw.columns, ["OPEN", "OpnPric", "OpenPrice"]),
        "high": pick_col(raw.columns, ["HIGH", "HghPric", "HighPrice"]),
        "low": pick_col(raw.columns, ["LOW", "LwPric", "LowPrice"]),
        "close": pick_col(raw.columns, ["CLOSE", "ClsPric", "ClosePrice"]),
        "volume": pick_col(raw.columns, ["TOTTRDQTY", "TtlTradgVol", "TotalTradedQuantity"]),
    }
    if not all(cols[k] for k in ["symbol", "open", "high", "low", "close", "volume"]):
        raise ValueError(f"Unrecognised NSE bhavcopy columns: {list(raw.columns)}")
    out = pd.DataFrame({
        "Symbol": raw[cols["symbol"]].astype(str).str.strip().str.upper(),
        "Series": raw[cols["series"]].astype(str).str.strip().str.upper() if cols["series"] else "",
        "Open": pd.to_numeric(raw[cols["open"]], errors="coerce"),
        "High": pd.to_numeric(raw[cols["high"]], errors="coerce"),
        "Low": pd.to_numeric(raw[cols["low"]], errors="coerce"),
        "Close": pd.to_numeric(raw[cols["close"]], errors="coerce"),
        "Volume": pd.to_numeric(raw[cols["volume"]], errors="coerce"),
    })
    out = out.dropna(subset=["Symbol", "Open", "High", "Low", "Close", "Volume"])
    if "Series" in out and (out["Series"] == "EQ").any():
        out = out[out["Series"] == "EQ"]
    return out.drop_duplicates("Symbol", keep="last").set_index("Symbol")


def load_snapshot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        first = df.columns[0]
        if str(first).startswith("Unnamed"):
            df = df.rename(columns={first: "Date"})
        else:
            raise ValueError(f"{path.name} has no Date column")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    if df["Date"].isna().any():
        raise ValueError(f"Invalid dates in {path}")
    for c in ["Open", "High", "Low", "Close", "Volume", "Adjusted_Close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").drop_duplicates("Date", keep="last")
    return df.set_index("Date", drop=False)


def future_share_factor(symbol: str, day: date) -> float:
    factor = 1.0
    for event in SHARE_ACTIONS:
        if event["symbol"] == symbol and day < date.fromisoformat(event["ex_date"]):
            factor *= float(event["factor"])
    return factor


def nonshare_price_factor(symbol: str, day: date) -> float:
    if symbol != "RELIANCE":
        return 1.0
    factor = 1.0
    if day < RIGHTS_EX_DATE:
        factor *= RIGHTS_PRICE_FACTOR
    if day < DEMERGER_EX_DATE:
        factor *= DEMERGER_PROVIDER_PRICE_FACTOR
    return factor


def get_nse_rows(cache_dir: Path) -> Dict[date, pd.DataFrame]:
    result = {}
    for path in sorted(cache_dir.glob("nse_cm_bhavcopy_*.zip")):
        match = re.search(r"(\d{8})", path.stem)
        if not match:
            continue
        day = datetime.strptime(match.group(1), "%Y%m%d").date()
        try:
            result[day] = parse_bhavcopy(path.read_bytes())
        except Exception as exc:
            print(f"WARNING: Could not parse {path.name}: {type(exc).__name__}: {exc}")
    return result


def build_action_registry() -> pd.DataFrame:
    rows = []
    for event in SHARE_ACTIONS:
        rows.append({
            "Symbol": event["symbol"], "Ex_Date": event["ex_date"],
            "Event_Type": event["action"], "Share_Count_Factor": event["factor"],
            "Historical_Price_Adjustment_Factor": 1.0,
            "Factor_Status": "VERIFIED_SHARE_COUNT_EVENT",
            "Notes": "Future share-count event used to translate vendor current-share basis to historical as-traded share units.",
        })
    rows += [
        {"Symbol": "RELIANCE", "Ex_Date": RIGHTS_EX_DATE.isoformat(),
         "Event_Type": "RIGHTS_1_FOR_15_AT_RS_1257", "Share_Count_Factor": 1.0,
         "Historical_Price_Adjustment_Factor": RIGHTS_PRICE_FACTOR,
         "Factor_Status": "THEORETICAL_EX_RIGHTS_FACTOR",
         "Notes": f"TERP=(15*{RIGHTS_PREVIOUS_CLOSE}+{RIGHTS_ISSUE_PRICE})/16={RIGHTS_TERP:.8f}; applies to Yahoo history before ex-date."},
        {"Symbol": "RELIANCE", "Ex_Date": DEMERGER_EX_DATE.isoformat(),
         "Event_Type": "DEMERGER_1_FOR_1_JIO_FINANCIAL", "Share_Count_Factor": 1.0,
         "Historical_Price_Adjustment_Factor": DEMERGER_PROVIDER_PRICE_FACTOR,
         "Factor_Status": "EMPIRICALLY_INFERRED_PROVIDER_FACTOR_REQUIRES_FURTHER_SOURCE_REVIEW",
         "Notes": "Observed factor from the stable Yahoo/NSE ratio transition around the demerger event window; not the 1:1 entitlement ratio and not used as a share-count multiplier."},
    ]
    return pd.DataFrame(rows)


def pct_error(a, b):
    if pd.isna(a) or pd.isna(b) or float(b) == 0:
        return np.nan
    return abs((float(a) - float(b)) / float(b)) * 100.0


def build_dual_basis(symbol: str, src: pd.DataFrame, nse_by_date: Dict[date, pd.DataFrame], repair_log: list):
    out = src.copy()
    out.insert(1, "Source_Symbol", symbol)
    # Keep vendor source fields explicitly named and unmodified.
    for src_col, dst_col in [("Open", "Yahoo_Quote_Open"), ("High", "Yahoo_Quote_High"),
                             ("Low", "Yahoo_Quote_Low"), ("Close", "Yahoo_Quote_Close"),
                             ("Volume", "Yahoo_Quote_Volume")]:
        out[dst_col] = pd.to_numeric(out[src_col], errors="coerce")

    derived = {
        "Technical_Open": [], "Technical_High": [], "Technical_Low": [],
        "Technical_Close": [], "Technical_Volume": [],
        "AsTraded_Open_Estimate": [], "AsTraded_High_Estimate": [],
        "AsTraded_Low_Estimate": [], "AsTraded_Close_Estimate": [],
        "AsTraded_Volume_Estimate": [], "Share_Count_Factor": [],
        "NonShare_Price_Adjustment_Factor": [], "AsTraded_Source": [],
        "Technical_Source": [], "Data_Quality_Flag": [],
    }
    for day, row in out.iterrows():
        if day in HOLIDAY_DATES:
            for key in derived:
                derived[key].append(np.nan if key not in ["AsTraded_Source", "Technical_Source", "Data_Quality_Flag"] else HOLIDAY_DATES[day])
            continue

        share_factor = future_share_factor(symbol, day) if symbol in STOCK_FILES else 1.0
        nonshare_factor = nonshare_price_factor(symbol, day) if symbol in STOCK_FILES else 1.0
        q = [parse_number(row.get(c)) for c in ["Open", "High", "Low", "Close"]]
        vol = parse_number(row.get("Volume"))
        is_repair = day == REPAIR_DATE and symbol in STOCK_FILES

        if is_repair:
            nse = nse_by_date.get(day)
            if nse is None or symbol not in nse.index:
                raise RuntimeError(f"Official NSE bhavcopy row missing for {symbol} on {day}; cannot repair this bar.")
            nr = nse.loc[symbol]
            raw = [parse_number(nr[c]) for c in ["Open", "High", "Low", "Close"]]
            raw_vol = parse_number(nr["Volume"])
            tech = [v / share_factor for v in raw]
            tech_vol = raw_vol * share_factor
            as_traded = raw
            as_traded_vol = raw_vol
            as_source = "NSE_OFFICIAL_BHAVCOPY_REPLACEMENT"
            tech_source = "NSE_OFFICIAL_BHAVCOPY_ON_VENDOR_SHARE_BASIS"
            quality = "REPLACED_INVALID_FLAT_ZERO_VOLUME_YAHOO_BAR"
            repair_log.append({
                "Symbol": symbol, "Date": day.isoformat(),
                "Original_Yahoo_Open": q[0], "Original_Yahoo_High": q[1],
                "Original_Yahoo_Low": q[2], "Original_Yahoo_Close": q[3],
                "Original_Yahoo_Volume": vol,
                "NSE_As_Traded_Open": raw[0], "NSE_As_Traded_High": raw[1],
                "NSE_As_Traded_Low": raw[2], "NSE_As_Traded_Close": raw[3],
                "NSE_As_Traded_Volume": raw_vol,
                "Share_Count_Factor": share_factor,
                "Technical_Open": tech[0], "Technical_High": tech[1],
                "Technical_Low": tech[2], "Technical_Close": tech[3],
                "Technical_Volume": tech_vol,
                "Status": "OFFICIAL_NSE_BAR_INSERTED_IN_DERIVED_DATA_ONLY",
            })
        else:
            # The vendor quote series is already adjusted to current share units
            # and, for Reliance, includes the observed rights/demerger price factors.
            tech = q
            tech_vol = vol
            as_traded = [v * share_factor / nonshare_factor if pd.notna(v) else np.nan for v in q]
            # Vendor volume is inversely scaled to preserve traded value, so reverse
            # the same non-share factor while undoing the share-count adjustment.
            as_traded_vol = vol * nonshare_factor / share_factor if pd.notna(vol) else np.nan
            if symbol in STOCK_FILES:
                as_source = "RECONSTRUCTED_FROM_VENDOR_OHLC_WITH_EVENT_FACTORS"
                tech_source = "YAHOO_QUOTE_OHLC_AS_RECEIVED"
            else:
                as_source = "YAHOO_INDEX_OHLC_AS_RECEIVED"
                tech_source = "YAHOO_INDEX_OHLC_AS_RECEIVED"
            if any(v is None or not np.isfinite(v) for v in q):
                quality = "MISSING_OR_INVALID_VENDOR_OHLC"
            else:
                quality = "OK_FROM_VENDOR_SERIES"

        vals = [
            ("Technical_Open", tech[0]), ("Technical_High", tech[1]),
            ("Technical_Low", tech[2]), ("Technical_Close", tech[3]),
            ("Technical_Volume", tech_vol),
            ("AsTraded_Open_Estimate", as_traded[0]), ("AsTraded_High_Estimate", as_traded[1]),
            ("AsTraded_Low_Estimate", as_traded[2]), ("AsTraded_Close_Estimate", as_traded[3]),
            ("AsTraded_Volume_Estimate", as_traded_vol),
            ("Share_Count_Factor", share_factor),
            ("NonShare_Price_Adjustment_Factor", nonshare_factor),
            ("AsTraded_Source", as_source), ("Technical_Source", tech_source),
            ("Data_Quality_Flag", quality),
        ]
        for key, value in vals:
            derived[key].append(value)

    for key, values in derived.items():
        out[key] = values
    # Remove official holiday observations from the new research dataset.
    out = out.loc[[d not in HOLIDAY_DATES for d in out.index]].copy()
    out = out.reset_index(drop=True)
    return out


def validate_against_nse(outputs: Dict[str, pd.DataFrame], nse_by_date: Dict[date, pd.DataFrame]):
    rows = []
    for day, nse in sorted(nse_by_date.items()):
        for symbol in STOCK_FILES:
            frame = outputs[symbol]
            rset = frame[frame["Date"] == day]
            if rset.empty or symbol not in nse.index:
                continue
            r = rset.iloc[0]
            nr = nse.loc[symbol]
            price_errors = {}
            for field, estimate_col, nse_col in [
                ("Open", "AsTraded_Open_Estimate", "Open"),
                ("High", "AsTraded_High_Estimate", "High"),
                ("Low", "AsTraded_Low_Estimate", "Low"),
                ("Close", "AsTraded_Close_Estimate", "Close"),
            ]:
                price_errors[field] = pct_error(r[estimate_col], nr[nse_col])
            max_price_error = max([v for v in price_errors.values() if pd.notna(v)], default=np.nan)
            volume_error = pct_error(r["AsTraded_Volume_Estimate"], nr["Volume"])
            is_repair = day == REPAIR_DATE
            rows.append({
                "Date": day.isoformat(), "Symbol": symbol,
                "Validation_Source": "DIRECT_NSE_REPLACEMENT" if is_repair else "RECONSTRUCTED_VENDOR_VS_NSE_SAMPLE",
                "NSE_Series": nse.loc[symbol].get("Series", ""),
                "Estimated_AsTraded_Open": r["AsTraded_Open_Estimate"], "NSE_Open": nr["Open"], "Open_Error_Pct": price_errors["Open"],
                "Estimated_AsTraded_High": r["AsTraded_High_Estimate"], "NSE_High": nr["High"], "High_Error_Pct": price_errors["High"],
                "Estimated_AsTraded_Low": r["AsTraded_Low_Estimate"], "NSE_Low": nr["Low"], "Low_Error_Pct": price_errors["Low"],
                "Estimated_AsTraded_Close": r["AsTraded_Close_Estimate"], "NSE_Close": nr["Close"], "Close_Error_Pct": price_errors["Close"],
                "Estimated_AsTraded_Volume": r["AsTraded_Volume_Estimate"], "NSE_Volume": nr["Volume"], "Volume_Error_Pct": volume_error,
                "Max_OHLC_Error_Pct": max_price_error,
                "Status": "PASS" if pd.notna(max_price_error) and max_price_error <= PRICE_TOLERANCE_PCT and pd.notna(volume_error) and volume_error <= VOLUME_TOLERANCE_PCT else "REVIEW",
            })
    return pd.DataFrame(rows)


def build_trade_overlap_review(trades_file: Optional[Path]):
    if trades_file is None or not trades_file.exists():
        return pd.DataFrame(), "V2.1 trades file not found; overlap review skipped."
    trades = pd.read_csv(trades_file)
    if not {"Symbol", "Entry_Date", "Exit_Date"}.issubset(trades.columns):
        return pd.DataFrame(), "V2.1 trades file lacks Symbol/Entry_Date/Exit_Date; overlap review skipped."
    event_specs = [
        ("TCS.NS", "2018-05-31", "BONUS_1_FOR_1", "Bonus share-count adjustment event."),
        ("INFY.NS", "2018-09-04", "BONUS_1_FOR_1", "Bonus share-count adjustment event."),
        ("HDFCBANK.NS", "2019-09-19", "STOCK_SPLIT_2_FOR_1", "Face-value split; share-count adjustment event."),
        ("HDFCBANK.NS", "2025-08-26", "BONUS_1_FOR_1", "Bonus share-count adjustment event."),
        ("ICICIBANK.NS", "2017-06-20", "BONUS_1_FOR_10", "Bonus share-count adjustment event."),
        ("RELIANCE.NS", "2017-09-07", "BONUS_1_FOR_1", "Bonus share-count adjustment event."),
        ("RELIANCE.NS", "2024-10-28", "BONUS_1_FOR_1", "Bonus share-count adjustment event."),
        ("RELIANCE.NS", "2020-05-13", "RIGHTS_1_FOR_15", "Rights entitlement may need modeling; issue requires subscription or sale of entitlement."),
        ("RELIANCE.NS", "2023-07-20", "JIO_FINANCIAL_DEMERGER_1_FOR_1", "Each RIL share received one JFS share; reference/discovered value Rs 261.85. Side asset not modeled by V2.1."),
    ]
    rows = []
    trades["Entry_Date"] = pd.to_datetime(trades["Entry_Date"], errors="coerce")
    trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"], errors="coerce")
    for symbol, ex_date, action, note in event_specs:
        ex = pd.Timestamp(ex_date)
        subset = trades[(trades["Symbol"].astype(str).str.upper() == symbol.upper()) &
                        (trades["Entry_Date"] < ex) & (trades["Exit_Date"] >= ex)]
        for _, r in subset.iterrows():
            rows.append({
                "Symbol": symbol, "Event_Ex_Date": ex_date, "Event": action,
                "Entry_Date": r["Entry_Date"].date().isoformat(), "Exit_Date": r["Exit_Date"].date().isoformat(),
                "Period": r.get("Period", ""), "Entry_Variant": r.get("Entry_Variant", ""),
                "Exit_Variant": r.get("Exit_Variant", ""), "Net_Return_Pct_V2_1": r.get("Net_Return_Pct", np.nan),
                "Review_Note": note,
            })
    return pd.DataFrame(rows), f"Reviewed V2.1 trades from {trades_file}."


def find_latest_trades_file() -> Optional[Path]:
    root = PROJECT_ROOT / "data" / "processed" / "backtest_engine_v21"
    candidates = sorted(root.glob("*/backtest_engine_v21_trades.csv"), key=lambda p: p.parent.name, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Build a dual-basis, corporate-action-aware data set (V2.1.3).")
    parser.add_argument("--input-dir", default=None, help="Folder with the six original input_bars CSV snapshots")
    parser.add_argument("--audit-dir", default=None, help="NSE event-window audit folder containing bhavcopy_cache")
    parser.add_argument("--trades-file", default=None, help="Optional V2.1 trades CSV for corporate-action overlap flags")
    args = parser.parse_args()

    input_dir = find_input_dir(args.input_dir)
    audit_dir = find_audit_dir(args.audit_dir)
    cache_dir = audit_dir / "bhavcopy_cache"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "data" / "processed" / "price_basis_reconstruction_v21_3" / run_id
    dual_dir = output_dir / "dual_basis_bars"
    source_dir = output_dir / "source_snapshots"
    dual_dir.mkdir(parents=True, exist_ok=False)
    source_dir.mkdir(parents=True, exist_ok=False)

    print("INDIAN STOCK ANALYZER — PRICE BASIS BUILDER V2.1.3")
    print(f"Input snapshots: {input_dir}")
    print(f"NSE audit/cache: {audit_dir}")
    print(f"Output folder:   {output_dir}")

    nse_by_date = get_nse_rows(cache_dir)
    print(f"Parsed NSE bhavcopy files: {len(nse_by_date)}")
    if REPAIR_DATE not in nse_by_date:
        raise RuntimeError("NSE cache does not contain 2025-03-18 bhavcopy; cannot insert replacement bars.")

    outputs: Dict[str, pd.DataFrame] = {}
    repair_log = []
    manifest = {"run_id": run_id, "input_dir": str(input_dir), "audit_dir": str(audit_dir), "input_files": {}, "output_files": {}}
    for symbol, filename in ALL_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing input snapshot: {path}")
        src = load_snapshot(path)
        src.to_csv(source_dir / filename, index=False)
        manifest["input_files"][filename] = {"sha256": sha256_file(path), "rows": int(len(src))}
        print(f"Building {symbol}: {len(src)} input rows")
        outputs[symbol] = build_dual_basis(symbol, src, nse_by_date, repair_log)
        outputs[symbol].to_csv(dual_dir / f"{symbol}_dual_basis.csv", index=False)

    validation = validate_against_nse(outputs, nse_by_date)
    validation.to_csv(output_dir / "nse_sample_validation.csv", index=False)
    if validation.empty:
        summary = pd.DataFrame(columns=["Symbol", "Sample_Rows", "Pass_Rows", "Review_Rows", "Max_OHLC_Error_Pct", "Max_Volume_Error_Pct"])
    else:
        summary = validation.groupby("Symbol").agg(
            Sample_Rows=("Status", "size"),
            Pass_Rows=("Status", lambda x: int((x == "PASS").sum())),
            Review_Rows=("Status", lambda x: int((x == "REVIEW").sum())),
            Max_OHLC_Error_Pct=("Max_OHLC_Error_Pct", "max"),
            Max_Volume_Error_Pct=("Volume_Error_Pct", "max"),
        ).reset_index()
    summary.to_csv(output_dir / "nse_validation_summary.csv", index=False)

    pd.DataFrame(repair_log).to_csv(output_dir / "2025-03-18_official_replacements.csv", index=False)
    build_action_registry().to_csv(output_dir / "corporate_action_factor_registry.csv", index=False)

    trades_path = Path(args.trades_file).expanduser().resolve() if args.trades_file else find_latest_trades_file()
    overlap, overlap_note = build_trade_overlap_review(trades_path)
    overlap.to_csv(output_dir / "trade_corporate_action_overlap_review.csv", index=False)

    for path in sorted(dual_dir.glob("*.csv")):
        manifest["output_files"][str(path.relative_to(output_dir))] = {"sha256": sha256_file(path), "rows": int(len(pd.read_csv(path)))}
    for path in [output_dir / "nse_sample_validation.csv", output_dir / "nse_validation_summary.csv",
                 output_dir / "2025-03-18_official_replacements.csv", output_dir / "corporate_action_factor_registry.csv",
                 output_dir / "trade_corporate_action_overlap_review.csv"]:
        manifest["output_files"][path.name] = {"sha256": sha256_file(path), "rows": int(len(pd.read_csv(path)))}
    (output_dir / "sha256_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""INDIAN STOCK ANALYZER — V2.1.3 DUAL-BASIS DATASET
Run: {run_id}
Input snapshots: {input_dir}
NSE audit folder: {audit_dir}

Purpose
- Preserve Yahoo quote OHLCV exactly as received.
- Add a technical-analysis OHLCV series on the vendor's share-adjusted basis.
- Reconstruct estimated historical as-traded OHLCV by reversing the verified future share-count factors and the Reliance rights/demerger provider price adjustments.
- Replace the known invalid 2025-03-18 zero-volume flat stock bars with official NSE OHLCV in this derived dataset only.
- Remove the 2026-01-15 holiday candles.

Important price-basis factors
- Reliance rights issue: 1-for-15 at Rs 1,257, ex-date 2020-05-13; theoretical ex-rights price factor {RIGHTS_PRICE_FACTOR:.12f}.
- Reliance / Jio Financial demerger: 1-for-1, ex-date 2023-07-20; observed vendor price factor {DEMERGER_PROVIDER_PRICE_FACTOR:.12f}. This factor is empirically inferred from Yahoo/NSE event-window comparisons and is not an entitlement/share-count factor.
- Share-count bonus/split events are listed separately in corporate_action_factor_registry.csv.

How to read dual_basis_bars
- Yahoo_Quote_*: source OHLCV preserved from the downloaded Yahoo snapshots.
- Technical_*: vendor-adjusted OHLCV for chart indicators; the 2025-03-18 bar is replaced by NSE as-traded values translated to the applicable share-count basis.
- AsTraded_*_Estimate: reconstructed nominal/as-traded estimate for execution research; 2025-03-18 uses direct official NSE data.
- Adjusted_Close remains a separate Yahoo total-return proxy. Dividend fields are retained, not converted in this version.
- Original input snapshots are copied unchanged under source_snapshots/.

Validation
- nse_sample_validation.csv compares the reconstructed as-traded estimates against every NSE bhavcopy in the supplied cache.
- nse_validation_summary.csv provides per-stock pass/review counts.
- Price tolerance: {PRICE_TOLERANCE_PCT:.2f}%; volume tolerance: {VOLUME_TOLERANCE_PCT:.2f}%.
- These are selected-date cross-checks, not proof that every daily bar in the 2016-2026 history is correct.

Corporate actions in backtests
- trade_corporate_action_overlap_review.csv flags existing V2.1 positions crossing rights/demerger/bonus events.
- V2.1 does not credit Jio Financial shares as a distributed side asset. Do not treat trades crossing the 2023-07-20 demerger as fully economically modeled until this is added or those trades are explicitly excluded.
- Rights entitlements also require appropriate treatment if a future trade ever spans the rights ex-date.

No backtest is run by this script. No main.py, V2, V2.1 script, or existing output is overwritten.
"""
    (output_dir / "README_V21_3.txt").write_text(readme, encoding="utf-8")

    print("\nValidation summary:")
    print(summary.to_string(index=False))
    print(f"\n18-Mar-2025 official replacements: {len(repair_log)}")
    print(f"Corporate-action overlap review rows: {len(overlap)} ({overlap_note})")
    print("\nSaved:")
    for path in [output_dir / "nse_sample_validation.csv", output_dir / "nse_validation_summary.csv",
                 output_dir / "2025-03-18_official_replacements.csv", output_dir / "corporate_action_factor_registry.csv",
                 output_dir / "trade_corporate_action_overlap_review.csv", output_dir / "README_V21_3.txt"]:
        print(path)
    print(f"Dual-basis bars: {dual_dir}")
    print(f"Original snapshots copied unchanged: {source_dir}")

    if not summary.empty and (summary["Review_Rows"] > 0).any():
        print("\nWARNING: Some NSE sample rows did not pass the configured tolerances. Review nse_sample_validation.csv before using this dataset.")
    else:
        print("\nAll supplied NSE sample comparisons passed configured price/volume tolerances.")
    print("Reminder: this builder does not certify unsampled dates or establish strategy profitability.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
