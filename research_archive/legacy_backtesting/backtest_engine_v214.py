#!/usr/bin/env python3
"""Indian Stock Analyzer — Backtest Engine V2.1.4.

Consumes the V2.1.3 dual-basis dataset. Indicators/signals are computed from
Technical_* prices; simulated entries, exits and daily marks use the estimated
AsTraded_*_Estimate prices. Corporate-action-crossing trades are conservatively
excluded until the full corporate action (including any distributed side asset)
is modeled. The script does not modify main.py, V2, V2.1, or the V2.1.3 inputs.

Required in the project root:
  main.py
  backtest_engine_audit_harness.py
  data/processed/price_basis_reconstruction_v21_3/<run>/dual_basis_bars/*.csv
  data/processed/price_basis_reconstruction_v21_3/<run>/corporate_action_factor_registry.csv

Run:
  python backtest_engine_v214.py
Optional:
  python backtest_engine_v214.py --input-dir "...\\dual_basis_bars"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import backtest_engine_audit_harness as audit

PROJECT_ROOT = Path(__file__).resolve().parent
STOCKS = {
    "TCS.NS": "TCS_dual_basis.csv",
    "INFY.NS": "INFY_dual_basis.csv",
    "RELIANCE.NS": "RELIANCE_dual_basis.csv",
    "HDFCBANK.NS": "HDFCBANK_dual_basis.csv",
    "ICICIBANK.NS": "ICICIBANK_dual_basis.csv",
}
NIFTY_FILE = "NSEI_dual_basis.csv"
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
POST_2024_START = pd.Timestamp("2025-01-01")
ACCEPTED_QUALITY_FLAGS = {
    "OK_FROM_VENDOR_SERIES",
    "REPLACED_INVALID_FLAT_ZERO_VOLUME_YAHOO_BAR",
    "OK_FROM_INDEX_VENDOR_SERIES",
    "NSE_INDEX_VENDOR_SERIES",
}

ENTRY_VARIANTS = [
    {"Entry_Variant": "BASELINE_ENTRIES", "Use_NIFTY_SMA200_Filter": False},
    {"Entry_Variant": "NIFTY_ABOVE_SMA200_ENTRIES", "Use_NIFTY_SMA200_Filter": True},
]
EXIT_VARIANTS = [
    {"Exit_Variant": "FIXED_5_SESSION", "Stop_ATR_Multiplier": None, "Target_ATR_Multiplier": None},
    {"Exit_Variant": "ATR_STOP_1_5", "Stop_ATR_Multiplier": 1.5, "Target_ATR_Multiplier": None},
    {"Exit_Variant": "ATR_TARGET_2_0", "Stop_ATR_Multiplier": None, "Target_ATR_Multiplier": 2.0},
    {"Exit_Variant": "ATR_STOP_1_5_TARGET_2_0", "Stop_ATR_Multiplier": 1.5, "Target_ATR_Multiplier": 2.0},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def newest_dual_basis_dir() -> Path:
    root = PROJECT_ROOT / "data" / "processed" / "price_basis_reconstruction_v21_3"
    candidates = sorted(root.glob("*/dual_basis_bars"), key=lambda p: p.parent.name, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find V2.1.3 dual_basis_bars under {root}. "
            "Pass --input-dir with the folder containing the six dual-basis CSVs."
        )
    return candidates[0]


def normalize_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if idx.isna().any():
        raise ValueError(f"{label}: invalid dates found.")
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    if out.index.has_duplicates:
        dupes = out.index[out.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{label}: duplicate dates found: {dupes[:5]}")
    return out.sort_index()


def load_dual_file(path: Path, symbol: str, is_index: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dual-basis input: {path}")
    raw = pd.read_csv(path)
    if "Date" not in raw.columns:
        raise ValueError(f"{path.name}: Date column is missing.")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    if raw["Date"].isna().any():
        raise ValueError(f"{path.name}: invalid Date values found.")
    raw = raw.set_index("Date")
    raw = normalize_frame(raw, path.name)

    technical_cols = ["Technical_Open", "Technical_High", "Technical_Low", "Technical_Close", "Technical_Volume"]
    execution_cols = [
        "AsTraded_Open_Estimate", "AsTraded_High_Estimate", "AsTraded_Low_Estimate",
        "AsTraded_Close_Estimate", "AsTraded_Volume_Estimate",
    ]
    needed = technical_cols + execution_cols + ["Adjusted_Close", "Dividend_Per_Share", "Share_Count_Factor"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: required columns missing: {missing}")

    for col in needed:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Indicators use continuous vendor technical OHLC. Execution uses the
    # reconstructed nominal/as-traded estimate. Keep both in the same frame.
    data = pd.DataFrame(index=raw.index)
    data["Open"] = raw["Technical_Open"]
    data["High"] = raw["Technical_High"]
    data["Low"] = raw["Technical_Low"]
    data["Close"] = raw["Technical_Close"]
    data["Volume"] = raw["Technical_Volume"]
    for col in technical_cols + execution_cols + [
        "Adjusted_Close", "Dividend_Per_Share", "Share_Count_Factor",
        "NonShare_Price_Adjustment_Factor", "AsTraded_Source", "Technical_Source", "Data_Quality_Flag",
    ]:
        if col in raw.columns:
            data[col] = raw[col]
    if "NonShare_Price_Adjustment_Factor" not in data:
        data["NonShare_Price_Adjustment_Factor"] = 1.0

    # Validate both price bases independently. Index execution fields are not
    # used to trade, but their technical fields form the market reference.
    tech = data[["Open", "High", "Low", "Close"]].copy()
    tech = audit.validate_ohlc(tech)
    exec_frame = data[
        ["AsTraded_Open_Estimate", "AsTraded_High_Estimate", "AsTraded_Low_Estimate", "AsTraded_Close_Estimate"]
    ].rename(columns={
        "AsTraded_Open_Estimate": "Open", "AsTraded_High_Estimate": "High",
        "AsTraded_Low_Estimate": "Low", "AsTraded_Close_Estimate": "Close",
    })
    audit.validate_ohlc(exec_frame)

    data["Execution_Scale"] = data["AsTraded_Close_Estimate"] / data["Technical_Close"]
    if data["Execution_Scale"].isna().any() or (data["Execution_Scale"] <= 0).any():
        raise ValueError(f"{path.name}: invalid execution-to-technical price scale.")
    if not is_index:
        flags = data.get("Data_Quality_Flag", pd.Series("", index=data.index)).fillna("").astype(str)
        bad = ~flags.isin(ACCEPTED_QUALITY_FLAGS)
        if bad.any():
            bad_dates = data.index[bad].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{symbol}: unapproved data-quality flags on {bad_dates[:8]}")
        # Reconstructed prices are estimates. Check data consistency, not an
        # assertion that every daily observation has exchange-source provenance.
        if data[["AsTraded_Open_Estimate", "AsTraded_High_Estimate", "AsTraded_Low_Estimate", "AsTraded_Close_Estimate"]].isna().any().any():
            raise ValueError(f"{symbol}: missing estimated execution OHLC.")
    data["Dividend_Per_Share"] = data["Dividend_Per_Share"].fillna(0.0)
    data["Share_Count_Factor"] = data["Share_Count_Factor"].fillna(1.0)
    return data


def load_events(metadata_dir: Path) -> pd.DataFrame:
    path = metadata_dir / "corporate_action_factor_registry.csv"
    if not path.exists():
        raise FileNotFoundError(f"Corporate-action registry not found: {path}")
    events = pd.read_csv(path)
    required = {"Symbol", "Ex_Date", "Event_Type"}
    if not required.issubset(events.columns):
        raise ValueError(f"Corporate-action registry missing columns: {sorted(required - set(events.columns))}")
    events = events.copy()
    events["Symbol"] = events["Symbol"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    events["Ex_Date"] = pd.to_datetime(events["Ex_Date"], errors="coerce")
    if events["Ex_Date"].isna().any():
        raise ValueError("Corporate-action registry contains an invalid Ex_Date.")
    return events.sort_values(["Symbol", "Ex_Date"]).reset_index(drop=True)


def periods(data: pd.DataFrame, start_pos: int) -> list[dict[str, Any]]:
    first = pd.Timestamp(data.index[start_pos])
    last = pd.Timestamp(data.index[-1])
    candidates = [
        {"Period": "FULL_AVAILABLE_WINDOW", "Start": first, "End": last},
        {"Period": "DEVELOPMENT_THROUGH_2024", "Start": first, "End": min(DEVELOPMENT_END, last)},
        {"Period": "PREVIOUSLY_REVIEWED_POST_2024", "Start": max(POST_2024_START, first), "End": last},
    ]
    return [p for p in candidates if p["Start"] <= p["End"]]


def find_event_in_window(events: pd.DataFrame, symbol: str, signal_date: pd.Timestamp, scheduled_exit: pd.Timestamp):
    normalized_symbol = str(symbol).upper().replace(".NS", "")
    local = events[events["Symbol"] == normalized_symbol]
    matched = local[(local["Ex_Date"] > signal_date) & (local["Ex_Date"] <= scheduled_exit)]
    if matched.empty:
        return None
    return matched.iloc[0]


def build_execution_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data[[
        "AsTraded_Open_Estimate", "AsTraded_High_Estimate",
        "AsTraded_Low_Estimate", "AsTraded_Close_Estimate",
    ]].rename(columns={
        "AsTraded_Open_Estimate": "Open", "AsTraded_High_Estimate": "High",
        "AsTraded_Low_Estimate": "Low", "AsTraded_Close_Estimate": "Close",
    })


def eligible_dividends_as_traded(data: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float:
    mask = (data.index > entry_date) & (data.index <= exit_date)
    # Yahoo's dividend field is on its adjusted share-count basis. Convert it
    # to the reconstructed historical share unit. Non-share price factors are
    # deliberately not applied to cash dividend entitlements.
    values = data.loc[mask, "Dividend_Per_Share"] * data.loc[mask, "Share_Count_Factor"]
    return float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())


def build_equity_curve(exec_bars: pd.DataFrame, trades: pd.DataFrame, period_start: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = audit.validate_ohlc(exec_bars)
    frame = frame.loc[frame.index >= period_start].copy()
    if frame.empty:
        raise ValueError("Cannot build equity curve for an empty period.")
    curve = pd.DataFrame(index=frame.index)
    curve.index.name = "Date"
    curve["AsTraded_Close"] = frame["Close"].astype(float)
    curve["Strategy_Equity"] = np.nan
    curve["In_Position"] = False
    current_equity = 1.0
    cursor = 0

    ordered = trades.sort_values("Entry_Date") if not trades.empty else trades
    for _, trade in ordered.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"])
        exit_date = pd.Timestamp(trade["Exit_Date"])
        if entry_date not in frame.index or exit_date not in frame.index:
            raise ValueError("Trade dates are missing from execution price bars.")
        entry_pos = int(frame.index.get_loc(entry_date))
        exit_pos = int(frame.index.get_loc(exit_date))
        if entry_pos < cursor:
            raise ValueError("Overlapping trades detected in equity curve.")
        if cursor < entry_pos:
            curve.iloc[cursor:entry_pos, curve.columns.get_loc("Strategy_Equity")] = current_equity
        entry = float(trade["Entry_Price_AsTraded"])
        cost_fraction = float(trade["Assumed_Round_Trip_Cost_Pct"]) / 100.0
        base_equity = current_equity
        for pos in range(entry_pos, exit_pos + 1):
            date = frame.index[pos]
            close_mark = float(trade["Exit_Price_AsTraded"]) if pos == exit_pos else float(frame.iloc[pos]["Close"])
            divs = eligible_dividends_as_traded_for_curve(trade, data_dates=frame.index, date=date)
            curve.at[date, "Strategy_Equity"] = base_equity * ((close_mark + divs) / entry - cost_fraction)
            curve.at[date, "In_Position"] = True
        current_equity = base_equity * (1.0 + float(trade["Net_Return_Pct"]) / 100.0)
        curve.at[exit_date, "Strategy_Equity"] = current_equity
        cursor = exit_pos + 1
    if cursor < len(frame):
        curve.iloc[cursor:, curve.columns.get_loc("Strategy_Equity")] = current_equity
    curve["Strategy_Equity"] = curve["Strategy_Equity"].ffill().fillna(1.0)
    curve["Equity_Peak"] = curve["Strategy_Equity"].cummax()
    curve["Daily_Simulated_Equity_Drawdown_Pct"] = (curve["Strategy_Equity"] / curve["Equity_Peak"] - 1.0) * 100.0
    stats = {
        "Final_Strategy_Equity": float(curve["Strategy_Equity"].iloc[-1]),
        "Compounded_Strategy_Return_Pct": float((curve["Strategy_Equity"].iloc[-1] - 1.0) * 100.0),
        "Maximum_Daily_Simulated_Equity_Drawdown_Pct": float(curve["Daily_Simulated_Equity_Drawdown_Pct"].min()),
        "Percent_Of_Sessions_In_Position": float(curve["In_Position"].mean() * 100.0),
    }
    return curve.reset_index(), stats


def eligible_dividends_as_traded_for_curve(trade: pd.Series, data_dates: pd.DatetimeIndex, date: pd.Timestamp) -> float:
    # Trade-level cumulative dividend snapshots are stored per trade/date in a
    # compact map serialized by _generate_trades; no source dataframe is needed.
    amounts = trade.get("Dividend_Cash_By_Date_JSON", "{}")
    try:
        lookup = json.loads(amounts) if isinstance(amounts, str) else {}
    except json.JSONDecodeError:
        lookup = {}
    return float(sum(v for d, v in lookup.items() if pd.Timestamp(d) <= date))


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "Number_Of_Trades": 0, "Win_Rate_Pct": np.nan,
            "Average_Gross_Trade_Return_Pct": np.nan, "Average_Net_Trade_Return_Pct": np.nan,
            "Median_Net_Trade_Return_Pct": np.nan, "Trade_Compounded_Return_Pct": 0.0,
            "Profit_Factor_Trade_Percent": np.nan, "Profit_Factor_Equity_Weighted": np.nan,
        }
    returns = pd.to_numeric(trades["Net_Return_Pct"], errors="coerce").dropna()
    gross = pd.to_numeric(trades["Gross_Return_Pct"], errors="coerce").dropna()
    pf = audit.calculate_profit_factor_metrics(returns.tolist())
    return {
        "Number_Of_Trades": int(len(returns)),
        "Win_Rate_Pct": float((returns > 0).mean() * 100.0),
        "Average_Gross_Trade_Return_Pct": float(gross.mean()) if len(gross) else np.nan,
        "Average_Net_Trade_Return_Pct": float(returns.mean()),
        "Median_Net_Trade_Return_Pct": float(returns.median()),
        "Trade_Compounded_Return_Pct": float(((1.0 + returns / 100.0).prod() - 1.0) * 100.0),
        **pf,
    }


def benchmark_return(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cost_pct: float) -> float:
    subset = bars.loc[(bars.index >= start) & (bars.index <= end)]
    values = pd.to_numeric(subset["Adjusted_Close"], errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
        values = pd.to_numeric(subset["Technical_Close"], errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
        return float("nan")
    return float((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0 - cost_pct)


def generate_trades(symbol: str, data: pd.DataFrame, start_pos: int, period: dict[str, Any],
                    entry_variant: dict[str, Any], exit_variant: dict[str, Any], analyzer: Any,
                    market_calendar: pd.DatetimeIndex, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holding = int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5))
    min_score = int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4))
    require_positive_rs = bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True))
    cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))
    exec_bars = build_execution_frame(data)
    calendar = pd.DatetimeIndex(market_calendar).sort_values().unique()
    data_dates = set(data.index)
    first_signal = max(start_pos, int(data.index.searchsorted(pd.Timestamp(period["Start"]))))
    trades: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    last_exit_pos = -1
    for signal_pos in range(first_signal, len(data)):
        signal_date = pd.Timestamp(data.index[signal_pos])
        if signal_date > pd.Timestamp(period["End"]):
            break
        if audit.should_skip_signal(signal_pos, last_exit_pos):
            continue
        calendar_pos = int(calendar.searchsorted(signal_date, side="right"))
        if calendar_pos + holding > len(calendar):
            continue
        holding_dates = pd.DatetimeIndex(calendar[calendar_pos:calendar_pos + holding])
        if len(holding_dates) != holding or any(day not in data_dates for day in holding_dates):
            continue
        entry_date = pd.Timestamp(holding_dates[0])
        scheduled_exit_date = pd.Timestamp(holding_dates[-1])
        if scheduled_exit_date > pd.Timestamp(period["End"]):
            continue
        entry_pos = int(data.index.get_loc(entry_date))
        time_exit_pos = int(data.index.get_loc(scheduled_exit_date))
        if not data.index[entry_pos:time_exit_pos + 1].equals(holding_dates):
            continue

        signal = data.iloc[signal_pos]
        score_value = analyzer.calculate_setup_score(signal)
        score = int(score_value) if pd.notna(score_value) else 0
        rel_strength = pd.to_numeric(signal.get("Relative_Strength_20D_Pct", np.nan), errors="coerce")
        if score < min_score or bool(signal.get("Breakdown_Confirmed", False)):
            continue
        if require_positive_rs and (pd.isna(rel_strength) or float(rel_strength) <= 0):
            continue
        nifty_close = pd.to_numeric(signal.get("NIFTY_Close", np.nan), errors="coerce")
        nifty_sma = pd.to_numeric(signal.get("NIFTY_SMA_200", np.nan), errors="coerce")
        if entry_variant["Use_NIFTY_SMA200_Filter"] and (
            pd.isna(nifty_close) or pd.isna(nifty_sma) or float(nifty_close) <= float(nifty_sma)
        ):
            continue

        # Do not simulate a trade across an unmodeled stock action. In
        # particular, this excludes a RIL trade that would receive JFS shares.
        event = find_event_in_window(events, symbol, signal_date, scheduled_exit_date)
        if event is not None:
            excluded.append({
                "Symbol": symbol, "Period": period["Period"],
                "Entry_Variant": entry_variant["Entry_Variant"], "Exit_Variant": exit_variant["Exit_Variant"],
                "Signal_Date": signal_date.date().isoformat(), "Entry_Date": entry_date.date().isoformat(),
                "Scheduled_Exit_Date": scheduled_exit_date.date().isoformat(),
                "Excluded_Event_Date": pd.Timestamp(event["Ex_Date"]).date().isoformat(),
                "Excluded_Event_Type": str(event["Event_Type"]),
                "Reason": "Potential holding window crosses a corporate action not fully modeled; trade excluded conservatively.",
            })
            continue

        # The ATR is computed on Technical_* basis. Convert it to the same
        # nominal execution-price units as the simulated AsTraded_* bars.
        signal_technical_close = float(signal["Technical_Close"])
        signal_scale = float(signal["Execution_Scale"])
        entry_scale = float(data.iloc[entry_pos]["Execution_Scale"])
        window_scales = data.loc[holding_dates, "Execution_Scale"].astype(float).to_numpy()
        scale_tol = max(1e-8, abs(signal_scale) * 1e-5)
        if abs(entry_scale - signal_scale) > scale_tol or np.max(np.abs(window_scales - signal_scale)) > scale_tol:
            excluded.append({
                "Symbol": symbol, "Period": period["Period"],
                "Entry_Variant": entry_variant["Entry_Variant"], "Exit_Variant": exit_variant["Exit_Variant"],
                "Signal_Date": signal_date.date().isoformat(), "Entry_Date": entry_date.date().isoformat(),
                "Scheduled_Exit_Date": scheduled_exit_date.date().isoformat(),
                "Excluded_Event_Date": "", "Excluded_Event_Type": "PRICE_BASIS_SCALE_CHANGE",
                "Reason": "Technical-to-execution price scale changed inside the candidate window.",
            })
            continue

        uses_atr = exit_variant["Stop_ATR_Multiplier"] is not None or exit_variant["Target_ATR_Multiplier"] is not None
        signal_atr_technical = pd.to_numeric(signal.get("ATR_14", np.nan), errors="coerce")
        if uses_atr and (pd.isna(signal_atr_technical) or float(signal_atr_technical) <= 0):
            continue
        signal_atr_execution = float(signal_atr_technical) * signal_scale if pd.notna(signal_atr_technical) else None
        entry_price = float(data.iloc[entry_pos]["AsTraded_Open_Estimate"])
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Invalid as-traded entry price for {symbol} on {entry_date.date()}.")
        exit_result = audit.simulate_long_exit(
            exec_bars, entry_pos=entry_pos, time_exit_pos=time_exit_pos, entry_price=entry_price,
            signal_atr=signal_atr_execution,
            stop_atr_multiplier=exit_variant["Stop_ATR_Multiplier"],
            target_atr_multiplier=exit_variant["Target_ATR_Multiplier"],
        )
        exit_date = pd.Timestamp(exit_result.exit_date)
        dividend_cash = eligible_dividends_as_traded(data, entry_date, exit_date)
        exit_price = float(exit_result.exit_price)
        gross_pct = ((exit_price + dividend_cash) / entry_price - 1.0) * 100.0
        net_pct = gross_pct - cost_pct
        # Cash dividends are also saved by date for correct daily MTM.
        div_mask = (data.index > entry_date) & (data.index <= exit_date) & (data["Dividend_Per_Share"] != 0)
        div_map = {}
        for div_date, div_row in data.loc[div_mask].iterrows():
            amount = float(div_row["Dividend_Per_Share"]) * float(div_row["Share_Count_Factor"])
            div_map[pd.Timestamp(div_date).date().isoformat()] = amount
        trades.append({
            "Symbol": symbol, "Period": period["Period"],
            "Entry_Variant": entry_variant["Entry_Variant"], "Exit_Variant": exit_variant["Exit_Variant"],
            "Signal_Date": signal_date.date().isoformat(), "Entry_Date": entry_date.date().isoformat(),
            "Exit_Date": exit_date.date().isoformat(), "Exit_Reason": exit_result.exit_reason,
            "Entry_Price_AsTraded": entry_price, "Exit_Price_AsTraded": exit_price,
            "Entry_Price_Technical": float(data.iloc[entry_pos]["Technical_Open"]),
            "Exit_Price_Technical_Close": float(data.loc[exit_date, "Technical_Close"]),
            "Execution_Scale_On_Signal": signal_scale,
            "Signal_ATR_Technical": float(signal_atr_technical) if pd.notna(signal_atr_technical) else np.nan,
            "Signal_ATR_Execution_Estimate": signal_atr_execution if signal_atr_execution is not None else np.nan,
            "Stop_Price_AsTraded": exit_result.stop_price,
            "Target_Price_AsTraded": exit_result.target_price,
            "Dividend_Per_Share_AsTraded_Estimate": dividend_cash,
            "Dividend_Cash_By_Date_JSON": json.dumps(div_map, separators=(",", ":")),
            "Holding_Sessions": int(data.index.get_loc(exit_date) - entry_pos + 1),
            "Scheduled_Max_Holding_Sessions": holding,
            "Setup_Score": score,
            "Relative_Strength_20D_Pct": float(rel_strength) if pd.notna(rel_strength) else np.nan,
            "NIFTY_Close_On_Signal_Technical": nifty_close,
            "NIFTY_SMA_200_On_Signal_Technical": nifty_sma,
            "Gross_Return_Pct": gross_pct, "Assumed_Round_Trip_Cost_Pct": cost_pct,
            "Net_Return_Pct": net_pct, "Outcome": "Win" if net_pct > 0 else "Loss/Flat",
            "Execution_Price_Source": str(data.iloc[entry_pos].get("AsTraded_Source", "")),
            "Technical_Price_Source": str(data.iloc[signal_pos].get("Technical_Source", "")),
        })
        last_exit_pos = int(data.index.get_loc(exit_date))
    return pd.DataFrame(trades), pd.DataFrame(excluded)


def summary_row(symbol: str, period: dict[str, Any], entry_variant: dict[str, Any], exit_variant: dict[str, Any],
                tech_data: pd.DataFrame, exec_bars: pd.DataFrame, nifty: pd.DataFrame,
                trades: pd.DataFrame, excluded: pd.DataFrame, analyzer: Any) -> tuple[dict[str, Any], pd.DataFrame]:
    start, end = pd.Timestamp(period["Start"]), pd.Timestamp(period["End"])
    window_exec = exec_bars.loc[(exec_bars.index >= start) & (exec_bars.index <= end)].copy()
    curve, curve_stats = build_equity_curve(window_exec, trades, start)
    cost = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))
    ret = pd.to_numeric(trades["Net_Return_Pct"], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    pf = audit.calculate_profit_factor_metrics(ret.dropna().tolist())
    gross = pd.to_numeric(trades["Gross_Return_Pct"], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    summary = {
        "Symbol": symbol, "Period": period["Period"],
        "Period_Start": start.date().isoformat(), "Period_End": end.date().isoformat(),
        "Entry_Variant": entry_variant["Entry_Variant"], "Exit_Variant": exit_variant["Exit_Variant"],
        "Use_NIFTY_SMA200_Filter": bool(entry_variant["Use_NIFTY_SMA200_Filter"]),
        "Holding_Sessions": int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5)),
        "Minimum_Setup_Score": int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4)),
        "Require_Positive_Relative_Strength": bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)),
        "Assumed_Round_Trip_Cost_Pct": cost,
        "Stock_Buy_Hold_Total_Return_Proxy_Pct": benchmark_return(tech_data, start, end, cost),
        "NIFTY_Buy_Hold_Total_Return_Proxy_Pct": benchmark_return(nifty, start, end, cost),
        **curve_stats,
        "Number_Of_Trades": int(len(ret.dropna())),
        "Win_Rate_Pct": float((ret.dropna() > 0).mean() * 100.0) if len(ret.dropna()) else np.nan,
        "Average_Gross_Trade_Return_Pct": float(gross.mean()) if len(gross.dropna()) else np.nan,
        "Average_Net_Trade_Return_Pct": float(ret.mean()) if len(ret.dropna()) else np.nan,
        "Median_Net_Trade_Return_Pct": float(ret.median()) if len(ret.dropna()) else np.nan,
        "Trade_Compounded_Return_Pct": float(((1 + ret.dropna() / 100).prod() - 1) * 100) if len(ret.dropna()) else 0.0,
        **pf,
        "Setups_Excluded_Corporate_Action_Or_Basis_Change": int(len(excluded)),
        "Input_Price_Basis": "Technical_* for indicators; AsTraded_*_Estimate for execution; reconstructed estimates not fully exchange-verified on every day",
    }
    if not curve.empty:
        curve["Symbol"] = symbol
        curve["Period"] = period["Period"]
        curve["Entry_Variant"] = entry_variant["Entry_Variant"]
        curve["Exit_Variant"] = exit_variant["Exit_Variant"]
    return summary, curve


def run(input_dir: Path) -> Path:
    input_dir = input_dir.expanduser().resolve()
    metadata_dir = input_dir.parent
    if not input_dir.exists():
        raise FileNotFoundError(f"Dual-basis directory not found: {input_dir}")
    events = load_events(metadata_dir)
    try:
        import main as analyzer
    except Exception as exc:
        raise RuntimeError(f"Could not import main.py from {PROJECT_ROOT}: {exc}") from exc

    output_dir = PROJECT_ROOT / "data" / "processed" / "backtest_engine_v214" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input_dual_basis").mkdir()
    shutil.copy2(metadata_dir / "corporate_action_factor_registry.csv", output_dir / "input_dual_basis" / "corporate_action_factor_registry.csv")

    print("\nINDIAN STOCK ANALYZER — BACKTEST ENGINE V2.1.4")
    print("Technical_* prices for indicators; AsTraded_*_Estimate prices for execution.")
    print("No source files or main.py are modified.")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    data_by_symbol: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    for symbol, filename in STOCKS.items():
        data = load_dual_file(input_dir / filename, symbol)
        data_by_symbol[symbol] = data
        quality_rows.append({
            "Symbol": symbol, "Rows": len(data), "First_Date": data.index.min().date().isoformat(),
            "Last_Date": data.index.max().date().isoformat(), "Invalid_Flag_Rows": int((~data["Data_Quality_Flag"].isin(ACCEPTED_QUALITY_FLAGS)).sum()),
            "Min_Execution_Scale": float(data["Execution_Scale"].min()), "Max_Execution_Scale": float(data["Execution_Scale"].max()),
        })
        shutil.copy2(input_dir / filename, output_dir / "input_dual_basis" / filename)

    nifty = load_dual_file(input_dir / NIFTY_FILE, "^NSEI", is_index=True)
    shutil.copy2(input_dir / NIFTY_FILE, output_dir / "input_dual_basis" / NIFTY_FILE)
    market_calendar = pd.DatetimeIndex(sorted(set().union(*(set(d.index) for d in [*data_by_symbol.values(), nifty]))))
    nifty_indicators = analyzer.calculate_indicators(nifty[["Open", "High", "Low", "Close", "Volume"]].copy())
    nifty_indicators = normalize_frame(nifty_indicators, "NIFTY indicators")

    summaries: list[dict[str, Any]] = []
    trade_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    excluded_parts: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for symbol, source_data in data_by_symbol.items():
        print(f"\nProcessing {symbol}...")
        try:
            tech = analyzer.calculate_indicators(source_data[["Open", "High", "Low", "Close", "Volume"]].copy())
            tech = normalize_frame(tech, f"{symbol} technical indicators")
            # Carry execution and event/dividend fields alongside signal rows.
            for col in [
                "Technical_Open", "Technical_High", "Technical_Low", "Technical_Close",
                "AsTraded_Open_Estimate", "AsTraded_High_Estimate", "AsTraded_Low_Estimate", "AsTraded_Close_Estimate",
                "AsTraded_Volume_Estimate", "Execution_Scale", "Dividend_Per_Share", "Share_Count_Factor",
                "AsTraded_Source", "Technical_Source", "Data_Quality_Flag", "Adjusted_Close",
            ]:
                tech[col] = source_data[col].reindex(tech.index)
            tech = analyzer.add_relative_strength(tech, nifty_indicators)
            tech = normalize_frame(tech, f"{symbol} with relative strength")
            tech["NIFTY_Close"] = nifty_indicators["Close"].reindex(tech.index)
            tech["NIFTY_SMA_200"] = nifty_indicators["SMA_200"].reindex(tech.index)
            ready = np.flatnonzero(tech["SMA_200"].notna().to_numpy())
            if not len(ready):
                raise ValueError("No SMA_200 warm-up values found.")
            configured_start = pd.Timestamp(getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03"))
            start_pos = max(int(ready[0]), int(tech.index.searchsorted(configured_start)))
            if start_pos >= len(tech):
                raise ValueError("Configured test start is beyond available data.")
            all_periods = periods(tech, start_pos)
            exec_bars = build_execution_frame(tech)
            print(f"  {len(tech)} bars: {tech.index.min().date()} to {tech.index.max().date()}")

            for period in all_periods:
                for entry_variant in ENTRY_VARIANTS:
                    for exit_variant in EXIT_VARIANTS:
                        trades, excluded = generate_trades(
                            symbol, tech, start_pos, period, entry_variant, exit_variant,
                            analyzer, market_calendar, events,
                        )
                        if not trades.empty:
                            trade_parts.append(trades)
                        if not excluded.empty:
                            excluded_parts.append(excluded)
                        summary, curve = summary_row(
                            symbol, period, entry_variant, exit_variant,
                            tech, exec_bars, nifty, trades, excluded, analyzer,
                        )
                        summaries.append(summary)
                        if not curve.empty:
                            curve_parts.append(curve)
                        print(
                            f"  {period['Period']:<31} {entry_variant['Entry_Variant']:<27} "
                            f"{exit_variant['Exit_Variant']:<25} trades={summary['Number_Of_Trades']:>3} "
                            f"excluded={summary['Setups_Excluded_Corporate_Action_Or_Basis_Change']:>2} "
                            f"avg_net={summary['Average_Net_Trade_Return_Pct']:>7.3f}% "
                            f"PF={summary['Profit_Factor_Equity_Weighted']:>5.2f} "
                            f"return={summary['Compounded_Strategy_Return_Pct']:>8.2f}%"
                        )
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc), "Traceback": traceback.format_exc()})
            print(f"  ERROR: {exc}")

    if not summaries:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
        raise RuntimeError("No summaries were generated. See errors.csv.")

    summary_df = pd.DataFrame(summaries)
    trades_df = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    curves_df = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    excluded_df = pd.concat(excluded_parts, ignore_index=True) if excluded_parts else pd.DataFrame()

    # Final safety assertion: no saved trade spans a registered corporate action.
    violations = []
    for _, tr in trades_df.iterrows():
        local = events[events["Symbol"] == str(tr["Symbol"]).replace(".NS", "")]
        entry, exit_ = pd.Timestamp(tr["Entry_Date"]), pd.Timestamp(tr["Exit_Date"])
        between = local[(local["Ex_Date"] > pd.Timestamp(tr["Signal_Date"])) & (local["Ex_Date"] <= exit_)]
        if not between.empty:
            violations.append({"Symbol": tr["Symbol"], "Entry_Date": tr["Entry_Date"], "Exit_Date": tr["Exit_Date"], "Event_Date": between.iloc[0]["Ex_Date"]})
    if violations:
        raise AssertionError(f"Trades crossed corporate actions despite exclusion policy: {violations[:5]}")

    summary_df.to_csv(output_dir / "backtest_engine_v214_summary.csv", index=False)
    trades_df.to_csv(output_dir / "backtest_engine_v214_trades.csv", index=False)
    curves_df.to_csv(output_dir / "backtest_engine_v214_equity_curves.csv", index=False)
    excluded_df.to_csv(output_dir / "backtest_engine_v214_excluded_events.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(output_dir / "backtest_engine_v214_data_quality.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)

    manifest = {
        "engine_version": "2.1.4",
        "input_dir": str(input_dir),
        "input_files_sha256": {name: sha256_file(input_dir / name) for name in [*STOCKS.values(), NIFTY_FILE]},
        "corporate_action_registry_sha256": sha256_file(metadata_dir / "corporate_action_factor_registry.csv"),
        "price_basis": {
            "signals_indicators": "Technical_Open/High/Low/Close/Volume",
            "execution": "AsTraded_Open/High/Low/Close_Estimate",
            "atr_conversion": "Signal-date Execution_Scale = AsTraded_Close_Estimate / Technical_Close",
            "dividends": "Dividend_Per_Share multiplied by Share_Count_Factor; no NonShare_Price_Adjustment_Factor is applied to cash dividends",
        },
        "corporate_action_policy": "Conservatively exclude candidate holding windows that cross any event in corporate_action_factor_registry.csv. This includes Reliance rights issue and Jio Financial demerger; distributed JFS asset is not synthetically valued.",
        "cost_pct": float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20)),
        "holding_sessions": int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5)),
        "minimum_setup_score": int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4)),
        "require_positive_relative_strength": bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)),
        "test_start_date": str(getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03")),
        "symbols": list(STOCKS.keys()),
        "notes": [
            "This is a separate engine; main.py is imported for signal rules but never edited.",
            "The AsTraded_* series is reconstructed and is not exchange-verified for every date.",
            "Jio Financial shares are not credited; trades that could span its demerger are excluded.",
            "Rights entitlements are not modeled; trades crossing the Reliance rights issue are excluded.",
            "The 0.20% round-trip cost remains an illustrative assumption, not a complete cost model.",
            "The 2025+ period has previously been reviewed and is not an untouched holdout.",
        ],
    }
    (output_dir / "backtest_engine_v214_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "methodology.txt").write_text(
        "V2.1.4 METHOD NOTES\n\n"
        "1. main.py supplies the existing indicator and setup-score rules. It is not modified.\n"
        "2. Technical_* prices are used for SMA, RSI, ATR, relative strength and NIFTY regime filters.\n"
        "3. AsTraded_*_Estimate prices are used for simulated entry, stop/target checks, exits and daily MTM.\n"
        "4. ATR is converted to the estimated execution price units using the signal-date AsTraded/Technical close ratio.\n"
        "5. Candidate trades are skipped if the scheduled holding window crosses any event in the supplied corporate-action registry, or if the price-basis scale changes. This deliberately excludes RIL rights/demerger crossing trades rather than pretending to model entitlements.\n"
        "6. Dividend-per-share values are translated by the share-count factor; non-share price factors are not applied to cash dividends. This dividend basis remains an estimate based on vendor event units.\n"
        "7. The buy-and-hold benchmark uses Adjusted_Close as a vendor total-return proxy.\n"
        "8. All five-stock tests retain the existing 5-session horizon, setup score threshold, positive relative-strength requirement, 0.20% round-trip cost assumption, and the two entry/four exit variants.\n"
        "9. This run is a data-basis integration test, not proof of a profitable strategy. The five-stock sample is not sufficient for a market-wide conclusion.\n",
        encoding="utf-8",
    )

    print("\nSaved reports:")
    for name in [
        "backtest_engine_v214_summary.csv", "backtest_engine_v214_trades.csv",
        "backtest_engine_v214_equity_curves.csv", "backtest_engine_v214_excluded_events.csv",
        "backtest_engine_v214_data_quality.csv", "backtest_engine_v214_manifest.json", "methodology.txt",
    ]:
        print(f"  {output_dir / name}")
    print(f"\nCompleted. Saved trades: {len(trades_df):,}; event/basis exclusions: {len(excluded_df):,}.")
    print("Reminder: V2.1.4 separates signal and execution price bases; it does not establish a trading edge.")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest on the V2.1.3 dual-basis dataset.")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Path to dual_basis_bars folder. Defaults to the latest V2.1.3 output.")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve() if args.input_dir else newest_dual_basis_dir()
    run(input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
