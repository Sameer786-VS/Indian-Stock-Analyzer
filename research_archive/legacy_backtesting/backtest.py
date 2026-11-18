"""Indian Stock Analyzer — corrected backtest engine prototype (v2).

This is a separate research script. It does NOT modify main.py or older
experiment scripts. It reuses the original signal-scoring functions from
main.py, but downloads quote OHLC into separate fields and preserves the
adjusted close and corporate actions separately.

Required files in the project folder:
  - main.py
  - backtest_engine_audit_harness.py
  - backtest_engine_v2.py (this file)

Run:
    python backtest_engine_v2.py

This is an audit/research prototype, not a proven strategy or trading system.
The upstream Yahoo split-adjustment basis still needs independent confirmation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import json
import math
import os
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
import requests

import backtest_engine_audit_harness as audit

SYMBOLS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
YAHOO_HOSTS = ("query2.finance.yahoo.com", "query1.finance.yahoo.com")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
POST_2024_START = pd.Timestamp("2025-01-01")

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


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates, and refuse duplicate rows rather than silently picking one."""
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    if out.index.has_duplicates:
        duplicates = out.index[out.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate daily bars found: {duplicates[:5]}")
    return out.sort_index()


def _exclude_incomplete_current_session(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Before 16:00 IST, exclude today's potentially incomplete daily candle."""
    out = _normalize_index(frame)
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Kolkata")
    else:
        now = pd.Timestamp(now)
        now = now.tz_localize("Asia/Kolkata") if now.tzinfo is None else now.tz_convert("Asia/Kolkata")
    if now.hour < 16:
        today = now.tz_localize(None).normalize()
        out = out.loc[out.index < today].copy()
    return out


def _download_yahoo_chart(symbol: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Download Yahoo chart payload while retaining quote OHLC and actions separately."""
    requester = session or requests.Session()
    errors: list[str] = []
    encoded_symbol = quote(symbol, safe="")
    for host in YAHOO_HOSTS:
        url = f"https://{host}/v8/finance/chart/{encoded_symbol}"
        for attempt in (1, 2):
            try:
                response = requester.get(
                    url,
                    params={"range": "10y", "interval": "1d", "events": "div,splits"},
                    headers=HEADERS,
                    timeout=(10, 25),
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt == 1:
                    time.sleep(1)
                    continue
                response.raise_for_status()
                payload = response.json()
                frame = audit.yahoo_chart_payload_to_frame(payload)
                frame["Data_Source"] = f"Yahoo chart endpoint ({host})"
                return frame
            except Exception as exc:
                errors.append(f"{host}, attempt {attempt}: {exc}")
                if attempt == 1:
                    time.sleep(0.5)
    raise RuntimeError("Yahoo chart download failed. " + " | ".join(errors))


def _download_yfinance_fallback(symbol: str) -> pd.DataFrame:
    """Fallback that requests unadjusted OHLC and actions as separate columns."""
    import yfinance as yf  # loaded only if direct Yahoo chart requests fail

    history = yf.Ticker(symbol).history(period="10y", interval="1d", auto_adjust=False, actions=True, timeout=25)
    if history is None or history.empty:
        raise RuntimeError("yfinance returned no rows.")
    if isinstance(history.columns, pd.MultiIndex):
        # yfinance versions may return (field, ticker) or (ticker, field).
        new_cols = []
        for col in history.columns:
            parts = [str(part) for part in col if str(part) != symbol]
            new_cols.append(parts[0] if parts else str(col[0]))
        history.columns = new_cols

    out = history.copy()
    renames = {"Adj Close": "Adjusted_Close", "Dividends": "Dividend_Per_Share", "Stock Splits": "Stock_Split_Ratio"}
    out = out.rename(columns=renames)
    if "Adjusted_Close" not in out:
        out["Adjusted_Close"] = np.nan
    if "Dividend_Per_Share" not in out:
        out["Dividend_Per_Share"] = 0.0
    if "Stock_Split_Ratio" not in out:
        out["Stock_Split_Ratio"] = np.nan
    out["Stock_Split_Ratio_Text"] = ""
    out["Data_Source"] = "yfinance history(auto_adjust=False, actions=True)"
    out = _normalize_index(out)
    out = audit.prepare_execution_price_frame(out)
    return out


def download_quote_bars(symbol: str) -> pd.DataFrame:
    """Direct Yahoo chart first, yfinance fallback. Keep quote OHLC distinct from Adj Close."""
    print(f"Downloading execution-price bars for {symbol}...")
    try:
        bars = _download_yahoo_chart(symbol)
    except Exception as direct_error:
        print(f"Direct Yahoo chart request failed: {direct_error}")
        try:
            bars = _download_yfinance_fallback(symbol)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not download {symbol}. Direct request error: {direct_error}; "
                f"yfinance fallback error: {fallback_error}"
            ) from fallback_error
    bars = _exclude_incomplete_current_session(bars)
    bars = audit.prepare_execution_price_frame(bars)
    if len(bars) < 260:
        raise ValueError(f"Only {len(bars)} usable sessions for {symbol}; insufficient warm-up history.")
    print(f"  {len(bars)} sessions: {bars.index[0].date()} to {bars.index[-1].date()}")
    return bars


def _load_analyzer():
    try:
        import main as analyzer
    except Exception as exc:
        raise RuntimeError(
            "Could not import main.py. Keep backtest_engine_v2.py beside your current main.py. "
            f"Original error: {exc}"
        ) from exc
    return analyzer


def _attach_indicators(raw_bars: pd.DataFrame, analyzer: Any) -> pd.DataFrame:
    # main.calculate_indicators uses the provided Open/High/Low/Close. Here Close
    # is kept on the provider quote-price basis, not overwritten by Adjusted_Close.
    indicators = analyzer.calculate_indicators(raw_bars.copy())
    return _normalize_index(indicators)


def _get_start_pos(data: pd.DataFrame, analyzer: Any) -> int:
    if "SMA_200" not in data.columns:
        raise ValueError("main.calculate_indicators did not create SMA_200.")
    ready = np.flatnonzero(data["SMA_200"].notna().to_numpy())
    if not len(ready):
        raise ValueError("No valid SMA_200 values; insufficient history.")
    configured_start = pd.Timestamp(getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03"))
    return max(int(ready[0]), int(data.index.searchsorted(configured_start)))


def _periods(data: pd.DataFrame, start_pos: int) -> list[dict[str, Any]]:
    full_start, last_date = pd.Timestamp(data.index[start_pos]), pd.Timestamp(data.index[-1])
    candidates = [
        {"Period": "FULL_AVAILABLE_WINDOW", "Start": full_start, "End": last_date},
        {"Period": "DEVELOPMENT_THROUGH_2024", "Start": full_start, "End": min(DEVELOPMENT_END, last_date)},
        {"Period": "PREVIOUSLY_REVIEWED_POST_2024", "Start": max(POST_2024_START, full_start), "End": last_date},
    ]
    return [p for p in candidates if p["Start"] <= p["End"]]


def _eligible_dividends(bars: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float:
    """Cash dividend per share on ex-dates after entry and through exit, if recorded."""
    if "Dividend_Per_Share" not in bars.columns:
        return 0.0
    eligible = (bars.index > entry_date) & (bars.index <= exit_date)
    amounts = pd.to_numeric(bars.loc[eligible, "Dividend_Per_Share"], errors="coerce").fillna(0.0)
    return float(amounts.sum())


def _build_equity_curve(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    start_pos: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Daily simulated equity includes dividends; exit days mark at simulated exit price."""
    frame = audit.validate_ohlc(bars)
    if not (0 <= start_pos < len(frame)):
        raise IndexError("start_pos is outside the price history.")
    dates = frame.index[start_pos:]
    if len(dates) == 0:
        empty_curve = pd.DataFrame(columns=["Date", "Close", "Strategy_Equity", "In_Position", "Equity_Peak", "Daily_Simulated_Equity_Drawdown_Pct"])
        return empty_curve, {
            "Final_Strategy_Equity": 1.0,
            "Compounded_Strategy_Return_Pct": 0.0,
            "Maximum_Daily_Simulated_Equity_Drawdown_Pct": 0.0,
            "Percent_Of_Sessions_In_Position": 0.0,
        }
    curve = pd.DataFrame(index=dates)
    curve.index.name = "Date"
    curve["Close"] = frame.loc[dates, "Close"].astype(float)
    curve["Strategy_Equity"] = np.nan
    curve["In_Position"] = False
    current_equity = 1.0
    cursor = start_pos
    ordered = trades.sort_values("Entry_Date") if not trades.empty else trades

    for _, trade in ordered.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"]).tz_localize(None).normalize()
        exit_date = pd.Timestamp(trade["Exit_Date"]).tz_localize(None).normalize()
        if entry_date not in frame.index or exit_date not in frame.index:
            raise ValueError("Trade dates are missing from saved input bars.")
        entry_pos, exit_pos = int(frame.index.get_loc(entry_date)), int(frame.index.get_loc(exit_date))
        if entry_pos < cursor:
            raise ValueError("Overlapping trades in equity-curve input.")
        if cursor < entry_pos:
            curve.loc[frame.index[cursor:entry_pos], "Strategy_Equity"] = current_equity

        entry_price = float(trade["Entry_Price"])
        cost_fraction = float(trade["Assumed_Round_Trip_Cost_Pct"]) / 100.0
        base_equity = current_equity
        for pos in range(entry_pos, exit_pos + 1):
            date = frame.index[pos]
            mark_price = float(trade["Exit_Price"]) if pos == exit_pos else float(frame.iloc[pos]["Close"])
            dividends_to_date = _eligible_dividends(frame, entry_date, date)
            curve.at[date, "Strategy_Equity"] = base_equity * (
                (mark_price + dividends_to_date) / entry_price - cost_fraction
            )
            curve.at[date, "In_Position"] = True

        current_equity = base_equity * (1.0 + float(trade["Net_Return_Pct"]) / 100.0)
        curve.at[exit_date, "Strategy_Equity"] = current_equity
        cursor = exit_pos + 1

    if cursor < len(frame):
        curve.loc[frame.index[cursor:], "Strategy_Equity"] = current_equity
    curve["Strategy_Equity"] = curve["Strategy_Equity"].ffill().fillna(1.0)
    curve["Equity_Peak"] = curve["Strategy_Equity"].cummax()
    curve["Daily_Simulated_Equity_Drawdown_Pct"] = (
        curve["Strategy_Equity"] / curve["Equity_Peak"] - 1.0
    ) * 100.0
    stats = {
        "Final_Strategy_Equity": float(curve["Strategy_Equity"].iloc[-1]),
        "Compounded_Strategy_Return_Pct": float((curve["Strategy_Equity"].iloc[-1] - 1.0) * 100.0),
        "Maximum_Daily_Simulated_Equity_Drawdown_Pct": float(curve["Daily_Simulated_Equity_Drawdown_Pct"].min()),
        "Percent_Of_Sessions_In_Position": float(curve["In_Position"].mean() * 100.0),
    }
    return curve.reset_index(), stats


def _trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
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


def _benchmark_return(bars: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, cost_pct: float) -> float:
    """Total-return proxy using provider Adjusted_Close when available."""
    subset = bars.loc[(bars.index >= start_date) & (bars.index <= end_date)]
    column = "Adjusted_Close" if "Adjusted_Close" in subset.columns else "Close"
    values = pd.to_numeric(subset[column], errors="coerce").dropna()
    if len(values) < 2 and column == "Adjusted_Close":
        values = pd.to_numeric(subset["Close"], errors="coerce").dropna()
    if len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
        return float("nan")
    return float((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0 - cost_pct)


def _generate_trades(
    symbol: str,
    data: pd.DataFrame,
    start_pos: int,
    period: dict[str, Any],
    entry_variant: dict[str, Any],
    exit_variant: dict[str, Any],
    analyzer: Any,
) -> pd.DataFrame:
    holding = int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5))
    min_score = int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4))
    require_positive_rs = bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True))
    cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))
    first_signal_pos = max(start_pos, int(data.index.searchsorted(pd.Timestamp(period["Start"]))))
    last_signal_pos = len(data) - holding - 1
    if first_signal_pos > last_signal_pos:
        return pd.DataFrame()

    trades: list[dict[str, Any]] = []
    last_exit_pos = -1
    for signal_pos in range(first_signal_pos, last_signal_pos + 1):
        signal_date = pd.Timestamp(data.index[signal_pos])
        if signal_date > pd.Timestamp(period["End"]):
            break
        # Signals on the exit date are allowed because the next entry is next day's open.
        if audit.should_skip_signal(signal_pos, last_exit_pos):
            continue
        signal = data.iloc[signal_pos]
        score_value = analyzer.calculate_setup_score(signal)
        score = int(score_value) if pd.notna(score_value) else 0
        rel_strength = pd.to_numeric(signal.get("Relative_Strength_20D_Pct", np.nan), errors="coerce")
        if score < min_score or bool(signal.get("Breakdown_Confirmed", False)):
            continue
        if require_positive_rs and (pd.isna(rel_strength) or float(rel_strength) <= 0):
            continue
        if entry_variant["Use_NIFTY_SMA200_Filter"]:
            nifty_close = pd.to_numeric(signal.get("NIFTY_Close", np.nan), errors="coerce")
            nifty_sma = pd.to_numeric(signal.get("NIFTY_SMA_200", np.nan), errors="coerce")
            if pd.isna(nifty_close) or pd.isna(nifty_sma) or float(nifty_close) <= float(nifty_sma):
                continue

        uses_atr = exit_variant["Stop_ATR_Multiplier"] is not None or exit_variant["Target_ATR_Multiplier"] is not None
        signal_atr = pd.to_numeric(signal.get("ATR_14", np.nan), errors="coerce")
        if uses_atr and (pd.isna(signal_atr) or float(signal_atr) <= 0):
            continue

        entry_pos = signal_pos + 1
        time_exit_pos = entry_pos + holding - 1
        if time_exit_pos >= len(data):
            break
        entry_date = pd.Timestamp(data.index[entry_pos])
        scheduled_exit_date = pd.Timestamp(data.index[time_exit_pos])
        if entry_date < pd.Timestamp(period["Start"]) or scheduled_exit_date > pd.Timestamp(period["End"]):
            continue
        entry_price = float(data.iloc[entry_pos]["Open"])
        if not math.isfinite(entry_price) or entry_price <= 0:
            raise ValueError(f"Invalid entry price for {symbol} on {entry_date.date()}.")

        exit_result = audit.simulate_long_exit(
            data,
            entry_pos=entry_pos,
            time_exit_pos=time_exit_pos,
            entry_price=entry_price,
            signal_atr=float(signal_atr) if pd.notna(signal_atr) else None,
            stop_atr_multiplier=exit_variant["Stop_ATR_Multiplier"],
            target_atr_multiplier=exit_variant["Target_ATR_Multiplier"],
        )
        exit_date = pd.Timestamp(exit_result.exit_date)
        dividends = _eligible_dividends(data, entry_date, exit_date)
        gross_pct = ((exit_result.exit_price + dividends) / entry_price - 1.0) * 100.0
        net_pct = gross_pct - cost_pct
        trades.append({
            "Symbol": symbol,
            "Period": period["Period"],
            "Entry_Variant": entry_variant["Entry_Variant"],
            "Exit_Variant": exit_variant["Exit_Variant"],
            "Signal_Date": signal_date.date().isoformat(),
            "Entry_Date": entry_date.date().isoformat(),
            "Exit_Date": exit_date.date().isoformat(),
            "Exit_Reason": exit_result.exit_reason,
            "Entry_Price": entry_price,
            "Exit_Price": float(exit_result.exit_price),
            "Signal_ATR_14": float(signal_atr) if pd.notna(signal_atr) else np.nan,
            "Stop_ATR_Multiplier": exit_variant["Stop_ATR_Multiplier"],
            "Target_ATR_Multiplier": exit_variant["Target_ATR_Multiplier"],
            "Stop_Price": exit_result.stop_price,
            "Target_Price": exit_result.target_price,
            "Dividend_Per_Share_Received": dividends,
            "Holding_Sessions": int(data.index.get_loc(exit_date) - entry_pos + 1),
            "Scheduled_Max_Holding_Sessions": holding,
            "Setup_Score": score,
            "Relative_Strength_20D_Pct": float(rel_strength) if pd.notna(rel_strength) else np.nan,
            "NIFTY_Close_On_Signal": signal.get("NIFTY_Close", np.nan),
            "NIFTY_SMA_200_On_Signal": signal.get("NIFTY_SMA_200", np.nan),
            "NIFTY_Above_SMA200_On_Signal": bool(
                pd.notna(signal.get("NIFTY_Close", np.nan))
                and pd.notna(signal.get("NIFTY_SMA_200", np.nan))
                and float(signal.get("NIFTY_Close")) > float(signal.get("NIFTY_SMA_200"))
            ),
            "Gross_Return_Pct": gross_pct,
            "Assumed_Round_Trip_Cost_Pct": cost_pct,
            "Net_Return_Pct": net_pct,
            "Outcome": "Win" if net_pct > 0 else "Loss/Flat",
        })
        last_exit_pos = int(data.index.get_loc(exit_date))
    return pd.DataFrame(trades)


def _summary_row(
    symbol: str,
    period: dict[str, Any],
    entry_variant: dict[str, Any],
    exit_variant: dict[str, Any],
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    start_pos: int,
    analyzer: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Build this period's equity independently; do not accidentally reuse the
    # full historical curve for development/post-2024 period summaries.
    period_start = pd.Timestamp(period["Start"])
    period_end = pd.Timestamp(period["End"])
    window_bars = bars.loc[(bars.index >= period_start) & (bars.index <= period_end)].copy()
    curve, curve_stats = _build_equity_curve(window_bars, trades, 0)
    cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))
    summary: dict[str, Any] = {
        "Symbol": symbol,
        "Period": period["Period"],
        "Period_Start": pd.Timestamp(period["Start"]).date().isoformat(),
        "Period_End": pd.Timestamp(period["End"]).date().isoformat(),
        "Entry_Variant": entry_variant["Entry_Variant"],
        "Exit_Variant": exit_variant["Exit_Variant"],
        "Use_NIFTY_SMA200_Filter": bool(entry_variant["Use_NIFTY_SMA200_Filter"]),
        "Holding_Sessions": int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5)),
        "Minimum_Setup_Score": int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4)),
        "Require_Positive_Relative_Strength": bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)),
        "Assumed_Round_Trip_Cost_Pct": cost_pct,
        "Stock_Buy_Hold_Total_Return_Proxy_Pct": _benchmark_return(bars, pd.Timestamp(period["Start"]), pd.Timestamp(period["End"]), cost_pct),
        "NIFTY_Buy_Hold_Total_Return_Proxy_Pct": np.nan,
        **curve_stats,
        **_trade_metrics(trades),
    }
    for reason in ("TIME_EXIT", "STOP_GAP_AT_OPEN", "ATR_STOP", "TARGET_GAP_AT_OPEN_LIMIT", "ATR_TARGET", "BOTH_TOUCHED_STOP_FIRST"):
        summary[f"Exit_Count_{reason}"] = int((trades["Exit_Reason"] == reason).sum()) if not trades.empty else 0
    if not curve.empty:
        curve["Symbol"] = symbol
        curve["Period"] = period["Period"]
        curve["Entry_Variant"] = entry_variant["Entry_Variant"]
        curve["Exit_Variant"] = exit_variant["Exit_Variant"]
    return summary, curve


def run_backtest() -> Path:
    analyzer = _load_analyzer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "data" / "processed" / "backtest_engine_v2" / timestamp
    inputs_dir = output_dir / "input_bars"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    print("\nINDIAN STOCK ANALYZER — BACKTEST ENGINE V2")
    print("Separate engine; main.py and earlier experiments remain unchanged.")
    print(f"Output folder: {output_dir.resolve()}")
    print("Price basis: quote OHLC kept separate from Adjusted_Close; split basis needs independent verification.")

    nifty_bars = download_quote_bars("^NSEI")
    audit.save_input_snapshot(
        nifty_bars, inputs_dir, "^NSEI", source="Yahoo chart endpoint / yfinance fallback",
        price_basis="Provider quote OHLC preserved; Adjusted_Close and actions separated; split basis still to be confirmed",
    )
    nifty_indicators = _attach_indicators(nifty_bars, analyzer)

    summaries: list[dict[str, Any]] = []
    trades_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    for symbol in SYMBOLS:
        print(f"\nProcessing {symbol}...")
        try:
            bars = download_quote_bars(symbol)
            audit.save_input_snapshot(
                bars, inputs_dir, symbol, source="Yahoo chart endpoint / yfinance fallback",
                price_basis="Provider quote OHLC preserved; Adjusted_Close and actions separated; split basis still to be confirmed",
            )
            data = _attach_indicators(bars, analyzer)
            data = _normalize_index(analyzer.add_relative_strength(data, nifty_bars))
            data["NIFTY_Close"] = nifty_indicators["Close"].reindex(data.index)
            data["NIFTY_SMA_200"] = nifty_indicators["SMA_200"].reindex(data.index)
            start_pos = _get_start_pos(data, analyzer)
            all_periods = _periods(data, start_pos)
            for period in all_periods:
                period_start = pd.Timestamp(period["Start"])
                period_end = pd.Timestamp(period["End"])
                for entry_variant in ENTRY_VARIANTS:
                    for exit_variant in EXIT_VARIANTS:
                        trades = _generate_trades(
                            symbol, data, start_pos, period, entry_variant, exit_variant, analyzer
                        )
                        if not trades.empty:
                            trades_parts.append(trades)
                        summary, curve = _summary_row(
                            symbol, period, entry_variant, exit_variant,
                            data, trades, start_pos, analyzer
                        )
                        # NIFTY benchmark always shares the same calendar window.
                        cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))
                        summary["NIFTY_Buy_Hold_Total_Return_Proxy_Pct"] = _benchmark_return(
                            nifty_bars, period_start, period_end, cost_pct
                        )
                        summary["Input_Bars_First_Date"] = bars.index[0].date().isoformat()
                        summary["Input_Bars_Last_Date"] = bars.index[-1].date().isoformat()
                        summary["Input_Price_Basis"] = "provider quote OHLC; dividend-adjusted close separate; split basis unverified"
                        summaries.append(summary)
                        if not curve.empty:
                            curve_parts.append(curve)
                        avg = summary["Average_Net_Trade_Return_Pct"]
                        pf = summary["Profit_Factor_Equity_Weighted"]
                        print(
                            f"  {period['Period']:<31} {entry_variant['Entry_Variant']:<27} "
                            f"{exit_variant['Exit_Variant']:<25} trades={summary['Number_Of_Trades']:>3} "
                            f"avg_net={avg:>7.3f}% PF_equity={pf:>5.2f} "
                            f"return={summary['Compounded_Strategy_Return_Pct']:>8.2f}% "
                            f"MDD={summary['Maximum_Daily_Simulated_Equity_Drawdown_Pct']:>7.2f}%"
                        )
        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc), "Traceback": traceback.format_exc()})
            print(f"  ERROR: {exc}")

    if not summaries:
        if errors:
            pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
        raise RuntimeError("No summaries generated; inspect errors.csv and terminal output.")

    summary_df = pd.DataFrame(summaries)
    trades_df = pd.concat(trades_parts, ignore_index=True) if trades_parts else pd.DataFrame()
    curves_df = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    summary_path = output_dir / "backtest_engine_v2_summary.csv"
    trades_path = output_dir / "backtest_engine_v2_trades.csv"
    curves_path = output_dir / "backtest_engine_v2_equity_curves.csv"
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)

    config = {
        "symbols": SYMBOLS,
        "entry_variants": ENTRY_VARIANTS,
        "exit_variants": EXIT_VARIANTS,
        "holding_sessions": int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5)),
        "minimum_setup_score": int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4)),
        "require_positive_relative_strength": bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)),
        "round_trip_cost_pct_assumption": float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20)),
        "configured_start_date": getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03"),
        "price_source": "Yahoo chart endpoint, yfinance fallback",
        "price_basis_warning": "Quote OHLC preserved; verify split-adjustment basis independently before relying on results.",
        "data_snapshot_dir": str(inputs_dir),
    }
    (output_dir / "engine_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    methodology = """BACKTEST ENGINE V2 — METHODOLOGY AND LIMITATIONS

- Original entry score, breakdown exclusion, positive 20-session relative-strength requirement, five-session maximum holding period, and 0.20 percentage-point round-trip cost are read from main.py.
- Signals are evaluated at the signal-day close and entered at the next session's Open.
- The same-exit-date signal is permitted; its entry is the next session's Open, so it does not overlap the previous position.
- Stop/target levels use ATR from the signal-day close. A stop gap exits at the opening price. A target gap is conservatively limited to the target price. If both levels are touched during a daily candle without an opening trigger, stop-first is assumed.
- Cash dividends recorded on ex-dates after entry and through exit are added to trade return. Corporate-action data and per-share dividend units should be checked against source records.
- Provider quote OHLC is kept separate from Adjusted_Close. The endpoint's split-adjustment convention has not been independently confirmed, so this is an improvement over local dividend-adjustment of OHLC but not yet certified as executable-price data.
- Every downloaded bar set is saved as CSV with a SHA-256 manifest. These snapshots make the run reproducible against the saved inputs.
- Equity curves use closing-price marks while open and simulated exit price on exit dates; field names explicitly say simulated-equity drawdown.
- Both percentage-based trade profit factor and equity-weighted profit factor are reported.
- Benchmarks use Adjusted_Close when available as a total-return proxy, not as an intraday execution series. Strategy exposure may be lower than buy-and-hold exposure.
- 2025 onward was already examined in prior experiments and is labelled PREVIOUSLY_REVIEWED_POST_2024, not a pristine holdout.
- The 0.20% cost is only an assumption and does not fully model brokerage, taxes/charges, slippage, spread, liquidity, market impact, or order execution.
- This experiment does not establish profitability or suitability for real-money trading.
"""
    (output_dir / "methodology.txt").write_text(methodology, encoding="utf-8")

    print("\nSaved reports:")
    print(f"  {summary_path.resolve()}")
    print(f"  {trades_path.resolve()}")
    print(f"  {curves_path.resolve()}")
    print(f"  {inputs_dir.resolve()} (CSV snapshots + SHA-256 manifests)")
    if errors:
        print(f"  {output_dir / 'errors.csv'}")
    print("\nReminder: this checks a backtest implementation, not whether the strategy has a reliable edge.")
    return output_dir


def main() -> int:
    run_backtest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
