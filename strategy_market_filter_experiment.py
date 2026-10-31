"""Test a NIFTY 50 200-day moving-average filter against the existing baseline.

Save beside main.py and run:
    python strategy_market_filter_experiment.py

The script does not modify main.py. It compares the original entry rules with
one additional rule: NIFTY 50 must close above its own SMA_200 on the signal day.
It reports a development period and a later out-of-sample period separately.
Historical backtests do not guarantee future performance.
"""

from datetime import datetime
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

try:
    import main as analyzer
except Exception as exc:
    raise SystemExit(
        "Could not import main.py. Save this script in the same folder as main.py, "
        "and check that your original project runs.\n"
        f"Details: {exc}"
    )

SYMBOLS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

# This split is defined before evaluating the new rule.
DEVELOPMENT_END_DATE = "2024-12-31"
OUT_OF_SAMPLE_START_DATE = "2025-01-01"

PERIODS = [
    {"Period": "DEVELOPMENT", "Start": getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03"), "End": DEVELOPMENT_END_DATE},
    {"Period": "OUT_OF_SAMPLE", "Start": OUT_OF_SAMPLE_START_DATE, "End": None},
]

VARIANTS = [
    {"Variant": "BASELINE", "Use_NIFTY_SMA200_Filter": False},
    {"Variant": "NIFTY_ABOVE_SMA200", "Use_NIFTY_SMA200_Filter": True},
]


def normalize_dates(frame):
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "Date"
    return out


def as_float(value):
    try:
        return float(value) if pd.notna(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def as_true(value):
    return bool(value) if pd.notna(value) else False


def get_start_position(data):
    if "SMA_200" not in data.columns:
        raise ValueError("SMA_200 is missing. Check indicator calculation in main.py.")
    ready = np.flatnonzero(data["SMA_200"].notna().to_numpy())
    if not len(ready):
        raise ValueError("Not enough data to calculate SMA_200.")
    fixed_start = pd.Timestamp(getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03"))
    return max(int(ready[0]), int(data.index.searchsorted(fixed_start)))


def evaluate_trades(symbol, data, start_pos, period, variant):
    """Generate non-overlapping trades within the requested period."""
    holding = int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5))
    min_score = int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4))
    require_positive_rs = bool(
        getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)
    )
    cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))

    period_start = pd.Timestamp(period["Start"])
    period_end = (
        pd.Timestamp(period["End"])
        if period.get("End") else data.index[-1]
    )
    start_signal_pos = max(start_pos, int(data.index.searchsorted(period_start)))
    last_possible_signal_pos = len(data) - holding - 1
    if start_signal_pos > last_possible_signal_pos:
        return pd.DataFrame()

    trades = []
    last_exit_pos = -1

    for signal_pos in range(start_signal_pos, last_possible_signal_pos + 1):
        signal_date = data.index[signal_pos]
        if signal_date > period_end:
            break
        if signal_pos <= last_exit_pos:
            continue

        signal = data.iloc[signal_pos]
        score = int(analyzer.calculate_setup_score(signal))
        relative_strength = as_float(signal.get("Relative_Strength_20D_Pct", np.nan))
        breakdown = as_true(signal.get("Breakdown_Confirmed", False))

        if score < min_score or breakdown:
            continue
        if require_positive_rs and (pd.isna(relative_strength) or relative_strength <= 0):
            continue

        nifty_close = as_float(signal.get("NIFTY_Close", np.nan))
        nifty_sma200 = as_float(signal.get("NIFTY_SMA_200", np.nan))
        nifty_above_sma200 = (
            pd.notna(nifty_close) and pd.notna(nifty_sma200)
            and nifty_close > nifty_sma200
        )
        if variant["Use_NIFTY_SMA200_Filter"] and not nifty_above_sma200:
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + holding - 1
        if exit_pos >= len(data):
            continue
        entry_date = data.index[entry_pos]
        exit_date = data.index[exit_pos]

        # A trade must start and finish inside its named evaluation period.
        if entry_date < period_start or exit_date > period_end:
            continue

        entry_price = as_float(data.iloc[entry_pos].get("Open", np.nan))
        exit_price = as_float(data.iloc[exit_pos].get("Close", np.nan))
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0 or exit_price <= 0:
            continue

        gross_return = (exit_price / entry_price - 1.0) * 100.0
        net_return = gross_return - cost_pct
        trades.append({
            "Symbol": symbol,
            "Period": period["Period"],
            "Variant": variant["Variant"],
            "Signal_Date": signal_date.date().isoformat(),
            "Entry_Date": entry_date.date().isoformat(),
            "Exit_Date": exit_date.date().isoformat(),
            "Signal_Year": int(signal_date.year),
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
            "Holding_Sessions": holding,
            "Signal_Setup_Score": score,
            "Signal_Relative_Strength_20D_Pct": relative_strength,
            "Signal_RSI_14": as_float(signal.get("RSI_14", np.nan)),
            "Signal_ATR_Pct": as_float(signal.get("ATR_Pct", np.nan)),
            "Signal_Relative_Volume_20D": as_float(signal.get("Relative_Volume_20D", np.nan)),
            "NIFTY_Close_On_Signal": nifty_close,
            "NIFTY_SMA_200_On_Signal": nifty_sma200,
            "NIFTY_Above_SMA200_On_Signal": bool(nifty_above_sma200),
            "Gross_Return_Pct": gross_return,
            "Assumed_Round_Trip_Cost_Pct": cost_pct,
            "Net_Return_Pct": net_return,
        })
        last_exit_pos = exit_pos

    return pd.DataFrame(trades)


def make_equity_curve(data, trades, period_start, period_end, symbol, period_name, variant_name, cost_pct):
    """Build a daily close mark-to-market curve with cash between trades."""
    period_data = data.loc[(data.index >= period_start) & (data.index <= period_end)].copy()
    if period_data.empty:
        return pd.DataFrame(columns=["Symbol", "Period", "Variant", "Date", "Strategy_Equity", "Daily_Drawdown_Pct", "In_Position"]), 1.0, 0.0

    equity = pd.Series(index=period_data.index, dtype=float)
    in_position = pd.Series(False, index=period_data.index, dtype=bool)
    current_equity = 1.0
    cursor = 0
    trades_sorted = trades.sort_values("Entry_Date") if not trades.empty else trades

    for _, trade in trades_sorted.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"])
        exit_date = pd.Timestamp(trade["Exit_Date"])
        if entry_date not in period_data.index or exit_date not in period_data.index:
            continue
        entry_pos = int(period_data.index.get_loc(entry_date))
        exit_pos = int(period_data.index.get_loc(exit_date))
        if entry_pos < cursor:
            raise ValueError("Overlapping or unordered trades found while building equity curve.")

        if cursor < entry_pos:
            equity.iloc[cursor:entry_pos] = current_equity

        entry_price = float(trade["Entry_Price"])
        for pos in range(entry_pos, exit_pos + 1):
            close = as_float(period_data.iloc[pos].get("Close", np.nan))
            if pd.notna(close) and entry_price > 0:
                equity.iloc[pos] = current_equity * (close / entry_price - cost_pct / 100.0)
                in_position.iloc[pos] = True
            else:
                equity.iloc[pos] = current_equity

        current_equity *= 1.0 + float(trade["Net_Return_Pct"]) / 100.0
        equity.iloc[exit_pos] = current_equity
        cursor = exit_pos + 1

    if cursor < len(period_data):
        equity.iloc[cursor:] = current_equity

    equity = equity.ffill().fillna(1.0)
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0) * 100.0
    curve = pd.DataFrame({
        "Symbol": symbol,
        "Period": period_name,
        "Variant": variant_name,
        "Date": period_data.index,
        "Strategy_Equity": equity.to_numpy(),
        "Equity_Peak": peak.to_numpy(),
        "Daily_Drawdown_Pct": drawdown.to_numpy(),
        "In_Position": in_position.to_numpy(),
    })
    final_equity = float(equity.iloc[-1])
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    return curve, final_equity, max_drawdown


def trade_metrics(trades):
    returns = pd.to_numeric(trades.get("Net_Return_Pct", pd.Series(dtype=float)), errors="coerce").dropna()
    if returns.empty:
        return {
            "Number_Of_Trades": 0,
            "Win_Rate_Pct": np.nan,
            "Average_Net_Trade_Return_Pct": np.nan,
            "Median_Net_Trade_Return_Pct": np.nan,
            "Profit_Factor": np.nan,
            "Trade_Compounded_Return_Pct": 0.0,
        }
    winners = returns[returns > 0]
    losers = returns[returns < 0]
    pf = float(winners.sum() / abs(losers.sum())) if len(losers) and abs(losers.sum()) > 0 else np.nan
    return {
        "Number_Of_Trades": int(len(returns)),
        "Win_Rate_Pct": float((returns > 0).mean() * 100.0),
        "Average_Net_Trade_Return_Pct": float(returns.mean()),
        "Median_Net_Trade_Return_Pct": float(returns.median()),
        "Profit_Factor": pf,
        "Trade_Compounded_Return_Pct": float(((1.0 + returns / 100.0).prod() - 1.0) * 100.0),
    }


def benchmark_returns(stock_data, nifty_data, period_start, period_end, cost_pct):
    """Use common stock/NIFTY dates for aligned close-to-close benchmarks."""
    stock_window = stock_data.loc[(stock_data.index >= period_start) & (stock_data.index <= period_end), "Close"]
    nifty_window = nifty_data.loc[(nifty_data.index >= period_start) & (nifty_data.index <= period_end), "Close"]
    common = stock_window.index.intersection(nifty_window.index)
    common = common[(common >= period_start) & (common <= period_end)]
    if len(common) < 2:
        return np.nan, np.nan, None, None
    first_date, last_date = common[0], common[-1]
    stock_first, stock_last = as_float(stock_window.loc[first_date]), as_float(stock_window.loc[last_date])
    nifty_first, nifty_last = as_float(nifty_window.loc[first_date]), as_float(nifty_window.loc[last_date])
    stock_net = ((stock_last / stock_first - 1.0) * 100.0 - cost_pct) if stock_first > 0 and stock_last > 0 else np.nan
    nifty_net = ((nifty_last / nifty_first - 1.0) * 100.0 - cost_pct) if nifty_first > 0 and nifty_last > 0 else np.nan
    return stock_net, nifty_net, first_date, last_date


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (Path(__file__).resolve().parent / "data" / "processed" / "market_filter_experiment" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nINDIAN STOCK ANALYZER — NIFTY 200-DAY TREND FILTER TEST")
    print("This script does not modify main.py or your previous experiment scripts.")
    print("Development period: configured start date through 2024-12-31")
    print("Out-of-sample period: 2025-01-01 through latest available data")
    print(f"Output folder: {output_dir.resolve()}\n")

    nifty_raw = analyzer.download_nifty50_data()
    if nifty_raw is None or nifty_raw.empty:
        raise SystemExit("Could not download NIFTY 50 data. Check your internet connection.")
    nifty_raw = normalize_dates(nifty_raw)
    nifty_indicators = normalize_dates(analyzer.calculate_indicators(nifty_raw))

    all_summaries = []
    all_trade_parts = []
    all_curve_parts = []
    errors = []

    for symbol in SYMBOLS:
        print(f"\nProcessing {symbol}...")
        try:
            raw = analyzer.download_stock_data(symbol)
            if raw is None or raw.empty:
                raise ValueError("No historical stock data returned.")
            raw = normalize_dates(raw)
            data = normalize_dates(analyzer.calculate_indicators(raw))
            data = normalize_dates(analyzer.add_relative_strength(data, nifty_raw))
            data["NIFTY_Close"] = nifty_indicators["Close"].reindex(data.index)
            data["NIFTY_SMA_200"] = nifty_indicators["SMA_200"].reindex(data.index)
            start_pos = get_start_position(data)
            cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))

            for period in PERIODS:
                period_start = pd.Timestamp(period["Start"])
                requested_end = pd.Timestamp(period["End"]) if period.get("End") else data.index[-1]
                period_end = min(requested_end, data.index[-1])
                if period_end <= period_start:
                    continue

                for variant in VARIANTS:
                    trades = evaluate_trades(symbol, data, start_pos, period, variant)
                    if not trades.empty:
                        all_trade_parts.append(trades)

                    curve, final_equity, max_dd = make_equity_curve(
                        data, trades, period_start, period_end,
                        symbol, period["Period"], variant["Variant"], cost_pct
                    )
                    if not curve.empty:
                        all_curve_parts.append(curve)

                    stock_bh, nifty_bh, benchmark_start, benchmark_end = benchmark_returns(
                        data, nifty_raw, period_start, period_end, cost_pct
                    )
                    summary = {
                        "Symbol": symbol,
                        "Period": period["Period"],
                        "Period_Start": period_start.date().isoformat(),
                        "Period_End": period_end.date().isoformat(),
                        "Variant": variant["Variant"],
                        "Use_NIFTY_200DMA_Filter": variant["Use_NIFTY_SMA200_Filter"],
                        "Holding_Sessions": int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5)),
                        "Minimum_Setup_Score": int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4)),
                        "Require_Positive_Relative_Strength": bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)),
                        "Assumed_Round_Trip_Cost_Pct": cost_pct,
                        "Actual_Benchmark_Window_Start": benchmark_start.date().isoformat() if benchmark_start is not None else None,
                        "Actual_Benchmark_Window_End": benchmark_end.date().isoformat() if benchmark_end is not None else None,
                        "Stock_Buy_Hold_Net_Return_Pct": stock_bh,
                        "NIFTY_Buy_Hold_Net_Return_Pct": nifty_bh,
                        "Final_Strategy_Equity": final_equity,
                        "Compounded_Strategy_Return_Pct": (final_equity - 1.0) * 100.0,
                        "Maximum_Daily_Close_MTM_Drawdown_Pct": max_dd,
                    }
                    summary.update(trade_metrics(trades))
                    all_summaries.append(summary)
                    avg_net = summary["Average_Net_Trade_Return_Pct"]
                    pf = summary["Profit_Factor"]
                    print(
                        f"  {period['Period']:<14} {variant['Variant']:<20} "
                        f"trades={summary['Number_Of_Trades']:>3} | "
                        f"avg net={avg_net:>7.3f}% | PF={pf:>5.2f} | "
                        f"strategy={summary['Compounded_Strategy_Return_Pct']:>8.2f}% | "
                        f"MDD={max_dd:>7.2f}%"
                    )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            errors.append({"Symbol": symbol, "Error": str(exc)})

    if not all_summaries:
        if errors:
            pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
        raise SystemExit("No summaries were generated. Review terminal errors.")

    summary_df = pd.DataFrame(all_summaries)
    summary_path = output_dir / "market_filter_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    trades_path = output_dir / "market_filter_trades.csv"
    trades_df = pd.concat(all_trade_parts, ignore_index=True) if all_trade_parts else pd.DataFrame()
    trades_df.to_csv(trades_path, index=False)

    equity_path = output_dir / "market_filter_equity_curves.csv"
    curve_df = pd.concat(all_curve_parts, ignore_index=True) if all_curve_parts else pd.DataFrame()
    curve_df.to_csv(equity_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)

    display_cols = [
        "Symbol", "Period", "Variant", "Number_Of_Trades",
        "Average_Net_Trade_Return_Pct", "Profit_Factor",
        "Compounded_Strategy_Return_Pct", "Maximum_Daily_Close_MTM_Drawdown_Pct",
        "Stock_Buy_Hold_Net_Return_Pct", "NIFTY_Buy_Hold_Net_Return_Pct",
    ]
    print("\nSUMMARY")
    print(summary_df[display_cols].round(2).to_string(index=False))
    print("\nSaved reports:")
    print(f"  {summary_path}")
    print(f"  {trades_path}")
    print(f"  {equity_path}")
    if errors:
        print(f"  {output_dir / 'errors.csv'}")
    print("\nImportant: compare out-of-sample results first. This filter is a hypothesis, not a proven improvement.")
    print("The NIFTY trend must be known at the signal day's close; entries are still next session Open.")
    print("Returns are simplified historical simulations and are not a guarantee of future performance.")


if __name__ == "__main__":
    main()
