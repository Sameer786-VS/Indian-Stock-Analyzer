"""\
INDIAN STOCK ANALYZER — RELATIVE-VOLUME FILTER EXPERIMENT

Save this file beside main.py and run:
    python strategy_relative_volume_experiment.py

Tests four controlled variants on five stocks:
  1. BASELINE
  2. NIFTY_ABOVE_SMA200
  3. RELATIVE_VOLUME_2X
  4. NIFTY_AND_VOLUME_2X

The volume rule requires signal-day volume to be at least 2.0 times
its prior 20-session average volume. The NIFTY filter requires NIFTY 50
close to be above its 200-day moving average on the signal day.

Development period ends 2024-12-31. Out-of-sample evaluation begins
2025-01-01. This script does not modify main.py.

Historical simulation only; no guarantee of future performance.
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
        "Could not import main.py. Save this file in the same folder as main.py.\n"
        f"Original error: {exc}"
    )

SYMBOLS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
DEVELOPMENT_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
VOLUME_THRESHOLD = 2.0

VARIANTS = [
    {
        "Variant": "BASELINE",
        "Use_NIFTY_SMA200_Filter": False,
        "Use_Relative_Volume_Filter": False,
    },
    {
        "Variant": "NIFTY_ABOVE_SMA200",
        "Use_NIFTY_SMA200_Filter": True,
        "Use_Relative_Volume_Filter": False,
    },
    {
        "Variant": "RELATIVE_VOLUME_2X",
        "Use_NIFTY_SMA200_Filter": False,
        "Use_Relative_Volume_Filter": True,
    },
    {
        "Variant": "NIFTY_AND_VOLUME_2X",
        "Use_NIFTY_SMA200_Filter": True,
        "Use_Relative_Volume_Filter": True,
    },
]

PERIODS = [
    {"Period": "DEVELOPMENT", "Start": None, "End": DEVELOPMENT_END},
    {"Period": "OUT_OF_SAMPLE", "Start": OOS_START, "End": None},
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


def as_bool(value):
    return bool(value) if pd.notna(value) else False


def get_start_position(data):
    if "SMA_200" not in data.columns:
        raise ValueError("SMA_200 is missing from indicator data.")
    ready = np.flatnonzero(data["SMA_200"].notna().to_numpy())
    if not len(ready):
        raise ValueError("There are not enough data to calculate SMA_200.")
    configured_start = pd.Timestamp(
        getattr(analyzer, "BACKTEST_TEST_START_DATE", "2022-08-03")
    )
    return max(int(ready[0]), int(data.index.searchsorted(configured_start)))


def generate_trades(symbol, data, start_pos, period, variant):
    holding = int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5))
    minimum_score = int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4))
    require_positive_rs = bool(
        getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)
    )
    cost_pct = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20))

    period_start = pd.Timestamp(period["Start"]) if period["Start"] is not None else data.index[start_pos]
    period_end = pd.Timestamp(period["End"]) if period["End"] is not None else data.index[-1]
    period_end = min(period_end, data.index[-1])
    first_signal_pos = max(start_pos, int(data.index.searchsorted(period_start)))
    last_signal_pos = len(data) - holding - 1

    if first_signal_pos > last_signal_pos or period_end < period_start:
        return pd.DataFrame()

    trades = []
    last_exit_pos = -1

    for signal_pos in range(first_signal_pos, last_signal_pos + 1):
        signal_date = data.index[signal_pos]
        if signal_date > period_end:
            break
        if signal_pos <= last_exit_pos:
            continue

        signal = data.iloc[signal_pos]
        score = int(analyzer.calculate_setup_score(signal))
        relative_strength = as_float(signal.get("Relative_Strength_20D_Pct", np.nan))
        relative_volume = as_float(signal.get("Relative_Volume_20D", np.nan))
        breakdown = as_bool(signal.get("Breakdown_Confirmed", False))
        nifty_close = as_float(signal.get("NIFTY_Close", np.nan))
        nifty_sma200 = as_float(signal.get("NIFTY_SMA_200", np.nan))

        nifty_above = (
            pd.notna(nifty_close)
            and pd.notna(nifty_sma200)
            and nifty_close > nifty_sma200
        )
        volume_confirmed = (
            pd.notna(relative_volume)
            and relative_volume >= VOLUME_THRESHOLD
        )

        # Original baseline entry requirements.
        if score < minimum_score or breakdown:
            continue
        if require_positive_rs and (
            pd.isna(relative_strength) or relative_strength <= 0
        ):
            continue

        # Controlled filters: each variant adds only the specified filter(s).
        if variant["Use_NIFTY_SMA200_Filter"] and not nifty_above:
            continue
        if variant["Use_Relative_Volume_Filter"] and not volume_confirmed:
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + holding - 1
        if exit_pos >= len(data):
            continue

        entry_date = data.index[entry_pos]
        exit_date = data.index[exit_pos]
        if entry_date < period_start or exit_date > period_end:
            continue

        entry_price = as_float(data.iloc[entry_pos].get("Open", np.nan))
        exit_price = as_float(data.iloc[exit_pos].get("Close", np.nan))
        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0 or exit_price <= 0:
            continue

        gross_pct = (exit_price / entry_price - 1.0) * 100.0
        net_pct = gross_pct - cost_pct

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
            "Setup_Score": score,
            "Relative_Strength_20D_Pct": relative_strength,
            "Relative_Volume_20D": relative_volume,
            "Volume_At_Least_2X": bool(volume_confirmed),
            "NIFTY_Close_On_Signal": nifty_close,
            "NIFTY_SMA_200_On_Signal": nifty_sma200,
            "NIFTY_Above_SMA200_On_Signal": bool(nifty_above),
            "Gross_Return_Pct": gross_pct,
            "Assumed_Round_Trip_Cost_Pct": cost_pct,
            "Net_Return_Pct": net_pct,
            "Outcome": "Win" if net_pct > 0 else "Loss/Flat",
        })
        last_exit_pos = exit_pos

    return pd.DataFrame(trades)


def make_equity_curve(data, trades, period_start, period_end, symbol, period_name, variant_name):
    window = data.loc[(data.index >= period_start) & (data.index <= period_end)].copy()
    cols = ["Symbol", "Period", "Variant", "Date", "Strategy_Equity", "Equity_Peak", "Daily_Drawdown_Pct", "In_Position"]
    if window.empty:
        return pd.DataFrame(columns=cols), 1.0, 0.0, 0.0

    holding = pd.Series(False, index=window.index)
    equity = pd.Series(index=window.index, dtype=float)
    current_equity = 1.0
    cursor = 0
    cost_fraction = float(getattr(analyzer, "BACKTEST_ROUND_TRIP_COST_PCT", 0.20)) / 100.0
    trades_sorted = trades.sort_values("Entry_Date") if not trades.empty else trades

    for _, trade in trades_sorted.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"])
        exit_date = pd.Timestamp(trade["Exit_Date"])
        if entry_date not in window.index or exit_date not in window.index:
            continue
        entry_pos = int(window.index.get_loc(entry_date))
        exit_pos = int(window.index.get_loc(exit_date))
        if entry_pos < cursor:
            raise ValueError("Overlapping trades detected in equity-curve calculation.")

        if cursor < entry_pos:
            equity.iloc[cursor:entry_pos] = current_equity

        entry_price = float(trade["Entry_Price"])
        for pos in range(entry_pos, exit_pos + 1):
            close = as_float(window.iloc[pos].get("Close", np.nan))
            if pd.notna(close) and entry_price > 0:
                equity.iloc[pos] = current_equity * (close / entry_price - cost_fraction)
                holding.iloc[pos] = True
            else:
                equity.iloc[pos] = current_equity

        current_equity *= 1.0 + float(trade["Net_Return_Pct"]) / 100.0
        equity.iloc[exit_pos] = current_equity
        cursor = exit_pos + 1

    if cursor < len(window):
        equity.iloc[cursor:] = current_equity

    equity = equity.ffill().fillna(1.0)
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0) * 100.0
    curve = pd.DataFrame({
        "Symbol": symbol,
        "Period": period_name,
        "Variant": variant_name,
        "Date": window.index,
        "Strategy_Equity": equity.to_numpy(),
        "Equity_Peak": peak.to_numpy(),
        "Daily_Drawdown_Pct": drawdown.to_numpy(),
        "In_Position": holding.to_numpy(),
    })
    return curve, float(equity.iloc[-1]), float(drawdown.min()), float(holding.mean() * 100.0)


def trade_metrics(trades):
    returns = pd.to_numeric(
        trades.get("Net_Return_Pct", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
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


def benchmark_returns(stock_data, nifty_data, start, end, cost_pct):
    stock_close = stock_data.loc[(stock_data.index >= start) & (stock_data.index <= end), "Close"]
    nifty_close = nifty_data.loc[(nifty_data.index >= start) & (nifty_data.index <= end), "Close"]
    common = stock_close.index.intersection(nifty_close.index)
    if len(common) < 2:
        return np.nan, np.nan
    first, last = common[0], common[-1]
    stock_start, stock_end = as_float(stock_close.loc[first]), as_float(stock_close.loc[last])
    nifty_start, nifty_end = as_float(nifty_close.loc[first]), as_float(nifty_close.loc[last])
    stock_return = (stock_end / stock_start - 1.0) * 100.0 - cost_pct if stock_start > 0 and stock_end > 0 else np.nan
    nifty_return = (nifty_end / nifty_start - 1.0) * 100.0 - cost_pct if nifty_start > 0 and nifty_end > 0 else np.nan
    return stock_return, nifty_return


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(__file__).resolve().parent
        / "data" / "processed" / "relative_volume_experiment" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nINDIAN STOCK ANALYZER — RELATIVE-VOLUME FILTER EXPERIMENT")
    print("This script does not modify main.py or previous experiment scripts.")
    print("Development period: configured start date through 2024-12-31")
    print("Out-of-sample period: 2025-01-01 through latest available data")
    print(f"Volume confirmation: Relative_Volume_20D >= {VOLUME_THRESHOLD:.1f}x")
    print(f"Output folder: {output_dir.resolve()}\n")

    print("Downloading NIFTY 50 data...")
    nifty_raw = analyzer.download_nifty50_data()
    if nifty_raw is None or nifty_raw.empty:
        raise SystemExit("Could not download NIFTY 50 data. Check your internet connection.")
    nifty_raw = normalize_dates(nifty_raw)
    nifty_indicators = normalize_dates(analyzer.calculate_indicators(nifty_raw))

    summaries = []
    trade_parts = []
    curve_parts = []
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
                period_start = pd.Timestamp(period["Start"]) if period["Start"] is not None else data.index[start_pos]
                period_end = pd.Timestamp(period["End"]) if period["End"] is not None else data.index[-1]
                period_end = min(period_end, data.index[-1])
                if period_end <= period_start:
                    continue

                for variant in VARIANTS:
                    trades = generate_trades(symbol, data, start_pos, period, variant)
                    if not trades.empty:
                        trade_parts.append(trades)

                    curve, final_equity, max_dd, exposure_pct = make_equity_curve(
                        data, trades, period_start, period_end, symbol,
                        period["Period"], variant["Variant"]
                    )
                    if not curve.empty:
                        curve_parts.append(curve)

                    stock_bh, nifty_bh = benchmark_returns(
                        data, nifty_raw, period_start, period_end, cost_pct
                    )
                    summary = {
                        "Symbol": symbol,
                        "Period": period["Period"],
                        "Period_Start": period_start.date().isoformat(),
                        "Period_End": period_end.date().isoformat(),
                        "Variant": variant["Variant"],
                        "Use_NIFTY_200DMA_Filter": variant["Use_NIFTY_SMA200_Filter"],
                        "Use_Relative_Volume_Filter": variant["Use_Relative_Volume_Filter"],
                        "Relative_Volume_Threshold": VOLUME_THRESHOLD,
                        "Holding_Sessions": int(getattr(analyzer, "BACKTEST_HOLDING_SESSIONS", 5)),
                        "Minimum_Setup_Score": int(getattr(analyzer, "BACKTEST_MIN_SETUP_SCORE", 4)),
                        "Require_Positive_Relative_Strength": bool(getattr(analyzer, "BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH", True)),
                        "Assumed_Round_Trip_Cost_Pct": cost_pct,
                        "Stock_Buy_Hold_Net_Return_Pct": stock_bh,
                        "NIFTY_Buy_Hold_Net_Return_Pct": nifty_bh,
                        "Final_Strategy_Equity": final_equity,
                        "Compounded_Strategy_Return_Pct": (final_equity - 1.0) * 100.0,
                        "Maximum_Daily_Close_MTM_Drawdown_Pct": max_dd,
                        "Percent_Of_Sessions_In_Position": exposure_pct,
                    }
                    summary.update(trade_metrics(trades))
                    summaries.append(summary)

                    avg = summary["Average_Net_Trade_Return_Pct"]
                    pf = summary["Profit_Factor"]
                    print(
                        f"  {period['Period']:<14} {variant['Variant']:<22} "
                        f"trades={summary['Number_Of_Trades']:>3} | "
                        f"avg net={avg:>7.3f}% | PF={pf:>5.2f} | "
                        f"strategy={summary['Compounded_Strategy_Return_Pct']:>8.2f}% | "
                        f"MDD={max_dd:>7.2f}% | exposure={exposure_pct:>5.1f}%"
                    )

        except Exception as exc:
            errors.append({"Symbol": symbol, "Error": str(exc)})
            print(f"  ERROR processing {symbol}: {exc}")
            traceback.print_exc()

    if not summaries:
        if errors:
            pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)
        raise SystemExit("No summaries generated. Review terminal errors.")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "relative_volume_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    trades_path = output_dir / "relative_volume_trades.csv"
    all_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    all_trades.to_csv(trades_path, index=False)

    equity_path = output_dir / "relative_volume_equity_curves.csv"
    all_curves = pd.concat(curve_parts, ignore_index=True) if curve_parts else pd.DataFrame()
    all_curves.to_csv(equity_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False)

    print("\nSUMMARY")
    display_cols = [
        "Symbol", "Period", "Variant", "Number_Of_Trades",
        "Win_Rate_Pct", "Average_Net_Trade_Return_Pct", "Profit_Factor",
        "Compounded_Strategy_Return_Pct", "Maximum_Daily_Close_MTM_Drawdown_Pct",
        "Percent_Of_Sessions_In_Position",
    ]
    print(summary_df[display_cols].round(2).to_string(index=False))
    print("\nSaved reports:")
    print(f"  {summary_path}")
    print(f"  {trades_path}")
    print(f"  {equity_path}")
    if errors:
        print(f"  {output_dir / 'errors.csv'}")
    print("\nCompare OUT_OF_SAMPLE first. A high historical return from a small sample is not proof of an edge.")
    print("This simplified simulation assumes fixed round-trip costs and does not include all execution effects.")


if __name__ == "__main__":
    main()
