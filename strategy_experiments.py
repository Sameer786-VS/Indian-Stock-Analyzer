"""Trade-path diagnostics for Indian Stock Analyzer.

Save beside main.py and run:
    python strategy_trade_diagnostics.py

This script leaves main.py unchanged. It downloads historical data through
main.py's existing functions and saves trade-path, yearly, feature-bucket,
and transaction-cost sensitivity reports.

MAE/MFE describe price excursions during each trade. They do NOT simulate
whether a stop-loss or target would have executed intraday.
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
        "Could not import main.py. Keep this file beside main.py and check "
        "that the existing project runs.\n"
        f"Details: {exc}"
    )

SYMBOLS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

VARIANTS = [
    {"Variant": "BASELINE", "Minimum_Setup_Score": 4,
     "Require_Confirmed_Breakout": False, "Holding_Sessions": 5},
    {"Variant": "SCORE_5", "Minimum_Setup_Score": 5,
     "Require_Confirmed_Breakout": False, "Holding_Sessions": 5},
    {"Variant": "BREAKOUT_REQUIRED", "Minimum_Setup_Score": 4,
     "Require_Confirmed_Breakout": True, "Holding_Sessions": 5},
    {"Variant": "HOLD_10", "Minimum_Setup_Score": 4,
     "Require_Confirmed_Breakout": False, "Holding_Sessions": 10},
]
COST_SCENARIOS_PCT = [0.20, 0.30, 0.50]


def normalize_dates(frame):
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "Date"
    return out


def num(value):
    try:
        return float(value) if pd.notna(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def get_start_position(data):
    if "SMA_200" not in data.columns:
        raise ValueError("SMA_200 is missing from indicator data.")
    ready = np.flatnonzero(data["SMA_200"].notna().to_numpy())
    if len(ready) == 0:
        raise ValueError("Not enough history to calculate SMA_200.")
    configured_start = pd.Timestamp(analyzer.BACKTEST_TEST_START_DATE)
    fixed_start = int(data.index.searchsorted(configured_start))
    return max(int(ready[0]), fixed_start)


def build_trades(symbol, data, start_pos, variant):
    holding = int(variant["Holding_Sessions"])
    min_score = int(variant["Minimum_Setup_Score"])
    require_breakout = bool(variant["Require_Confirmed_Breakout"])
    require_rs = bool(analyzer.BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH)
    base_cost = float(analyzer.BACKTEST_ROUND_TRIP_COST_PCT)

    trades = []
    last_exit_pos = -1
    last_signal_pos = len(data) - holding - 1

    for signal_pos in range(start_pos, last_signal_pos + 1):
        if signal_pos <= last_exit_pos:
            continue

        signal = data.iloc[signal_pos]
        score = int(analyzer.calculate_setup_score(signal))
        rs = num(signal.get("Relative_Strength_20D_Pct", np.nan))
        breakout = bool(signal.get("Breakout_Confirmed", False))
        breakdown = bool(signal.get("Breakdown_Confirmed", False))

        if score < min_score or breakdown:
            continue
        if require_rs and (pd.isna(rs) or rs <= 0):
            continue
        if require_breakout and not breakout:
            continue

        entry_pos = signal_pos + 1
        exit_pos = entry_pos + holding - 1
        if exit_pos >= len(data):
            break

        entry_row = data.iloc[entry_pos]
        exit_row = data.iloc[exit_pos]
        entry_price = num(entry_row.get("Open", np.nan))
        exit_price = num(exit_row.get("Close", np.nan))
        if (pd.isna(entry_price) or pd.isna(exit_price)
                or entry_price <= 0 or exit_price <= 0):
            continue

        path = data.iloc[entry_pos:exit_pos + 1]
        path_high = pd.to_numeric(path["High"], errors="coerce").dropna()
        path_low = pd.to_numeric(path["Low"], errors="coerce").dropna()
        mfe = ((path_high.max() / entry_price) - 1.0) * 100 if len(path_high) else np.nan
        mae = ((path_low.min() / entry_price) - 1.0) * 100 if len(path_low) else np.nan
        gross = ((exit_price / entry_price) - 1.0) * 100

        stock_close = num(signal.get("Close", np.nan))
        sma50 = num(signal.get("SMA_50", np.nan))
        sma200 = num(signal.get("SMA_200", np.nan))
        nifty_close = num(signal.get("NIFTY_Close", np.nan))
        nifty_sma200 = num(signal.get("NIFTY_SMA_200", np.nan))

        trades.append({
            "Symbol": symbol,
            "Variant": variant["Variant"],
            "Signal_Date": data.index[signal_pos].date().isoformat(),
            "Entry_Date": data.index[entry_pos].date().isoformat(),
            "Exit_Date": data.index[exit_pos].date().isoformat(),
            "Signal_Year": int(data.index[signal_pos].year),
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
            "Holding_Sessions": holding,
            "Signal_Setup_Score": score,
            "Signal_RSI_14": num(signal.get("RSI_14", np.nan)),
            "Signal_ATR_Pct": num(signal.get("ATR_Pct", np.nan)),
            "Signal_Relative_Volume_20D": num(signal.get("Relative_Volume_20D", np.nan)),
            "Signal_Return_20D_Pct": num(signal.get("Return_20D_Pct", np.nan)),
            "Signal_Relative_Strength_20D_Pct": rs,
            "Signal_Close_Above_SMA50": bool(pd.notna(sma50) and pd.notna(stock_close) and stock_close > sma50),
            "Signal_Close_Above_SMA200": bool(pd.notna(sma200) and pd.notna(stock_close) and stock_close > sma200),
            "Signal_SMA50_Above_SMA200": bool(pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200),
            "Signal_NIFTY_Above_SMA200": bool(pd.notna(nifty_close) and pd.notna(nifty_sma200) and nifty_close > nifty_sma200),
            "Signal_Confirmed_Breakout": breakout,
            "Gross_Return_Pct": gross,
            "Assumed_Round_Trip_Cost_Pct": base_cost,
            "Net_Return_Pct": gross - base_cost,
            "MFE_Max_Favorable_Excursion_Pct": mfe,
            "MAE_Max_Adverse_Excursion_Pct": mae,
            "Reached_Plus_1Pct_Intratrade": bool(pd.notna(mfe) and mfe >= 1.0),
            "Reached_Minus_1Pct_Intratrade": bool(pd.notna(mae) and mae <= -1.0),
        })
        last_exit_pos = exit_pos

    return pd.DataFrame(trades)


def metrics_for_returns(returns):
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if r.empty:
        return {
            "Trade_Count": 0, "Win_Rate_Pct": np.nan,
            "Average_Net_Return_Pct": np.nan,
            "Median_Net_Return_Pct": np.nan, "Profit_Factor": np.nan,
            "Trade_Compounded_Return_Pct": np.nan,
        }
    winners = r[r > 0]
    losers = r[r < 0]
    pf = winners.sum() / abs(losers.sum()) if len(losers) and abs(losers.sum()) > 0 else np.nan
    return {
        "Trade_Count": len(r),
        "Win_Rate_Pct": float((r > 0).mean() * 100),
        "Average_Net_Return_Pct": float(r.mean()),
        "Median_Net_Return_Pct": float(r.median()),
        "Profit_Factor": float(pf) if pd.notna(pf) else np.nan,
        "Trade_Compounded_Return_Pct": float(((1 + r / 100).prod() - 1) * 100),
    }


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "data" / "processed" / "trade_diagnostics" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nINDIAN STOCK ANALYZER — TRADE DIAGNOSTICS")
    print("This script does not modify main.py.")
    print(f"Output folder: {out_dir.resolve()}\n")

    nifty_raw = analyzer.download_nifty50_data()
    if nifty_raw is None or nifty_raw.empty:
        raise SystemExit("Could not download NIFTY 50 data. Check internet and try again.")
    nifty_raw = normalize_dates(nifty_raw)
    nifty_indicators = normalize_dates(analyzer.calculate_indicators(nifty_raw))

    trades_parts = []
    errors = []

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        try:
            raw = analyzer.download_stock_data(symbol)
            if raw is None or raw.empty:
                raise ValueError("No historical data returned.")
            raw = normalize_dates(raw)
            data = normalize_dates(analyzer.calculate_indicators(raw))
            data = normalize_dates(analyzer.add_relative_strength(data, nifty_raw))

            # Attach NIFTY trend state on matching dates for each signal.
            data["NIFTY_Close"] = nifty_indicators["Close"].reindex(data.index)
            data["NIFTY_SMA_200"] = nifty_indicators["SMA_200"].reindex(data.index)
            start_pos = get_start_position(data)

            for variant in VARIANTS:
                result = build_trades(symbol, data, start_pos, variant)
                if not result.empty:
                    trades_parts.append(result)
                print(f"  {variant['Variant']}: {len(result)} trades")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            errors.append({"Symbol": symbol, "Error": str(exc)})

    if not trades_parts:
        raise SystemExit("No trades were generated. Check the error messages above.")

    trades = pd.concat(trades_parts, ignore_index=True)
    trades_path = out_dir / "trade_path_diagnostics.csv"
    trades.to_csv(trades_path, index=False)

    # Overall summary for each stock and variant.
    summary_rows = []
    for (symbol, variant), group in trades.groupby(["Symbol", "Variant"]):
        row = {"Symbol": symbol, "Variant": variant}
        row.update(metrics_for_returns(group["Net_Return_Pct"]))
        row.update({
            "Average_MFE_Pct": group["MFE_Max_Favorable_Excursion_Pct"].mean(),
            "Median_MFE_Pct": group["MFE_Max_Favorable_Excursion_Pct"].median(),
            "Average_MAE_Pct": group["MAE_Max_Adverse_Excursion_Pct"].mean(),
            "Median_MAE_Pct": group["MAE_Max_Adverse_Excursion_Pct"].median(),
            "Pct_Trades_Reaching_Plus_1Pct": group["Reached_Plus_1Pct_Intratrade"].mean() * 100,
            "Pct_Trades_Touching_Minus_1Pct": group["Reached_Minus_1Pct_Intratrade"].mean() * 100,
        })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "stock_variant_diagnostics.csv", index=False)

    # Year-by-year performance by signal year; this is trade-based, not portfolio return.
    yearly_rows = []
    for (variant, year), group in trades.groupby(["Variant", "Signal_Year"]):
        row = {"Variant": variant, "Signal_Year": int(year), "Stocks_With_Trades": int(group["Symbol"].nunique())}
        row.update(metrics_for_returns(group["Net_Return_Pct"]))
        yearly_rows.append(row)
    pd.DataFrame(yearly_rows).sort_values(["Variant", "Signal_Year"]).to_csv(
        out_dir / "yearly_trade_performance.csv", index=False
    )

    # Simple feature-bucket report to look for recurring patterns, not to choose
    # a rule from the same data without a later independent test.
    features = {
        "RSI": pd.cut(trades["Signal_RSI_14"], [-np.inf, 50, 60, 70, np.inf],
                       labels=["Below 50", "50-60", "60-70", "Above 70"]),
        "Relative_Strength_20D": pd.cut(trades["Signal_Relative_Strength_20D_Pct"],
                                          [-np.inf, 0, 5, 10, np.inf],
                                          labels=["<=0%", "0-5%", "5-10%", ">10%"]),
        "Relative_Volume": pd.cut(trades["Signal_Relative_Volume_20D"],
                                    [-np.inf, 1, 1.5, 2, np.inf],
                                    labels=["Below 1.0x", "1.0-1.5x", "1.5-2.0x", "At least 2.0x"]),
        "Stock_Close_Above_SMA200": trades["Signal_Close_Above_SMA200"].map({True: "Yes", False: "No"}),
        "NIFTY_Above_SMA200": trades["Signal_NIFTY_Above_SMA200"].map({True: "Yes", False: "No"}),
        "Stock_SMA50_Above_SMA200": trades["Signal_SMA50_Above_SMA200"].map({True: "Yes", False: "No"}),
    }
    bucket_rows = []
    for feature_name, bucket_series in features.items():
        temp = trades.copy()
        temp["Feature_Bucket"] = bucket_series.astype("object")
        for (variant, bucket), group in temp.dropna(subset=["Feature_Bucket"]).groupby(["Variant", "Feature_Bucket"], observed=True):
            row = {"Variant": variant, "Feature": feature_name, "Feature_Bucket": str(bucket)}
            row.update(metrics_for_returns(group["Net_Return_Pct"]))
            row["Average_MFE_Pct"] = group["MFE_Max_Favorable_Excursion_Pct"].mean()
            row["Average_MAE_Pct"] = group["MAE_Max_Adverse_Excursion_Pct"].mean()
            bucket_rows.append(row)
    pd.DataFrame(bucket_rows).to_csv(out_dir / "feature_bucket_diagnostics.csv", index=False)

    # Transaction cost sensitivity, holding signals and exits constant.
    cost_rows = []
    for (symbol, variant), group in trades.groupby(["Symbol", "Variant"]):
        for cost in COST_SCENARIOS_PCT:
            returns = group["Gross_Return_Pct"] - cost
            row = {"Symbol": symbol, "Variant": variant, "Assumed_Round_Trip_Cost_Pct": cost}
            row.update(metrics_for_returns(returns))
            cost_rows.append(row)
    pd.DataFrame(cost_rows).to_csv(out_dir / "cost_sensitivity.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "errors.csv", index=False)

    print("\nSaved reports:")
    for name in [
        "trade_path_diagnostics.csv",
        "stock_variant_diagnostics.csv",
        "yearly_trade_performance.csv",
        "feature_bucket_diagnostics.csv",
        "cost_sensitivity.csv",
    ]:
        path = out_dir / name
        if path.exists():
            print(f"  {path}")
    if errors:
        print(f"  {out_dir / 'errors.csv'}")

    print("\nImportant:")
    print("- MAE/MFE are price excursions; they do not simulate stop/target fills.")
    print("- Cost sensitivity keeps the same trade entries and exits fixed.")
    print("- Results are historical research, not predictions or live trading advice.")


if __name__ == "__main__":
    main()
