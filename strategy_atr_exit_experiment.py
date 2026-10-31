""" 
Indian Stock Analyzer — ATR Stop-Loss / Profit-Target Experiment

Save beside main.py and run:
    python strategy_atr_exit_experiment.py

This is a separate experiment. It does not modify main.py.

Design:
- Reuses the actual data-download, indicator, relative-strength and scoring
  functions from main.py.
- Tests five stocks: TCS, Infosys, Reliance, HDFC Bank and ICICI Bank.
- Compares the original entry rules with the same entries plus a NIFTY 50
  close-above-SMA200 filter.
- Tests four exit rules: fixed 5-session exit, 1.5 ATR stop, 2 ATR target,
  and 1.5 ATR stop + 2 ATR target. Any ATR is taken from the signal-day close,
  so the level is known before the next-session-open entry.
- Uses the main.py fixed round-trip cost assumption (0.20 percentage points).
- Splits development (configured start date through 2024-12-31) and a later
  evaluation period (2025-01-01 onward). The latter has already been inspected
  in earlier experiments, so it is not a pristine untouched holdout anymore.

Execution assumptions:
- Enter at next session open.
- Stop/target are checked using daily OHLC. If both stop and target are hit
  on the same daily bar, assume the stop hit first (conservative).
- A stop gap through its level fills at that day's open. Otherwise fill at stop.
- A profit target fills at the target price; no price improvement is assumed.
- If no stop/target is hit, exit at the close of the 5th holding session.
- A strategy can hold at most one position at a time.

This is exploratory research, not investment advice or a claim of profitability.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

try:
    import main as analyzer
except Exception as exc:
    raise SystemExit(
        "Could not import main.py. Place this script beside main.py and ensure "
        "its dependencies are installed.\n"
        f"Import error: {exc}"
    )


SYMBOLS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
MAX_HOLDING_SESSIONS = int(analyzer.BACKTEST_HOLDING_SESSIONS)
COST_PCT = float(analyzer.BACKTEST_ROUND_TRIP_COST_PCT)

DEVELOPMENT_END = pd.Timestamp("2024-12-31")
OUT_OF_SAMPLE_START = pd.Timestamp("2025-01-01")

ENTRY_FILTERS = [
    {
        "Entry_Filter": "BASELINE_ENTRIES",
        "Require_NIFTY_Above_SMA200": False,
        "Description": "Original main.py entry rules",
    },
    {
        "Entry_Filter": "NIFTY_FILTERED_ENTRIES",
        "Require_NIFTY_Above_SMA200": True,
        "Description": "Original entry rules plus NIFTY 50 close > SMA200 on signal date",
    },
]

EXIT_RULES = [
    {
        "Exit_Rule": "FIXED_5_SESSION",
        "Stop_ATR_Multiplier": None,
        "Target_ATR_Multiplier": None,
        "Description": "Original time-based exit at close of session 5",
    },
    {
        "Exit_Rule": "ATR_STOP_1_5",
        "Stop_ATR_Multiplier": 1.5,
        "Target_ATR_Multiplier": None,
        "Description": "Exit at 1.5 x signal-day ATR below entry, or time exit",
    },
    {
        "Exit_Rule": "ATR_TARGET_2_0",
        "Stop_ATR_Multiplier": None,
        "Target_ATR_Multiplier": 2.0,
        "Description": "Exit at 2.0 x signal-day ATR above entry, or time exit",
    },
    {
        "Exit_Rule": "ATR_STOP_1_5_TARGET_2_0",
        "Stop_ATR_Multiplier": 1.5,
        "Target_ATR_Multiplier": 2.0,
        "Description": "1.5 x ATR stop plus 2.0 x ATR target, max 5 sessions",
    },
]

PERIODS = [
    {
        "Period": "DEVELOPMENT",
        "Start": pd.Timestamp(analyzer.BACKTEST_TEST_START_DATE),
        "End": DEVELOPMENT_END,
    },
    {
        "Period": "OUT_OF_SAMPLE_EVALUATION",
        "Start": OUT_OF_SAMPLE_START,
        "End": None,  # latest common available date for each stock
    },
]


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with timezone-naive normalized dates."""
    result = frame.copy()
    index = pd.DatetimeIndex(result.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result = result[~result.index.duplicated(keep="last")]
    result.index.name = "Date"
    return result.sort_index()


def safe_symbol(symbol: str) -> str:
    return symbol.replace(".", "_").replace("^", "").replace("/", "_").replace("\\", "_")


def build_nifty_features(nifty_prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate NIFTY SMA200 using the same indicator implementation as main.py."""
    nifty_prices = normalize_dates(nifty_prices)
    nifty = analyzer.calculate_indicators(nifty_prices)
    return normalize_dates(nifty)


def get_period_positions(data: pd.DataFrame, period: dict) -> tuple[int, int, int]:
    """Return period-start, signal-start and period-end positional indices."""
    start_stamp = pd.Timestamp(period["Start"]).normalize()
    if period["End"] is None:
        end_pos = len(data) - 1
    else:
        end_stamp = pd.Timestamp(period["End"]).normalize()
        end_pos = int(data.index.searchsorted(end_stamp, side="right")) - 1
        end_pos = min(end_pos, len(data) - 1)

    period_start_pos = int(data.index.searchsorted(start_stamp, side="left"))
    if period_start_pos > end_pos or period_start_pos >= len(data):
        raise ValueError(f"No stock sessions found in period beginning {start_stamp.date()}.")

    ready_positions = np.flatnonzero(data["SMA_200"].notna().to_numpy())
    if len(ready_positions) == 0:
        raise ValueError("SMA_200 is unavailable; insufficient historical data.")

    signal_start_pos = max(period_start_pos, int(ready_positions[0]))
    if signal_start_pos > end_pos:
        raise ValueError(f"SMA_200 is not ready within period {period['Period']}.")

    return period_start_pos, signal_start_pos, end_pos


def get_nifty_window_return(nifty: pd.DataFrame, start_date: pd.Timestamp,
                            end_date: pd.Timestamp) -> float:
    """Close-to-close NIFTY return over the stock period's available aligned dates."""
    closes = nifty["Close"].reindex([start_date, end_date]).dropna()
    if len(closes) != 2 or closes.iloc[0] <= 0:
        return np.nan
    return float((closes.iloc[1] / closes.iloc[0] - 1.0) * 100.0 - COST_PCT)


def trade_path_statistics(data: pd.DataFrame, entry_pos: int, exit_pos: int,
                          entry_price: float) -> tuple[float, float]:
    """Daily-bar estimates of maximum adverse/favorable excursion, in percent."""
    path = data.iloc[entry_pos:exit_pos + 1]
    if path.empty or entry_price <= 0:
        return np.nan, np.nan
    mae = (float(path["Low"].min()) / entry_price - 1.0) * 100.0
    mfe = (float(path["High"].max()) / entry_price - 1.0) * 100.0
    return mae, mfe


def simulate_variant(
    symbol: str,
    data: pd.DataFrame,
    nifty: pd.DataFrame,
    period: dict,
    period_start_pos: int,
    signal_start_pos: int,
    period_end_pos: int,
    entry_filter: dict,
    exit_rule: dict,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run one stock/period/entry-filter/exit-rule configuration."""
    max_hold = MAX_HOLDING_SESSIONS
    stop_mult = exit_rule["Stop_ATR_Multiplier"]
    target_mult = exit_rule["Target_ATR_Multiplier"]
    requires_rs = bool(analyzer.BACKTEST_REQUIRE_POSITIVE_RELATIVE_STRENGTH)
    minimum_score = int(analyzer.BACKTEST_MIN_SETUP_SCORE)

    # Require enough sessions for the full maximum holding period even if
    # an exit rule later closes the position sooner. This keeps variants comparable.
    last_signal_pos = period_end_pos - max_hold
    if last_signal_pos < signal_start_pos:
        raise ValueError(f"Not enough sessions in {period['Period']} for a {max_hold}-session test.")

    nifty_close = nifty["Close"].reindex(data.index)
    nifty_sma200 = nifty["SMA_200"].reindex(data.index)
    nifty_above_sma200 = (nifty_close > nifty_sma200) & nifty_close.notna() & nifty_sma200.notna()

    trades = []
    last_exit_pos = -1

    for signal_pos in range(signal_start_pos, last_signal_pos + 1):
        if signal_pos <= last_exit_pos:
            continue

        signal = data.iloc[signal_pos]
        score = int(analyzer.calculate_setup_score(signal))
        rel_strength = signal.get("Relative_Strength_20D_Pct", np.nan)
        breakdown = bool(signal.get("Breakdown_Confirmed", False))

        if score < minimum_score or breakdown:
            continue
        if requires_rs and (pd.isna(rel_strength) or rel_strength <= 0):
            continue
        if entry_filter["Require_NIFTY_Above_SMA200"] and not bool(nifty_above_sma200.iloc[signal_pos]):
            continue

        atr_signal = pd.to_numeric(pd.Series([signal.get("ATR_14", np.nan)]), errors="coerce").iloc[0]
        if (stop_mult is not None or target_mult is not None) and (pd.isna(atr_signal) or atr_signal <= 0):
            continue

        entry_pos = signal_pos + 1
        final_hold_pos = entry_pos + max_hold - 1
        if final_hold_pos > period_end_pos:
            continue

        entry_row = data.iloc[entry_pos]
        entry_price = float(entry_row["Open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        # ATR is sampled from the signal-day close; no future ATR is used to set levels.
        stop_price = (entry_price - float(stop_mult) * float(atr_signal)) if stop_mult is not None else np.nan
        target_price = (entry_price + float(target_mult) * float(atr_signal)) if target_mult is not None else np.nan

        exit_pos = final_hold_pos
        exit_price = float(data.iloc[final_hold_pos]["Close"])
        exit_reason = "TIME_EXIT"

        for position in range(entry_pos, final_hold_pos + 1):
            bar = data.iloc[position]
            open_price = float(bar["Open"])
            low_price = float(bar["Low"])
            high_price = float(bar["High"])

            stop_hit = stop_mult is not None and low_price <= stop_price
            target_hit = target_mult is not None and high_price >= target_price

            # Conservative handling when both levels are touched on one daily bar:
            # assume the stop executes first, because daily OHLC cannot reveal order.
            if stop_hit:
                exit_pos = position
                if open_price <= stop_price:
                    exit_price = open_price  # adverse gap through stop
                    exit_reason = "STOP_GAP_EXIT"
                else:
                    exit_price = float(stop_price)
                    exit_reason = "STOP_LOSS"
                break

            if target_hit:
                exit_pos = position
                exit_price = float(target_price)  # no opening-price improvement assumed
                exit_reason = "PROFIT_TARGET"
                break

            if position == final_hold_pos:
                exit_pos = position
                exit_price = float(bar["Close"])
                exit_reason = "TIME_EXIT"

        if not np.isfinite(exit_price) or exit_price <= 0:
            continue

        gross_return = (exit_price / entry_price - 1.0) * 100.0
        net_return = gross_return - COST_PCT
        mae_pct, mfe_pct = trade_path_statistics(data, entry_pos, exit_pos, entry_price)

        trades.append({
            "Symbol": symbol,
            "Period": period["Period"],
            "Entry_Filter": entry_filter["Entry_Filter"],
            "Exit_Rule": exit_rule["Exit_Rule"],
            "Signal_Date": data.index[signal_pos].date().isoformat(),
            "Entry_Date": data.index[entry_pos].date().isoformat(),
            "Exit_Date": data.index[exit_pos].date().isoformat(),
            "Entry_Price": entry_price,
            "Exit_Price": exit_price,
            "Exit_Reason": exit_reason,
            "Holding_Sessions": int(exit_pos - entry_pos + 1),
            "Maximum_Holding_Sessions": max_hold,
            "Signal_Setup_Score": score,
            "Signal_Relative_Strength_20D_Pct": rel_strength,
            "Signal_Relative_Volume_20D": signal.get("Relative_Volume_20D", np.nan),
            "Signal_RSI_14": signal.get("RSI_14", np.nan),
            "Signal_ATR_14": atr_signal,
            "Stop_ATR_Multiplier": stop_mult,
            "Target_ATR_Multiplier": target_mult,
            "Stop_Price": stop_price,
            "Target_Price": target_price,
            "Signal_NIFTY_Close": nifty_close.iloc[signal_pos],
            "Signal_NIFTY_SMA200": nifty_sma200.iloc[signal_pos],
            "NIFTY_Above_SMA200_On_Signal": bool(nifty_above_sma200.iloc[signal_pos]),
            "Gross_Return_Pct": gross_return,
            "Assumed_Round_Trip_Cost_Pct": COST_PCT,
            "Net_Return_Pct": net_return,
            "MAE_Pct_Daily_Bar_Estimate": mae_pct,
            "MFE_Pct_Daily_Bar_Estimate": mfe_pct,
            "Outcome": "Win" if net_return > 0 else "Loss/Flat",
        })
        last_exit_pos = exit_pos

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=[
            "Symbol", "Period", "Entry_Filter", "Exit_Rule", "Signal_Date",
            "Entry_Date", "Exit_Date", "Entry_Price", "Exit_Price", "Exit_Reason",
            "Holding_Sessions", "Gross_Return_Pct", "Net_Return_Pct",
            "Assumed_Round_Trip_Cost_Pct",
        ])

    # Build a period-level daily equity curve. Mark positions to daily close,
    # but on exit day mark at the actual simulated exit price (stop/target/close).
    period_dates = data.index[period_start_pos:period_end_pos + 1]
    equity_curve = pd.DataFrame(index=period_dates)
    equity_curve.index.name = "Date"
    equity_curve["Symbol"] = symbol
    equity_curve["Period"] = period["Period"]
    equity_curve["Entry_Filter"] = entry_filter["Entry_Filter"]
    equity_curve["Exit_Rule"] = exit_rule["Exit_Rule"]
    equity_curve["Close"] = data.loc[period_dates, "Close"].astype(float)
    equity_curve["Strategy_Equity"] = np.nan
    equity_curve["In_Position"] = False

    current_equity = 1.0
    cursor_pos = period_start_pos
    trade_close_equities = [1.0]

    for _, trade in trades_df.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"]).normalize()
        exit_date = pd.Timestamp(trade["Exit_Date"]).normalize()
        entry_pos = int(data.index.get_loc(entry_date))
        exit_pos = int(data.index.get_loc(exit_date))

        if entry_pos < cursor_pos:
            raise ValueError("Trade overlap detected while building the equity curve.")

        if cursor_pos < entry_pos:
            cash_dates = data.index[cursor_pos:entry_pos]
            equity_curve.loc[cash_dates, "Strategy_Equity"] = current_equity

        for position in range(entry_pos, exit_pos + 1):
            date = data.index[position]
            if position == exit_pos:
                mark_price = float(trade["Exit_Price"])
            else:
                mark_price = float(data.iloc[position]["Close"])
            marked_equity = current_equity * (
                mark_price / float(trade["Entry_Price"]) - COST_PCT / 100.0
            )
            equity_curve.at[date, "Strategy_Equity"] = marked_equity
            equity_curve.at[date, "In_Position"] = True

        net_fraction = float(trade["Net_Return_Pct"]) / 100.0
        current_equity *= 1.0 + net_fraction
        equity_curve.at[exit_date, "Strategy_Equity"] = current_equity
        trade_close_equities.append(current_equity)
        cursor_pos = exit_pos + 1

    if cursor_pos <= period_end_pos:
        cash_dates = data.index[cursor_pos:period_end_pos + 1]
        equity_curve.loc[cash_dates, "Strategy_Equity"] = current_equity

    equity_curve["Strategy_Equity"] = equity_curve["Strategy_Equity"].ffill().fillna(1.0)
    equity_curve["Equity_Peak"] = equity_curve["Strategy_Equity"].cummax()
    equity_curve["Daily_Drawdown_Pct"] = (
        equity_curve["Strategy_Equity"] / equity_curve["Equity_Peak"] - 1.0
    ) * 100.0
    daily_mdd = float(equity_curve["Daily_Drawdown_Pct"].min()) if not equity_curve.empty else 0.0

    trade_close_series = pd.Series(trade_close_equities, dtype=float)
    trade_close_mdd = float(((trade_close_series / trade_close_series.cummax()) - 1.0).min() * 100.0)

    net = pd.to_numeric(trades_df["Net_Return_Pct"], errors="coerce").dropna()
    gross = pd.to_numeric(trades_df["Gross_Return_Pct"], errors="coerce").dropna()
    winners = net[net > 0]
    losers = net[net < 0]
    if len(losers) and abs(float(losers.sum())) > 0:
        profit_factor = float(winners.sum() / abs(losers.sum()))
    else:
        profit_factor = np.nan

    total_position_sessions = int(equity_curve["In_Position"].sum())
    period_session_count = int(len(equity_curve))
    exposure_pct = (total_position_sessions / period_session_count * 100.0) if period_session_count else 0.0

    stock_start_close = float(data.iloc[period_start_pos]["Close"])
    stock_end_close = float(data.iloc[period_end_pos]["Close"])
    stock_bh_gross = (stock_end_close / stock_start_close - 1.0) * 100.0 if stock_start_close > 0 else np.nan
    stock_bh_net = stock_bh_gross - COST_PCT if pd.notna(stock_bh_gross) else np.nan

    period_start_date = data.index[period_start_pos]
    period_end_date = data.index[period_end_pos]
    nifty_bh_net = get_nifty_window_return(nifty, period_start_date, period_end_date)

    exit_counts = trades_df["Exit_Reason"].value_counts() if not trades_df.empty else pd.Series(dtype=int)
    summary = {
        "Symbol": symbol,
        "Period": period["Period"],
        "Period_Start_Date": period_start_date.date().isoformat(),
        "Period_End_Date": period_end_date.date().isoformat(),
        "Entry_Filter": entry_filter["Entry_Filter"],
        "Exit_Rule": exit_rule["Exit_Rule"],
        "Entry_Rule_Description": entry_filter["Description"],
        "Exit_Rule_Description": exit_rule["Description"],
        "Maximum_Holding_Sessions": max_hold,
        "Minimum_Setup_Score": minimum_score,
        "Require_Positive_Relative_Strength": requires_rs,
        "Require_NIFTY_Above_SMA200": entry_filter["Require_NIFTY_Above_SMA200"],
        "Assumed_Round_Trip_Cost_Pct": COST_PCT,
        "Number_Of_Trades": int(len(net)),
        "Win_Rate_Pct": float((net > 0).mean() * 100.0) if len(net) else np.nan,
        "Average_Gross_Trade_Return_Pct": float(gross.mean()) if len(gross) else np.nan,
        "Average_Net_Trade_Return_Pct": float(net.mean()) if len(net) else np.nan,
        "Median_Net_Trade_Return_Pct": float(net.median()) if len(net) else np.nan,
        "Profit_Factor": profit_factor,
        "Compounded_Strategy_Return_Pct": float((current_equity - 1.0) * 100.0),
        "Final_Strategy_Equity": float(current_equity),
        "Maximum_Trade_Close_Drawdown_Pct": trade_close_mdd,
        "Maximum_Daily_Close_MTM_Drawdown_Pct": daily_mdd,
        "Percent_Of_Sessions_In_Position": exposure_pct,
        "Average_MAE_Pct_Daily_Bar_Estimate": float(trades_df["MAE_Pct_Daily_Bar_Estimate"].mean()) if len(net) else np.nan,
        "Average_MFE_Pct_Daily_Bar_Estimate": float(trades_df["MFE_Pct_Daily_Bar_Estimate"].mean()) if len(net) else np.nan,
        "Stop_Loss_Exits": int(exit_counts.get("STOP_LOSS", 0)),
        "Stop_Gap_Exits": int(exit_counts.get("STOP_GAP_EXIT", 0)),
        "Profit_Target_Exits": int(exit_counts.get("PROFIT_TARGET", 0)),
        "Time_Exits": int(exit_counts.get("TIME_EXIT", 0)),
        "Stock_Buy_And_Hold_Gross_Return_Pct_Fixed_Window": stock_bh_gross,
        "Stock_Buy_And_Hold_Net_Return_Pct_Fixed_Window": stock_bh_net,
        "NIFTY50_Buy_And_Hold_Net_Return_Pct_Fixed_Window": nifty_bh_net,
    }

    return trades_df, summary, equity_curve.reset_index()


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "data" / "processed" / "atr_exit_experiment"
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("INDIAN STOCK ANALYZER — ATR STOP-LOSS / PROFIT-TARGET EXPERIMENT")
    print("=" * 76)
    print("This script does not modify main.py or previous experiment scripts.")
    print(f"Configured start date: {analyzer.BACKTEST_TEST_START_DATE}")
    print("Development period: configured start date through 2024-12-31")
    print("Evaluation period: 2025-01-01 through latest available data")
    print(f"Maximum holding period: {MAX_HOLDING_SESSIONS} sessions")
    print(f"Assumed round-trip cost: {COST_PCT:.2f} percentage points")
    print(f"Output folder: {output_dir.resolve()}\n")

    print("Downloading NIFTY 50 data...")
    nifty_prices = analyzer.download_nifty50_data()
    if nifty_prices is None or nifty_prices.empty:
        raise SystemExit("Could not download NIFTY 50 data. Check the connection and try again.")
    nifty = build_nifty_features(nifty_prices)

    all_summaries = []
    all_trades = []
    all_curves = []
    errors = []

    for symbol in SYMBOLS:
        print("\n" + "-" * 76)
        print(f"PROCESSING {symbol}")
        print("-" * 76)
        try:
            prices = analyzer.download_stock_data(symbol)
            if prices is None or prices.empty:
                raise ValueError("No historical stock data returned.")
            prices = normalize_dates(prices)
            data = analyzer.calculate_indicators(prices)
            data = analyzer.add_relative_strength(data, nifty)
            data = normalize_dates(data)

            # Reindex NIFTY features to stock dates for signal-date-only checks.
            nifty_aligned = nifty.reindex(data.index)
            nifty_aligned["Close"] = nifty["Close"].reindex(data.index)
            nifty_aligned["SMA_200"] = nifty["SMA_200"].reindex(data.index)

            for period in PERIODS:
                try:
                    period_start_pos, signal_start_pos, period_end_pos = get_period_positions(data, period)
                except ValueError as period_error:
                    errors.append({"Symbol": symbol, "Period": period["Period"], "Error": str(period_error)})
                    print(f"  Skipping {period['Period']}: {period_error}")
                    continue

                actual_period = dict(period)
                actual_period["End"] = data.index[period_end_pos]
                print(
                    f"  {actual_period['Period']}: "
                    f"{data.index[period_start_pos].date()} to "
                    f"{data.index[period_end_pos].date()}"
                )

                for entry_filter in ENTRY_FILTERS:
                    for exit_rule in EXIT_RULES:
                        try:
                            trades_df, summary, curve_df = simulate_variant(
                                symbol=symbol,
                                data=data,
                                nifty=nifty_aligned,
                                period=actual_period,
                                period_start_pos=period_start_pos,
                                signal_start_pos=signal_start_pos,
                                period_end_pos=period_end_pos,
                                entry_filter=entry_filter,
                                exit_rule=exit_rule,
                            )
                            all_summaries.append(summary)
                            if not trades_df.empty:
                                all_trades.append(trades_df)
                            all_curves.append(curve_df)

                            avg_net = summary["Average_Net_Trade_Return_Pct"]
                            pf = summary["Profit_Factor"]
                            ret = summary["Compounded_Strategy_Return_Pct"]
                            mdd = summary["Maximum_Daily_Close_MTM_Drawdown_Pct"]
                            print(
                                f"    {entry_filter['Entry_Filter']:<24} "
                                f"{exit_rule['Exit_Rule']:<27} "
                                f"trades={summary['Number_Of_Trades']:>3} | "
                                f"avg net={avg_net:>7.3f}% | "
                                f"PF={pf:>5.2f} | "
                                f"strategy={ret:>8.2f}% | MDD={mdd:>7.2f}%"
                            )
                        except Exception as variant_error:
                            errors.append({
                                "Symbol": symbol,
                                "Period": actual_period["Period"],
                                "Entry_Filter": entry_filter["Entry_Filter"],
                                "Exit_Rule": exit_rule["Exit_Rule"],
                                "Error": str(variant_error),
                            })
                            print(
                                f"    ERROR {entry_filter['Entry_Filter']} / "
                                f"{exit_rule['Exit_Rule']}: {variant_error}"
                            )
                            traceback.print_exc()

        except Exception as stock_error:
            errors.append({"Symbol": symbol, "Error": str(stock_error)})
            print(f"ERROR processing {symbol}: {stock_error}")
            traceback.print_exc()

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = output_dir / "atr_exit_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        aggregate_df = (
            summary_df.groupby(["Period", "Entry_Filter", "Exit_Rule"], dropna=False)
            .agg(
                Stock_Count=("Symbol", "nunique"),
                Total_Trades=("Number_Of_Trades", "sum"),
                Average_Stock_Return_Pct=("Compounded_Strategy_Return_Pct", "mean"),
                Median_Stock_Return_Pct=("Compounded_Strategy_Return_Pct", "median"),
                Average_Stock_Max_Daily_Drawdown_Pct=("Maximum_Daily_Close_MTM_Drawdown_Pct", "mean"),
                Average_Profit_Factor=("Profit_Factor", "mean"),
                Average_Trade_Net_Return_Pct=("Average_Net_Trade_Return_Pct", "mean"),
                Average_Exposure_Pct=("Percent_Of_Sessions_In_Position", "mean"),
            )
            .reset_index()
        )
        aggregate_path = output_dir / "atr_exit_aggregate_summary.csv"
        aggregate_df.to_csv(aggregate_path, index=False)

        print("\n" + "=" * 120)
        print("AGGREGATE SUMMARY — SIMPLE AVERAGES OF SEPARATE STOCK TESTS")
        print("=" * 120)
        print(aggregate_df.round(3).to_string(index=False))
        print(f"\nDetailed summary: {summary_path.resolve()}")
        print(f"Aggregate summary: {aggregate_path.resolve()}")

    if all_trades:
        trades_path = output_dir / "atr_exit_trades.csv"
        pd.concat(all_trades, ignore_index=True).to_csv(trades_path, index=False)
        print(f"Trade-level report: {trades_path.resolve()}")
    else:
        print("\nNo trades were generated for any configuration.")

    if all_curves:
        equity_path = output_dir / "atr_exit_equity_curves.csv"
        pd.concat(all_curves, ignore_index=True).to_csv(equity_path, index=False)
        print(f"Equity curves: {equity_path.resolve()}")

    if errors:
        errors_path = output_dir / "atr_exit_errors.csv"
        pd.DataFrame(errors).to_csv(errors_path, index=False)
        print(f"Error report: {errors_path.resolve()}")

    print("\n" + "=" * 76)
    print("ATR EXIT EXPERIMENT FINISHED")
    print("=" * 76)
    print(f"Results folder: {output_dir.resolve()}")
    print("Please review out-of-sample results first, but remember earlier experiments")
    print("have already examined 2025 onward, so that period is no longer untouched.")
    print("Daily OHLC cannot reveal exact intraday order; same-day stop+target hits")
    print("are handled conservatively by assuming the stop happens first.")
    print("The 0.20% round-trip cost is only an assumption and omits many real costs.")
    print("This experiment does not prove future profitability.")


if __name__ == "__main__":
    main()
