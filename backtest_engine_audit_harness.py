"""Indian Stock Analyzer — isolated backtest-engine audit harness.

Purpose
-------
Run deterministic regression tests for the highest-priority backtesting
methodology issues without importing main.py, downloading market data, or
changing any existing project file.

This is a test harness and reusable reference implementation, NOT a replacement
for main.py and NOT proof that the strategy is profitable. It must pass its
synthetic tests before any of these ideas are integrated into the project.

Run from the project folder:
    python backtest_engine_audit_harness.py

Dependencies: pandas and numpy (already used by this project), plus Python's
standard library. No internet connection is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

OHLC = ("Open", "High", "Low", "Close")


@dataclass(frozen=True)
class ExitResult:
    exit_pos: int
    exit_date: str
    exit_price: float
    exit_reason: str
    stop_price: Optional[float]
    target_price: Optional[float]


def normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted, date-normalized copy and reject duplicate dates.

    Quietly deduplicating source bars can hide a data problem, so this audit
    harness rejects duplicate daily dates instead of choosing one implicitly.
    """
    if frame is None or frame.empty:
        raise ValueError("Price data is empty.")
    out = frame.copy()
    try:
        idx = pd.DatetimeIndex(out.index)
    except Exception as exc:  # pragma: no cover - defensive error path
        raise ValueError("Price data must have a datetime-like index.") from exc
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = out.index.name or "Date"
    if out.index.has_duplicates:
        duplicate_dates = out.index[out.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Duplicate daily dates found: {duplicate_dates[:5]}")
    return out.sort_index()


def validate_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate positive and internally consistent daily OHLC bars."""
    out = normalize_daily_frame(frame)
    missing = [column for column in OHLC if column not in out.columns]
    if missing:
        raise ValueError(f"Required OHLC columns are missing: {missing}")

    for column in OHLC:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    invalid = out[list(OHLC)].isna().any(axis=1)
    invalid |= (out[list(OHLC)] <= 0).any(axis=1)
    invalid |= out["High"] < out["Low"]
    invalid |= out["High"] < out[["Open", "Close"]].max(axis=1)
    invalid |= out["Low"] > out[["Open", "Close"]].min(axis=1)
    if invalid.any():
        bad_dates = out.index[invalid].strftime("%Y-%m-%d").tolist()
        raise ValueError(
            "Invalid or incomplete OHLC bar(s); refusing to guess whether an "
            f"exit was triggered. Dates: {bad_dates[:5]}"
        )
    return out


def yahoo_chart_payload_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Parse a Yahoo chart JSON payload without applying Adj Close / Close to OHLC.

    The provider quote fields are copied as received. Adjusted close and basic
    corporate-action fields are retained separately. The split-adjustment basis
    of the provider OHLC must still be confirmed; this parser does not apply a
    split transformation itself.
    """
    chart = (payload or {}).get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo chart payload contains no result rows.")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [])
    if not timestamps or not quote_data:
        raise ValueError("Yahoo chart payload is missing timestamps or OHLC quote data.")

    row_count = len(timestamps)

    def fit(values: Optional[Iterable[Any]]) -> list[Any]:
        values = list(values or [])
        return (values + [None] * max(0, row_count - len(values)))[:row_count]

    timezone_name = result.get("meta", {}).get("exchangeTimezoneName")
    date_index = pd.to_datetime(timestamps, unit="s", utc=True)
    if timezone_name:
        try:
            date_index = date_index.tz_convert(timezone_name)
        except (TypeError, ValueError):
            pass
    # Strip timezone only after conversion to the exchange timezone, preserving
    # the exchange's actual local calendar date.
    date_index = date_index.tz_localize(None).normalize()

    quote_row = quote_data[0]
    frame = pd.DataFrame(
        {column: fit(quote_row.get(column.lower())) for column in OHLC + ("Volume",)},
        index=date_index,
    )
    frame.index.name = "Date"

    adjusted_rows = result.get("indicators", {}).get("adjclose") or []
    if adjusted_rows:
        frame["Adjusted_Close"] = fit(adjusted_rows[0].get("adjclose"))
    else:
        frame["Adjusted_Close"] = np.nan

    frame["Dividend_Per_Share"] = 0.0
    frame["Stock_Split_Ratio"] = np.nan
    frame["Stock_Split_Ratio_Text"] = ""
    events = result.get("events") or {}

    def local_event_date(timestamp: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(int(timestamp), unit="s", tz="UTC")
        if timezone_name:
            try:
                stamp = stamp.tz_convert(timezone_name)
            except (TypeError, ValueError):
                pass
        return stamp.tz_localize(None).normalize()

    for event_timestamp, event in (events.get("dividends") or {}).items():
        event_date = local_event_date(event_timestamp)
        if event_date in frame.index:
            frame.at[event_date, "Dividend_Per_Share"] = pd.to_numeric(
                event.get("amount"), errors="coerce"
            )

    for event_timestamp, event in (events.get("splits") or {}).items():
        event_date = local_event_date(event_timestamp)
        if event_date not in frame.index:
            continue
        numerator = pd.to_numeric(event.get("numerator"), errors="coerce")
        denominator = pd.to_numeric(event.get("denominator"), errors="coerce")
        if pd.notna(numerator) and pd.notna(denominator) and denominator != 0:
            frame.at[event_date, "Stock_Split_Ratio"] = float(numerator / denominator)
        frame.at[event_date, "Stock_Split_Ratio_Text"] = str(event.get("splitRatio", ""))

    for column in OHLC + ("Volume", "Adjusted_Close", "Dividend_Per_Share", "Stock_Split_Ratio"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = validate_ohlc(frame)
    frame["Price_Basis_Note"] = (
        "Yahoo quote OHLC copied without local dividend adjustment; "
        "Adjusted_Close and corporate actions stored separately"
    )
    return frame


def prepare_execution_price_frame(
    quote_bars: pd.DataFrame,
    adjusted_close: Optional[pd.Series] = None,
    dividends: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Keep quote OHLC separate from dividend-adjusted close and dividends.

    IMPORTANT: `quote_bars` must be taken from the provider quote fields BEFORE
    the existing main.py adjustment which multiplies OHLC by Adj Close / Close.
    That adjustment is irreversible once the original OHLC has been overwritten.

    This function deliberately does NOT multiply Open/High/Low/Close by an
    adjustment factor. `Adjusted_Close` is stored separately for total-return
    research/benchmarking; it is not used to trigger stop/target levels.

    Split-adjustment conventions of the upstream provider still need to be
    confirmed against corporate-action records before treating the series as
    executable prices. This helper does not itself rewrite split history.
    """
    out = validate_ohlc(quote_bars)

    if adjusted_close is None:
        for candidate in ("Adjusted_Close", "Adj_Close", "Adj Close"):
            if candidate in out.columns:
                adjusted_close = out[candidate].copy()
                break

    if adjusted_close is not None:
        adjusted = pd.to_numeric(adjusted_close.reindex(out.index), errors="coerce")
        out["Adjusted_Close"] = adjusted
    elif "Adjusted_Close" not in out.columns:
        out["Adjusted_Close"] = np.nan

    if dividends is not None:
        out["Dividend_Per_Share"] = pd.to_numeric(
            dividends.reindex(out.index), errors="coerce"
        ).fillna(0.0)
    elif "Dividend_Per_Share" not in out.columns:
        out["Dividend_Per_Share"] = 0.0

    out["Price_Basis_Note"] = (
        "Provider quote OHLC preserved without local dividend adjustment; "
        "Adjusted_Close kept separately"
    )
    return out


def should_skip_signal(signal_pos: int, last_exit_pos: int) -> bool:
    """Whether a signal should be skipped because the prior trade is still open.

    A signal is evaluated after the close. If the previous position exited on
    that same date, the new signal can enter at the following session's Open.
    Thus only signals strictly BEFORE the prior exit date are skipped.
    """
    return int(signal_pos) < int(last_exit_pos)


def _valid_bar(bar: pd.Series, date_label: str) -> None:
    values = {column: pd.to_numeric(bar.get(column), errors="coerce") for column in OHLC}
    if any(pd.isna(value) or value <= 0 for value in values.values()):
        raise ValueError(f"Invalid OHLC bar on {date_label}; cannot simulate execution safely.")
    if values["High"] < max(values["Open"], values["Close"], values["Low"]):
        raise ValueError(f"Inconsistent High on {date_label}.")
    if values["Low"] > min(values["Open"], values["Close"], values["High"]):
        raise ValueError(f"Inconsistent Low on {date_label}.")


def simulate_long_exit(
    bars: pd.DataFrame,
    entry_pos: int,
    time_exit_pos: int,
    entry_price: float,
    signal_atr: Optional[float],
    stop_atr_multiplier: Optional[float] = None,
    target_atr_multiplier: Optional[float] = None,
) -> ExitResult:
    """Simulate a long trade exit from daily OHLC using conservative rules.

    Rules:
    - ATR is fixed from the signal-date close, before the entry session.
    - A stop gap through the stop level exits at the day's Open (worse price).
    - A target gap exits at the target level, not at the more favourable Open.
    - If both stop and target trade through on the same candle, assume stop first.
    - Missing/inconsistent OHLC in the holding window raises instead of silently
      assuming that no stop or target was hit.
    - If no level is reached, exit at the scheduled session's Close.
    """
    frame = normalize_daily_frame(bars)
    missing = [column for column in OHLC if column not in frame.columns]
    if missing:
        raise ValueError(f"Required OHLC columns are missing: {missing}")
    if not (0 <= entry_pos <= time_exit_pos < len(frame)):
        raise IndexError("Entry/exit positions are outside the supplied price data.")
    if not math.isfinite(float(entry_price)) or float(entry_price) <= 0:
        raise ValueError("Entry price must be finite and positive.")

    uses_atr = stop_atr_multiplier is not None or target_atr_multiplier is not None
    if uses_atr:
        if signal_atr is None or not math.isfinite(float(signal_atr)) or float(signal_atr) <= 0:
            raise ValueError("A positive signal-day ATR is required for ATR exits.")

    stop_price = (
        float(entry_price) - float(stop_atr_multiplier) * float(signal_atr)
        if stop_atr_multiplier is not None else None
    )
    target_price = (
        float(entry_price) + float(target_atr_multiplier) * float(signal_atr)
        if target_atr_multiplier is not None else None
    )
    if stop_price is not None and stop_price <= 0:
        raise ValueError("Calculated stop price is not positive; check entry/ATR inputs.")

    for pos in range(entry_pos, time_exit_pos + 1):
        bar = frame.iloc[pos]
        date_label = pd.Timestamp(frame.index[pos]).date().isoformat()
        _valid_bar(bar, date_label)
        open_price = float(bar["Open"])
        high_price = float(bar["High"])
        low_price = float(bar["Low"])

        # Opening gaps are assessed before the intraday high/low range.
        if stop_price is not None and open_price <= stop_price:
            return ExitResult(pos, date_label, open_price, "STOP_GAP_AT_OPEN", stop_price, target_price)
        if target_price is not None and open_price >= target_price:
            # No favourable price improvement is assumed for the target.
            return ExitResult(pos, date_label, target_price, "TARGET_GAP_AT_OPEN_LIMIT", stop_price, target_price)

        stop_hit = stop_price is not None and low_price <= stop_price
        target_hit = target_price is not None and high_price >= target_price
        if stop_hit and target_hit:
            return ExitResult(pos, date_label, float(stop_price), "BOTH_TOUCHED_STOP_FIRST", stop_price, target_price)
        if stop_hit:
            return ExitResult(pos, date_label, float(stop_price), "ATR_STOP", stop_price, target_price)
        if target_hit:
            return ExitResult(pos, date_label, float(target_price), "ATR_TARGET", stop_price, target_price)

    close_price = float(frame.iloc[time_exit_pos]["Close"])
    date_label = pd.Timestamp(frame.index[time_exit_pos]).date().isoformat()
    return ExitResult(time_exit_pos, date_label, close_price, "TIME_EXIT", stop_price, target_price)


def calculate_trade_return_pct(
    entry_price: float, exit_price: float, round_trip_cost_pct: float
) -> tuple[float, float]:
    """Return (gross %, net %) where cost is deducted in percentage points."""
    values = (entry_price, exit_price, round_trip_cost_pct)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Entry, exit, and cost must be finite numbers.")
    if entry_price <= 0 or exit_price <= 0 or round_trip_cost_pct < 0:
        raise ValueError("Prices must be positive and cost must be non-negative.")
    gross_pct = (float(exit_price) / float(entry_price) - 1.0) * 100.0
    return gross_pct, gross_pct - float(round_trip_cost_pct)


def calculate_profit_factor_metrics(net_returns_pct: Iterable[float]) -> dict[str, float]:
    """Calculate both percentage-based and sequential-equity-weighted profit factor.

    The equity-weighted version assumes the full account equity is allocated to
    each non-overlapping trade and compounds after each net trade return. That
    is the same basic account-allocation convention used by the prototype's
    compounded equity curve; alternative position sizing would change this PF.
    """
    returns = np.asarray(list(net_returns_pct), dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return {"Profit_Factor_Trade_Percent": float("nan"), "Profit_Factor_Equity_Weighted": float("nan")}

    raw_wins = returns[returns > 0].sum()
    raw_losses = abs(returns[returns < 0].sum())
    pf_trade_pct = float(raw_wins / raw_losses) if raw_losses > 0 else float("nan")

    equity = 1.0
    cash_wins = 0.0
    cash_losses = 0.0
    for net_return in returns:
        cash_pnl = equity * float(net_return) / 100.0
        if cash_pnl > 0:
            cash_wins += cash_pnl
        elif cash_pnl < 0:
            cash_losses += abs(cash_pnl)
        equity *= 1.0 + float(net_return) / 100.0

    pf_equity = float(cash_wins / cash_losses) if cash_losses > 0 else float("nan")
    return {
        "Profit_Factor_Trade_Percent": pf_trade_pct,
        "Profit_Factor_Equity_Weighted": pf_equity,
    }


def build_simulated_equity_curve(
    bars: pd.DataFrame, trades: Iterable[Mapping[str, Any]], start_pos: int = 0
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a reproducible daily equity/drawdown curve for non-overlapping trades.

    On non-exit days inside a trade, the account is marked using that day's Close.
    On the exit date it is marked at the recorded simulated exit price. The
    drawdown field is therefore named Daily_Simulated_Equity_Drawdown_Pct rather
    than claiming every date is a close-only mark-to-market.
    """
    frame = validate_ohlc(bars)
    if not (0 <= start_pos < len(frame)):
        raise IndexError("start_pos is outside the supplied price data.")

    trade_list = [dict(trade) for trade in trades]
    trade_list.sort(key=lambda item: pd.Timestamp(item["Entry_Date"]))
    dates = frame.index[start_pos:]
    curve = pd.DataFrame(index=dates)
    curve.index.name = "Date"
    curve["Close"] = frame.loc[dates, "Close"].astype(float)
    curve["Strategy_Equity"] = np.nan
    curve["In_Position"] = False

    current_equity = 1.0
    cursor_pos = start_pos
    for trade in trade_list:
        entry_date = pd.Timestamp(trade["Entry_Date"]).tz_localize(None).normalize()
        exit_date = pd.Timestamp(trade["Exit_Date"]).tz_localize(None).normalize()
        if entry_date not in frame.index or exit_date not in frame.index:
            raise ValueError("A trade entry/exit date is missing from the supplied daily bars.")
        entry_pos = int(frame.index.get_loc(entry_date))
        exit_pos = int(frame.index.get_loc(exit_date))
        if entry_pos < start_pos or exit_pos < entry_pos:
            raise ValueError("Trade entry/exit ordering is invalid.")
        if entry_pos < cursor_pos:
            raise ValueError("Overlapping trades detected in simulated-equity curve.")

        if cursor_pos < entry_pos:
            curve.loc[frame.index[cursor_pos:entry_pos], "Strategy_Equity"] = current_equity

        entry_price = float(trade["Entry_Price"])
        exit_price = float(trade["Exit_Price"])
        cost_pct = float(trade["Assumed_Round_Trip_Cost_Pct"])
        gross_pct, expected_net_pct = calculate_trade_return_pct(entry_price, exit_price, cost_pct)
        net_pct = float(trade["Net_Return_Pct"])
        if not math.isclose(net_pct, expected_net_pct, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(
                "Trade Net_Return_Pct does not reconcile to recorded entry, exit, and cost."
            )

        trade_base_equity = current_equity
        cost_fraction = cost_pct / 100.0
        for pos in range(entry_pos, exit_pos + 1):
            date = frame.index[pos]
            mark_price = exit_price if pos == exit_pos else float(frame.iloc[pos]["Close"])
            curve.at[date, "Strategy_Equity"] = trade_base_equity * (
                mark_price / entry_price - cost_fraction
            )
            curve.at[date, "In_Position"] = True

        current_equity = trade_base_equity * (1.0 + net_pct / 100.0)
        curve.at[exit_date, "Strategy_Equity"] = current_equity
        cursor_pos = exit_pos + 1

    if cursor_pos < len(frame):
        curve.loc[frame.index[cursor_pos:], "Strategy_Equity"] = current_equity

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


def save_input_snapshot(
    bars: pd.DataFrame,
    output_dir: str | Path,
    symbol: str,
    source: str = "unspecified",
    price_basis: str = "quote OHLC, not locally dividend-adjusted; verify split basis",
) -> tuple[Path, Path]:
    """Write a dated OHLC input snapshot and SHA-256 manifest for reproducibility."""
    frame = normalize_daily_frame(bars)
    safe_symbol = re.sub(r"[^A-Za-z0-9_-]+", "_", symbol).strip("_") or "symbol"
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{safe_symbol}_input_bars.csv"
    manifest_path = folder / f"{safe_symbol}_input_manifest.json"
    frame.to_csv(csv_path, index=True, date_format="%Y-%m-%d", float_format="%.10g")
    digest = sha256(csv_path.read_bytes()).hexdigest()
    manifest = {
        "symbol": symbol,
        "source": source,
        "price_basis": price_basis,
        "row_count": int(len(frame)),
        "first_date": frame.index[0].date().isoformat(),
        "last_date": frame.index[-1].date().isoformat(),
        "columns": list(frame.columns),
        "csv_file": csv_path.name,
        "csv_sha256": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return csv_path, manifest_path


# ---------------------------------------------------------------------------
# Synthetic regression tests — no internet access and no live market data.
# ---------------------------------------------------------------------------

def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(rows), freq="B", name="Date")
    frame = pd.DataFrame(rows, columns=list(OHLC), index=dates)
    frame["Volume"] = 1000
    return frame


class BacktestEngineAuditTests(unittest.TestCase):
    def test_same_exit_date_signal_is_allowed(self) -> None:
        self.assertFalse(should_skip_signal(signal_pos=10, last_exit_pos=10))

    def test_signal_before_prior_exit_is_skipped(self) -> None:
        self.assertTrue(should_skip_signal(signal_pos=9, last_exit_pos=10))

    def test_stop_gap_exits_at_open(self) -> None:
        bars = _bars([(100, 102, 99, 101), (90, 92, 88, 91), (91, 92, 90, 91)])
        result = simulate_long_exit(bars, 1, 2, 100, 5, stop_atr_multiplier=1.5)
        self.assertEqual(result.exit_reason, "STOP_GAP_AT_OPEN")
        self.assertAlmostEqual(result.exit_price, 90.0)

    def test_intraday_stop_fills_at_stop_level(self) -> None:
        bars = _bars([(100, 102, 99, 101), (99, 101, 91, 95), (95, 97, 94, 96)])
        result = simulate_long_exit(bars, 1, 2, 100, 5, stop_atr_multiplier=1.5)
        self.assertEqual(result.exit_reason, "ATR_STOP")
        self.assertAlmostEqual(result.exit_price, 92.5)

    def test_intraday_target_fills_at_target_level(self) -> None:
        bars = _bars([(100, 102, 99, 101), (101, 112, 100, 110), (110, 112, 108, 111)])
        result = simulate_long_exit(bars, 1, 2, 100, 5, target_atr_multiplier=2.0)
        self.assertEqual(result.exit_reason, "ATR_TARGET")
        self.assertAlmostEqual(result.exit_price, 110.0)

    def test_target_gap_does_not_assume_favourable_open_fill(self) -> None:
        bars = _bars([(100, 102, 99, 101), (115, 118, 114, 116), (116, 117, 115, 116)])
        result = simulate_long_exit(bars, 1, 2, 100, 5, target_atr_multiplier=2.0)
        self.assertEqual(result.exit_reason, "TARGET_GAP_AT_OPEN_LIMIT")
        self.assertAlmostEqual(result.exit_price, 110.0)
        self.assertLess(result.exit_price, float(bars.iloc[1]["Open"]))

    def test_both_levels_hit_same_day_assumes_stop_first(self) -> None:
        bars = _bars([(100, 102, 99, 101), (100, 112, 90, 105), (105, 106, 104, 105)])
        result = simulate_long_exit(
            bars, 1, 2, 100, 5, stop_atr_multiplier=1.5, target_atr_multiplier=2.0
        )
        self.assertEqual(result.exit_reason, "BOTH_TOUCHED_STOP_FIRST")
        self.assertAlmostEqual(result.exit_price, 92.5)

    def test_time_exit_uses_scheduled_close(self) -> None:
        bars = _bars([(100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)])
        result = simulate_long_exit(bars, 1, 2, 101, 1, None, None)
        self.assertEqual(result.exit_reason, "TIME_EXIT")
        self.assertEqual(result.exit_pos, 2)
        self.assertAlmostEqual(result.exit_price, 103.0)

    def test_invalid_candle_fails_closed(self) -> None:
        bars = _bars([(100, 102, 99, 101), (100, 95, 90, 94), (94, 96, 93, 95)])
        with self.assertRaisesRegex(ValueError, "Inconsistent High"):
            simulate_long_exit(bars, 1, 2, 100, 5, stop_atr_multiplier=1.5)

    def test_quote_ohlc_is_not_dividend_adjusted(self) -> None:
        quote = _bars([(100, 105, 98, 102), (102, 107, 101, 106)])
        adjusted = pd.Series([96.0, 100.0], index=quote.index)
        prepared = prepare_execution_price_frame(quote, adjusted_close=adjusted)
        self.assertAlmostEqual(float(prepared.iloc[0]["Open"]), 100.0)
        self.assertAlmostEqual(float(prepared.iloc[0]["Close"]), 102.0)
        self.assertAlmostEqual(float(prepared.iloc[0]["Adjusted_Close"]), 96.0)
        self.assertNotAlmostEqual(float(prepared.iloc[0]["Close"]), float(prepared.iloc[0]["Adjusted_Close"]))

    def test_yahoo_payload_parser_preserves_quote_ohlc_and_events(self) -> None:
        stamp = int(pd.Timestamp("2025-01-02", tz="UTC").timestamp())
        payload = {
            "chart": {
                "result": [{
                    "timestamp": [stamp],
                    "meta": {"exchangeTimezoneName": "UTC"},
                    "indicators": {
                        "quote": [{"open": [100.0], "high": [110.0], "low": [95.0], "close": [105.0], "volume": [1000]}],
                        "adjclose": [{"adjclose": [99.0]}],
                    },
                    "events": {
                        "dividends": {str(stamp): {"amount": 1.5}},
                        "splits": {str(stamp): {"numerator": 2, "denominator": 1, "splitRatio": "2:1"}},
                    },
                }],
                "error": None,
            }
        }
        parsed = yahoo_chart_payload_to_frame(payload)
        self.assertAlmostEqual(float(parsed.iloc[0]["Open"]), 100.0)
        self.assertAlmostEqual(float(parsed.iloc[0]["Close"]), 105.0)
        self.assertAlmostEqual(float(parsed.iloc[0]["Adjusted_Close"]), 99.0)
        self.assertAlmostEqual(float(parsed.iloc[0]["Dividend_Per_Share"]), 1.5)
        self.assertAlmostEqual(float(parsed.iloc[0]["Stock_Split_Ratio"]), 2.0)
        self.assertEqual(parsed.iloc[0]["Stock_Split_Ratio_Text"], "2:1")

    def test_cost_is_deducted_as_percentage_points(self) -> None:
        gross, net = calculate_trade_return_pct(100.0, 103.0, 0.20)
        self.assertAlmostEqual(gross, 3.0)
        self.assertAlmostEqual(net, 2.8)

    def test_profit_factor_variants_are_both_reported(self) -> None:
        metrics = calculate_profit_factor_metrics([10.0, -5.0, 5.0, -10.0])
        self.assertIn("Profit_Factor_Trade_Percent", metrics)
        self.assertIn("Profit_Factor_Equity_Weighted", metrics)
        self.assertFalse(math.isclose(
            metrics["Profit_Factor_Trade_Percent"],
            metrics["Profit_Factor_Equity_Weighted"],
            rel_tol=1e-6,
        ))

    def test_next_session_entry_after_exit_is_not_overlapping(self) -> None:
        bars = _bars([
            (100, 101, 99, 100),
            (100, 101, 89, 90),
            (95, 98, 94, 97),
            (97, 102, 96, 101),
            (101, 103, 100, 102),
        ])
        trades = [
            {
                "Entry_Date": bars.index[0], "Exit_Date": bars.index[1],
                "Entry_Price": 100.0, "Exit_Price": 90.0,
                "Assumed_Round_Trip_Cost_Pct": 0.20, "Net_Return_Pct": -10.2,
            },
            {
                "Entry_Date": bars.index[2], "Exit_Date": bars.index[3],
                "Entry_Price": 95.0, "Exit_Price": 101.0,
                "Assumed_Round_Trip_Cost_Pct": 0.20,
                "Net_Return_Pct": (101.0 / 95.0 - 1.0) * 100.0 - 0.20,
            },
        ]
        curve, stats = build_simulated_equity_curve(bars, trades)
        self.assertEqual(len(curve), len(bars))
        self.assertGreater(stats["Final_Strategy_Equity"], 0)
        self.assertLessEqual(stats["Maximum_Daily_Simulated_Equity_Drawdown_Pct"], 0.0)
        self.assertGreater(stats["Percent_Of_Sessions_In_Position"], 0.0)

    def test_overlapping_trades_are_rejected(self) -> None:
        bars = _bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
        common = {
            "Entry_Price": 100.0, "Exit_Price": 100.0,
            "Assumed_Round_Trip_Cost_Pct": 0.20, "Net_Return_Pct": -0.20,
        }
        trades = [
            {**common, "Entry_Date": bars.index[0], "Exit_Date": bars.index[1]},
            {**common, "Entry_Date": bars.index[1], "Exit_Date": bars.index[2]},
        ]
        with self.assertRaisesRegex(ValueError, "Overlapping trades"):
            build_simulated_equity_curve(bars, trades)

    def test_snapshot_writes_hash_manifest(self) -> None:
        bars = _bars([(100, 102, 99, 101), (101, 103, 100, 102)])
        with tempfile.TemporaryDirectory() as folder:
            csv_path, manifest_path = save_input_snapshot(
                bars, folder, "TEST.NS", source="synthetic-test"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(csv_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual(manifest["csv_sha256"], sha256(csv_path.read_bytes()).hexdigest())


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BacktestEngineAuditTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\nAudit harness only: no market data was downloaded and no project files were changed.")
    print("A passing suite validates only the synthetic cases included here; it does not validate historical Yahoo data or profitability.")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
