"""Stage 3.6: explainable stock research and peer-relative context.

This module does not change the existing setup-score rules. It adds transparent
research descriptions and improves stock-versus-industry peer comparisons.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any) -> float:
    """Convert a value to a finite float, otherwise return NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _flag(value: Any) -> bool:
    """Safely interpret optional boolean-like values."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def enrich_scan_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Add plain-language evidence fields without changing Setup_Score.

    The five core scoring rules, breakout bonus, breakdown penalty and final
    Setup_Score are copied from the existing scanner and are not recalculated.
    """
    result = dict(record)

    close = _number(result.get("Close"))
    sma20 = _number(result.get("SMA_20"))
    sma50 = _number(result.get("SMA_50"))
    sma200 = _number(result.get("SMA_200"))
    return20 = _number(result.get("Return_20D_Pct"))
    rsi = _number(result.get("RSI_14"))
    rel_nifty = _number(result.get("Relative_Strength_20D_Pct"))
    rel_volume = _number(result.get("Relative_Volume_20D"))

    # Count only rules for which the required input data exists.
    availability = [
        np.isfinite(close) and np.isfinite(sma20),
        np.isfinite(sma20) and np.isfinite(sma50),
        np.isfinite(close) and np.isfinite(sma200),
        np.isfinite(return20),
        np.isfinite(rsi),
    ]
    available_count = int(sum(availability))
    result["Core_Rule_Available_Count"] = available_count
    result["Core_Rule_Completeness_Pct"] = round(available_count / 5 * 100, 2)

    core_keys = [
        "Score_Close_Above_SMA20",
        "Score_SMA20_Above_SMA50",
        "Score_Close_Above_SMA200",
        "Score_Positive_Return_20D",
        "Score_RSI_50_70",
    ]
    core_points = sum(int(_number(result.get(key)) == 1) for key in core_keys)
    breakout_value = _number(result.get("Score_Breakout_Bonus"))
    penalty_value = _number(result.get("Score_Breakdown_Penalty"))
    breakout_bonus = int(breakout_value) if np.isfinite(breakout_value) else 0
    breakdown_penalty = int(penalty_value) if np.isfinite(penalty_value) else 0
    setup_score = _number(result.get("Setup_Score"))

    result["Core_Rules_Passed"] = core_points
    result["Core_Score_Explanation"] = (
        f"{core_points}/5 core rules passed; breakout bonus {breakout_bonus:+d}; "
        f"breakdown penalty {breakdown_penalty:+d}; "
        f"existing setup score {setup_score:g}" if np.isfinite(setup_score)
        else f"{core_points}/5 core rules passed; setup score unavailable"
    )

    # Trend evidence.
    if not all(np.isfinite(x) for x in (close, sma20, sma50)):
        trend = "Insufficient trend data"
    elif close > sma20 > sma50:
        trend = (
            "Positive long-term alignment"
            if np.isfinite(sma200) and close > sma200
            else "Positive short/medium-term alignment"
        )
    elif close < sma20 < sma50:
        trend = "Negative short/medium-term alignment"
    else:
        trend = "Mixed moving-average alignment"
    result["Research_Trend_State"] = trend

    # Momentum and RSI evidence are deliberately kept descriptive.
    if not np.isfinite(return20) or not np.isfinite(rsi):
        momentum = "Insufficient momentum data"
    elif return20 > 0 and 50 <= rsi <= 70:
        momentum = "Positive momentum; RSI in the 50–70 zone"
    elif return20 > 0 and rsi > 70:
        momentum = "Positive return; RSI elevated above 70"
    elif return20 < 0 and rsi < 50:
        momentum = "Weak momentum; negative return and RSI below 50"
    elif return20 < 0:
        momentum = "Negative 20-session return; RSI needs context"
    else:
        momentum = "Mixed or sideways momentum"
    result["Research_Momentum_State"] = momentum

    # Benchmark relative strength.
    if not np.isfinite(rel_nifty):
        benchmark_state = "Unavailable"
    elif rel_nifty > 0.5:
        benchmark_state = "Outperforming NIFTY 50"
    elif rel_nifty < -0.5:
        benchmark_state = "Underperforming NIFTY 50"
    else:
        benchmark_state = "Broadly in line with NIFTY 50"
    result["Research_Benchmark_State"] = benchmark_state

    # Volume context.
    if not np.isfinite(rel_volume):
        volume_state = "Unavailable"
    elif rel_volume >= 1.5:
        volume_state = "High volume participation"
    elif rel_volume >= 1.0:
        volume_state = "At or above average volume"
    else:
        volume_state = "Below average volume"
    result["Research_Volume_State"] = volume_state

    breakout = _flag(result.get("Breakout_Confirmed"))
    breakdown = _flag(result.get("Breakdown_Confirmed"))
    raw_breakout = _flag(result.get("Breakout_60D"))
    raw_breakdown = _flag(result.get("Breakdown_60D"))
    if breakdown:
        price_action = "Volume-confirmed breakdown flag"
    elif breakout:
        price_action = "Volume-confirmed breakout flag"
    elif raw_breakdown:
        price_action = "Potential breakdown; volume confirmation absent"
    elif raw_breakout:
        price_action = "Potential breakout; volume confirmation absent"
    else:
        price_action = "No fresh 60-session breakout or breakdown"
    result["Research_Price_Action_State"] = price_action

    positive: list[str] = []
    cautions: list[str] = []
    missing: list[str] = []

    if np.isfinite(close) and np.isfinite(sma20):
        (positive if close > sma20 else cautions).append(
            "price above SMA 20" if close > sma20 else "price below SMA 20"
        )
    else:
        missing.append("SMA 20 comparison")

    if np.isfinite(sma20) and np.isfinite(sma50):
        (positive if sma20 > sma50 else cautions).append(
            "SMA 20 above SMA 50" if sma20 > sma50 else "SMA 20 below SMA 50"
        )
    else:
        missing.append("SMA 20/SMA 50 relationship")

    if np.isfinite(close) and np.isfinite(sma200):
        (positive if close > sma200 else cautions).append(
            "price above SMA 200" if close > sma200 else "price below SMA 200"
        )
    else:
        missing.append("SMA 200 comparison")

    if np.isfinite(return20):
        (positive if return20 > 0 else cautions).append(
            "positive 20-session return" if return20 > 0
            else "non-positive 20-session return"
        )
    else:
        missing.append("20-session return")

    if np.isfinite(rsi):
        if 50 <= rsi <= 70:
            positive.append("RSI between 50 and 70")
        elif rsi > 70:
            cautions.append("RSI above 70; momentum may be extended")
        elif rsi < 50:
            cautions.append("RSI below 50")
    else:
        missing.append("RSI 14")

    if np.isfinite(rel_nifty):
        if rel_nifty > 0.5:
            positive.append("outperforming NIFTY 50")
        elif rel_nifty < -0.5:
            cautions.append("underperforming NIFTY 50")
    else:
        missing.append("NIFTY 50 relative strength")

    if np.isfinite(rel_volume):
        if rel_volume >= 1.5:
            positive.append("volume at least 1.5x its prior average")
        elif rel_volume < 1.0:
            cautions.append("volume below its prior average")
    else:
        missing.append("relative volume")

    if breakout:
        positive.append("volume-confirmed 60-session breakout")
    if breakdown:
        cautions.append("volume-confirmed 60-session breakdown")
    if raw_breakout and not breakout:
        cautions.append("breakout lacks volume confirmation")
    if raw_breakdown and not breakdown:
        cautions.append("breakdown lacks volume confirmation")

    if positive:
        positives_text = "; ".join(positive)
    else:
        positives_text = "No clear positive evidence from the available rules"
    if cautions:
        cautions_text = "; ".join(cautions)
    else:
        cautions_text = "No major caution flagged by the available rules"
    if missing:
        missing_text = "; ".join(missing)
    else:
        missing_text = "None"

    result["Research_Positive_Evidence"] = positives_text
    result["Research_Caution_Evidence"] = cautions_text
    result["Research_Missing_Evidence"] = missing_text

    if breakdown:
        conclusion = "Downside-risk evidence present — review carefully"
    elif (
        trend.startswith("Positive")
        and np.isfinite(return20) and return20 > 0
        and np.isfinite(rel_nifty) and rel_nifty > 0.5
    ):
        conclusion = "Constructive research profile"
    elif (
        trend.startswith("Negative")
        and np.isfinite(rel_nifty) and rel_nifty < -0.5
    ):
        conclusion = "Weak research profile"
    else:
        conclusion = "Mixed research evidence"
    result["Research_Conclusion"] = conclusion
    result["Research_Use_Limitation"] = (
        "Descriptive screening evidence only; not a probability, forecast, "
        "buy/sell signal or investment recommendation."
    )
    return result


def add_peer_comparisons(
    stock_results: pd.DataFrame,
    sector_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each stock with other scanned names in its industry.

    The stock itself is excluded from its peer average. Sector-wide returns
    remain unchanged. Small and single-constituent groups are clearly flagged.
    """
    required = {"Symbol", "NSE_Industry", "Return_20D_Pct"}
    missing = sorted(required - set(stock_results.columns))
    if missing:
        raise ValueError("Peer comparison is missing columns: " + ", ".join(missing))

    stocks = stock_results.copy()
    stocks["Return_20D_Pct"] = pd.to_numeric(
        stocks["Return_20D_Pct"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)

    peer_returns: list[float] = []
    peer_counts: list[int] = []
    peer_statuses: list[str] = []
    for _, stock in stocks.iterrows():
        peers = stocks[
            (stocks["NSE_Industry"] == stock["NSE_Industry"])
            & (stocks["Symbol"] != stock["Symbol"])
        ]
        peer_values = peers["Return_20D_Pct"].dropna()
        peer_count = int(peer_values.count())
        peer_mean = float(peer_values.mean()) if peer_count else np.nan
        peer_counts.append(peer_count)
        peer_returns.append(peer_mean)
        if peer_count == 0:
            peer_statuses.append("Unavailable — no other scanned peers")
        elif peer_count == 1:
            peer_statuses.append("Limited — one scanned peer")
        elif peer_count < 4:
            peer_statuses.append("Small peer sample — interpret cautiously")
        else:
            peer_statuses.append("Peer comparison available")

    stocks["Sector_Peer_Constituent_Count"] = peer_counts
    stocks["Sector_Peer_Return_20D_Pct"] = peer_returns
    stocks["Stock_vs_Sector_20D_Pct"] = (
        stocks["Return_20D_Pct"] - stocks["Sector_Peer_Return_20D_Pct"]
    )
    stocks["Stock_vs_Sector_Status"] = [
        _peer_strength_label(value)
        for value in stocks["Stock_vs_Sector_20D_Pct"]
    ]
    stocks["Peer_Comparison_Quality"] = peer_statuses

    sectors = sector_summary.copy()
    if not sectors.empty and "Expected_Constituents" in sectors.columns:
        sample_status = []
        for _, sector in sectors.iterrows():
            expected = int(_number(sector.get("Expected_Constituents")) or 0)
            scanned = int(_number(sector.get("Scanned_Constituents")) or 0)
            if expected <= 1:
                label = "Single constituent — not a broad industry sample"
            elif expected <= 3:
                label = "Small industry group — interpret cautiously"
            else:
                label = "Broader industry group"
            if scanned < expected:
                label = "Partial coverage; " + label
            sample_status.append(label)
        sectors["Industry_Sample_Interpretation"] = sample_status
    return stocks, sectors


def _peer_strength_label(value: Any, neutral_band: float = 0.25) -> str:
    number = _number(value)
    if not np.isfinite(number):
        return "Unavailable"
    if number > neutral_band:
        return "Outperforming other industry constituents"
    if number < -neutral_band:
        return "Underperforming other industry constituents"
    return "Broadly in line with other industry constituents"


def build_research_report_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact, readable selection for the extra research CSV."""
    preferred = [
        "Symbol", "Company_Name", "NSE_Industry", "Latest_Date", "Close",
        "Setup_Score", "Core_Rules_Passed", "Core_Rule_Available_Count",
        "Core_Rule_Completeness_Pct", "Core_Score_Explanation",
        "Research_Trend_State", "Research_Momentum_State",
        "Research_Benchmark_State", "Research_Volume_State",
        "Research_Price_Action_State", "Research_Conclusion",
        "Research_Positive_Evidence", "Research_Caution_Evidence",
        "Research_Missing_Evidence", "Sector_Return_20D_Pct",
        "Sector_Relative_Strength_20D_Pct", "Sector_Peer_Return_20D_Pct",
        "Stock_vs_Sector_20D_Pct", "Stock_vs_Sector_Status",
        "Sector_Peer_Constituent_Count", "Peer_Comparison_Quality",
        "Sector_Constituent_Coverage_Pct", "Industry_Sample_Interpretation",
        "Research_Use_Limitation",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns].copy()
