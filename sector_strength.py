"""Transparent sector/industry strength calculations for the NIFTY 50 scanner.

Sector returns are equal-weighted averages of the available NIFTY 50
constituent returns for the supplied snapshot. They are research proxies,
not official NSE sector-index returns and not market-cap weighted.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np
import pandas as pd

REQUIRED_RESULT_COLUMNS = {
    "Symbol",
    "NSE_Industry",
    "Daily_Return_Pct",
    "Return_5D_Pct",
    "Return_20D_Pct",
    "NIFTY50_Return_20D_Pct",
}


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return float(values.mean()) if not values.empty else np.nan


def _strength_label(value: object, neutral_band: float = 0.5) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not np.isfinite(number):
        return "Unavailable"
    if number > neutral_band:
        return "Outperforming NIFTY 50"
    if number < -neutral_band:
        return "Underperforming NIFTY 50"
    return "Broadly in line with NIFTY 50"


def _stock_sector_label(value: object, neutral_band: float = 0.25) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not np.isfinite(number):
        return "Unavailable"
    if number > neutral_band:
        return "Outperforming own industry"
    if number < -neutral_band:
        return "Underperforming own industry"
    return "Broadly in line with own industry"


def build_sector_strength_report(
    stock_results: pd.DataFrame,
    universe_metadata: Mapping[str, Mapping[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (enriched_stock_results, sector_summary).

    `stock_results` must contain one latest scan row per successful stock.
    `universe_metadata` must describe the full dated NIFTY 50 snapshot, so
    coverage percentages use the expected constituent count per industry,
    rather than only the stocks that happened to download successfully.
    """
    missing = sorted(REQUIRED_RESULT_COLUMNS - set(stock_results.columns))
    if missing:
        raise ValueError(
            "Stock scan results are missing required columns: "
            + ", ".join(missing)
        )
    if not universe_metadata:
        raise ValueError("Full NIFTY 50 universe metadata is required.")

    stocks = stock_results.copy()
    if stocks["Symbol"].astype(str).duplicated().any():
        duplicates = sorted(
            stocks.loc[
                stocks["Symbol"].astype(str).duplicated(keep=False), "Symbol"
            ].astype(str).unique()
        )
        raise ValueError("Duplicate stock rows supplied: " + ", ".join(duplicates))

    # Derive expected sector membership from the full constituent snapshot.
    expected_by_sector = Counter()
    for metadata in universe_metadata.values():
        industry = str(metadata.get("NSE_Industry", "")).strip()
        if industry:
            expected_by_sector[industry] += 1
    if not expected_by_sector:
        raise ValueError("Universe metadata contains no industry classifications.")

    # Use the known snapshot classification, rather than trusting an arbitrary
    # industry label from a caller-provided results frame.
    metadata_by_symbol = {
        str(symbol).upper(): metadata
        for symbol, metadata in universe_metadata.items()
    }
    industry_values = []
    for symbol in stocks["Symbol"].astype(str):
        metadata = metadata_by_symbol.get(symbol.upper(), {})
        industry_values.append(str(metadata.get("NSE_Industry", "")).strip())
    stocks["NSE_Industry"] = industry_values

    valid_stocks = stocks[stocks["NSE_Industry"].isin(expected_by_sector)].copy()
    numeric_columns = [
        "Daily_Return_Pct",
        "Return_5D_Pct",
        "Return_20D_Pct",
        "NIFTY50_Return_20D_Pct",
    ]
    for column in numeric_columns:
        valid_stocks[column] = pd.to_numeric(
            valid_stocks[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    # Every sector in the dated snapshot appears in the summary, even if all
    # of its constituent downloads failed; coverage makes this explicit.
    rows: list[dict[str, object]] = []
    for industry, expected_count in sorted(expected_by_sector.items()):
        group = valid_stocks[valid_stocks["NSE_Industry"] == industry]
        scanned_count = int(group["Symbol"].nunique())
        sector_1d = _safe_mean(group["Daily_Return_Pct"])
        sector_5d = _safe_mean(group["Return_5D_Pct"])
        sector_20d = _safe_mean(group["Return_20D_Pct"])
        nifty_20d = _safe_mean(group["NIFTY50_Return_20D_Pct"])
        relative_20d = (
            sector_20d - nifty_20d
            if np.isfinite(sector_20d) and np.isfinite(nifty_20d)
            else np.nan
        )

        advancers_1d = int((group["Daily_Return_Pct"] > 0).sum())
        positive_20d = int((group["Return_20D_Pct"] > 0).sum())
        outperformers_20d = int(
            (group["Return_20D_Pct"] > group["NIFTY50_Return_20D_Pct"]).sum()
        )
        above_sma20 = (
            int(group["Close_Above_SMA20"].fillna(False).astype(bool).sum())
            if "Close_Above_SMA20" in group.columns else np.nan
        )
        breakouts = (
            int(group["Breakout_Confirmed"].fillna(False).astype(bool).sum())
            if "Breakout_Confirmed" in group.columns else np.nan
        )

        coverage = scanned_count / expected_count * 100 if expected_count else 0.0
        rows.append({
            "NSE_Industry": industry,
            "Expected_Constituents": int(expected_count),
            "Scanned_Constituents": scanned_count,
            "Coverage_Pct": coverage,
            "Sector_Return_1D_Pct": sector_1d,
            "Sector_Return_5D_Pct": sector_5d,
            "Sector_Return_20D_Pct": sector_20d,
            "NIFTY50_Return_20D_Pct": nifty_20d,
            "Sector_Relative_Strength_20D_Pct": relative_20d,
            "Advancers_1D_Count": advancers_1d,
            "Positive_Return_20D_Count": positive_20d,
            "Outperforming_NIFTY_20D_Count": outperformers_20d,
            "Above_SMA20_Count": above_sma20,
            "Confirmed_Breakout_Count": breakouts,
            "Coverage_Status": (
                "FULL" if scanned_count == expected_count
                else "PARTIAL" if scanned_count > 0
                else "NO DATA"
            ),
            "Strength_Status": _strength_label(relative_20d),
        })

    sector_summary = pd.DataFrame(rows)
    sector_summary = sector_summary.sort_values(
        ["Sector_Relative_Strength_20D_Pct", "Sector_Return_20D_Pct"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    sector_summary.insert(
        0, "Sector_Rank_20D",
        pd.Series(
            np.where(
                sector_summary["Sector_Relative_Strength_20D_Pct"].notna(),
                np.arange(1, len(sector_summary) + 1),
                np.nan,
            ),
            dtype="float64",
        ),
    )

    sector_lookup = sector_summary.set_index("NSE_Industry").to_dict("index")
    stock_sector_returns = []
    stock_sector_relative = []
    stock_vs_sector = []
    stock_vs_sector_status = []
    stock_sector_coverage = []
    for _, row in stocks.iterrows():
        industry = row["NSE_Industry"]
        sector = sector_lookup.get(industry, {})
        sector_return = sector.get("Sector_Return_20D_Pct", np.nan)
        sector_relative = sector.get("Sector_Relative_Strength_20D_Pct", np.nan)
        try:
            stock_return = float(row["Return_20D_Pct"])
        except (TypeError, ValueError):
            stock_return = np.nan
        difference = (
            stock_return - sector_return
            if np.isfinite(stock_return) and pd.notna(sector_return)
            else np.nan
        )
        stock_sector_returns.append(sector_return)
        stock_sector_relative.append(sector_relative)
        stock_vs_sector.append(difference)
        stock_vs_sector_status.append(_stock_sector_label(difference))
        stock_sector_coverage.append(sector.get("Coverage_Pct", np.nan))

    stocks["Sector_Return_20D_Pct"] = stock_sector_returns
    stocks["Sector_Relative_Strength_20D_Pct"] = stock_sector_relative
    stocks["Stock_vs_Sector_20D_Pct"] = stock_vs_sector
    stocks["Stock_vs_Sector_Status"] = stock_vs_sector_status
    stocks["Sector_Constituent_Coverage_Pct"] = stock_sector_coverage
    return stocks, sector_summary
