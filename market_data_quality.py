
"""Advanced daily market-data quality checks for Indian Stock Analyzer."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from market_data_manager import validate_ohlcv

INDIA_TIMEZONE = "Asia/Kolkata"

DEFAULT_EXTREME_RETURN_PCT = 35.0
DEFAULT_HIGH_MISSING_VOLUME_PCT = 10.0
DEFAULT_HIGH_ZERO_VOLUME_PCT = 2.0
DEFAULT_STALE_SESSION_TOLERANCE = 1


def _normalize_calendar(values: Iterable) -> pd.DatetimeIndex:
    """Normalize a benchmark calendar to Indian market dates."""

    dates = pd.DatetimeIndex(
        pd.to_datetime(list(values), errors="coerce")
    )

    if dates.isna().any():
        raise ValueError(
            "Reference market calendar contains invalid dates."
        )

    if dates.tz is not None:
        dates = dates.tz_convert(
            INDIA_TIMEZONE
        ).tz_localize(None)

    return dates.normalize().drop_duplicates().sort_values()


def assess_market_data(
    frame: pd.DataFrame,
    symbol: str,
    reference_calendar: Optional[Iterable] = None,
    *,
    as_of: Optional[date] = None,
    extreme_return_pct: float = DEFAULT_EXTREME_RETURN_PCT,
    high_missing_volume_pct: float = DEFAULT_HIGH_MISSING_VOLUME_PCT,
    high_zero_volume_pct: float = DEFAULT_HIGH_ZERO_VOLUME_PCT,
    stale_session_tolerance: int = DEFAULT_STALE_SESSION_TOLERANCE,
) -> dict:
    """Validate daily market data and return a quality report."""

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("A stock symbol is required.")

    if extreme_return_pct <= 0:
        raise ValueError(
            "Extreme return threshold must be positive."
        )

    if not 0 <= high_missing_volume_pct <= 100:
        raise ValueError(
            "Missing-volume threshold must be between 0 and 100."
        )

    if not 0 <= high_zero_volume_pct <= 100:
        raise ValueError(
            "Zero-volume threshold must be between 0 and 100."
        )

    if stale_session_tolerance < 0:
        raise ValueError(
            "Stale-session tolerance cannot be negative."
        )

    # Stage 3.1 structural OHLCV validation.
    data = validate_ohlcv(frame)

    first_date = data.index.min()
    latest_date = data.index.max()
    total_rows = len(data)

    warnings = []

    if total_rows < 200:
        warnings.append(
            f"Only {total_rows} rows available; "
            "SMA 200 may not be ready."
        )

    # Unusually large daily price changes.
    returns = data["Close"].pct_change() * 100

    extreme_mask = returns.abs() > extreme_return_pct
    extreme_dates = data.index[extreme_mask.fillna(False)]
    extreme_count = int(extreme_mask.sum())

    if extreme_count:
        sample = ", ".join(
            d.strftime("%Y-%m-%d")
            for d in extreme_dates[:5]
        )

        warnings.append(
            f"{extreme_count} daily returns exceeded "
            f"+/-{extreme_return_pct}%: {sample}. "
            "Review corporate actions or source data."
        )

    # Volume analysis.
    volume = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    )

    missing_volume_count = int(volume.isna().sum())
    zero_volume_count = int((volume == 0).sum())

    missing_volume_pct = (
        missing_volume_count / total_rows * 100
        if total_rows else 0.0
    )

    zero_volume_pct = (
        zero_volume_count / total_rows * 100
        if total_rows else 0.0
    )

    is_index = symbol.startswith("^")

    if (
        not is_index
        and missing_volume_pct > high_missing_volume_pct
    ):
        warnings.append(
            f"Missing volume is {missing_volume_pct:.2f}% of rows."
        )

    if (
        not is_index
        and zero_volume_pct > high_zero_volume_pct
    ):
        warnings.append(
            f"Zero volume is {zero_volume_pct:.2f}% of rows."
        )

    # Detect suspicious flat-price, zero-volume bars.
    flat_zero_mask = (
        (data["Open"] == data["High"])
        & (data["High"] == data["Low"])
        & (data["Low"] == data["Close"])
        & (volume.fillna(-1) == 0)
    )

    flat_zero_count = int(flat_zero_mask.sum())

    if not is_index and flat_zero_count:
        warnings.append(
            f"{flat_zero_count} flat OHLC rows have zero volume."
        )

    # Benchmark calendar and freshness analysis.
    calendar_gap_count = None
    calendar_gap_dates = []
    latest_session_lag = None

    freshness_status = "NOT_ASSESSED_NO_REFERENCE_CALENDAR"
    reference_latest = None

    if reference_calendar is not None:

        calendar = _normalize_calendar(reference_calendar)

        if len(calendar) == 0:
            warnings.append("Reference calendar is empty.")

        else:
            reference_latest = calendar.max()

            expected_dates = calendar[
                (calendar >= first_date)
                & (calendar <= latest_date)
            ]

            missing_dates = expected_dates.difference(data.index)

            calendar_gap_count = len(missing_dates)

            calendar_gap_dates = [
                d.strftime("%Y-%m-%d")
                for d in missing_dates[:10]
            ]

            if calendar_gap_count:
                warnings.append(
                    f"{calendar_gap_count} benchmark session dates "
                    "are missing from the stock history."
                )

            later_sessions = calendar[calendar > latest_date]

            latest_session_lag = len(later_sessions)

            if latest_date > reference_latest:

                freshness_status = "AHEAD_OF_REFERENCE"

                warnings.append(
                    "Stock data ends after the benchmark's latest date."
                )

            elif latest_session_lag == 0:

                freshness_status = "CURRENT_TO_REFERENCE"

            elif latest_session_lag <= stale_session_tolerance:

                freshness_status = "WITHIN_TOLERANCE"

            else:

                freshness_status = "STALE_VS_REFERENCE"

                warnings.append(
                    f"Stock data trails the benchmark by "
                    f"{latest_session_lag} sessions."
                )

    # Future-date check.
    current_date = (
        as_of
        or datetime.now(
            ZoneInfo(INDIA_TIMEZONE)
        ).date()
    )

    if latest_date.date() > current_date:
        warnings.append(
            f"Latest market date {latest_date.date()} is "
            f"later than as-of date {current_date}."
        )

    # Read metadata supplied by Stage 3.1.
    metadata = frame.attrs.get(
        "market_data", {}
    )

    quality_status = (
        "WARN" if warnings else "PASS"
    )

    return {
        "Symbol": symbol,
        "Quality_Status": quality_status,
        "Rows": total_rows,

        "First_Market_Date": (
            first_date.date().isoformat()
        ),

        "Latest_Market_Date": (
            latest_date.date().isoformat()
        ),

        "Reference_Latest_Market_Date": (
            reference_latest.date().isoformat()
            if reference_latest is not None else ""
        ),

        "Latest_Session_Lag": latest_session_lag,
        "Freshness_Status": freshness_status,

        "Calendar_Gap_Count": calendar_gap_count,

        "Calendar_Gap_Example_Dates": (
            ";".join(calendar_gap_dates)
        ),

        "Missing_Volume_Count": missing_volume_count,
        "Missing_Volume_Pct": round(
            missing_volume_pct, 4
        ),

        "Zero_Volume_Count": zero_volume_count,
        "Zero_Volume_Pct": round(
            zero_volume_pct, 4
        ),

        "Flat_Zero_Volume_Bar_Count": (
            flat_zero_count
        ),

        "Extreme_Daily_Return_Count": (
            extreme_count
        ),

        "Extreme_Daily_Return_Threshold_Pct": (
            float(extreme_return_pct)
        ),

        "Source": metadata.get(
            "source", "unknown"
        ),

        "Cache_Status": metadata.get(
            "cache_status", "unknown"
        ),

        "Downloaded_At_UTC": metadata.get(
            "downloaded_at_utc", ""
        ),

        "Price_Basis": metadata.get(
            "price_basis", "not supplied"
        ),

        "Warning_Count": len(warnings),

        "Warnings": " | ".join(warnings),
    }


def save_quality_reports(
    reports: list[dict],
    output_dir: str,
    run_label: str = "market_data_quality",
    timestamp: Optional[str] = None,
) -> tuple[str, str]:
    """Save timestamped and latest quality reports."""

    if not reports:
        raise ValueError(
            "No quality reports were supplied."
        )

    folder = Path(output_dir)

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = (
        timestamp
        or datetime.now(
            ZoneInfo(INDIA_TIMEZONE)
        ).strftime("%Y%m%d_%H%M%S")
    )

    safe_label = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in run_label
    )

    archived_path = (
        folder / f"{safe_label}_{stamp}.csv"
    )

    latest_path = (
        folder / f"{safe_label}_latest.csv"
    )

    report_frame = pd.DataFrame(reports)

    report_frame.to_csv(
        archived_path,
        index=False
    )

    report_frame.to_csv(
        latest_path,
        index=False
    )

    return (
        str(archived_path),
        str(latest_path)
    )
