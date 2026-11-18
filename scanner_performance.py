"""Small, testable performance helpers for the stock scanner.

This module keeps concurrency limited to independent per-symbol data downloads.
Indicator calculations, scoring rules, and report generation remain in main.py.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class ScanDownloadResult:
    """Result of downloading one symbol during a multi-stock scan."""

    symbol: str
    data: Optional[pd.DataFrame]
    error: Optional[str]
    elapsed_seconds: float


def download_symbols_concurrently(
    symbols: Sequence[str],
    downloader: Callable[[str], pd.DataFrame],
    max_workers: int = 3,
) -> list[ScanDownloadResult]:
    """Download distinct symbols concurrently, isolate errors, preserve order.

    Each symbol is submitted at most once. Results are returned in the same
    order as the first occurrence of each symbol, even if downloads finish in
    a different order. The caller remains responsible for validating the
    returned data and handling errors.
    """

    if isinstance(max_workers, bool) or int(max_workers) < 1:
        raise ValueError("max_workers must be an integer of at least 1.")

    ordered_symbols: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = str(raw_symbol).strip().upper()
        if not symbol:
            raise ValueError("Scan symbols cannot contain empty values.")
        if symbol not in seen:
            ordered_symbols.append(symbol)
            seen.add(symbol)

    if not ordered_symbols:
        return []

    worker_count = min(int(max_workers), len(ordered_symbols))
    results: dict[str, ScanDownloadResult] = {}

    def run_one(symbol: str) -> ScanDownloadResult:
        started = perf_counter()
        try:
            frame = downloader(symbol)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Downloader did not return a pandas DataFrame.")
            if frame.empty:
                raise ValueError("Downloader returned an empty DataFrame.")
            return ScanDownloadResult(
                symbol=symbol,
                data=frame,
                error=None,
                elapsed_seconds=perf_counter() - started,
            )
        except Exception as exc:  # isolate one symbol without aborting the batch
            return ScanDownloadResult(
                symbol=symbol,
                data=None,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=perf_counter() - started,
            )

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="stock-scan",
    ) as executor:
        future_to_symbol = {
            executor.submit(run_one, symbol): symbol
            for symbol in ordered_symbols
        }
        for future in as_completed(future_to_symbol):
            result = future.result()
            results[result.symbol] = result

    return [results[symbol] for symbol in ordered_symbols]


def calculate_benchmark_returns(
    benchmark_data: Optional[pd.DataFrame],
    period: int = 20,
) -> Optional[pd.Series]:
    """Calculate benchmark N-session close returns once for reuse by a scan."""

    if benchmark_data is None or benchmark_data.empty:
        return None
    if "Close" not in benchmark_data.columns:
        raise ValueError("Benchmark data is missing the Close column.")
    if isinstance(period, bool) or int(period) < 1:
        raise ValueError("period must be an integer of at least 1.")

    frame = benchmark_data.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
    if index.isna().any():
        raise ValueError("Benchmark data contains invalid dates.")
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    close = pd.to_numeric(frame["Close"], errors="coerce")
    returns = close.pct_change(int(period)) * 100.0
    returns.name = f"Benchmark_Return_{int(period)}D_Pct"
    return returns
