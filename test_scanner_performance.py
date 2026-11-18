"""Offline tests for Stage 3.3 scanner performance helpers."""
from __future__ import annotations

import threading
import time
import unittest

import pandas as pd

from scanner_performance import (
    calculate_benchmark_returns,
    download_symbols_concurrently,
)


def sample_frame(close_values=(100.0, 101.0, 102.0)) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-01", periods=len(close_values), name="Date")
    return pd.DataFrame({"Close": close_values}, index=index)


class ScannerPerformanceTests(unittest.TestCase):
    def test_downloads_return_in_input_order(self):
        def downloader(symbol):
            # Deliberately finish in a different order from the input.
            time.sleep({"AAA.NS": 0.03, "BBB.NS": 0.01, "CCC.NS": 0.02}[symbol])
            return sample_frame()

        results = download_symbols_concurrently(
            ["AAA.NS", "BBB.NS", "CCC.NS"], downloader, max_workers=3
        )
        self.assertEqual(
            [result.symbol for result in results],
            ["AAA.NS", "BBB.NS", "CCC.NS"],
        )
        self.assertTrue(all(result.data is not None for result in results))
        self.assertTrue(all(result.error is None for result in results))

    def test_one_download_failure_does_not_abort_other_symbols(self):
        def downloader(symbol):
            if symbol == "BAD.NS":
                raise RuntimeError("simulated provider failure")
            return sample_frame()

        results = download_symbols_concurrently(
            ["GOOD1.NS", "BAD.NS", "GOOD2.NS"], downloader, max_workers=2
        )
        by_symbol = {result.symbol: result for result in results}
        self.assertIsNotNone(by_symbol["GOOD1.NS"].data)
        self.assertIsNone(by_symbol["BAD.NS"].data)
        self.assertIn("simulated provider failure", by_symbol["BAD.NS"].error)
        self.assertIsNotNone(by_symbol["GOOD2.NS"].data)

    def test_duplicate_symbols_are_downloaded_once(self):
        calls = []
        lock = threading.Lock()

        def downloader(symbol):
            with lock:
                calls.append(symbol)
            return sample_frame()

        results = download_symbols_concurrently(
            ["tcs.ns", "INFY.NS", "TCS.NS"], downloader, max_workers=2
        )
        self.assertEqual(calls.count("TCS.NS"), 1)
        self.assertEqual([result.symbol for result in results], ["TCS.NS", "INFY.NS"])

    def test_empty_input_returns_empty_results(self):
        results = download_symbols_concurrently([], lambda symbol: sample_frame())
        self.assertEqual(results, [])

    def test_invalid_worker_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            download_symbols_concurrently(["TCS.NS"], lambda symbol: sample_frame(), 0)

    def test_empty_or_non_dataframe_download_is_recorded_as_error(self):
        results = download_symbols_concurrently(
            ["EMPTY.NS", "WRONG.NS"],
            lambda symbol: pd.DataFrame() if symbol == "EMPTY.NS" else None,
            max_workers=2,
        )
        self.assertTrue(all(result.data is None for result in results))
        self.assertIn("empty DataFrame", results[0].error)
        self.assertIn("pandas DataFrame", results[1].error)

    def test_benchmark_returns_are_calculated_for_reuse(self):
        frame = sample_frame((100.0, 105.0, 110.0))
        returns = calculate_benchmark_returns(frame, period=1)
        self.assertAlmostEqual(float(returns.iloc[1]), 5.0)
        self.assertAlmostEqual(float(returns.iloc[2]), (110.0 / 105.0 - 1.0) * 100.0)

    def test_benchmark_returns_normalize_timezone_and_duplicate_dates(self):
        frame = sample_frame((100.0, 101.0, 102.0))
        frame.index = frame.index.tz_localize("Asia/Kolkata")
        returns = calculate_benchmark_returns(frame, period=1)
        self.assertIsNone(returns.index.tz)
        self.assertTrue(returns.index.is_monotonic_increasing)

    def test_missing_benchmark_returns_none(self):
        self.assertIsNone(calculate_benchmark_returns(None))
        self.assertIsNone(calculate_benchmark_returns(pd.DataFrame()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
