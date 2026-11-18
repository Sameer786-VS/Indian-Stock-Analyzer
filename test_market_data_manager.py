"""Offline tests for market_data_manager.py; no market-data network calls."""
from __future__ import annotations

from datetime import date, timedelta
import tempfile
import unittest

import pandas as pd

from market_data_manager import (
    MarketDataManager,
    merge_price_history,
    parse_yahoo_chart_payload,
    validate_ohlcv,
)


def sample_bars(start: str = "2025-01-01", count: int = 70, close: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=count, name="Date")
    frame = pd.DataFrame(index=idx)
    frame["Open"] = close
    frame["High"] = close + 2
    frame["Low"] = close - 2
    frame["Close"] = close + 1
    frame["Volume"] = 1000
    return frame


class MarketDataManagerTests(unittest.TestCase):
    def test_validates_and_sorts_daily_ohlcv(self):
        frame = sample_bars().iloc[::-1]
        result = validate_ohlcv(frame, min_rows=60)
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertEqual(len(result), 70)

    def test_rejects_invalid_ohlc(self):
        frame = sample_bars()
        frame.iloc[4, frame.columns.get_loc("High")] = 1
        with self.assertRaisesRegex(ValueError, "Invalid OHLCV"):
            validate_ohlcv(frame)

    def test_merge_new_rows_win_on_overlapping_dates(self):
        old = sample_bars(count=5, close=100)
        incoming = sample_bars(start=old.index[-2].date().isoformat(), count=4, close=200)
        merged = merge_price_history(old, incoming)
        self.assertEqual(len(merged), 7)
        self.assertEqual(float(merged.loc[old.index[-1], "Close"]), 201.0)

    def test_initial_download_then_fresh_cache_hit(self):
        calls = []

        def fetcher(symbol, start_date):
            calls.append((symbol, start_date))
            return sample_bars()

        with tempfile.TemporaryDirectory() as folder:
            manager = MarketDataManager(folder, fetcher=fetcher, max_cache_age_hours=24)
            first = manager.get_history("TCS.NS", min_rows=60)
            second = manager.get_history("TCS.NS", min_rows=60)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.attrs["market_data"]["cache_status"], "FULL_DOWNLOAD")
            self.assertEqual(second.attrs["market_data"]["cache_status"], "FRESH_CACHE")
            self.assertEqual(len(second), 70)

    def test_force_refresh_uses_incremental_overlap_and_merges(self):
        calls = []
        original = sample_bars()

        def fetcher(symbol, start_date):
            calls.append(start_date)
            if start_date is None:
                return original
            # Return the final overlapping day plus a new session.
            return sample_bars(start=original.index[-1].date().isoformat(), count=2, close=150)

        with tempfile.TemporaryDirectory() as folder:
            manager = MarketDataManager(folder, fetcher=fetcher, max_cache_age_hours=24)
            manager.get_history("INFY.NS", min_rows=60)
            updated = manager.get_history("INFY.NS", min_rows=60, force_refresh=True)
            self.assertEqual(len(calls), 2)
            self.assertIsNotNone(calls[1])
            self.assertEqual(updated.attrs["market_data"]["cache_status"], "INCREMENTAL_UPDATE")
            self.assertEqual(len(updated), 71)
            self.assertEqual(float(updated.iloc[-2]["Close"]), 151.0)

    def test_failed_refresh_preserves_valid_stale_cache(self):
        def fetcher(symbol, start_date):
            if start_date is None:
                return sample_bars()
            raise RuntimeError("simulated connection failure")

        with tempfile.TemporaryDirectory() as folder:
            manager = MarketDataManager(folder, fetcher=fetcher, max_cache_age_hours=0)
            manager.get_history("RELIANCE.NS", min_rows=60)
            stale = manager.get_history("RELIANCE.NS", min_rows=60, force_refresh=True)
            self.assertEqual(len(stale), 70)
            self.assertEqual(stale.attrs["market_data"]["cache_status"], "STALE_CACHE_FALLBACK")


    def test_failed_full_refresh_preserves_existing_cache(self):
        def fetcher(symbol, start_date):
            if start_date is None and hasattr(fetcher, "initial_done"):
                raise RuntimeError("simulated full refresh failure")
            fetcher.initial_done = True
            return sample_bars()

        with tempfile.TemporaryDirectory() as folder:
            manager = MarketDataManager(folder, fetcher=fetcher, max_cache_age_hours=24)
            manager.get_history("TCS.NS", min_rows=60)
            stale = manager.get_history("TCS.NS", min_rows=60, full_refresh=True)
            self.assertEqual(len(stale), 70)
            self.assertEqual(stale.attrs["market_data"]["cache_status"], "STALE_CACHE_FALLBACK")

    def test_yahoo_parser_keeps_adjusted_analytical_basis(self):
        stamp = int(pd.Timestamp("2025-01-02", tz="UTC").timestamp())
        payload = {
            "chart": {
                "result": [{
                    "timestamp": [stamp],
                    "meta": {"exchangeTimezoneName": "Asia/Kolkata"},
                    "indicators": {
                        "quote": [{"open": [100], "high": [110], "low": [95], "close": [105], "volume": [500]}],
                        "adjclose": [{"adjclose": [99]}],
                    },
                }],
                "error": None,
            }
        }
        result = parse_yahoo_chart_payload(payload)
        self.assertAlmostEqual(float(result.iloc[0]["Open"]), 100 * 99 / 105)
        self.assertAlmostEqual(float(result.iloc[0]["Close"]), 99)
        self.assertEqual(result.index[0].date().isoformat(), "2025-01-02")

    def test_cache_metadata_checksum_detects_modified_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = MarketDataManager(folder, fetcher=lambda symbol, start: sample_bars())
            manager.get_history("HDFCBANK.NS", min_rows=60)
            from pathlib import Path
            cached_csv = Path(folder) / "HDFCBANK_NS.csv"
            cached_csv.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                manager._read_cache("HDFCBANK.NS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
