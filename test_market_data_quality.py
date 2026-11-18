"""Offline tests for Stage 3.2 market-data quality diagnostics."""
from __future__ import annotations

from datetime import date
import tempfile
import unittest

import pandas as pd

from market_data_quality import assess_market_data, save_quality_reports


def sample_bars(count: int = 230, start: str = "2025-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=count, name="Date")
    frame = pd.DataFrame(index=idx)
    frame["Open"] = 100.0
    frame["High"] = 102.0
    frame["Low"] = 99.0
    frame["Close"] = 101.0
    frame["Volume"] = 1000.0
    return frame


class MarketDataQualityTests(unittest.TestCase):
    def test_clean_stock_data_passes_without_warnings(self):
        frame = sample_bars()
        report = assess_market_data(frame, "TCS.NS", as_of=date(2026, 1, 1))
        self.assertEqual(report["Quality_Status"], "PASS")
        self.assertEqual(report["Rows"], 230)
        self.assertEqual(report["Warning_Count"], 0)

    def test_extreme_price_move_is_flagged_not_modified(self):
        frame = sample_bars()
        frame.iloc[100, frame.columns.get_loc("Close")] = 150.0
        frame.iloc[100, frame.columns.get_loc("High")] = 152.0
        report = assess_market_data(frame, "TCS.NS", as_of=date(2026, 1, 1))
        self.assertEqual(report["Quality_Status"], "WARN")
        self.assertGreater(report["Extreme_Daily_Return_Count"], 0)
        self.assertEqual(float(frame.iloc[100]["Close"]), 150.0)

    def test_benchmark_calendar_detects_missing_sessions(self):
        frame = sample_bars(count=10)
        frame = frame.drop(frame.index[4])
        calendar = pd.bdate_range(frame.index.min(), frame.index.max())
        report = assess_market_data(
            frame, "INFY.NS", reference_calendar=calendar, as_of=date(2026, 1, 1)
        )
        self.assertEqual(report["Calendar_Gap_Count"], 1)
        self.assertEqual(report["Quality_Status"], "WARN")

    def test_benchmark_calendar_detects_stale_latest_date(self):
        frame = sample_bars(count=10)
        calendar = pd.bdate_range(frame.index.min(), periods=14)
        report = assess_market_data(
            frame, "HDFCBANK.NS", reference_calendar=calendar, as_of=date(2026, 1, 1)
        )
        self.assertEqual(report["Freshness_Status"], "STALE_VS_REFERENCE")
        self.assertGreater(report["Latest_Session_Lag"], 1)

    def test_one_session_lag_is_within_tolerance(self):
        frame = sample_bars(count=10)
        calendar = pd.bdate_range(frame.index.min(), periods=11)
        report = assess_market_data(
            frame, "TCS.NS", reference_calendar=calendar, as_of=date(2026, 1, 1)
        )
        self.assertEqual(report["Freshness_Status"], "WITHIN_TOLERANCE")
        self.assertEqual(report["Latest_Session_Lag"], 1)

    def test_short_history_warns_about_sma200(self):
        report = assess_market_data(sample_bars(count=90), "NEWCO.NS", as_of=date(2026, 1, 1))
        self.assertEqual(report["Quality_Status"], "WARN")
        self.assertTrue(any("SMA 200" in report["Warnings"] for _ in [0]))

    def test_missing_volume_on_index_is_recorded_without_warning(self):
        frame = sample_bars()
        frame["Volume"] = float("nan")
        report = assess_market_data(frame, "^NSEI", as_of=date(2026, 1, 1))
        self.assertEqual(report["Missing_Volume_Count"], len(frame))
        self.assertEqual(report["Quality_Status"], "PASS")

    def test_zero_volume_stock_rows_are_reported(self):
        frame = sample_bars()
        frame.loc[frame.index[:8], "Volume"] = 0
        frame.loc[frame.index[:2], ["Open", "High", "Low", "Close"]] = 100
        report = assess_market_data(frame, "RELIANCE.NS", as_of=date(2026, 1, 1))
        self.assertEqual(report["Zero_Volume_Count"], 8)
        self.assertEqual(report["Flat_Zero_Volume_Bar_Count"], 2)
        self.assertEqual(report["Quality_Status"], "WARN")

    def test_future_market_date_is_flagged(self):
        frame = sample_bars(count=2, start="2025-01-01")
        report = assess_market_data(frame, "TCS.NS", as_of=date(2024, 12, 31))
        self.assertEqual(report["Quality_Status"], "WARN")
        self.assertIn("later than as-of date", report["Warnings"])

    def test_structurally_invalid_ohlc_is_rejected(self):
        frame = sample_bars()
        frame.iloc[3, frame.columns.get_loc("High")] = 50
        with self.assertRaisesRegex(ValueError, "Invalid OHLCV"):
            assess_market_data(frame, "TCS.NS", as_of=date(2026, 1, 1))

    def test_quality_reports_save_archived_and_latest_files(self):
        report = assess_market_data(sample_bars(), "TCS.NS", as_of=date(2026, 1, 1))
        with tempfile.TemporaryDirectory() as folder:
            archived, latest = save_quality_reports(
                [report], folder, run_label="test_quality", timestamp="20260101_120000"
            )
            self.assertTrue(pd.io.common.file_exists(archived))
            self.assertTrue(pd.io.common.file_exists(latest))
            loaded = pd.read_csv(latest)
            self.assertEqual(loaded.iloc[0]["Symbol"], "TCS.NS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
