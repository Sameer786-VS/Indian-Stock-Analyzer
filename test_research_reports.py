"""Unit tests for Stage 3.7 research report generation."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research_reports import (
    create_stock_chart,
    generate_scan_report,
    generate_stock_report,
    safe_filename,
)


def sample_history(rows: int = 230) -> pd.DataFrame:
    index = pd.bdate_range("2025-12-01", periods=rows)
    close = pd.Series(np.linspace(100, 142, rows) + np.sin(np.arange(rows) / 5), index=index)
    frame = pd.DataFrame(index=index)
    frame["Close"] = close
    frame["Open"] = close * .998
    frame["High"] = close * 1.01
    frame["Low"] = close * .99
    frame["Volume"] = 100000 + (np.arange(rows) % 17) * 3000
    frame["SMA_20"] = close.rolling(20).mean()
    frame["SMA_50"] = close.rolling(50).mean()
    frame["SMA_200"] = close.rolling(200).mean()
    frame["RSI_14"] = 55 + np.sin(np.arange(rows) / 7) * 8
    frame["ATR_14"] = 1.5
    frame["ATR_Pct"] = frame["ATR_14"] / close * 100
    frame["Support_60D"] = close.rolling(60).min()
    frame["Resistance_60D"] = close.rolling(60).max()
    frame["Avg_Volume_20D"] = frame["Volume"].rolling(20).mean()
    frame["Relative_Volume_20D"] = frame["Volume"] / frame["Avg_Volume_20D"]
    frame["Daily_Return_Pct"] = close.pct_change() * 100
    frame["Return_5D_Pct"] = close.pct_change(5) * 100
    frame["Return_20D_Pct"] = close.pct_change(20) * 100
    frame["Relative_Strength_20D_Pct"] = 2.4
    return frame


class TestResearchReports(unittest.TestCase):
    def test_safe_filename(self):
        self.assertEqual(safe_filename("HDFC Bank.NS"), "HDFC_BANK_NS")
        self.assertEqual(safe_filename("***"), "REPORT")

    def test_create_stock_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stock.png"
            result = create_stock_chart("TEST.NS", sample_history(), path)
            self.assertEqual(result, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_generate_stock_report_creates_archive_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {
                "Symbol": "TEST.NS", "Company_Name": "Test Company",
                "Setup_Score": 4, "Score_Close_Above_SMA20": 1,
                "Score_SMA20_Above_SMA50": 1, "Score_Close_Above_SMA200": 1,
                "Score_Positive_Return_20D": 1, "Score_RSI_50_70": 0,
                "Score_Breakout_Bonus": 0, "Score_Breakdown_Penalty": 0,
                "Research_Trend_State": "Positive alignment",
                "Research_Conclusion": "Mixed research evidence",
                "Stock_vs_Sector_20D_Pct": 1.2,
                "Peer_Comparison_Quality": "Adequate sample",
            }
            paths = generate_stock_report(
                "TEST.NS", sample_history(), record,
                root / "reports", root / "charts", timestamp="20261111_120000",
            )
            for key in ("archive_html", "latest_html", "archive_chart", "latest_chart"):
                self.assertTrue(paths[key].exists(), key)
            content = paths["latest_html"].read_text(encoding="utf-8")
            self.assertIn("Original score breakdown", content)
            self.assertIn("not a probability", content)
            self.assertIn("stock_TEST_NS_latest.png", content)

    def test_generate_nifty_scan_report_creates_sector_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame([
                {"Symbol": "AAA.NS", "Company_Name": "AAA Ltd", "Close": 100,
                 "Return_20D_Pct": 8.0, "Relative_Strength_20D_Pct": 2.0,
                 "Close_Above_SMA20": True, "Setup_Score": 4},
                {"Symbol": "BBB.NS", "Company_Name": "BBB Ltd", "Close": 80,
                 "Return_20D_Pct": 3.0, "Relative_Strength_20D_Pct": -1.0,
                 "Close_Above_SMA20": False, "Setup_Score": 2},
            ])
            sectors = pd.DataFrame([
                {"NSE_Industry": "IT", "Sector_Rank_20D": 1,
                 "Scanned_Constituents": 4, "Coverage_Pct": 100,
                 "Sector_Return_20D_Pct": 9.0,
                 "Sector_Relative_Strength_20D_Pct": 3.0,
                 "Strength_Status": "Outperforming"},
            ])
            paths = generate_scan_report(
                frame, sectors, root / "reports", root / "charts",
                scan_label="NIFTY 50", timestamp="20261111_120001",
            )
            self.assertTrue(paths["latest_html"].exists())
            self.assertTrue(paths["chart_sector"].exists())
            html = paths["latest_html"].read_text(encoding="utf-8")
            self.assertIn("Industry-group strength", html)
            self.assertIn("not official NSE sector-index", html)


if __name__ == "__main__":
    unittest.main()
