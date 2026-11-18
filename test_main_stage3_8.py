"""Offline tests for the Stage 3.8 persistence integration helpers."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import db_manager_v2 as db
import main_stage3_8 as stage38
from test_db_manager_v2 import SCHEMA


class Stage38PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "stage38_test.db"
        connection = sqlite3.connect(db.DB_PATH)
        connection.executescript(SCHEMA)
        connection.commit()
        connection.close()
        stage38.PERSISTENCE_STATS.update({
            "stock_price_rows": 0,
            "benchmark_price_rows": 0,
            "research_rows": 0,
            "scan_rows": 0,
            "sector_rows": 0,
            "write_errors": 0,
        })

    def tearDown(self):
        db.DB_PATH = self.old_path
        self.temp_dir.cleanup()

    @staticmethod
    def sample_frame():
        frame = pd.DataFrame(
            [{"Open": 100.0, "High": 105.0, "Low": 99.0,
              "Close": 103.0, "Volume": 500}],
            index=pd.to_datetime(["2026-11-16"]),
        )
        frame.attrs["market_data"] = {
            "source": "yahoo_chart_api",
            "downloaded_at_utc": "2026-11-17T00:00:00+00:00",
        }
        return frame

    def test_stock_history_persists_with_nse_yahoo_mapping(self):
        count = stage38.persist_downloaded_data("TCS.NS", self.sample_frame())
        self.assertEqual(count, 1)
        row = db.get_stock_prices("TCS", include_provenance=True)[0]
        self.assertEqual(row[0], "TCS")
        self.assertEqual(row[7], stage38.DATABASE_PRICE_BASIS)
        self.assertEqual(row[8], "yahoo_chart_api")
        self.assertEqual(db.get_all_stocks()[0][1], "TCS")
        self.assertEqual(db.get_all_stocks()[0][5], "TCS.NS")

    def test_benchmark_is_saved_separately_from_stocks(self):
        count = stage38.persist_downloaded_data("^NSEI", self.sample_frame())
        self.assertEqual(count, 1)
        with db.get_connection() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM benchmark_prices").fetchone()[0], 1
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0], 0
            )

    def test_scan_research_and_sector_outputs_persist(self):
        scanner = pd.DataFrame([{
            "Symbol": "TCS.NS", "Latest_Date": "2026-11-16",
            "Company_Name": "Tata Consultancy Services", "NSE_Industry": "IT",
            "Close": 103.0, "RSI_14": 55.0, "SMA_20": 100.0,
            "SMA_50": 98.0, "SMA_200": 90.0,
            "Relative_Strength_20D_Pct": 2.1, "Setup_Score": 5,
            "Trend": "Positive alignment", "Assessment": "Monitor",
            "Breakout_Confirmed": False, "High_Volume_Session": True,
            "Data_Source": "yahoo_chart_api",
            "Price_Basis": stage38.DATABASE_PRICE_BASIS,
        }])
        sectors = pd.DataFrame([{
            "NSE_Industry": "IT", "Sector_Return_20D_Pct": 3.0,
            "Sector_Relative_Strength_20D_Pct": 1.2,
            "Strength_Status": "Outperforming NIFTY 50",
            "Scanned_Constituents": 1, "Expected_Constituents": 2,
            "Coverage_Pct": 50.0, "Coverage_Status": "PARTIAL",
            "Sector_Rank_20D": 1.0,
        }])
        saved = stage38.persist_scan_outputs(
            scanner, sectors, "Test NIFTY scan", "2026-11-17"
        )
        self.assertEqual(saved, {"scan_rows": 1, "research_rows": 1, "sector_rows": 1})
        with db.get_connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM research_results").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM sector_data").fetchone()[0], 1)
            saved_date = connection.execute(
                "SELECT analysis_date FROM sector_data WHERE sector_name = 'IT'"
            ).fetchone()[0]
            self.assertEqual(saved_date, "2026-11-17")

    def test_sector_analysis_date_comes_from_latest_market_data_date(self):
        scanner = pd.DataFrame([
            {"Latest_Date": "2026-11-16"},
            {"Latest_Date": "2026-11-17"},
        ])
        resolved = stage38._resolve_sector_analysis_date(scanner)
        self.assertEqual(resolved, "2026-11-17")

    def test_sector_analysis_date_requires_market_data_date(self):
        scanner = pd.DataFrame([{"Symbol": "TCS.NS"}])
        with self.assertRaisesRegex(ValueError, "Latest_Date"):
            stage38._resolve_sector_analysis_date(scanner)

    def test_empty_scan_does_not_write_records(self):
        empty = pd.DataFrame()
        saved = stage38.persist_scan_outputs(empty)
        self.assertEqual(saved, {"scan_rows": 0, "research_rows": 0, "sector_rows": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
