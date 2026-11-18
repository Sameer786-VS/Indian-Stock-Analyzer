"""Offline tests for db_manager_v2.py; uses a temporary SQLite database."""
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

import db_manager_v2 as db


SCHEMA = """
CREATE TABLE stocks (
 stock_id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL UNIQUE,
 company_name TEXT NOT NULL, exchange TEXT DEFAULT 'NSE', industry TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP, yahoo_symbol TEXT
);
CREATE UNIQUE INDEX idx_stocks_yahoo_symbol ON stocks(yahoo_symbol)
 WHERE yahoo_symbol IS NOT NULL AND TRIM(yahoo_symbol) <> '';
CREATE TABLE historical_prices (
 price_id INTEGER PRIMARY KEY AUTOINCREMENT, stock_id INTEGER NOT NULL,
 price_date TEXT NOT NULL, open_price REAL NOT NULL, high_price REAL NOT NULL,
 low_price REAL NOT NULL, close_price REAL NOT NULL, volume INTEGER DEFAULT 0,
 price_basis TEXT, data_source TEXT, retrieved_at_utc TEXT,
 FOREIGN KEY(stock_id) REFERENCES stocks(stock_id), UNIQUE(stock_id,price_date)
);
CREATE TABLE benchmarks (
 benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
 benchmark_symbol TEXT NOT NULL UNIQUE, benchmark_name TEXT NOT NULL,
 exchange TEXT DEFAULT 'NSE', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 provider_symbol TEXT
);
CREATE TABLE benchmark_prices (
 benchmark_price_id INTEGER PRIMARY KEY AUTOINCREMENT, benchmark_id INTEGER NOT NULL,
 price_date TEXT NOT NULL, open_price REAL NOT NULL, high_price REAL NOT NULL,
 low_price REAL NOT NULL, close_price REAL NOT NULL, volume INTEGER DEFAULT 0,
 price_basis TEXT, data_source TEXT, retrieved_at_utc TEXT,
 FOREIGN KEY(benchmark_id) REFERENCES benchmarks(benchmark_id),
 UNIQUE(benchmark_id,price_date)
);
CREATE TABLE research_results (
 research_id INTEGER PRIMARY KEY AUTOINCREMENT, stock_id INTEGER NOT NULL,
 analysis_date TEXT NOT NULL, close_price REAL, rsi REAL, sma_20 REAL,
 sma_50 REAL, sma_200 REAL, relative_strength REAL, research_score REAL,
 trend TEXT, recommendation TEXT, data_source TEXT, price_basis TEXT,
 FOREIGN KEY(stock_id) REFERENCES stocks(stock_id)
);
CREATE TABLE scan_results (
 scan_id INTEGER PRIMARY KEY AUTOINCREMENT, stock_id INTEGER NOT NULL,
 scan_date TEXT NOT NULL, signal_type TEXT, signal_strength REAL,
 close_price REAL, breakout_detected INTEGER DEFAULT 0,
 volume_confirmation INTEGER DEFAULT 0, notes TEXT, data_source TEXT,
 price_basis TEXT, FOREIGN KEY(stock_id) REFERENCES stocks(stock_id)
);
CREATE TABLE sector_data (
 sector_id INTEGER PRIMARY KEY AUTOINCREMENT, sector_name TEXT NOT NULL,
 analysis_date TEXT NOT NULL, average_return REAL, relative_strength REAL,
 strength_score REAL, trend TEXT, stocks_count INTEGER DEFAULT 0, notes TEXT,
 data_source TEXT, price_basis TEXT, UNIQUE(sector_name,analysis_date)
);
CREATE TABLE execution_prices (
 execution_price_id INTEGER PRIMARY KEY AUTOINCREMENT,
 stock_id INTEGER NOT NULL, price_date TEXT NOT NULL, open_price REAL NOT NULL,
 high_price REAL NOT NULL, low_price REAL NOT NULL, close_price REAL NOT NULL,
 volume INTEGER DEFAULT 0, price_basis TEXT NOT NULL DEFAULT 'as_traded_execution',
 data_source TEXT NOT NULL, retrieved_at_utc TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 FOREIGN KEY(stock_id) REFERENCES stocks(stock_id), UNIQUE(stock_id,price_date)
);
CREATE TABLE schema_migrations (
 version INTEGER PRIMARY KEY, description TEXT NOT NULL,
 applied_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
PRAGMA user_version=2;
"""


class DatabaseManagerV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_path = db.DB_PATH
        db.DB_PATH = Path(self.temp_dir.name) / "test.db"
        con = sqlite3.connect(db.DB_PATH)
        con.executescript(SCHEMA)
        con.execute("INSERT INTO stocks(symbol,company_name,exchange,industry,yahoo_symbol) VALUES('TCS','TCS','NSE','IT','TCS.NS')")
        con.commit()
        con.close()

    def tearDown(self):
        db.DB_PATH = self.old_path
        self.temp_dir.cleanup()

    def test_connection_and_schema_version(self):
        db.test_connection()

    def test_upsert_stock_history_and_provenance(self):
        frame = pd.DataFrame(
            [{"Open": 100, "High": 105, "Low": 99, "Close": 103, "Volume": 500}],
            index=pd.to_datetime(["2026-11-16"]),
        )
        n = db.save_stock_history("TCS.NS", frame,
            metadata={"NSE_Symbol":"TCS", "Company_Name":"Tata Consultancy Services", "NSE_Industry":"IT"},
            source_info={"source":"yahoo_chart_api", "price_basis":"adjusted analytical OHLC", "downloaded_at_utc":"2026-11-17T00:00:00+00:00"})
        self.assertEqual(n, 1)
        row = db.get_stock_prices("TCS", include_provenance=True)[0]
        self.assertEqual(row[1], "2026-11-16")
        self.assertEqual(row[7], "adjusted analytical OHLC")
        self.assertEqual(row[8], "yahoo_chart_api")

    def test_reject_nan_and_infinity_prices(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            db.save_historical_price("TCS", "2026-11-16", float("nan"), 105, 99, 103)
        with self.assertRaisesRegex(ValueError, "finite"):
            db.save_historical_price("TCS", "2026-11-16", 100, float("inf"), 99, 103)

    def test_reject_bad_ohlc_and_volume(self):
        with self.assertRaisesRegex(ValueError, "High price"):
            db.save_historical_price("TCS", "2026-11-16", 100, 99, 98, 101)
        with self.assertRaisesRegex(ValueError, "whole number"):
            db.save_historical_price("TCS", "2026-11-16", 100, 105, 99, 103, 1.5)
        with self.assertRaisesRegex(ValueError, "finite"):
            db.save_historical_price("TCS", "2026-11-16", 100, 105, 99, 103, float("inf"))

    def test_invalid_date_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid YYYY-MM-DD"):
            db.save_historical_price("TCS", "2026-02-30", 100, 105, 99, 103)

    def test_duplicate_price_date_is_upserted(self):
        db.save_historical_price("TCS", "2026-11-16", 100, 105, 99, 103, 500)
        db.save_historical_price("TCS", "2026-11-16", 101, 106, 100, 104, 600)
        with db.get_connection() as con:
            count = con.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0]
            close = con.execute("SELECT close_price FROM historical_prices").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(close, 104)

    def test_benchmark_history_goes_to_benchmark_table(self):
        frame = pd.DataFrame(
            [{"Open": 20000, "High": 20100, "Low": 19900, "Close": 20050, "Volume": None}],
            index=pd.to_datetime(["2026-11-16"]),
        )
        self.assertEqual(db.save_benchmark_history(frame), 1)
        with db.get_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM benchmark_prices").fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0], 0)

    def test_research_and_scan_results_append(self):
        record = {"Symbol":"TCS.NS", "Latest_Date":"2026-11-16", "Close":103,
                  "RSI_14":55, "SMA_20":100, "SMA_50":98, "SMA_200":90,
                  "Relative_Strength_20D_Pct":2.1, "Setup_Score":5,
                  "Trend":"Positive alignment", "Assessment":"Monitor",
                  "Data_Source":"yahoo_chart_api", "Price_Basis":"adjusted analytical OHLC"}
        db.save_research_result(record, metadata={"NSE_Symbol":"TCS", "Company_Name":"TCS"})
        self.assertEqual(db.save_scan_results([record], "unit test"), 1)
        with db.get_connection() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM research_results").fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0], 1)

    def test_sector_results_upsert(self):
        rows = [{"NSE_Industry":"IT", "Sector_Return_20D_Pct":3.0,
                 "Sector_Relative_Strength_20D_Pct":1.2,
                 "Strength_Status":"Outperforming NIFTY 50",
                 "Scanned_Constituents":5, "Expected_Constituents":6,
                 "Coverage_Pct":83.33, "Coverage_Status":"PARTIAL",
                 "Sector_Rank_20D":float("nan")}]
        db.save_sector_results(rows, "2026-11-16", {"source":"scanner"})
        db.save_sector_results(rows, "2026-11-16", {"source":"scanner"})
        with db.get_connection() as con:
            count = con.execute("SELECT COUNT(*) FROM sector_data").fetchone()[0]
        self.assertEqual(count, 1)

    def test_missing_volume_is_allowed(self):
        db.save_historical_price("TCS", "2026-11-16", 100, 105, 99, 103, None)
        with db.get_connection() as con:
            volume = con.execute("SELECT volume FROM historical_prices").fetchone()[0]
        self.assertIsNone(volume)


if __name__ == "__main__":
    unittest.main(verbosity=2)
