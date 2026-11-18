"""Offline tests for the dated NIFTY 50 constituent registry."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from nifty50_universe import (
    DEFAULT_UNIVERSE_PATH,
    build_nifty50_alias_map,
    load_nifty50_symbols,
    load_nifty50_universe,
    lookup_nifty50_symbol,
)


class Nifty50UniverseTests(unittest.TestCase):
    def test_bundled_universe_has_50_unique_constituents(self):
        rows = load_nifty50_universe()
        symbols = [row["Yahoo_Symbol"] for row in rows]
        self.assertEqual(len(rows), 50)
        self.assertEqual(len(set(symbols)), 50)
        self.assertEqual(len(set(row["NSE_Symbol"] for row in rows)), 50)
        self.assertTrue(all(symbol.endswith(".NS") for symbol in symbols))

    def test_all_rows_have_one_snapshot_date_and_sector(self):
        rows = load_nifty50_universe()
        self.assertEqual({row["Constituent_As_Of"] for row in rows}, {"2026-10-28"})
        self.assertTrue(all(row["NSE_Industry"].strip() for row in rows))
        self.assertTrue(all(row["Index_Name"] == "NIFTY 50" for row in rows))

    def test_names_and_symbols_resolve_to_yahoo_tickers(self):
        aliases = build_nifty50_alias_map()
        self.assertEqual(lookup_nifty50_symbol("Coal India", aliases), "COALINDIA.NS")
        self.assertEqual(lookup_nifty50_symbol("TCS", aliases), "TCS.NS")
        self.assertEqual(lookup_nifty50_symbol("M&M", aliases), "M&M.NS")
        self.assertEqual(lookup_nifty50_symbol("Mahindra and Mahindra", aliases), "M&M.NS")
        self.assertEqual(lookup_nifty50_symbol("Tata Motors", aliases), "TMPV.NS")
        self.assertEqual(lookup_nifty50_symbol("TMPV", aliases), "TMPV.NS")

    def test_symbol_list_matches_universe_order(self):
        rows = load_nifty50_universe()
        self.assertEqual(load_nifty50_symbols(), [row["Yahoo_Symbol"] for row in rows])

    def test_duplicate_symbol_is_rejected(self):
        rows = load_nifty50_universe()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "duplicate.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                rows[1]["Yahoo_Symbol"] = rows[0]["Yahoo_Symbol"]
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "duplicate Yahoo symbols"):
                load_nifty50_universe(path)

    def test_missing_constituent_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_nifty50_universe(DEFAULT_UNIVERSE_PATH.parent / "missing.csv")

    def test_unexpected_row_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 49"):
            load_nifty50_universe(expected_count=49)


if __name__ == "__main__":
    unittest.main(verbosity=2)
