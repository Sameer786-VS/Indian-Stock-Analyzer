"""Offline tests for equal-weighted NIFTY 50 industry-strength analysis."""
import unittest

import numpy as np
import pandas as pd

from sector_strength import build_sector_strength_report


class SectorStrengthTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "AAA.NS": {"NSE_Industry": "Technology"},
            "BBB.NS": {"NSE_Industry": "Technology"},
            "CCC.NS": {"NSE_Industry": "Financial Services"},
            "DDD.NS": {"NSE_Industry": "Healthcare"},  # intentionally not scanned
        }
        self.results = pd.DataFrame([
            {
                "Symbol": "AAA.NS", "NSE_Industry": "Wrong label",
                "Daily_Return_Pct": 2.0, "Return_5D_Pct": 5.0,
                "Return_20D_Pct": 10.0, "NIFTY50_Return_20D_Pct": 4.0,
                "Close_Above_SMA20": True, "Breakout_Confirmed": True,
            },
            {
                "Symbol": "BBB.NS", "NSE_Industry": "Wrong label",
                "Daily_Return_Pct": -1.0, "Return_5D_Pct": 1.0,
                "Return_20D_Pct": 2.0, "NIFTY50_Return_20D_Pct": 4.0,
                "Close_Above_SMA20": False, "Breakout_Confirmed": False,
            },
            {
                "Symbol": "CCC.NS", "NSE_Industry": "Wrong label",
                "Daily_Return_Pct": 0.5, "Return_5D_Pct": -1.0,
                "Return_20D_Pct": 1.0, "NIFTY50_Return_20D_Pct": 4.0,
                "Close_Above_SMA20": True, "Breakout_Confirmed": False,
            },
        ])

    def test_sector_returns_are_equal_weighted_and_relative_to_nifty(self):
        stocks, sectors = build_sector_strength_report(self.results, self.metadata)
        tech = sectors.set_index("NSE_Industry").loc["Technology"]
        self.assertAlmostEqual(tech["Sector_Return_20D_Pct"], 6.0)
        self.assertAlmostEqual(tech["Sector_Relative_Strength_20D_Pct"], 2.0)
        self.assertEqual(tech["Strength_Status"], "Outperforming NIFTY 50")

    def test_stock_is_compared_with_its_own_sector(self):
        stocks, _ = build_sector_strength_report(self.results, self.metadata)
        a = stocks.set_index("Symbol").loc["AAA.NS"]
        b = stocks.set_index("Symbol").loc["BBB.NS"]
        self.assertAlmostEqual(a["Stock_vs_Sector_20D_Pct"], 4.0)
        self.assertEqual(a["Stock_vs_Sector_Status"], "Outperforming own industry")
        self.assertAlmostEqual(b["Stock_vs_Sector_20D_Pct"], -4.0)
        self.assertEqual(b["Stock_vs_Sector_Status"], "Underperforming own industry")

    def test_coverage_uses_full_universe_expected_members(self):
        _, sectors = build_sector_strength_report(self.results, self.metadata)
        tech = sectors.set_index("NSE_Industry").loc["Technology"]
        self.assertEqual(tech["Expected_Constituents"], 2)
        self.assertEqual(tech["Scanned_Constituents"], 2)
        self.assertEqual(tech["Coverage_Pct"], 100.0)
        self.assertEqual(tech["Coverage_Status"], "FULL")

    def test_unscanned_sector_is_retained_with_no_data(self):
        _, sectors = build_sector_strength_report(self.results, self.metadata)
        healthcare = sectors.set_index("NSE_Industry").loc["Healthcare"]
        self.assertEqual(healthcare["Expected_Constituents"], 1)
        self.assertEqual(healthcare["Scanned_Constituents"], 0)
        self.assertEqual(healthcare["Coverage_Status"], "NO DATA")
        self.assertTrue(np.isnan(healthcare["Sector_Return_20D_Pct"]))

    def test_sector_advancers_and_breadth_counts(self):
        _, sectors = build_sector_strength_report(self.results, self.metadata)
        tech = sectors.set_index("NSE_Industry").loc["Technology"]
        self.assertEqual(tech["Advancers_1D_Count"], 1)
        self.assertEqual(tech["Positive_Return_20D_Count"], 2)
        self.assertEqual(tech["Outperforming_NIFTY_20D_Count"], 1)
        self.assertEqual(tech["Above_SMA20_Count"], 1)
        self.assertEqual(tech["Confirmed_Breakout_Count"], 1)

    def test_stock_industry_is_taken_from_constituent_metadata(self):
        stocks, _ = build_sector_strength_report(self.results, self.metadata)
        self.assertEqual(
            stocks.set_index("Symbol").loc["AAA.NS", "NSE_Industry"],
            "Technology",
        )

    def test_duplicate_stock_rows_are_rejected(self):
        duplicate = pd.concat([self.results, self.results.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate stock rows"):
            build_sector_strength_report(duplicate, self.metadata)

    def test_missing_required_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_sector_strength_report(self.results.drop(columns=["Return_20D_Pct"]), self.metadata)

    def test_empty_universe_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Full NIFTY 50 universe metadata"):
            build_sector_strength_report(self.results, {})

    def test_missing_values_do_not_crash_sector_calculation(self):
        sample = self.results.copy()
        sample.loc[0, "Return_20D_Pct"] = np.nan
        stocks, sectors = build_sector_strength_report(sample, self.metadata)
        tech = sectors.set_index("NSE_Industry").loc["Technology"]
        self.assertAlmostEqual(tech["Sector_Return_20D_Pct"], 2.0)
        self.assertTrue(np.isnan(stocks.set_index("Symbol").loc["AAA.NS", "Stock_vs_Sector_20D_Pct"]))

    def test_sector_summary_is_ranked_by_relative_strength(self):
        _, sectors = build_sector_strength_report(self.results, self.metadata)
        ranks = sectors.dropna(subset=["Sector_Rank_20D"]).set_index("NSE_Industry")["Sector_Rank_20D"]
        self.assertLess(ranks["Technology"], ranks["Financial Services"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
