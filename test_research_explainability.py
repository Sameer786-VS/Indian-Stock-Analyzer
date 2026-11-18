"""Offline tests for Stage 3.6 explainability and peer comparisons."""
import unittest

import numpy as np
import pandas as pd

from research_explainability import (
    add_peer_comparisons,
    enrich_scan_record,
)


class ExplainabilityTests(unittest.TestCase):
    def sample_record(self):
        return {
            "Symbol": "AAA.NS",
            "Close": 120.0,
            "SMA_20": 115.0,
            "SMA_50": 110.0,
            "SMA_200": 100.0,
            "Return_20D_Pct": 8.0,
            "RSI_14": 62.0,
            "Relative_Strength_20D_Pct": 3.0,
            "Relative_Volume_20D": 1.7,
            "Breakout_60D": True,
            "Breakout_Confirmed": True,
            "Breakdown_60D": False,
            "Breakdown_Confirmed": False,
            "Setup_Score": 7,
            "Score_Close_Above_SMA20": 1,
            "Score_SMA20_Above_SMA50": 1,
            "Score_Close_Above_SMA200": 1,
            "Score_Positive_Return_20D": 1,
            "Score_RSI_50_70": 1,
            "Score_Breakout_Bonus": 2,
            "Score_Breakdown_Penalty": 0,
        }

    def test_explainability_does_not_change_original_setup_score(self):
        record = enrich_scan_record(self.sample_record())
        self.assertEqual(record["Setup_Score"], 7)
        self.assertEqual(record["Core_Rules_Passed"], 5)
        self.assertIn("5/5", record["Core_Score_Explanation"])

    def test_positive_research_profile_is_explained(self):
        record = enrich_scan_record(self.sample_record())
        self.assertEqual(record["Research_Conclusion"], "Constructive research profile")
        self.assertIn("outperforming NIFTY 50", record["Research_Positive_Evidence"])
        self.assertEqual(record["Research_Trend_State"], "Positive long-term alignment")

    def test_breakdown_is_flagged_as_downside_risk(self):
        sample = self.sample_record()
        sample["Breakdown_Confirmed"] = True
        record = enrich_scan_record(sample)
        self.assertEqual(
            record["Research_Conclusion"],
            "Downside-risk evidence present — review carefully",
        )

    def test_missing_inputs_are_reported_not_fabricated(self):
        sample = self.sample_record()
        sample["SMA_200"] = np.nan
        sample["RSI_14"] = np.nan
        record = enrich_scan_record(sample)
        self.assertIn("SMA 200 comparison", record["Research_Missing_Evidence"])
        self.assertIn("RSI 14", record["Research_Missing_Evidence"])
        self.assertLess(record["Core_Rule_Completeness_Pct"], 100)

    def test_high_rsi_is_caution_not_positive_evidence(self):
        sample = self.sample_record()
        sample["RSI_14"] = 78
        record = enrich_scan_record(sample)
        self.assertIn("RSI above 70", record["Research_Caution_Evidence"])
        self.assertNotIn("RSI between 50 and 70", record["Research_Positive_Evidence"])

    def test_underperforming_benchmark_is_caution(self):
        sample = self.sample_record()
        sample["Relative_Strength_20D_Pct"] = -2
        record = enrich_scan_record(sample)
        self.assertEqual(record["Research_Benchmark_State"], "Underperforming NIFTY 50")
        self.assertIn("underperforming NIFTY 50", record["Research_Caution_Evidence"])

    def test_missing_relative_volume_is_reported(self):
        sample = self.sample_record()
        sample["Relative_Volume_20D"] = np.nan
        record = enrich_scan_record(sample)
        self.assertEqual(record["Research_Volume_State"], "Unavailable")
        self.assertIn("relative volume", record["Research_Missing_Evidence"])


class PeerComparisonTests(unittest.TestCase):
    def setUp(self):
        self.stocks = pd.DataFrame([
            {"Symbol": "AAA.NS", "NSE_Industry": "Technology", "Return_20D_Pct": 12.0},
            {"Symbol": "BBB.NS", "NSE_Industry": "Technology", "Return_20D_Pct": 8.0},
            {"Symbol": "CCC.NS", "NSE_Industry": "Technology", "Return_20D_Pct": 4.0},
            {"Symbol": "DDD.NS", "NSE_Industry": "Healthcare", "Return_20D_Pct": 6.0},
        ])
        self.sectors = pd.DataFrame([
            {"NSE_Industry": "Technology", "Expected_Constituents": 4, "Scanned_Constituents": 3},
            {"NSE_Industry": "Healthcare", "Expected_Constituents": 1, "Scanned_Constituents": 1},
        ])

    def test_peer_comparison_excludes_the_stock_itself(self):
        stocks, _ = add_peer_comparisons(self.stocks, self.sectors)
        rows = stocks.set_index("Symbol")
        self.assertAlmostEqual(rows.loc["AAA.NS", "Sector_Peer_Return_20D_Pct"], 6.0)
        self.assertAlmostEqual(rows.loc["AAA.NS", "Stock_vs_Sector_20D_Pct"], 6.0)

    def test_peer_count_is_correct(self):
        stocks, _ = add_peer_comparisons(self.stocks, self.sectors)
        rows = stocks.set_index("Symbol")
        self.assertEqual(rows.loc["AAA.NS", "Sector_Peer_Constituent_Count"], 2)
        self.assertEqual(rows.loc["DDD.NS", "Sector_Peer_Constituent_Count"], 0)

    def test_single_stock_industry_has_no_peer_return(self):
        stocks, _ = add_peer_comparisons(self.stocks, self.sectors)
        row = stocks.set_index("Symbol").loc["DDD.NS"]
        self.assertTrue(np.isnan(row["Sector_Peer_Return_20D_Pct"]))
        self.assertEqual(row["Stock_vs_Sector_Status"], "Unavailable")

    def test_small_industry_sample_is_flagged(self):
        _, sectors = add_peer_comparisons(self.stocks, self.sectors)
        technology = sectors.set_index("NSE_Industry").loc["Technology"]
        healthcare = sectors.set_index("NSE_Industry").loc["Healthcare"]
        self.assertIn("Partial coverage", technology["Industry_Sample_Interpretation"])
        self.assertIn("Single constituent", healthcare["Industry_Sample_Interpretation"])

    def test_peer_comparison_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            add_peer_comparisons(self.stocks.drop(columns=["Return_20D_Pct"]), self.sectors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
