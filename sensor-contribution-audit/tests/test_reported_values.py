from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensor_contribution_audit.reported import reported_values


class ReportedTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.report = reported_values()
        self.tables = self.report["tables"]

    def test_explicit_source_and_statistical_scope(self):
        self.assertTrue(self.report["reported_values_only"])
        self.assertFalse(self.report["new_statistics"])
        self.assertEqual(len(self.report["source_map"]["source_articles"]), 2)
        self.assertTrue(all(s["article_licence"] == "CC BY 4.0" for s in self.report["source_map"]["source_articles"]))

    def test_author_reported_cnr_ratios(self):
        rows = self.tables["Ortega2022_HbO_CNR_reported.csv"]
        self.assertEqual([r["reported_fold_change"] for r in rows], ["3.94", "3.91"])
        self.assertEqual([r["after_normalised"] for r in rows], ["3.94", "3.91"])
        self.assertTrue(all(r["normalisation_rule"] == "use_author_reported_fold_not_recompute_from_rounded_means" for r in rows))

    def test_concentration_and_od_branches_remain_separate(self):
        concentration = self.tables["Ortega2022_classification_tests_reported.csv"]
        od = self.tables["Ortega2022_OD_tests_context_only.csv"]
        self.assertEqual([r["reported_p"] for r in concentration], ["0.056", "0.17"])
        self.assertEqual([r["reported_p"] for r in od], ["0.03", "0.15"])
        self.assertTrue(all(r["new_statistics"] == "false" for r in concentration))

    def test_separate_peak_windows_are_preserved(self):
        rows = self.tables["Shin2018_WG_peaks_reported.csv"]
        self.assertEqual([(r["reported_peak_accuracy_percent"], r["peak_window_right_end_s"]) for r in rows], [("76.9", "10"), ("74.3", "14"), ("80.7", "11")])
        self.assertTrue(all(r["same_latency_comparison"] == "false" for r in rows))
        self.assertTrue(all(r["window_width_s"] == "5" for r in rows))


if __name__ == "__main__":
    unittest.main()
