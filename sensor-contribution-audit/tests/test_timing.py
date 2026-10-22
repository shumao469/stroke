from pathlib import Path
import sys
import unittest
import math
import hashlib
import json
from importlib.resources import files

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensor_contribution_audit.timing import (
    coherent_amplitude, expected_mean_phasor, jitter_budget_ms,
    legacy_heatmap_grid, retention_contour,
)
from sensor_contribution_audit.validation import validate_reference
from sensor_contribution_audit.dimensions import dimension_audit


class TimingTests(unittest.TestCase):
    def test_zero_jitter_retains_unit_coherent_amplitude(self):
        for f in (0, 1, 10, 20, 40):
            self.assertEqual(coherent_amplitude(f, 0), 1)

    def test_zero_frequency_retains_unit_amplitude(self):
        self.assertEqual(coherent_amplitude(0, 20), 1)

    def test_explicit_milliseconds_conversion(self):
        expected = math.exp(-0.5 * (2 * math.pi * 20 * 0.005) ** 2)
        self.assertAlmostEqual(coherent_amplitude(20, 5), expected, places=15)

    def test_monotonic_frequency(self):
        values = [coherent_amplitude(f, 10) for f in range(1, 41)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_monotonic_jitter(self):
        values = [coherent_amplitude(20, s) for s in range(21)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_fixed_offset_preserves_expected_modulus(self):
        for offset in (-20, 0, 5, 100):
            self.assertAlmostEqual(abs(expected_mean_phasor(20, 5, offset)), coherent_amplitude(20, 5), places=15)

    def test_budget_substitution(self):
        for f in (1, 10, 20, 40):
            for eta in (0.5, 0.9, 1):
                self.assertAlmostEqual(coherent_amplitude(f, jitter_budget_ms(f, eta)), eta, places=14)

    def test_reported_budget_rounding(self):
        self.assertEqual(round(jitter_budget_ms(20), 2), 3.65)
        self.assertEqual(round(jitter_budget_ms(40), 2), 1.83)

    def test_exact_840_cell_grid(self):
        rows = legacy_heatmap_grid()
        self.assertEqual(len(rows), 840)
        self.assertEqual({(r["frequency_hz"], r["relative_jitter_sd_ms"]) for r in rows}, {(f, s) for f in range(1, 41) for s in range(21)})

    def test_analytic_contour(self):
        contour = retention_contour()
        self.assertEqual(len(contour), 200)
        self.assertAlmostEqual(contour[0]["relative_jitter_sd_ms"], 20, places=13)
        self.assertEqual(contour[-1]["frequency_hz"], 40)
        for r in contour:
            self.assertAlmostEqual(coherent_amplitude(r["frequency_hz"], r["relative_jitter_sd_ms"]), 0.9, places=14)

    def test_retained_analytical_reference(self):
        report = validate_reference()
        self.assertEqual(report["reference_cells"], 840)
        self.assertLessEqual(report["maximum_absolute_difference"], 1e-14)
        self.assertTrue(report["reference_grid_pass"])
        self.assertTrue(report["budget_substitution_pass"])

    def test_reference_provenance_hash(self):
        base = files("sensor_contribution_audit").joinpath("data")
        manifest = json.loads(base.joinpath("reference_manifest.json").read_text(encoding="utf-8"))
        actual = hashlib.sha256(base.joinpath(manifest["file"]).read_bytes()).hexdigest()
        self.assertEqual(actual, manifest["sha256"])

    def test_rejects_invalid_scalar_inputs(self):
        for f, s in [(-1, 0), (20, -1), (math.nan, 0), (1, math.inf), (True, 1), (1, "5")]:
            with self.subTest(f=f, s=s), self.assertRaises(ValueError):
                coherent_amplitude(f, s)

    def test_budget_domain(self):
        for f, eta in [(0, 0.9), (-1, 0.9), (1, 0), (1, 1.1), (1, math.nan)]:
            with self.subTest(f=f, eta=eta), self.assertRaises(ValueError):
                jitter_budget_ms(f, eta)

    def test_contour_sample_domain(self):
        for samples in (0, 1, True, 2.5):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                retention_contour(samples=samples)


class DimensionTests(unittest.TestCase):
    def test_affine_feature_and_parameter_dimensions(self):
        r = dimension_audit(12, 8)
        self.assertEqual(r["combined_feature_dimension"], 20)
        self.assertEqual(r["coefficient_dimensions"], {"w_x": 12, "w_z": 8})
        self.assertEqual(r["affine_total_parameter_count"], 21)
        self.assertEqual(r["score_dimension"], 1)

    def test_nonlinear_readout_dimension_is_explicit(self):
        r = dimension_audit(12, 8, 6, 4, 2, 2)["encoder_fusion_input"]
        self.assertEqual(r["declared_concatenated_readout_input_dimension"], 14)
        self.assertIsNone(r["encoder_and_fusion_parameter_count"])

    def test_dimension_domain(self):
        for a, b in [(0, 2), (2, -1), (True, 2), (1.5, 2)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                dimension_audit(a, b)

    def test_partial_latent_specification_rejected(self):
        with self.assertRaises(ValueError):
            dimension_audit(2, 3, h_x=2)
        with self.assertRaises(ValueError):
            dimension_audit(2, 3, quality=2)


if __name__ == "__main__":
    unittest.main()
