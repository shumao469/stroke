from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sensor_contribution_audit.cli import main
from sensor_contribution_audit.plotting import write_svg, write_pdf, write_png


class CommandLineTests(unittest.TestCase):
    def test_checkout_entry_point(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_audit.py"), "validate"], check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["reference_cells"], 840)
        self.assertTrue(report["reference_grid_pass"])

    def test_dimension_cli(self):
        result = subprocess.run([sys.executable, str(ROOT / "run_audit.py"), "dimensions", "--d-x", "12", "--d-z", "8"], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["affine_total_parameter_count"], 21)

    def test_deterministic_core_outputs(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            self.assertEqual(main(["generate", "--out", first]), 0)
            self.assertEqual(main(["generate", "--out", second]), 0)
            for name in ("gaussian_relative_jitter_heatmap.csv", "retention_0p9_contour.csv", "analytical_summary.json", "timing_budget.svg"):
                self.assertEqual((Path(first) / name).read_bytes(), (Path(second) / name).read_bytes())

    def test_svg_is_editable_vector(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "figure.svg"
            write_svg(path)
            root = ET.parse(path).getroot()
            ns = {"s": "http://www.w3.org/2000/svg"}
            self.assertEqual(len(root.findall("s:rect", ns)), 941)
            self.assertEqual(len(root.findall("s:image", ns)), 0)
            self.assertGreater(len(root.findall("s:text", ns)), 10)
            self.assertEqual(root.attrib["width"], "180mm")


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("reportlab", "pypdfium2", "PIL")), "Optional plotting dependencies are unavailable")
class OptionalRenderingTests(unittest.TestCase):
    def test_pdf_and_png_are_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            for stem in ("first", "second"):
                write_pdf(p / f"{stem}.pdf")
                write_png(p / f"{stem}.pdf", p / f"{stem}.png")
            self.assertEqual((p / "first.pdf").read_bytes(), (p / "second.pdf").read_bytes())
            self.assertEqual((p / "first.png").read_bytes(), (p / "second.png").read_bytes())

    def test_png_has_print_resolution(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            write_pdf(p / "figure.pdf")
            write_png(p / "figure.pdf", p / "figure.png")
            with Image.open(p / "figure.png") as image:
                self.assertGreaterEqual(image.width, 2125)
                self.assertGreaterEqual(image.height, 1417)
                self.assertAlmostEqual(image.info["dpi"][0], 300, places=1)


if __name__ == "__main__":
    unittest.main()
