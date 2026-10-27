from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_checkout_info_is_explicit(self):
        run = subprocess.run([sys.executable, "-B", str(ROOT / "run_strokeweave.py"), "info"],
                             check=True, capture_output=True, text=True)
        value = json.loads(run.stdout)
        self.assertEqual(value["name"], "StrokeWeave")
        self.assertFalse(value["patient_execution"])
        self.assertFalse(value["training_enabled"])
        self.assertEqual(value["controller_superiority"], "not established")

    def test_checkout_demo_and_idempotent_repeat(self):
        with tempfile.TemporaryDirectory() as state:
            command = [sys.executable, "-B", str(ROOT / "run_strokeweave.py"), "demo", "--state-dir", state]
            first = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            second = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
            self.assertEqual((first["complete"], first["referred"]), (2, 3))
            self.assertEqual(first["batch"]["job_ids"], second["batch"]["job_ids"])
            self.assertTrue(second["batch"]["reused"])

    def test_lazy_import_does_not_require_scientific_stack(self):
        code = ("import sys; from strokeweave import numerics; "
                "assert not {'numpy','scipy','torch','prefect'}.intersection(sys.modules)")
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([sys.executable, "-B", "-c", code], check=True,
                       capture_output=True, text=True, env=environment)


if __name__ == "__main__":
    unittest.main()
