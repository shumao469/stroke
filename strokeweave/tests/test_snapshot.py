import html
import io
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from strokeweave.workflow import WorkflowStore


class EvidenceSnapshotTests(unittest.TestCase):
    def test_zip_reports_use_one_snapshot_even_if_queue_can_change(self):
        with tempfile.TemporaryDirectory() as runtime:
            store = WorkflowStore(runtime)
            try:
                batch = store.submit({"request_key": "snapshot_test_01", "tasks": [{"recipe": "generated_volume"}]})
                job = batch["job_ids"][0]
                before = store.report(job)
                store.run_pending()
                after = store.report(job)
                self.assertNotEqual(before["state"], after["state"])
                with patch.object(store, "report", side_effect=[before, after]) as loader:
                    archive = store.export(job)
                    self.assertEqual(loader.call_count, 1)
                with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
                    json_report = json.loads(zipped.read("report.json"))
                    page = zipped.read("report.html").decode()
                    html_report = json.loads(html.unescape(page.split("<pre>", 1)[1].split("</pre>", 1)[0]))
                self.assertEqual(json_report, html_report)
            finally:
                store.close()

    def test_invalidated_report_includes_its_new_audit_event(self):
        with tempfile.TemporaryDirectory() as runtime:
            store = WorkflowStore(runtime)
            try:
                batch = store.submit({"request_key": "snapshot_test_02", "tasks": [{"recipe": "generated_volume"}]})
                job = batch["job_ids"][0]
                store.run_pending()
                (store.tasks / job / "result.json").write_text("{}")
                report = store.report(job)
                self.assertEqual(report["state"], "REFER")
                self.assertEqual(report["events"][-1]["event"], "ARTIFACT_INVALIDATED")
                self.assertTrue(report["prior_result_preserved"])
                self.assertIsNone(report["result"])
            finally:
                store.close()

    def test_costs_recorded_on_success_and_failed_worker_paths(self):
        with tempfile.TemporaryDirectory() as runtime:
            store = WorkflowStore(runtime)
            try:
                batch = store.submit({"request_key": "snapshot_test_03", "tasks": [
                    {"recipe": "generated_volume"}, {"recipe": "persistent_failure"}]})
                store.run_pending()
                for job in batch["job_ids"]:
                    report = store.report(job)
                    last = report["events"][-1]
                    self.assertIn(last["event"], ("COMPLETE", "REFER"))
                    self.assertGreaterEqual(last["payload"]["worker_path_wall_seconds"], 0)
            finally:
                store.close()
