"""Generated-only integration and boundary tests; no external model or data."""
import io
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from strokeweave.app import make_server
from strokeweave.workflow import (ContractError, WorkflowStore, generated_tool,
                                 plan_task, validate_output)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WorkflowStore(Path(self.temp.name) / "runtime")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def submit(self, recipe="generated_volume", key="test_request_01"):
        return self.store.submit({"request_key": key, "tasks": [{"recipe": recipe}]})

    def test_measurement_is_analytic_and_correct(self):
        result = validate_output(generated_tool("generated_volume", 1))
        self.assertEqual(result["foreground_voxels"], 24)
        self.assertEqual(result["volume_mL"], 0.576)

    def test_batch_and_audit_chain(self):
        batch = self.submit()
        self.store.run_pending()
        value = self.store.detail(batch["job_ids"][0])
        self.assertEqual(value["state"], "COMPLETE")
        self.assertEqual([e["event"] for e in value["events"]], ["QUEUED", "RUNNING", "CHARGE", "VALIDATED", "COMPLETE"])
        self.assertFalse(value["patient_execution"])
        self.assertFalse(value["training_enabled"])

    def test_idempotency_and_conflict(self):
        first = self.submit()
        self.assertEqual(self.submit()["job_ids"], first["job_ids"])
        self.assertTrue(self.submit()["reused"])
        with self.assertRaises(FileExistsError):
            self.submit("fallback_recovery")

    def test_bounded_recovery_charges_failure(self):
        job = self.submit("fallback_recovery")["job_ids"][0]
        self.store.run_pending()
        value = self.store.detail(job)
        self.assertEqual(value["state"], "COMPLETE")
        self.assertEqual(value["attempts"], 2)
        self.assertEqual(sum(e["event"] == "FAILED" for e in value["events"]), 1)

    def test_persistent_failure_stops(self):
        job = self.submit("persistent_failure")["job_ids"][0]
        self.store.run_pending()
        value = self.store.detail(job)
        self.assertEqual((value["state"], value["attempts"]), ("REFER", 2))
        self.assertIsNone(value["result"])

    def test_probability_and_unit_rejected(self):
        for i, recipe in enumerate(("invalid_probability", "unknown_units")):
            job = self.submit(recipe, "reject_case_" + str(i))["job_ids"][0]
            self.store.run_pending()
            value = self.store.detail(job)
            self.assertEqual(value["state"], "REFER")
            self.assertEqual(value["attempts"], 1)

    def test_result_tamper_refers_and_preserves(self):
        job = self.submit()["job_ids"][0]
        self.store.run_pending()
        artifact = self.store.tasks / job / "result.json"
        artifact.write_text('{"changed":true}')
        value = self.store.detail(job)
        self.assertEqual(value["state"], "REFER")
        self.assertIsNone(value["result"])
        self.assertEqual(artifact.read_text(), '{"changed":true}')

    def test_journal_tamper_fails_closed(self):
        job = self.submit()["job_ids"][0]
        with sqlite3.connect(self.store.db) as db:
            db.execute("UPDATE events SET event_hash='bad' WHERE job=?", (job,))
        with self.assertRaises(ContractError):
            self.store.detail(job)

    def test_restart_preserves_charged_attempts(self):
        job = self.submit("persistent_failure")["job_ids"][0]
        with sqlite3.connect(self.store.db) as db:
            db.execute("UPDATE jobs SET state='RUNNING',attempts=1 WHERE id=?", (job,))
        self.store.close()
        recovered = WorkflowStore(self.store.root)
        try:
            recovered.run_pending()
            value = recovered.detail(job)
            self.assertEqual((value["state"], value["attempts"]), ("REFER", 2))
            self.assertTrue(any(e["event"] == "RECOVERY_QUEUED" for e in value["events"]))
        finally:
            recovered.close()

    def test_runtime_owner_cannot_be_stolen(self):
        with self.assertRaises(RuntimeError):
            WorkflowStore(self.store.root)

    def test_zip_whitelist_and_html(self):
        job = self.submit()["job_ids"][0]
        self.store.run_pending()
        with zipfile.ZipFile(io.BytesIO(self.store.export(job))) as archive:
            self.assertEqual(set(archive.namelist()), {"report.json", "report.html", "SCOPE.txt"})
            report = json.loads(archive.read("report.json"))
            self.assertEqual(report["state"], "COMPLETE")
        self.assertIn(b"Not clinical", self.store.report_html(job).replace(b"not clinical", b"Not clinical"))

    def test_no_paths_or_arbitrary_recipe(self):
        for task in ({"recipe": "generated_volume", "path": "/secret"}, {"recipe": "../escape"}):
            with self.assertRaises(ValueError):
                self.store.submit({"request_key": "invalid_task_01", "tasks": [task]})
        for job in ("../escape", "0" * 31, "G" * 32):
            with self.assertRaises(ValueError):
                self.store.detail(job)

    def test_rule_router_boundaries(self):
        self.assertEqual(plan_task("Run all synthetic demonstrations")["status"], "READY")
        self.assertEqual(len(plan_task("Run all synthetic demonstrations")["tasks"]), 5)
        self.assertEqual(plan_task("Run synthetic fallback recovery")["tasks"], [{"recipe": "fallback_recovery"}])
        self.assertEqual(plan_task("Analyze a patient MRI")["status"], "BLOCKED")
        self.assertEqual(plan_task("Run synthetic DeepISLES")["status"], "BLOCKED")
        self.assertEqual(plan_task("Read /mnt/data/input.nii")["status"], "BLOCKED")
        self.assertEqual(plan_task("Hello there")["status"], "NEEDS_CLARIFICATION")


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = WorkflowStore(self.temp.name)
        self.server = make_server(self.store, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.store.close()
        self.temp.cleanup()

    def request(self, path, value=None, headers=None):
        body = None if value is None else json.dumps(value).encode()
        base = {"Content-Type": "application/json", "X-StrokeWeave-Request": "local-research-v1"}
        base.update(headers or {})
        return self.http.open(urllib.request.Request(self.base + path, body, base), timeout=5)

    def test_health_static_and_batch_api(self):
        with self.request("/api/health") as response:
            self.assertFalse(json.load(response)["patient_execution"])
        with self.request("/") as response:
            self.assertIn(b"StrokeWeave", response.read())
            self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
        with self.request("/api/batches", {"request_key": "http_request_01", "tasks": [{"recipe": "generated_volume"}]}) as response:
            self.assertEqual(response.status, 201)
            batch = json.load(response)
        self.store.run_pending()
        with self.request("/api/jobs/" + batch["job_ids"][0] + "/report.json") as response:
            self.assertEqual(json.load(response)["state"], "COMPLETE")

    def test_cross_origin_and_host_rejected(self):
        for headers in ({"Origin": "https://external.example"}, {"Host": "external.example"}):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.request("/api/health", headers=headers)
            self.assertEqual(ctx.exception.code, 403)

    def test_traversal_and_patient_inputs_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request("/../workflow.py")
        self.assertEqual(ctx.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.request("/api/batches", {"request_key": "invalid_http_01", "tasks": [{"recipe": "generated_volume", "path": "/patient"}]})
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
