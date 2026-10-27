"""Portable generated-input workflow, formerly developed under TACTIC.

This module does not accept imaging files, load checkpoints, train models or
make clinical decisions. Its technical fixtures are not patient validation.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

RECIPES = {
    "generated_volume": "Generated spatial volume",
    "fallback_recovery": "Bounded fallback recovery",
    "invalid_probability": "Invalid probability rejection",
    "unknown_units": "Unknown unit rejection",
    "persistent_failure": "Persistent failure and bounded referral",
}
TERMINAL = {"COMPLETE", "REFER", "ERROR"}
LIMITS = {"attempt_budget": 2, "batch_size": 24, "request_bytes": 32768}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def tool_cards() -> list[dict[str, Any]]:
    return [
        {"id": "generated_spatial_tool", "enabled": True,
         "scope": "Analytic generated array; physical mask-volume measurement",
         "training_exposure": "None; deterministic technical fixture"},
        {"id": "DeepISLES", "enabled": False, "status": "EXTERNAL_PLUGIN_DISABLED",
         "scope": "Not bundled, loaded or qualified by this portable release",
         "training_exposure": "ISLES-exposed; cannot be called unseen on ISLES"},
        {"id": "SynthStroke", "enabled": False, "status": "EXTERNAL_PLUGIN_DISABLED",
         "scope": "Not bundled, loaded or qualified by this portable release",
         "training_exposure": "ATLAS-derived exposure; separate qualification required"},
    ]


def plan_task(text: str) -> dict[str, Any]:
    """Rule-based intent preview, not an LLM or a clinical planner."""
    if not isinstance(text, str) or not 3 <= len(text) <= 2000:
        raise ValueError("Task text must contain 3 to 2000 characters")
    low = text.casefold()
    state, recipes = "NEEDS_CLARIFICATION", []
    reason = "Explicitly request a generated or synthetic demonstration."
    if re.search(r"patient|clinical|diagnos|treat|thrombo|患者|病人|诊断|治疗", low):
        state, reason = "BLOCKED", "Patient and clinical tasks are not admitted."
    elif re.search(r"[a-z]:[\\/]|/mnt/|https?://|\.nii|\.dcm|\.csv|\.py\b|[;`]|\b(?:bash|sudo|exec|powershell)\b", low):
        state, reason = "BLOCKED", "External paths, URLs and commands are disabled."
    elif re.search(r"unseal|override|bypass|risk threshold|cal/hold|test label|解封|绕过|放宽", low):
        state, reason = "BLOCKED", "Requests cannot change admission or sealed-label boundaries."
    elif "deepisles" in low or "synthstroke" in low:
        state, reason = "BLOCKED", "Real-model plugins are not enabled in this release."
    elif re.search(r"synthetic|generated|phantom|demo|合成|演示", low):
        if re.search(r"\ball\b|\bsuite\b|全部", low):
            recipes = list(RECIPES)
        elif "persistent" in low:
            recipes = ["persistent_failure"]
        elif "probability" in low or "invalid" in low:
            recipes = ["invalid_probability"]
        elif "unit" in low:
            recipes = ["unknown_units"]
        elif re.search(r"recover|fallback|恢复", low):
            recipes = ["fallback_recovery"]
        elif re.search(r"volume|spatial|geometry|measure|体积", low):
            recipes = ["generated_volume"]
        if recipes:
            state = "READY"
            reason = "Preview only. Confirm to queue these app-generated tasks."
        else:
            reason = "Supported demos: volume, recovery, invalid probability, units, or all scenarios."
    return {"status": state, "engine": "rule_based_router_not_LLM",
            "explanation": reason, "tasks": [{"recipe": r} for r in recipes],
            "request_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "patient_execution": False, "training_enabled": False}


class ContractError(ValueError):
    """Generated tool output violates a fixed technical contract."""


def generated_tool(recipe: str, attempt: int) -> dict[str, Any]:
    """A real deterministic local tool; faults are explicit generated tests."""
    if recipe not in RECIPES:
        raise ValueError("Unregistered recipe")
    if recipe == "persistent_failure" or (recipe == "fallback_recovery" and attempt == 1):
        raise RuntimeError("Injected generated-tool failure")
    shape = [8, 9, 7]
    values = [float(1 <= x < 4 and 2 <= y < 6 and 1 <= z < 3)
              for x in range(shape[0]) for y in range(shape[1]) for z in range(shape[2])]
    if recipe == "invalid_probability":
        values[0] = float("nan")
    return {"source": "analytic_generated_fixture_v1", "generated_only": True,
            "shape": shape, "spacing": [2.0, 3.0, 4.0],
            "unit": "unknown" if recipe == "unknown_units" else "mm",
            "probability": values, "version": "generated-spatial-v1"}


def validate_output(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
            "source", "generated_only", "shape", "spacing", "unit", "probability", "version"}:
        raise ContractError("Unexpected output schema")
    if (value["generated_only"] is not True or value["source"] != "analytic_generated_fixture_v1"
            or value["version"] != "generated-spatial-v1"):
        raise ContractError("Invalid source or tool version")
    if value["unit"] != "mm":
        raise ContractError("Unknown spatial units")
    if value["shape"] != [8, 9, 7] or value["spacing"] != [2.0, 3.0, 4.0]:
        raise ContractError("Unexpected generated spatial geometry")
    probabilities = value["probability"]
    if not isinstance(probabilities, list) or len(probabilities) != 504:
        raise ContractError("Invalid output shape")
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
           for x in probabilities):
        raise ContractError("Non-finite or out-of-range probability")
    count = sum(x >= 0.5 for x in probabilities)
    return {"foreground_voxels": count, "voxel_volume_mm3": 24.0,
            "volume_mL": count * 24.0 / 1000.0, "output_sha256": digest(value),
            "semantics": "Thresholded volume of an analytic fixture; not a lesion-accuracy result"}


class WorkflowStore:
    """Single-worker persistent queue with bounded, non-resetting attempts."""

    def __init__(self, runtime: str | Path):
        self.root = Path(runtime).absolute()
        if self.root.is_symlink():
            raise ValueError("Runtime must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.tasks = self.root / "tasks"
        if self.tasks.is_symlink():
            raise ValueError("Task directory must not be a symlink")
        self.tasks.mkdir(exist_ok=True)
        self.db = self.root / "queue.sqlite3"
        if self.db.is_symlink():
            raise ValueError("Database must not be a symlink")
        owner_path = self.root / "owner.lock"
        if owner_path.is_symlink():
            raise ValueError("Owner lock must not be a symlink")
        self.owner = owner_path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if owner_path.stat().st_size == 0:
                    self.owner.write(b"0")
                    self.owner.flush()
                self.owner.seek(0)
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("Runtime already owned by another process; do not recover active work") from None
        self.lock = threading.RLock()
        self.wake, self.stopping = threading.Event(), threading.Event()
        self.thread: threading.Thread | None = None
        with self._connection() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,request_key TEXT UNIQUE,payload_hash TEXT);
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,batch_id TEXT,recipe TEXT,state TEXT,
                attempts INTEGER,reason TEXT,result TEXT,created REAL,updated REAL);
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,
                record TEXT,event_hash TEXT);
            """)
            for row in db.execute("SELECT id FROM jobs WHERE state='RUNNING'").fetchall():
                db.execute("UPDATE jobs SET state='QUEUED',reason=? WHERE id=?",
                           ("Process interrupted; consumed attempts retained", row["id"]))
                self._event(db, row["id"], "RECOVERY_QUEUED", {"budget_reset": False})

    @contextmanager
    def _connection(self):
        # Also serialize handlers against final shutdown/owner release. A queued
        # HTTP handler must never reopen a closed owner's database connection.
        with self.lock:
            if self.owner.closed:
                raise RuntimeError("Workflow store is closed")
            db = sqlite3.connect(self.db, timeout=20)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            try:
                with db:
                    yield db
            finally:
                db.close()

    def _event(self, db, job: str, event: str, payload: dict[str, Any]):
        previous = db.execute("SELECT event_hash FROM events WHERE job=? ORDER BY seq DESC LIMIT 1", (job,)).fetchone()
        record = {"job": job, "event": event, "stamp": time.time(), "payload": payload,
                  "previous_hash": previous[0] if previous else "0" * 64}
        db.execute("INSERT INTO events(job,record,event_hash) VALUES(?,?,?)",
                   (job, canonical(record).decode(), digest(record)))
        return {**record, "event_hash": digest(record)}

    def _task_dir(self, job: str) -> Path:
        if not isinstance(job, str) or re.fullmatch(r"[a-f0-9]{32}", job) is None:
            raise ValueError("Invalid job identifier")
        folder = self.tasks / job
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.tasks):
            raise ValueError("Task path escapes runtime")
        return folder

    def submit(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"request_key", "tasks"}:
            raise ValueError("Only request_key and registered generated tasks are accepted")
        key, tasks = value["request_key"], value["tasks"]
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_-]{8,64}", key) is None:
            raise ValueError("Invalid idempotency key")
        if not isinstance(tasks, list) or not 1 <= len(tasks) <= LIMITS["batch_size"]:
            raise ValueError("A batch must contain 1 to 24 tasks")
        if any(not isinstance(t, dict) or set(t) != {"recipe"} or
               not isinstance(t["recipe"], str) or t["recipe"] not in RECIPES for t in tasks):
            raise ValueError("Unregistered task; paths, patient inputs and commands are disabled")
        with self.lock, self._connection() as db:
            old = db.execute("SELECT * FROM batches WHERE request_key=?", (key,)).fetchone()
            if old:
                if old["payload_hash"] != digest(tasks):
                    raise FileExistsError("Idempotency key already belongs to another request")
                jobs = [r[0] for r in db.execute("SELECT id FROM jobs WHERE batch_id=? ORDER BY created,id", (old["id"],))]
                return {"batch_id": old["id"], "job_ids": jobs, "reused": True}
            batch, jobs = uuid.uuid4().hex, []
            db.execute("INSERT INTO batches VALUES(?,?,?)", (batch, key, digest(tasks)))
            for task in tasks:
                job, now = uuid.uuid4().hex, time.time()
                jobs.append(job)
                db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                           (job, batch, task["recipe"], "QUEUED", 0, "", None, now, now))
                self._event(db, job, "QUEUED", {"recipe": task["recipe"], "generated_only": True})
        self.wake.set()
        return {"batch_id": batch, "job_ids": jobs, "reused": False}

    def run_one(self) -> bool:
        with self.lock, self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state='QUEUED' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return False
            job, recipe = row["id"], row["recipe"]
            db.execute("UPDATE jobs SET state='RUNNING',updated=? WHERE id=?", (time.time(), job))
            self._event(db, job, "RUNNING", {"attempts_already_charged": row["attempts"]})
        start = time.perf_counter()
        accepted, reason, result = False, "Attempt budget exhausted", None
        for attempt in range(row["attempts"] + 1, LIMITS["attempt_budget"] + 1):
            with self.lock, self._connection() as db:
                db.execute("UPDATE jobs SET attempts=?,updated=? WHERE id=?", (attempt, time.time(), job))
                self._event(db, job, "CHARGE", {"attempt": attempt, "tool": "generated_spatial_tool"})
            try:
                output = generated_tool(recipe, attempt)
                result = validate_output(output)
                result.update(generated_only=True, patient_execution=False, training_enabled=False,
                              actual_wall_seconds=time.perf_counter() - start)
                folder = self._task_dir(job)
                folder.mkdir(exist_ok=True)
                artifact = folder / "result.json"
                if artifact.is_symlink():
                    raise ContractError("Symlink result rejected")
                encoded = canonical(result)
                if artifact.exists():
                    # Crash recovery never silently overwrites a previous artifact.
                    raise ContractError("Existing artifact requires explicit review")
                with artifact.open("xb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                with self.lock, self._connection() as db:
                    self._event(db, job, "VALIDATED", {"output_sha256": result["output_sha256"]})
                    self._event(db, job, "COMPLETE", {"artifact_sha256": hashlib.sha256(encoded).hexdigest(),
                                                       "worker_path_wall_seconds": time.perf_counter() - start})
                    db.execute("UPDATE jobs SET state='COMPLETE',result=?,reason='',updated=? WHERE id=?",
                               (encoded.decode(), time.time(), job))
                accepted = True
                break
            except Exception as exc:
                reason = type(exc).__name__ + ": " + str(exc)
                with self.lock, self._connection() as db:
                    self._event(db, job, "REJECTED" if isinstance(exc, ContractError) else "FAILED",
                                {"attempt": attempt, "reason": reason})
                if isinstance(exc, ContractError):
                    break
        if not accepted:
            with self.lock, self._connection() as db:
                self._event(db, job, "REFER", {"reason": reason, "budget_reset": False,
                                               "worker_path_wall_seconds": time.perf_counter() - start})
                db.execute("UPDATE jobs SET state='REFER',reason=?,updated=? WHERE id=?", (reason, time.time(), job))
        return True

    def run_pending(self):
        while self.run_one():
            pass

    def start(self):
        if self.owner.closed:
            raise RuntimeError("Workflow store is closed")
        if self.thread is not None:
            raise RuntimeError("Worker already started")
        def work():
            while not self.stopping.is_set():
                try:
                    if self.run_one():
                        continue
                except Exception:
                    # A damaged store is not converted into a successful job.
                    self.stopping.set()
                    break
                self.wake.wait(0.5)
                self.wake.clear()
        self.thread = threading.Thread(target=work, name="strokeweave-generated-worker", daemon=True)
        self.thread.start()

    def close(self, timeout: float = 5.0):
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("Shutdown timeout must be finite and non-negative")
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                # Do not permit another owner while a blocked worker may still write.
                # Caller can retry close after its bounded operation has returned.
                raise TimeoutError("Worker still alive; runtime ownership retained")
        with self.lock:
            if not self.owner.closed:
                self.owner.close()

    def detail(self, job: str) -> dict[str, Any]:
        folder = self._task_dir(job)
        with self.lock, self._connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job,)).fetchone()
            if row is None:
                raise KeyError("Unknown job")
            value = dict(row)
            records = db.execute("SELECT record,event_hash FROM events WHERE job=? ORDER BY seq", (job,)).fetchall()
            previous, trace = "0" * 64, []
            for item in records:
                record = json.loads(item["record"])
                if record["previous_hash"] != previous or digest(record) != item["event_hash"]:
                    raise ContractError("Audit journal integrity failed")
                previous = item["event_hash"]
                trace.append({**record, "event_hash": previous})
            value["result"] = json.loads(value["result"]) if value["result"] else None
            if value["state"] == "COMPLETE":
                try:
                    artifact = folder / "result.json"
                    if artifact.is_symlink():
                        raise ContractError("Result symlink rejected")
                    raw = artifact.read_bytes()
                    completed = [e for e in trace if e["event"] == "COMPLETE"]
                    if (len(completed) != 1 or hashlib.sha256(raw).hexdigest() != completed[0]["payload"]["artifact_sha256"]
                            or json.loads(raw) != value["result"]):
                        raise ContractError("Accepted artifact changed")
                except (OSError, ValueError, KeyError) as exc:
                    reason = "Artifact integrity failure: " + str(exc)
                    trace.append(self._event(db, job, "ARTIFACT_INVALIDATED",
                                             {"reason": reason, "prior_receipt_preserved": True}))
                    db.execute("UPDATE jobs SET state='REFER',reason=?,updated=? WHERE id=?", (reason, time.time(), job))
                    value.update(state="REFER", reason=reason)
                    # Retain old receipt in storage, do not display it as accepted.
                    value["prior_result_preserved"] = True
                    value["result"] = None
            elif value["state"] != "COMPLETE":
                value["result"] = None
            value.update(events=trace, generated_only=True, patient_execution=False,
                         training_enabled=False, protected_labels_sealed=True,
                         attempt_budget=LIMITS["attempt_budget"])
            return value

    def listing(self) -> list[dict[str, Any]]:
        with self.lock, self._connection() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs ORDER BY created DESC,id DESC LIMIT 200")]
        return [self.detail(job) for job in ids]

    def report(self, job: str) -> dict[str, Any]:
        value = self.detail(job)
        value["release"] = "StrokeWeave research prototype (TACTIC lineage)"
        value["limits"] = ["Generated inputs only; no patient validation", "No clinical diagnosis or treatment",
                           "No performance superiority established", "Real-model plugins disabled",
                           "Hash chains are integrity checks, not authenticated signatures"]
        return value

    def report_html(self, job: str) -> bytes:
        return self.render_report_html(self.report(job))

    @staticmethod
    def render_report_html(report: dict[str, Any]) -> bytes:
        esc = lambda x: html.escape(str(x))
        return ("<!doctype html><html lang=en><meta charset=utf-8><title>StrokeWeave audit</title>"
                "<style>body{font:16px system-ui;max-width:1000px;margin:3rem auto;padding:1rem;color:#183842}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef4f5;padding:1rem}"
                "aside{padding:1rem;background:#fff2d8}</style><h1>StrokeWeave · research audit</h1>"
                "<aside>Generated technical demonstration only. Not clinical evidence.</aside>"
                f"<h2>{esc(report['state'])}</h2><pre>{esc(json.dumps(report, indent=2, ensure_ascii=False))}</pre></html>").encode()

    def export(self, job: str) -> bytes:
        report = self.report(job)
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("report.json", canonical(report))
            archive.writestr("report.html", self.render_report_html(report))
            archive.writestr("SCOPE.txt", "Generated-input research audit only. No patient files, credentials, checkpoints or clinical validation.\n")
        return target.getvalue()


def run_generated_demo(state_dir: str | Path) -> dict[str, Any]:
    """Execute the fixed five-scenario generated suite, without a web server."""
    store = WorkflowStore(state_dir)
    try:
        batch = store.submit({"request_key": "strokeweave_generated_demo_v1",
                              "tasks": [{"recipe": recipe} for recipe in RECIPES]})
        store.run_pending()
        jobs = [store.detail(job) for job in batch["job_ids"]]
        return {"name": "StrokeWeave", "generated_only": True,
                "patient_execution": False, "training_enabled": False,
                "batch": batch, "jobs": jobs,
                "complete": sum(job["state"] == "COMPLETE" for job in jobs),
                "referred": sum(job["state"] == "REFER" for job in jobs)}
    finally:
        store.close()
