import tempfile
import threading
import unittest

from strokeweave.workflow import WorkflowStore


class ShutdownOwnershipTests(unittest.TestCase):
    def test_shutdown_timeout_retains_owner_until_worker_has_stopped(self):
        with tempfile.TemporaryDirectory() as runtime:
            store = WorkflowStore(runtime)
            entered, release = threading.Event(), threading.Event()

            def blocked_operation():
                entered.set()
                release.wait(2)
                return False

            store.run_one = blocked_operation
            store.start()
            self.assertTrue(entered.wait(1))
            try:
                with self.assertRaisesRegex(TimeoutError, "ownership retained"):
                    store.close(timeout=0.01)
                self.assertFalse(store.owner.closed)
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    WorkflowStore(runtime)
            finally:
                release.set()
                store.close(timeout=3)
            self.assertTrue(store.owner.closed)
            replacement = WorkflowStore(runtime)
            replacement.close()

    def test_bad_shutdown_timeouts_are_rejected(self):
        with tempfile.TemporaryDirectory() as runtime:
            store = WorkflowStore(runtime)
            try:
                for value in (-1, float("inf"), float("nan"), True, "5"):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        store.close(timeout=value)
            finally:
                store.close()

    def test_closed_store_cannot_reopen_database_or_submit(self):
        with tempfile.TemporaryDirectory() as runtime:
            store = WorkflowStore(runtime)
            batch = store.submit({"request_key": "shutdown_test_01", "tasks": [{"recipe": "generated_volume"}]})
            job = batch["job_ids"][0]
            store.close()
            for operation in (
                lambda: store.submit({"request_key": "shutdown_test_02", "tasks": [{"recipe": "generated_volume"}]}),
                store.run_one,
                lambda: store.detail(job),
                store.listing,
            ):
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    operation()
            replacement = WorkflowStore(runtime)
            try:
                self.assertEqual(len(replacement.listing()), 1)
                replacement.run_pending()
                self.assertEqual(replacement.detail(job)["state"], "COMPLETE")
            finally:
                replacement.close()
