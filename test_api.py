#!/usr/bin/env python3
"""Tests for benchy.api lock logic — no network (fetch + upstream are monkeypatched).

Run:  python3 test_api.py
"""
import json, os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import api, fetch_benchmarks as fb


class TestLock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # redirect data dir + lockfile into the temp dir
        self._data, fb.DATA = fb.DATA, os.path.join(self.tmp.name, "data")
        self._lock, api.LOCK_PATH = api.LOCK_PATH, os.path.join(self.tmp.name, "lock.json")
        os.makedirs(fb.DATA, exist_ok=True)
        # a real registry key to lock against
        self.key = "logic"
        self.dataset = fb.dataset_of(self.key)
        # fake upstream sha + fake fetch that writes deterministic rows
        self._usha, api.upstream_sha = api.upstream_sha, lambda ds, timeout=20: "sha_v1"
        self._fetch = fb.fetch
        self.rows = '{"question":"q","options":{"A":"x","B":"y"},"answer_idx":"A"}\n'
        def fake_fetch(key, revision=None):
            with open(api._data_path(key), "w") as f:
                f.write(self.rows)
            self.last_revision = revision
            return 1
        fb.fetch = fake_fetch
        self.last_revision = "unset"

    def tearDown(self):
        fb.DATA, api.LOCK_PATH, api.upstream_sha, fb.fetch = \
            self._data, self._lock, self._usha, self._fetch
        self.tmp.cleanup()

    def test_lock_on_first_fetch(self):
        p = api.fetch(self.key)
        self.assertTrue(os.path.exists(p))
        lock = json.load(open(api.LOCK_PATH))
        ent = lock["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v1")
        self.assertEqual(ent["rows"], 1)
        self.assertTrue(ent["content_sha"])

    def test_fetch_uses_pinned_revision(self):
        api.fetch(self.key)                 # locks at sha_v1
        os.remove(api._data_path(self.key))  # force a re-fetch
        api.upstream_sha = lambda ds, timeout=20: "sha_v2"   # upstream moved
        api.fetch(self.key)
        self.assertEqual(self.last_revision, "sha_v1")  # fetched the PINNED rev, not v2

    def test_drift_detection(self):
        api.fetch(self.key)
        # upstream changes its content under the same pin
        self.rows = '{"question":"DIFFERENT","options":{"A":"z"},"answer_idx":"A"}\n'
        os.remove(api._data_path(self.key))
        with self.assertRaises(api.LockError):
            api.fetch(self.key)

    def test_relock_accepts_new_content(self):
        api.fetch(self.key)
        self.rows = '{"question":"NEW","options":{"A":"z"},"answer_idx":"A"}\n'
        os.remove(api._data_path(self.key))
        api.upstream_sha = lambda ds, timeout=20: "sha_v2"
        api.fetch(self.key, update=True)    # relock
        ent = json.load(open(api.LOCK_PATH))["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v2")
        self.assertIsNone(self.last_revision)  # update ignores the old pin

    def test_prelock_no_download(self):
        called = []
        fb.fetch = lambda *a, **k: called.append(1)
        api.prelock([self.key])
        self.assertEqual(called, [])         # prelock must not download data
        ent = json.load(open(api.LOCK_PATH))["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v1")
        self.assertIsNone(ent["content_sha"])

    def test_complete_prelock_on_present_file(self):
        api.prelock([self.key])              # revision-only
        with open(api._data_path(self.key), "w") as f:
            f.write(self.rows)               # data appears (e.g. fetched elsewhere)
        api.fetch(self.key)                  # should fill content_sha without re-fetch error
        ent = json.load(open(api.LOCK_PATH))["benchmarks"][self.key]
        self.assertTrue(ent["content_sha"])

    def test_stable_surface(self):
        self.assertEqual(api.API_VERSION, 1)
        for fn in ("registry", "meta", "keys", "fetch", "lock_status", "data_path"):
            self.assertTrue(callable(getattr(api, fn)))
        self.assertIn(self.key, [b["key"] for b in api.registry()])


if __name__ == "__main__":
    unittest.main(verbosity=2)
