#!/usr/bin/env python3
"""Tests for benchy.api lock logic — no network (fetch + upstream are monkeypatched).

Run:  python3 test_api.py
"""
import json, os, sys, tempfile, threading, time, unittest

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
            with open(api._data_path(key), "w", encoding="utf-8", newline="\n") as f:
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
        lock = json.load(open(api.LOCK_PATH, encoding="utf-8"))
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
        ent = json.load(open(api.LOCK_PATH, encoding="utf-8"))["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v2")
        self.assertIsNone(self.last_revision)  # update ignores the old pin

    def test_prelock_no_download(self):
        called = []
        fb.fetch = lambda *a, **k: called.append(1)
        api.prelock([self.key])
        self.assertEqual(called, [])         # prelock must not download data
        ent = json.load(open(api.LOCK_PATH, encoding="utf-8"))["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v1")
        self.assertIsNone(ent["content_sha"])

    def test_complete_prelock_on_present_file(self):
        api.prelock([self.key])              # revision-only
        with open(api._data_path(self.key), "w", encoding="utf-8", newline="\n") as f:
            f.write(self.rows)               # data appears (e.g. fetched elsewhere)
        api.fetch(self.key)                  # should fill content_sha without re-fetch error
        ent = json.load(open(api.LOCK_PATH, encoding="utf-8"))["benchmarks"][self.key]
        self.assertTrue(ent["content_sha"])

    def test_prelock_keeps_pin_when_offline(self):
        api.prelock([self.key])              # pins sha_v1
        api.upstream_sha = lambda ds, timeout=20: None       # upstream unreachable
        api.prelock([self.key])
        ent = json.load(open(api.LOCK_PATH, encoding="utf-8"))["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v1")      # pin survives the outage

    def test_corrupt_lockfile_read_path_never_renames(self):
        # READ paths (dashboard /api/benchmarks, runners' check_dataset_lock) may catch a
        # writer mid-flight: _load_lock must be PURE — warn + fresh-empty, never quarantine
        with open(api.LOCK_PATH, "w", encoding="utf-8") as f:
            f.write("{not json")
        lock = api._load_lock()
        self.assertEqual(lock["benchmarks"], {})             # fresh-empty view
        self.assertTrue(os.path.exists(api.LOCK_PATH))       # file left exactly in place…
        self.assertFalse(os.path.exists(api.LOCK_PATH + ".corrupt"))
        with open(api.LOCK_PATH, encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not json")          # …byte-identical

    def test_corrupt_lockfile_quarantined(self):
        # CLI WRITE path (api.py lock): a genuinely corrupt lockfile is re-checked and
        # then moved aside to .corrupt before the new pins are written
        with open(api.LOCK_PATH, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(SystemExit) as cm:
            api._cmd_lock([self.key])
        self.assertEqual(cm.exception.code, 0)               # lock succeeded
        self.assertTrue(os.path.exists(api.LOCK_PATH + ".corrupt"))  # bad file kept aside
        with open(api.LOCK_PATH + ".corrupt", encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not json")
        lock = json.load(open(api.LOCK_PATH, encoding="utf-8"))      # fresh valid lockfile
        self.assertIn(self.key, lock["benchmarks"])

    def test_quarantine_skips_healthy_lockfile(self):
        api.fetch(self.key)                                  # valid lockfile on disk
        api._quarantine_corrupt_lock()                       # re-checks: load succeeds → no-op
        self.assertTrue(os.path.exists(api.LOCK_PATH))
        self.assertFalse(os.path.exists(api.LOCK_PATH + ".corrupt"))

    def test_concurrent_writers_keep_both_keys(self):
        # two `api.py lock` writers whose downloads overlap: each loaded the lock before
        # the other saved. _write_lock_entry's fresh read-modify-write under the write
        # sentinel must keep BOTH pins (the old whole-dict save dropped one).
        keys = ["logic", "arc_challenge"]
        barrier = threading.Barrier(2)
        def slow_fetch(key, revision=None):
            with open(api._data_path(key), "w", encoding="utf-8", newline="\n") as f:
                f.write(self.rows)
            barrier.wait(timeout=5)      # both writers are mid-"download" simultaneously
            return 1
        fb.fetch = slow_fetch
        errs = []
        def worker(k):
            try:
                api.fetch(k)
            except Exception as e:       # pragma: no cover - surfaced via assertEqual
                errs.append((k, e))
        ts = [threading.Thread(target=worker, args=(k,)) for k in keys]
        for t in ts: t.start()
        for t in ts: t.join(timeout=15)
        self.assertEqual(errs, [])
        lock = json.load(open(api.LOCK_PATH, encoding="utf-8"))
        for k in keys:
            self.assertIn(k, lock["benchmarks"], f"writer for '{k}' lost its pin")
            self.assertTrue(lock["benchmarks"][k]["content_sha"])
        self.assertFalse(os.path.exists(api.LOCK_PATH + ".write.lock"))  # sentinel released

    def test_stale_write_sentinel_is_reclaimed(self):
        sentinel = api.LOCK_PATH + ".write.lock"
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write("99999 dead-writer\n")
        old = time.time() - 600                              # > the 5 min staleness window
        os.utime(sentinel, (old, old))
        p = api.fetch(self.key)                              # must reclaim, not hang or raise
        self.assertTrue(os.path.exists(p))
        self.assertFalse(os.path.exists(sentinel))           # removed in finally
        lock = json.load(open(api.LOCK_PATH, encoding="utf-8"))
        self.assertIn(self.key, lock["benchmarks"])

    def test_live_write_sentinel_blocks_writer(self):
        sentinel = api.LOCK_PATH + ".write.lock"
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()} live\n")                 # fresh: held by a live writer
        old_t, api._SENTINEL_TIMEOUT_S = api._SENTINEL_TIMEOUT_S, 0.2
        try:
            with self.assertRaises(api.LockError):
                api.fetch(self.key)
        finally:
            api._SENTINEL_TIMEOUT_S = old_t
            os.remove(sentinel)

    def test_cached_content_mismatch(self):
        api.fetch(self.key)                  # locks the content hash
        with open(api._data_path(self.key), "w", encoding="utf-8", newline="\n") as f:
            f.write('{"question":"TAMPERED","options":{"A":"z"},"answer_idx":"A"}\n')
        with self.assertRaises(api.LockError):
            api.fetch(self.key)              # cached rows no longer match the lock

    def test_verify_false_skips_check(self):
        api.fetch(self.key)
        with open(api._data_path(self.key), "w", encoding="utf-8", newline="\n") as f:
            f.write('{"question":"TAMPERED","options":{"A":"z"},"answer_idx":"A"}\n')
        p = api.fetch(self.key, verify=False)  # no LockError
        self.assertTrue(os.path.exists(p))

    def test_cmd_verify_restores_original_on_fetch_failure(self):
        api.fetch(self.key)                  # data + lock present
        fb.fetch = lambda *a, **k: None      # re-fetch fails (e.g. offline)
        with self.assertRaises(SystemExit) as cm:
            api._cmd_verify([self.key])
        self.assertEqual(cm.exception.code, 1)               # non-zero exit
        path = api._data_path(self.key)
        self.assertTrue(os.path.exists(path))                # original restored…
        self.assertFalse(os.path.exists(path + ".bak"))      # …and the .bak consumed
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), self.rows)

    def test_fetch_benchmarks_cli_goes_through_lock(self):
        # F9: a manual `python3 fetch_benchmarks.py <key>` must not bypass the lock
        # contract — main() delegates to the same pinned fetch+lock as `api.py lock`
        old_argv, sys.argv = sys.argv, ["fetch_benchmarks.py", self.key]
        try:
            with self.assertRaises(SystemExit) as cm:
                fb.main()
            self.assertEqual(cm.exception.code, 0)
        finally:
            sys.argv = old_argv
        ent = json.load(open(api.LOCK_PATH, encoding="utf-8"))["benchmarks"][self.key]
        self.assertEqual(ent["upstream_sha"], "sha_v1")      # pinned…
        self.assertTrue(ent["content_sha"])                  # …and content-hashed

    def test_stable_surface(self):
        self.assertEqual(api.API_VERSION, 1)
        for fn in ("registry", "meta", "keys", "fetch", "lock_status", "data_path"):
            self.assertTrue(callable(getattr(api, fn)))
        self.assertIn(self.key, [b["key"] for b in api.registry()])


if __name__ == "__main__":
    unittest.main(verbosity=2)
