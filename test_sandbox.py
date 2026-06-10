#!/usr/bin/env python3
"""Tests pinning eval_code.run_tests's containment mitigations — offline, no model.

Covers the three behaviors that make code-exec runs survivable: (1) on timeout the WHOLE
process group dies, so a candidate that fork()s a child cannot outlive the harness;
(2) the normal completion path (pass and fail) is unchanged; (3) RLIMIT_FSIZE kills a
candidate that tries to write an oversized file. POSIX-only by design — that is the only
platform where the killpg/rlimit code paths exist.

run_tests takes an explicit timeout parameter so these tests use a 2s wall timeout
instead of the production default (BENCHY_CODE_TIMEOUT / 12s); total wall time ~3-4s.

Run:  python3 test_sandbox.py
"""
import os, shutil, signal, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the runner's main() refuses to start without this opt-in; set it before import so the
# module under test sees the same environment a sanctioned code run would
os.environ["BENCHY_ALLOW_CODE_EXEC"] = "1"
import eval_code

TASK = {"task_id": "sandbox/1", "prompt": "", "tests": "", "entry_point": ""}


def task_with_tests(tests):
    t = dict(TASK)
    t["tests"] = tests
    return t


def pid_gone(pid, wait_s=3.0):
    """Poll briefly until `pid` no longer exists (killed + reaped by init)."""
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False                # exists, owned by someone else (never ours)
        time.sleep(0.05)
    return False


@unittest.skipUnless(os.name == "posix", "process groups + resource rlimits are POSIX-only")
class TestTimeoutKillsProcessGroup(unittest.TestCase):
    def test_forked_child_does_not_survive_timeout(self):
        # the candidate forks a 60s sleeper, reports its pid, then sleeps past the
        # timeout itself: the old subprocess.run(timeout=) escape in one program
        d = tempfile.mkdtemp(prefix="benchy_sandbox_")
        self.addCleanup(shutil.rmtree, d, True)
        pidfile = os.path.join(d, "child.pid")
        cand = (
            "import os, time\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "with open(%r, 'w') as f:\n"
            "    f.write(str(pid))\n"
            "time.sleep(60)\n" % pidfile
        )
        t0 = time.time()
        ok, err = eval_code.run_tests(cand, TASK, timeout=2)
        self.assertFalse(ok)
        self.assertIn("timeout", err)
        self.assertLess(time.time() - t0, 10, "timeout kill took far longer than the 2s budget")
        with open(pidfile, encoding="utf-8") as f:
            child = int(f.read().strip())
        self.assertTrue(pid_gone(child),
                        "forked child %d outlived the harness timeout — killpg escape" % child)


@unittest.skipUnless(os.name == "posix", "process groups + resource rlimits are POSIX-only")
class TestNormalPathUnchanged(unittest.TestCase):
    def test_passing_candidate_passes(self):
        ok, err = eval_code.run_tests("def add(a, b):\n    return a + b",
                                      task_with_tests("assert add(1, 2) == 3"), timeout=3)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_failing_candidate_reports_last_error_line(self):
        ok, err = eval_code.run_tests("def add(a, b):\n    return a - b",
                                      task_with_tests("assert add(1, 2) == 3"), timeout=3)
        self.assertFalse(ok)
        self.assertIn("AssertionError", err)


@unittest.skipUnless(os.name == "posix", "process groups + resource rlimits are POSIX-only")
class TestSigtermKillsCandidateGroup(unittest.TestCase):
    def test_sigterm_to_harness_kills_candidate_and_its_fork(self):
        # Pins the F15 fix: the wall-clock timeout was the ONLY stop path that killed the
        # candidate group — a dashboard stop_all SIGTERM (or Ctrl-C) orphaned it. Spawn a
        # real harness process (install_signal_handlers + run_tests on a long candidate),
        # SIGTERM it mid-run, and assert the candidate's forked child died with it.
        d = tempfile.mkdtemp(prefix="benchy_sandbox_")
        self.addCleanup(shutil.rmtree, d, True)
        pidfile = os.path.join(d, "child.pid")
        candfile = os.path.join(d, "cand_src.py")
        driver = os.path.join(d, "driver.py")
        with open(candfile, "w", encoding="utf-8") as f:
            f.write(
                "import os, time\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    time.sleep(60)\n"
                "    os._exit(0)\n"
                "with open(%r, 'w') as f:\n"
                "    f.write(str(pid))\n"
                "time.sleep(60)\n" % pidfile
            )
        with open(driver, "w", encoding="utf-8") as f:
            f.write(
                "import os, sys\n"
                "sys.path.insert(0, %r)\n"
                "os.environ['BENCHY_ALLOW_CODE_EXEC'] = '1'\n"
                "import eval_code\n"
                "eval_code.install_signal_handlers()\n"
                "with open(%r, encoding='utf-8') as f:\n"
                "    cand = f.read()\n"
                "task = {'task_id': 'sig/1', 'prompt': '', 'tests': '', 'entry_point': ''}\n"
                "eval_code.run_tests(cand, task, timeout=30)\n" % (HERE, candfile)
            )
        harness = subprocess.Popen([sys.executable, driver], cwd=d,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.time() + 4.0        # candidate up + forked: pidfile appears
            while time.time() < deadline and not os.path.exists(pidfile):
                time.sleep(0.05)
            self.assertTrue(os.path.exists(pidfile), "candidate never started under the harness")
            with open(pidfile, encoding="utf-8") as f:
                child = int(f.read().strip())
            harness.send_signal(signal.SIGTERM)
            harness.wait(timeout=5)
            self.assertEqual(harness.returncode, -signal.SIGTERM,
                             "harness did not exit with default SIGTERM semantics")
            self.assertTrue(pid_gone(child),
                            "candidate's forked child %d survived SIGTERM to the harness "
                            "— stop_all/Ctrl-C orphan regression" % child)
        finally:
            if harness.poll() is None:
                harness.kill()
                harness.wait()


@unittest.skipUnless(os.name == "posix", "process groups + resource rlimits are POSIX-only")
class TestFsizeLimit(unittest.TestCase):
    def test_oversized_write_is_killed(self):
        # tries to write 80MB > the 64MB RLIMIT_FSIZE ceiling: the kernel SIGXFSZ-kills
        # the candidate mid-write, so it never reaches exit 0 → scored as a clean FAIL
        d = tempfile.mkdtemp(prefix="benchy_sandbox_")
        self.addCleanup(shutil.rmtree, d, True)
        big = os.path.join(d, "big.bin")
        cand = (
            "with open(%r, 'wb') as f:\n"
            "    for _ in range(20):\n"
            "        f.write(b'\\0' * (4 * 1024 * 1024))\n" % big
        )
        ok, err = eval_code.run_tests(cand, TASK, timeout=3)
        self.assertFalse(ok, "candidate wrote 80MB and exited 0 — RLIMIT_FSIZE not applied")
        self.assertNotIn("timeout", err)        # killed by the rlimit, not the wall clock
        if os.path.exists(big):                 # whatever landed before SIGXFSZ stayed <= 64MB
            self.assertLessEqual(os.path.getsize(big), 64 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
