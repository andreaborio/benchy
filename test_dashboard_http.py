#!/usr/bin/env python3
"""HTTP characterization tests for dashboard.py — fully offline, no model server.

Importing dashboard must not bind a port (binding happens only under __main__); each test
class redirects the module's state paths into a fresh tempdir, rebinds STATS to it and
serves dashboard.H on an EPHEMERAL port (never :8050, where a real dashboard may live).
server_up is stubbed to False so no test ever probes a model server. Pins: every GET route
answers clean JSON on an empty results tree, the populated routes are well-formed, and the
Host/Origin/CSRF guard matrix (DNS-rebinding + cross-site POSTs) fails closed.

Run:  python3 test_dashboard_http.py
"""
import http.client, json, os, socket, sys, tempfile, threading, unittest
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchy_common as bc
import benchy_stats
import dashboard

_PATCHED = ("RESULTS", "DATA", "RUNS", "PERF", "LIVE", "STREAM", "CONFIG_PATH", "HISTORY",
            "STATS", "DASH_PORT", "server_up")


class DashboardCase(unittest.TestCase):
    """Shared scaffolding: tempdir state + a live dashboard.H server on an ephemeral port."""

    @classmethod
    def populate(cls):
        """Subclass hook: drop synthetic runs/details into cls.results before testing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = os.path.join(cls.tmp.name, "results")
        cls.data = os.path.join(cls.tmp.name, "data")
        os.makedirs(os.path.join(cls.results, "details"))
        os.makedirs(cls.data)
        cls._saved = {n: getattr(dashboard, n) for n in _PATCHED}
        dashboard.RESULTS = cls.results
        dashboard.DATA = cls.data
        dashboard.RUNS = os.path.join(cls.results, "runs.jsonl")
        dashboard.PERF = os.path.join(cls.results, "perf.jsonl")
        dashboard.LIVE = os.path.join(cls.results, "live.json")
        dashboard.STREAM = os.path.join(cls.results, "stream.jsonl")
        dashboard.CONFIG_PATH = os.path.join(cls.tmp.name, "config.json")
        dashboard.HISTORY = os.path.join(cls.results, "metrics.jsonl")
        dashboard.STATS = benchy_stats.Stats(cls.results)
        dashboard.server_up = lambda: False        # never probe a real model server from tests
        cls.populate()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), dashboard.H)   # port 0 = ephemeral
        cls.port = cls.httpd.server_address[1]
        dashboard.DASH_PORT = cls.port             # the Host allowlist must match what we bound
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        for n, v in cls._saved.items():
            setattr(dashboard, n, v)
        cls.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            h = dict(headers or {})
            if body is not None:
                h.setdefault("Content-Type", "application/json")
                body = json.dumps(body)
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            return r.status, r, r.read()
        finally:
            conn.close()

    def get(self, path, headers=None):
        return self.request("GET", path, headers=headers)

    def get_json(self, path):
        status, r, raw = self.get(path)
        self.assertIn("application/json", r.getheader("Content-Type", ""), path)
        return status, json.loads(raw)


class TestEmptyState(DashboardCase):
    """Every API route answers cleanly with NO runs/config/logs on disk."""

    JSON_ROUTES = ("/api/runs", "/api/perf", "/api/meta", "/api/config", "/api/benchmarks",
                   "/api/summary", "/api/compare", "/api/sys", "/api/activity", "/api/history",
                   "/api/live", "/api/stream", "/api/live_details", "/api/log",
                   "/api/server/status", "/api/pulse")

    def test_api_routes_valid_json(self):
        for route in self.JSON_ROUTES:
            with self.subTest(route=route):
                status, obj = self.get_json(route)
                self.assertEqual(status, 200)
                self.assertIsInstance(obj, (dict, list))

    def test_empty_state_shapes(self):
        self.assertEqual(self.get_json("/api/runs")[1], [])
        self.assertEqual(self.get_json("/api/config")[1], {})       # tempdir config.json absent
        self.assertEqual(self.get_json("/api/live")[1], {"running": False})
        s = self.get_json("/api/summary")[1]
        for key in ("benchmarks", "per_run", "by_benchmark", "macro", "bias"):
            self.assertIn(key, s)
        self.assertEqual(s["per_run"], [])                          # no runs -> no rows
        meta = self.get_json("/api/meta")[1]
        self.assertEqual(meta["benchmarks"], [])                    # tempdir data/ is empty
        cmp_ = self.get_json("/api/compare")[1]                     # no params -> clean refusal
        self.assertFalse(cmp_["ok"])
        self.assertIn("error", cmp_)

    def test_index_page_csp_and_csrf(self):
        status, r, raw = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", r.getheader("Content-Type", ""))
        self.assertEqual(r.getheader("Content-Security-Policy"), dashboard.CSP)
        # assets are vendored: the policy allows NO third-party origin at all
        self.assertEqual(dashboard.CSP,
                         "default-src 'none'; script-src 'self' 'unsafe-inline'; "
                         "style-src 'self' 'unsafe-inline'; font-src 'self'; "
                         "connect-src 'self'; img-src 'self' data:; "
                         "base-uri 'none'; frame-ancestors 'none'")
        for host in (b"jsdelivr.net", b"fonts.googleapis.com", b"fonts.gstatic.com"):
            self.assertNotIn(host, raw)                             # page is fully offline
        self.assertIn(dashboard.CSRF_TOKEN.encode(), raw)           # token injected...
        self.assertNotIn(b"__BENCHY_CSRF__", raw)                   # ...placeholder gone

    def test_report_is_markdown(self):
        status, r, raw = self.get("/api/report")
        self.assertEqual(status, 200)
        self.assertIn("text/markdown", r.getheader("Content-Type", ""))
        self.assertIn("attachment", r.getheader("Content-Disposition", ""))
        self.assertIn(b"benchmark report", raw)

    def test_details_without_file_is_clean_404(self):
        status, obj = self.get_json("/api/details")
        self.assertEqual(status, 404)
        self.assertIn("error", obj)

    def test_unknown_route_404(self):
        status, _, _ = self.get("/nope")
        self.assertEqual(status, 404)


class TestStaticVendor(DashboardCase):
    """/static/vendor/<name> — vendored third-party assets: exact-name allowlist only
    (no path interpretation), correct Content-Type, day-long cache; anything else 404s."""

    def test_vendor_assets_served(self):
        for name, ctype in dashboard.VENDOR_FILES.items():
            with self.subTest(name=name):
                status, r, raw = self.get("/static/vendor/" + name)
                self.assertEqual(status, 200)
                self.assertEqual(r.getheader("Content-Type"), ctype)
                self.assertEqual(r.getheader("Cache-Control"), "max-age=86400")
                self.assertGreater(len(raw), 1000)                   # a real asset, not a stub

    def test_vendor_traversal_404(self):
        for path in ("/static/vendor/..%2Fdashboard.py", "/static/vendor/../dashboard.py",
                     "/static/vendor/..", "/static/vendor/sub/chart.umd.min.js",
                     "/static/vendor/nosuch.js", "/static/vendor/"):
            with self.subTest(path=path):
                status, _, _ = self.get(path)
                self.assertEqual(status, 404)
        status, _ = self.get_json("/api/runs")                       # still serving afterwards
        self.assertEqual(status, 200)


class TestPopulated(DashboardCase):
    """Synthetic runs.jsonl + details files served back through the API."""

    DETAILS_A = "synth__nothink__A1.jsonl"
    DETAILS_B = "synth__nothink__B1.jsonl"

    @classmethod
    def populate(cls):
        def rec(tag, details, acc, correct):
            return {"ts": "2026-06-10T12:00:00", "tag": tag, "benchmark": "synth",
                    "mode": bc.MODE_NOTHINK, "kind": "mcq", "errors": 0, "n": 4,
                    "correct": correct, "accuracy": acc, "n_options": 4,
                    "letter_dist": {"A": 2, "B": 2}, "details": details,
                    "shuffle_options": True, "data_sha": "feedfacecafe", "model": "m-q4"}
        with open(os.path.join(cls.results, "runs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec("tA", cls.DETAILS_A, 75.0, 3)) + "\n")
            f.write(json.dumps(rec("tB", cls.DETAILS_B, 75.0, 3)) + "\n")
        oks = {cls.DETAILS_A: [True, True, True, False],   # discordant on q2 and q4 ->
               cls.DETAILS_B: [True, False, True, True]}   # b=1, c=1, p=1.0
        for fn, flags in oks.items():
            with open(os.path.join(cls.results, "details", fn), "w", encoding="utf-8", newline="\n") as f:
                for i, ok in enumerate(flags):
                    f.write(json.dumps({"i": i + 1, "question": "q%d" % (i + 1), "ok": ok,
                                        "pred": "A", "gold": "A"}) + "\n")

    def test_runs_roundtrip(self):
        status, runs = self.get_json("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual([r["tag"] for r in runs], ["tA", "tB"])
        self.assertEqual(runs[0]["details"], self.DETAILS_A)

    def test_summary_well_formed(self):
        _, s = self.get_json("/api/summary")
        rows = [x for x in s["per_run"] if x["benchmark"] == "synth"]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNotNone(row["ci_lo"])
            self.assertIsNotNone(row["ci_hi"])
            self.assertTrue(row["ci_lo"] <= 75.0 <= row["ci_hi"])
            self.assertTrue(row["small_n"])                          # n=4 -> wide-CI flag
        self.assertIsNotNone(s["by_benchmark"]["synth"]["nothink"])
        self.assertEqual(s["macro"]["nothink_mean"], 75.0)

    def test_details_route(self):
        status, rows = self.get_json("/api/details?file=" + self.DETAILS_A)
        self.assertEqual(status, 200)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["question"], "q1")

    def test_details_traversal_is_clean_404_and_server_survives(self):
        for fn in ("..", "..%2F..%2Fruns.jsonl", "nosuch.jsonl"):
            with self.subTest(file=fn):
                status, obj = self.get_json("/api/details?file=" + fn)
                self.assertEqual(status, 404)
                self.assertIn("error", obj)
        status, _ = self.get_json("/api/runs")                       # still serving afterwards
        self.assertEqual(status, 200)

    def test_compare_route(self):
        _, v = self.get_json("/api/compare?a=tA&b=tB&bench=synth")
        self.assertTrue(v["ok"])
        self.assertEqual((v["a_better"], v["b_better"], v["n_common"]), (1, 1, 4))
        self.assertEqual(v["p_value"], 1.0)
        self.assertFalse(v["significant"])


class TestPulse(DashboardCase):
    """/api/pulse — the consolidated poll: one payload per tick, since-cursor history increments,
    and the sys/activity/server routes all serving the sampler's cached snapshot."""

    KEYS = ("live", "stream", "server", "sys", "activity", "hist", "now")

    @classmethod
    def populate(cls):
        # seed the sampler snapshot synchronously under THIS class's stubs — exactly what
        # __main__ does before serve_forever; request handlers only ever READ _SNAP
        dashboard._SNAP["cur"] = dashboard._snapshot()
        cls._hist_saved = list(dashboard.HIST)
        dashboard.HIST.clear()
        for i in range(1, 6):                        # five sampler rows: t = 1.0 … 5.0
            dashboard.HIST.append({"t": float(i), "tps": 10.0 * i, "up": False})

    @classmethod
    def tearDownClass(cls):
        dashboard.HIST.clear()
        dashboard.HIST.extend(cls._hist_saved)
        dashboard._SNAP["cur"] = None
        super().tearDownClass()

    def test_pulse_shape(self):
        status, d = self.get_json("/api/pulse")
        self.assertEqual(status, 200)
        for key in self.KEYS:
            self.assertIn(key, d)
        self.assertEqual(d["live"], {"running": False})      # empty results tree
        self.assertEqual(d["server"], {"up": False})         # stubbed server_up, via the snapshot
        self.assertIsInstance(d["sys"], dict)
        self.assertIsInstance(d["activity"], dict)
        self.assertEqual(d["stream"]["live"], d["live"])     # stream embeds the same live.json read

    def test_since_cursor_semantics(self):
        _, first = self.get_json("/api/pulse")               # since absent -> buffer tail (≤300)
        self.assertEqual([r["t"] for r in first["hist"]], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(first["now"], 5.0)
        _, inc = self.get_json("/api/pulse?since=3")         # strictly newer rows only
        self.assertEqual([r["t"] for r in inc["hist"]], [4.0, 5.0])
        self.assertEqual(inc["now"], 5.0)
        _, caught = self.get_json("/api/pulse?since=%s" % inc["now"])   # cursor echo -> no overlap
        self.assertEqual(caught["hist"], [])
        self.assertEqual(caught["now"], 5.0)
        _, bad = self.get_json("/api/pulse?since=bogus")     # malformed cursor degrades to 0
        self.assertEqual(len(bad["hist"]), 5)

    def test_polled_routes_serve_the_cached_snapshot(self):
        # /api/sys, /api/activity and /api/server/status answer from the same cached sampler
        # snapshot the pulse carries — no per-request subprocess probing
        _, d = self.get_json("/api/pulse")
        self.assertEqual(d["sys"], self.get_json("/api/sys")[1])
        self.assertEqual(d["activity"], self.get_json("/api/activity")[1])
        self.assertEqual(d["server"], self.get_json("/api/server/status")[1])

    def test_unseeded_fallback_computes_without_caching(self):
        # when the sampler has never run (import-only contexts), _latest_snap computes a
        # snapshot on demand but must NOT publish it: the sampler thread is the sole writer
        # of _SNAP, so a slow request can never clobber a newer sampler snapshot at startup
        seeded = dashboard._SNAP["cur"]
        try:
            dashboard._SNAP["cur"] = None
            snap = dashboard._latest_snap()
            for key in ("sys", "activity", "server"):
                self.assertIn(key, snap)
            self.assertIsNone(dashboard._SNAP["cur"])    # fallback did not cache
        finally:
            dashboard._SNAP["cur"] = seeded


class TestReportFlags(DashboardCase):
    """make_report exports runner verdicts honestly: an errors column plus INVALID /
    mode-suspect / unlocked markers — while legacy records (none of the additive
    fields) still render as clean rows."""

    @classmethod
    def populate(cls):
        recs = [
            # legacy record: predates errors/invalid/mode_suspect/locked — must render clean
            {"ts": "2026-06-01T10:00:00", "tag": "legacy", "benchmark": "oldbench",
             "mode": bc.MODE_NOTHINK, "n": 50, "accuracy": 80.0, "sec_per_q": 1.2, "notes": ""},
            # >5% of requests failed: runner marked the run invalid
            {"ts": "2026-06-02T10:00:00", "tag": "bad", "benchmark": "synth",
             "mode": bc.MODE_NOTHINK, "n": 88, "accuracy": 62.0, "sec_per_q": 1.0,
             "errors": 12, "invalid": True, "locked": True},
            # labelled thinking but the server ignored the flag; dataset unpinned
            {"ts": "2026-06-03T10:00:00", "tag": "fakethink", "benchmark": "synth",
             "mode": bc.MODE_THINKING, "n": 100, "accuracy": 70.0, "sec_per_q": 2.0,
             "errors": 0, "mode_suspect": True, "locked": False},
        ]
        with open(os.path.join(cls.results, "runs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def report(self):
        status, _, raw = self.get("/api/report")
        self.assertEqual(status, 200)
        return raw.decode("utf-8")

    def test_degraded_runs_are_flagged(self):
        text = self.report()
        self.assertIn("| date | benchmark | tag | mode | N | accuracy | errors | s/q | notes |", text)
        self.assertIn("**62.0%** ⚠ INVALID (12 err)", text)            # invalid run cannot read clean
        self.assertIn("| %s ⚠ |" % bc.MODE_THINKING, text)             # mode_suspect marks the mode cell
        self.assertIn("| synth † |", text)                             # locked: false marks the benchmark
        self.assertIn("NOT comparable", text)                          # footnote explains INVALID
        self.assertIn("benchmarks.lock.json", text)                    # footnote explains the † marker
        line = next(l for l in text.splitlines() if "| bad |" in l)
        self.assertIn("| 12 |", line)                                  # errors column carries the count

    def test_legacy_record_renders_clean(self):
        line = next(l for l in self.report().splitlines() if "| legacy |" in l)
        self.assertIn("**80.0%**", line)
        self.assertNotIn("⚠", line)
        self.assertNotIn("†", line)
        self.assertIn("| **80.0%** |  | 1.2 |", line)                  # additive errors column just empty


class TestFetchSingleFlight(DashboardCase):
    """/api/benchmarks/fetch in-process gate: concurrent requests spawn ONE fetch process
    (the check-then-act is atomic under _FETCH_GATE — no TOCTOU double-spawn)."""

    def test_concurrent_fetches_spawn_one_process(self):
        class FakeProc:
            def poll(self):
                return None                                            # "still running" forever

        class FakeFB:
            REGISTRY = {"benchk": {}}

        spawned = []

        def fake_popen(*a, **kw):
            spawned.append(a)
            return FakeProc()

        saved = (dashboard.FB, dashboard.subprocess.Popen, dashboard.PROC.get("fetch"))
        dashboard.FB, dashboard.subprocess.Popen = FakeFB, fake_popen
        dashboard.PROC["fetch"] = None
        try:
            results, n = [], 8
            barrier = threading.Barrier(n)
            def go():
                barrier.wait()                                         # maximize the race window
                results.append(dashboard.fetch_benchmarks_async(["benchk"]))
            threads = [threading.Thread(target=go) for _ in range(n)]
            for t in threads: t.start()
            for t in threads: t.join(timeout=10)
            self.assertEqual(len(spawned), 1)                          # exactly one api.py lock spawn
            self.assertEqual(sum(1 for r in results if r.get("ok")), 1)
            for r in results:
                if not r.get("ok"):
                    self.assertIn("already running", r["error"])
        finally:
            dashboard.FB, dashboard.subprocess.Popen = saved[0], saved[1]
            dashboard.PROC["fetch"] = saved[2]


class TestGuardMatrix(DashboardCase):
    """The DNS-rebinding / CSRF guard fails closed on every spoofable input."""

    def test_get_good_host_ok(self):
        status, _, _ = self.get("/")                                  # http.client sends 127.0.0.1:port
        self.assertEqual(status, 200)

    def test_get_localhost_host_ok(self):
        status, _, _ = self.get("/", headers={"Host": "localhost:%d" % self.port})
        self.assertEqual(status, 200)

    def test_get_evil_host_403(self):
        status, _, _ = self.get("/", headers={"Host": "evil.com"})
        self.assertEqual(status, 403)

    def test_get_wrong_port_host_403(self):
        status, _, _ = self.get("/", headers={"Host": "127.0.0.1:1"})
        self.assertEqual(status, 403)

    def test_get_missing_host_403(self):
        # raw HTTP/1.0 without a Host header at all — must fail CLOSED, not open
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as s:
            s.sendall(b"GET / HTTP/1.0\r\n\r\n")
            reply = s.makefile("rb").read()
        self.assertTrue(reply.startswith(b"HTTP/1.0 403"), reply[:60])

    def test_get_evil_origin_403(self):
        status, _, _ = self.get("/api/runs", headers={"Origin": "http://evil.com"})
        self.assertEqual(status, 403)

    def test_get_evil_referer_403(self):
        status, _, _ = self.get("/api/runs", headers={"Referer": "http://evil.com/x"})
        self.assertEqual(status, 403)

    def test_get_malformed_origin_clean_403(self):
        # "http://[::1" makes urlparse raise ValueError — the guard must answer a clean 403,
        # never let the exception kill the request mid-handler
        for hdr in ("Origin", "Referer"):
            with self.subTest(header=hdr):
                status, _, _ = self.get("/api/runs", headers={hdr: "http://[::1"})
                self.assertEqual(status, 403)
        status, _ = self.get_json("/api/runs")                        # still serving afterwards
        self.assertEqual(status, 200)

    def test_get_same_origin_full_match_ok(self):
        # positive control for full-origin enforcement: scheme+host+port all match
        status, _, _ = self.get("/api/runs", headers={"Origin": "http://127.0.0.1:%d" % self.port})
        self.assertEqual(status, 200)
        status, _, _ = self.get("/api/runs", headers={"Referer": "http://localhost:%d/index" % self.port})
        self.assertEqual(status, 200)

    def test_get_loopback_origin_wrong_port_403(self):
        # a loopback HOSTNAME alone is not same-origin: the port must match too
        wrong = 9999 if self.port != 9999 else 9998
        status, _, _ = self.get("/api/runs", headers={"Origin": "http://127.0.0.1:%d" % wrong})
        self.assertEqual(status, 403)

    def test_get_loopback_origin_wrong_scheme_403(self):
        # ...and so must the scheme (we only ever serve plain http on loopback)
        status, _, _ = self.get("/api/runs", headers={"Origin": "https://127.0.0.1:%d" % self.port})
        self.assertEqual(status, 403)

    def test_get_portless_loopback_origin_403(self):
        # "http://localhost" (implicit :80) is a DIFFERENT origin than localhost:<port>
        status, _, _ = self.get("/api/runs", headers={"Origin": "http://localhost"})
        self.assertEqual(status, 403)

    def test_post_without_csrf_403(self):
        status, _, _ = self.request("POST", "/api/run", body={"benchmark": "x"})
        self.assertEqual(status, 403)

    def test_post_bad_csrf_403(self):
        status, _, _ = self.request("POST", "/api/run", body={"benchmark": "x"},
                                    headers={"X-Benchy-CSRF": "0" * 32})
        self.assertEqual(status, 403)

    def test_post_token_but_evil_host_403(self):
        status, _, _ = self.request("POST", "/api/run", body={"benchmark": "x"},
                                    headers={"X-Benchy-CSRF": dashboard.CSRF_TOKEN,
                                             "Host": "evil.com"})
        self.assertEqual(status, 403)

    def test_post_token_but_cross_port_origin_403(self):
        # a valid token cannot ride along with a non-same-origin Origin header
        wrong = 9999 if self.port != 9999 else 9998
        status, _, _ = self.request("POST", "/api/run", body={"benchmark": "x"},
                                    headers={"X-Benchy-CSRF": dashboard.CSRF_TOKEN,
                                             "Origin": "http://127.0.0.1:%d" % wrong})
        self.assertEqual(status, 403)

    def test_post_token_malformed_origin_403(self):
        status, _, _ = self.request("POST", "/api/run", body={"benchmark": "x"},
                                    headers={"X-Benchy-CSRF": dashboard.CSRF_TOKEN,
                                             "Origin": "http://[::1"})
        self.assertEqual(status, 403)

    def test_post_with_token_passes_guard(self):
        # positive control: the same POST with the real token reaches the handler
        # (and is then refused by start_eval because no model server is up)
        status, r, raw = self.request("POST", "/api/run", body={"benchmark": "x"},
                                      headers={"X-Benchy-CSRF": dashboard.CSRF_TOKEN})
        self.assertEqual(status, 200)
        obj = json.loads(raw)
        self.assertFalse(obj["ok"])
        self.assertIn("server", obj["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
