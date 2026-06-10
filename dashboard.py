#!/usr/bin/env python3
"""benchy — local LLM benchmark explorer & live dashboard.

  python3 dashboard.py [port]      # default 8050  ->  http://127.0.0.1:8050

Stdlib backend (no pip deps); the SPA ships alongside in dashboard.html with its
third-party assets (Chart.js, marked, DOMPurify, highlight.js — see NOTICE) vendored
under static/vendor/, so the dashboard is fully offline (re-inline the page with
make_dist.py for a single-file copy). Lets you:
 - watch a live Q/A stream + real-time accuracy chart,
 - watch the model's real-time decode activity (tokens/s, phase, throughput),
 - launch the model server and an eval run from the UI,
 - explore accuracy + performance benchmarks and export a Markdown report.

Nothing about the model or host is hardcoded: the model id is read from the server's
/v1/models endpoint and host specs from the OS. Optional comparison baselines and a
display title live in a git-ignored config.json (written by the in-UI guided setup).
"""
import json, os, sys, re, math, datetime, subprocess, threading, time, collections, platform, secrets
import urllib.request as ureq
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import fetch_benchmarks as FB   # benchmark registry + downloader (optional)
    import api as API               # lockfile contract: pinned fetch + content hashes (optional)
except Exception:
    FB = API = None
import benchy_common as bc                # shared runner contract: settings(), MODE_*/KIND_* constants, write_live
from benchy_common import write_live      # atomic JSON write (tmp + os.replace)
import benchy_stats                       # extracted stats core (Wilson/χ²/McNemar/summary) — headless-importable
from benchy_stats import read_jsonl, is_rubric, wilson, _chi2_uniform, _mcnemar_p   # re-exported; same names as pre-split

HERE = os.path.dirname(os.path.abspath(__file__))
# BENCHY_RESULTS points the dashboard at another checkout's results/ (e.g. to watch a
# run launched from a different benchy copy); everything live/runs/details follows it.
RESULTS = os.environ.get("BENCHY_RESULTS") or os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(RESULTS, "runs.jsonl")
PERF = os.path.join(RESULTS, "perf.jsonl")
LIVE = os.path.join(RESULTS, "live.json")
STREAM = os.path.join(RESULTS, "stream.jsonl")
CONFIG_PATH = os.path.join(HERE, "config.json")  # optional local settings (git-ignored); written by guided setup
SERVER_BASE = bc.settings()["server_base"]  # OpenAI-compatible server: env BENCHY_SERVER > config.json server_base > bc.DEFAULT_SERVER
CHAT_URL = SERVER_BASE + "/v1/chat/completions"
MODELS_URL = SERVER_BASE + "/v1/models"
try: SERVER_PORT = urlparse(SERVER_BASE).port or 8000
except ValueError: SERVER_PORT = 8000       # malformed port in the URL — fall back to the conventional default
DS4 = os.environ.get("DS4_DIR", os.path.expanduser("~/ds4"))  # optional ds4 checkout: fallback for the start-server button when config.json has no server.cmd
PROC = {"eval": None, "server": None, "fetch": None}
_FETCH_GATE = threading.Lock()  # atomic test-and-set around the fetch check+spawn (ThreadingHTTPServer: two rapid POSTs must not double-spawn)
ALLOW_CODE_EXEC = os.environ.get("BENCHY_ALLOW_CODE_EXEC", "").lower() in ("1", "true", "yes", "on")  # gate model-code execution
CSRF_TOKEN = secrets.token_hex(16)  # per-launch token; required as X-Benchy-CSRF on every POST (CSRF/DNS-rebind guard)
DASH_PORT = 8050                     # overwritten in __main__ from argv; used for the Host allowlist
HISTORY = os.path.join(RESULTS, "metrics.jsonl")
HIST = collections.deque(maxlen=900)   # ~30 min @ 2s — survives browser refresh; persisted to metrics.jsonl

def load_config():
    """Optional local settings (config.json, git-ignored): reference baselines, display
    title, server start command, etc. Empty by default so a fresh clone runs fully generic."""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)

def load_shipped_references():
    """Published frontier-model baselines shipped with the repo (references.json, git-tracked,
    cited). Keys starting with '_' are notes. Merged under any config.json references."""
    try:
        with open(os.path.join(HERE, "references.json"), encoding="utf-8") as f:
            d = json.load(f) or {}
        return {k: v for k, v in d.items() if not k.startswith("_") and isinstance(v, list)}
    except Exception:
        return {}

SHIPPED_REFS = load_shipped_references()

def merged_references(cfg):
    """Shipped published baselines + the user's own (config.json). Per benchmark, a user
    entry with the same label overrides the shipped one; otherwise both are shown."""
    out = {b: list(lst) for b, lst in SHIPPED_REFS.items()}
    for b, lst in (cfg.get("references") or {}).items():
        if not isinstance(lst, list): continue
        labels = {x.get("label") for x in lst}
        out[b] = [x for x in out.get(b, []) if x.get("label") not in labels] + list(lst)
    return out

def detect_host():
    """Best-effort host facts from the OS (no config, nothing hardcoded): total RAM,
    CPU, OS — for the System-card denominator and the Environment chips."""
    info = {"cores": os.cpu_count(), "machine": platform.machine(), "system": platform.system()}
    try:
        if platform.system() == "Darwin":
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, encoding="utf-8", timeout=3).stdout.strip()
            if mem: info["mem_total_gb"] = round(int(mem) / 1e9, 1)
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, encoding="utf-8", timeout=3).stdout.strip()
            if cpu: info["cpu"] = cpu
            mac = platform.mac_ver()[0]
            if mac: info["os"] = "macOS " + mac
        else:
            try:
                with open("/proc/meminfo", encoding="utf-8") as f:
                    for ln in f:
                        if ln.startswith("MemTotal"):
                            info["mem_total_gb"] = round(int(ln.split()[1]) / 1e6, 1); break
            except Exception:
                pass
            info["os"] = " ".join(x for x in (platform.system(), platform.release()) if x)
    except Exception:
        pass
    return info

DETECTED = {"model": None, "host": detect_host()}   # model filled in by the metrics sampler when the server is up

def detect_model():
    """The model id the server advertises on /v1/models (OpenAI-compatible), so the UI
    shows whatever model is actually loaded — never a hardcoded name."""
    try:
        with ureq.urlopen(MODELS_URL, timeout=2) as r:
            data = json.load(r).get("data") or []
        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        return ids[0] if ids else None
    except Exception:
        return None

def model_id():
    """Model id to send in chat/eval payloads: explicit BENCHY_MODEL override > detected > 'default'."""
    return os.environ.get("BENCHY_MODEL") or DETECTED.get("model") or "default"

def capability_flags():
    """Honest capability flags for the UI (on /api/meta), so panels whose data source
    simply doesn't exist here say so instead of sitting empty forever:
      darwin            — host metrics (vm_stat/sysctl memory, swap) are macOS-only
      ds4_log_telemetry — decode t/s & phase come from parsing a ds4-style results/server.log
                          (written when the server is launched from the dashboard); a
                          generic OpenAI-compatible server has no such log
      server_managed    — this dashboard launched the model server process itself"""
    telem = False
    try:
        log = os.path.join(RESULTS, "server.log")
        if os.path.exists(log):
            with open(log, "rb") as f:
                f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 14000))
                tail = f.read().decode("utf-8", "ignore")
            telem = bool(re.search(r"(chunk|avg)=[\d.]+ t/s", tail))
    except Exception:
        telem = False
    pr = PROC.get("server")
    return {"darwin": platform.system() == "Darwin",
            "ds4_log_telemetry": telem,
            "server_managed": bool(pr is not None and pr.poll() is None)}

def meta_payload():
    """Identity/config for the UI — all auto-detected or from the optional config.json.
    No static model/hardware/benchmark values are baked in."""
    cfg = load_config()
    host = DETECTED.get("host") or {}
    model = DETECTED.get("model") or cfg.get("model_name")
    env = {}
    if model: env["Model"] = model
    machine = " · ".join(str(x) for x in (host.get("cpu") or host.get("machine"),
                         (str(host["mem_total_gb"]) + " GB RAM") if host.get("mem_total_gb") else None,
                         host.get("os")) if x)
    if machine: env["Machine"] = machine
    env.update(cfg.get("env") or {})   # optional user-supplied chips (engine, quant, build…)
    return {
        "title": cfg.get("title") or "benchy",
        "subtitle": cfg.get("subtitle"),   # None -> the HTML keeps its generic default
        "model": model, "host": host, "env": env, "server": SERVER_BASE,
        "flags": capability_flags(),
        "references": merged_references(cfg),
        "options": cfg.get("options") or {},
        "baseline_tag": cfg.get("baseline_tag") or "baseline",
        "primary_ref": cfg.get("primary_ref"),
        "configured": bool(cfg),
    }

def read_live():
    if not os.path.exists(LIVE): return {"running": False}
    try:
        d = json.load(open(LIVE, encoding="utf-8"))
        if d.get("running") and d.get("ts"):
            ts = d["ts"]
            age = ((datetime.datetime.now() - datetime.datetime.fromisoformat(ts)).total_seconds()
                   if isinstance(ts, str) else datetime.datetime.now().timestamp() - float(ts))
            if age > 120:
                # live.json is written once per question; a long thinking question can exceed 120s.
                # Don't declare the run dead while the model is still actively decoding (server.log fresh).
                lp = os.path.join(RESULTS, "server.log")
                decoding = False
                try:
                    decoding = os.path.exists(lp) and (datetime.datetime.now().timestamp() - os.path.getmtime(lp)) < 30
                except Exception:
                    decoding = False
                if not decoding:
                    d["running"] = False; d["stale"] = True
        return d
    except Exception: return {"running": False}

def _per_q_latency(evs):
    """eval_mcq writes per-question `t`; healthbench (pre-fix) writes CUMULATIVE `t`
    (elapsed since run start). If the series is monotonic non-decreasing it's cumulative
    -> return per-question diffs; otherwise it's already per-question -> return as-is."""
    ts = [e.get("t") for e in evs]
    vals = [t for t in ts if isinstance(t, (int, float))]
    cumulative = len(vals) >= 3 and all(b >= a for a, b in zip(vals, vals[1:]))
    out, prev = [], 0.0
    for t in ts:
        if not isinstance(t, (int, float)): out.append(None); continue
        out.append(round(t - prev, 1) if cumulative else t); prev = t
    return out, cumulative

def read_stream():
    evs = read_jsonl(STREAM); live = read_live()
    perq, cumulative = _per_q_latency(evs)
    for e, q in zip(evs, perq): e["lat_q"] = q   # corrected per-question latency
    return {"running": bool(live.get("running")), "live": live, "events": evs[-40:],
            "series": [[e.get("i"), e.get("acc")] for e in evs],
            "lat": [[e.get("i"), q] for e, q in zip(evs, perq)],
            "lat_cumulative_fixed": cumulative}

def benchmarks():
    if not os.path.isdir(DATA): return []
    return sorted(f[:-6] for f in os.listdir(DATA)
                  if f.endswith(".jsonl") and os.path.getsize(os.path.join(DATA, f)) > 0)

def bench_fit(bench, path):
    """Which runner a benchmark needs ('mcq' | 'code' | 'rubric'). The declarative registry
    (benchmarks/*.json, via fetch_benchmarks) is authoritative when the benchmark is
    registered; healthbench_* subsets are rubric-graded (manual registry entries — no spec
    file); first-row sniffing (bench_kind) survives ONLY as a fallback for BYO files."""
    try:
        if FB is not None and bench in FB.REGISTRY:
            return FB.fit_of(FB.REGISTRY[bench])
    except Exception:
        pass
    if is_rubric(bench):
        return bc.KIND_RUBRIC
    return bench_kind(path)

def bench_kind(path):
    """Peek a benchmark file's first row to pick the runner: 'code' (has tests/entry_point)
    vs 'mcq' (has options) — the fallback for files the registry doesn't know (see bench_fit)."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                if "options" in row: return "mcq"
                if "tests" in row or "entry_point" in row: return "code"
                return "mcq"
    except Exception:
        pass
    return "mcq"

def server_up():
    try:
        with ureq.urlopen(MODELS_URL, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def server_pid():
    """PID of whatever process is listening on the server port — works for any
    OpenAI-compatible server, not just ds4. Used for live RSS/CPU."""
    try:
        out = subprocess.run(["lsof", "-ti", "tcp:%d" % SERVER_PORT, "-sTCP:LISTEN"],
                             capture_output=True, encoding="utf-8", timeout=3).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None

def start_server():
    """Start the model server from the UI. Reads server.cmd/server.cwd from config.json
    (any OpenAI-compatible server); falls back to a ds4 checkout if one is present."""
    if server_up(): return {"ok": True, "already": True}
    DETECTED["model"] = None   # a (re)start can load a different model — re-detect once it's up
    sc = load_config().get("server") or {}
    cmd, cwd, shell = sc.get("cmd"), sc.get("cwd") or HERE, False
    if not cmd:
        if os.path.exists(os.path.join(DS4, "ds4-server")):   # ds4 dev fallback
            cmd = ["./ds4-server", "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
                   "--ctx", "8192", "--port", str(SERVER_PORT)]
            cwd = DS4
        else:
            return {"ok": False, "error": "no start command configured — start your OpenAI-compatible "
                    "server manually, or set \"server\": {\"cmd\": [...]} in config.json (or DS4_DIR for ds4)."}
    if isinstance(cmd, str): shell = True
    log = open(os.path.join(RESULTS, "server.log"), "a", encoding="utf-8")
    try:
        PROC["server"] = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=log, start_new_session=True, shell=shell)
    except Exception as e:
        return {"ok": False, "error": "could not start server: %s" % e}
    return {"ok": True, "starting": True}

def start_eval(p):
    if read_live().get("running"): return {"ok": False, "error": "a run is already in progress"}
    if not server_up(): return {"ok": False, "error": "model server not reachable on :%d — start it first" % SERVER_PORT}
    bench = re.sub(r"[^A-Za-z0-9_.-]", "", str(p.get("benchmark", "")))
    path = os.path.join(DATA, bench + ".jsonl")
    if not bench or not os.path.exists(path): return {"ok": False, "error": "benchmark not found: " + (bench or "(none)")}
    try: n = max(0, int(p.get("n", 25)))   # 0 = all rows (unified across all runners)
    except Exception: n = 25
    mode = "think" if p.get("mode") == bc.MODE_THINKING else "nothink"   # shared CLI token (think|nothink)
    tag = (re.sub(r"[^A-Za-z0-9_.-]", "", str(p.get("tag", "run"))) or "run")[:40]
    fit = bench_fit(bench, path)
    if fit == bc.KIND_CODE and not ALLOW_CODE_EXEC:
        return {"ok": False, "error": "code-execution benchmarks run model-written code on this host and are "
                "disabled by default. Relaunch the dashboard with BENCHY_ALLOW_CODE_EXEC=1 to enable them."}
    # unified runner CLI (benchy_common.parse_run_args): BENCH N MODE TAG [--seed INT]
    if fit == bc.KIND_RUBRIC:
        # rubric-graded, not MCQ — needs the HealthBench runner (grader via .apikey); BENCH = subset
        args = [sys.executable, os.path.join(HERE, "healthbench.py"), bench.replace("healthbench_", ""), str(n), mode, tag]
    elif fit == bc.KIND_CODE:
        # code-generation tasks — generate, then EXECUTE against tests (pass@1)
        args = [sys.executable, os.path.join(HERE, "eval_code.py"), path, str(n), mode, tag]
    else:
        args = [sys.executable, os.path.join(HERE, "eval_mcq.py"), path, str(n), mode, tag]
    log = open(os.path.join(RESULTS, "eval.log"), "a", encoding="utf-8")
    PROC["eval"] = subprocess.Popen(args, cwd=HERE, stdout=log, stderr=log, start_new_session=True)
    return {"ok": True, "started": {"benchmark": bench, "n": n, "mode": mode, "tag": tag}}

def stop_eval():
    pr = PROC.get("eval")
    if pr and pr.poll() is None:
        try: pr.terminate()
        except Exception: pass
        return {"ok": True, "stopped": True}
    return {"ok": True, "stopped": False}

def fetch_benchmarks_async(keys):
    """Download one or more registry benchmarks into data/ (detached) — through the api.py
    lock contract, NOT the raw fetcher: `api.py lock <key> ...` pins each set in
    benchmarks.lock.json and verifies its content hash (partial fetch = fatal). Failures
    land in fetch.log and exit non-zero, surfaced by fetch_outcome() on /api/benchmarks."""
    if FB is None: return {"ok": False, "error": "fetch_benchmarks.py not importable"}
    valid = [k for k in keys if k in FB.REGISTRY]
    if not valid: return {"ok": False, "error": "no known benchmark keys in request"}
    # single-flight, atomically: the running-check and the spawn+set happen under one lock,
    # so two rapid POSTs can never both pass the check and double-spawn `api.py lock`
    # (in-process gate; api.py adds its own cross-process lockfile safety)
    if not _FETCH_GATE.acquire(blocking=False):
        return {"ok": False, "error": "a fetch is already running"}
    try:
        if PROC.get("fetch") and PROC["fetch"].poll() is None:
            return {"ok": False, "error": "a fetch is already running"}
        os.makedirs(RESULTS, exist_ok=True)
        log = open(os.path.join(RESULTS, "fetch.log"), "w", encoding="utf-8")
        PROC["fetch"] = subprocess.Popen([sys.executable, os.path.join(HERE, "api.py"), "lock"] + valid,
                                         cwd=HERE, stdout=log, stderr=log, start_new_session=True)
    finally:
        _FETCH_GATE.release()
    return {"ok": True, "fetching": valid}

def fetch_outcome():
    """State of the last dashboard-launched fetch for /api/benchmarks: fetching (still
    running), fetch_rc, and on failure the '! <key>: <error>' lines api.py lock left in
    fetch.log — so a LockError / partial fetch surfaces in the UI instead of vanishing."""
    pr = PROC.get("fetch")
    if not pr:
        return {"fetching": False}
    if pr.poll() is None:
        return {"fetching": True}
    out = {"fetching": False, "fetch_rc": pr.returncode}
    if pr.returncode != 0:
        try:
            tail = open(os.path.join(RESULTS, "fetch.log"), encoding="utf-8").read()[-4000:]
            errs = [ln.strip() for ln in tail.splitlines() if ln.strip().startswith("!")]
            out["fetch_error"] = ("; ".join(errs))[:500] or "fetch exited with code %s" % pr.returncode
        except Exception:
            out["fetch_error"] = "fetch exited with code %s" % pr.returncode
    return out

_LOCK_SHA_CACHE = {}   # path -> ((mtime_ns, size), sha): hash a few-MB file once per change

def _cached_sha(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    sig = (st.st_mtime_ns, st.st_size)
    hit = _LOCK_SHA_CACHE.get(path)
    if hit and hit[0] == sig:
        return hit[1]
    sha = bc.sha256_file(path)
    _LOCK_SHA_CACHE[path] = (sig, sha)
    return sha

def lock_states():
    """Per-benchmark lockfile state for the UI — NO network and REPORT-ONLY: this GET path
    must never mutate benchmarks.lock.json (corrupt-lockfile quarantine/rewrite belongs to
    the api.py write commands; api._load_lock is side-effect-free by contract). Any loader
    failure — corrupt file, unexpected shape — degrades to {} (everything 'unpinned').
    States: 'locked' (entry pins a content_sha; the data file, when present, matches it),
    'drift' (file present but its hash no longer matches the pin), 'unpinned' (no entry /
    no content hash yet). Hashes are cached by (mtime, size) so the poll loop stays cheap."""
    if API is None:
        return {}
    try:
        entries = (API._load_lock() or {}).get("benchmarks") or {}
    except Exception:
        return {}   # unreadable lock -> report nothing as pinned; never quarantine/rewrite here
    out = {}
    for k, ent in entries.items():
        want = (ent or {}).get("content_sha")
        if not want:
            out[k] = "unpinned"; continue
        got = _cached_sha(os.path.join(DATA, k + ".jsonl"))
        out[k] = "locked" if (got is None or got == want) else "drift"
    return out

EVAL_PROC_PATTERNS = ("eval_mcq.py", "eval_code.py", "healthbench.py", "fetch_benchmarks.py",
                      "api.py lock",   # the dashboard's fetch path (fetch_benchmarks_async)
                      "run_sweep.sh", "cache_sweep.sh", "next_eval.sh", "healthbench_chain.sh")

def stop_all():
    """Forcefully stop ALL eval/chain processes — even those launched outside the dashboard."""
    killed = []
    for pat in EVAL_PROC_PATTERNS:
        try:
            if subprocess.run(["pkill", "-f", pat], capture_output=True).returncode == 0:
                killed.append(pat)
        except Exception: pass
    pr = PROC.get("eval")
    if pr and pr.poll() is None:
        try: pr.terminate()
        except Exception: pass
    # only mark the run dead in live.json once nothing matching is actually still alive —
    # a survivor would keep writing live.json and the UI must keep showing it as running
    time.sleep(0.3)   # give SIGTERM a beat to land
    survivors = []
    for pat in EVAL_PROC_PATTERNS:
        try:
            if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
                survivors.append(pat)
        except Exception: pass
    if survivors:
        return {"ok": False, "killed": killed, "still_running": survivors,
                "error": "still running after kill: " + ", ".join(survivors)}
    try: write_live(LIVE, {"running": False})
    except Exception: pass
    return {"ok": True, "killed": killed}

def kill_server():
    DETECTED["model"] = None   # whatever replaces this server may advertise a different model
    pid = server_pid()
    try:
        if pid:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True)
            return {"ok": True, "killed": pid}
        subprocess.run(["pkill", "-9", "-f", "ds4-server"], capture_output=True)   # ds4 dev fallback
        return {"ok": True, "killed": "server"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

STATS = benchy_stats.Stats(RESULTS)   # the extracted stats core, bound to this dashboard's results dir

def _details_for(tag, bench, mode=None):
    """Most-complete details file for a (tag, benchmark[, mode]) run — benchy_stats.Stats._details_for."""
    return STATS._details_for(tag, bench, mode)

def paired_compare(tag_a, tag_b, bench, mode=None):
    """McNemar paired comparison of two runs — benchy_stats.Stats.paired_compare."""
    return STATS.paired_compare(tag_a, tag_b, bench, mode)

def summary():
    """Server-side stats over runs.jsonl x optional reference baselines (Wilson CIs, gaps,
    macro-avg, answer-bias) — benchy_stats.Stats.summary, fed the config-derived references,
    option counts and primary-ref label exactly as before the split."""
    cfg = load_config()
    return STATS.summary(refs_all=merged_references(cfg),
                         opts_map=cfg.get("options") or {},
                         primary_label=cfg.get("primary_ref"))

def sysmetrics():
    """Live system + model-server metrics: server RSS, CPU, system memory, decode t/s."""
    m = {"server_up": server_up()}
    pid = server_pid()
    m["pid"] = pid
    if pid:
        try:
            r = subprocess.run(["ps", "-o", "rss=,%cpu=", "-p", str(pid)], capture_output=True, encoding="utf-8", timeout=3).stdout.split()
            if len(r) >= 2:
                m["rss_gb"] = round(int(r[0]) / 1048576, 1); m["cpu"] = float(r[1])
        except Exception: pass
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, encoding="utf-8", timeout=3).stdout
        psize = 16384
        mp = re.search(r"page size of (\d+)", vs)
        if mp: psize = int(mp.group(1))
        def pg(key):
            mm = re.search(re.escape(key) + r"[^\d]*(\d+)", vs); return int(mm.group(1)) if mm else 0
        m["sys"] = {"free_gb": round(pg("Pages free") * psize / 1e9, 1),
                    "active_gb": round(pg("Pages active") * psize / 1e9, 1),
                    "wired_gb": round(pg("Pages wired down") * psize / 1e9, 1),
                    "compressed_gb": round(pg("occupied by compressor") * psize / 1e9, 1)}
    except Exception: pass
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, encoding="utf-8", timeout=3).stdout
        ms = re.search(r"used = ([\d.]+)M", sw)
        if ms: m["swap_gb"] = round(float(ms.group(1)) / 1024, 1)
    except Exception: pass
    try:
        log = os.path.join(RESULTS, "server.log")
        if os.path.exists(log):
            with open(log, "rb") as f:
                f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 4000)); tail = f.read().decode("utf-8", "ignore")
            avg = re.findall(r"avg=([\d.]+) t/s", tail)
            gen = re.findall(r"gen=(\d+)", tail)
            if avg: m["decode_tps"] = float(avg[-1])
            if gen: m["gen_tokens"] = int(gen[-1])
            m["thinking"] = tail.rfind("THINKING") > tail.rfind("finish=")
    except Exception: pass
    return m

def activity():
    """Parse the tail of server.log into a live decode timeline for the LLM-activity view.

    Returns the current generation's per-chunk throughput series (resets each answer),
    the live phase (prefill / thinking / generating / done / idle), current gen-token
    count, instantaneous + average t/s, and the last completed generation's stats.
    Everything is derived from the real server log — no fabricated numbers.
    """
    out = {"server_up": server_up(), "phase": "idle", "thinking": False, "gen": 0,
           "chunk_tps": None, "avg_tps": None, "series": [], "last_finish": None, "age": None}
    log = os.path.join(RESULTS, "server.log")
    if not os.path.exists(log): return out
    try:
        out["age"] = round(datetime.datetime.now().timestamp() - os.path.getmtime(log), 1)
        with open(log, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 14000)); tail = f.read().decode("utf-8", "ignore")
        lines = [l for l in tail.splitlines() if l.strip()]
        # last completed generation (scan everything we have)
        for ln in lines:
            mf = re.search(r"gen=(\d+) finish=(\S+) ([\d.]+)s", ln)
            if mf:
                out["last_finish"] = {"tokens": int(mf.group(1)), "finish": mf.group(2),
                                      "secs": round(float(mf.group(3)), 1)}
        # current generation = lines after the last "prompt start"
        start = 0
        for idx in range(len(lines) - 1, -1, -1):
            if "prompt start" in lines[idx]: start = idx; break
        gen = 0; chunk_tps = avg = None; thinking = False; series = []
        for ln in lines[start:]:
            md = re.search(r"gen=(\d+).*?chunk=([\d.]+) t/s avg=([\d.]+) t/s", ln)
            if md:
                gen = int(md.group(1)); chunk_tps = float(md.group(2)); avg = float(md.group(3))
                th = "THINKING" in ln; thinking = th
                series.append({"gen": gen, "tps": chunk_tps, "thinking": th})
        out["series"] = series[-60:]; out["gen"] = gen
        out["chunk_tps"] = chunk_tps; out["avg_tps"] = avg; out["thinking"] = thinking
        last = lines[-1] if lines else ""
        recent = out["age"] is not None and out["age"] < 12
        if not out["server_up"] or not recent: out["phase"] = "idle"
        elif "finish=" in last: out["phase"] = "done"
        elif series: out["phase"] = "thinking" if thinking else "generating"
        else: out["phase"] = "prefill"
    except Exception:
        pass
    return out

_SNAP = {"cur": None}   # latest sampler snapshot; atomically swapped (single dict assignment) by _sample_metrics

def _snapshot():
    """Produce ONE sys+activity+server snapshot. This is the only place the polled GET data
    is computed — every subprocess probe (lsof/ps/vm_stat/sysctl) and the /v1/models poke
    happens here, in the background sampler thread, never per HTTP request."""
    m = sysmetrics(); a = activity()
    return {"sys": m, "activity": a, "server": {"up": bool(m.get("server_up"))},
            "t": round(time.time(), 1)}

def _latest_snap():
    """The sampler's last snapshot: /api/sys, /api/activity, /api/server/status and /api/pulse
    all serve this cached dict — zero subprocess spawns and zero :8000 probes per request.
    Live this never misses: __main__ seeds _SNAP synchronously before serve_forever. The
    None branch exists only for import-only contexts (e.g. the offline tests, where the
    sampler thread never starts): it computes a snapshot on demand and returns it WITHOUT
    caching — the sampler stays the sole writer of _SNAP, so a slow request thread can
    never overwrite a newer sampler snapshot with an older probe of its own."""
    snap = _SNAP["cur"]
    return snap if snap is not None else _snapshot()

def pulse_payload(since=0.0):
    """GET /api/pulse — the whole 1.5s browser tick in ONE response (replaces the old
    live/stream/activity/history/server-status/sys fan-out): live.json, the stream feed,
    the sampler's cached sys/activity/server snapshot, and the metrics ring-buffer rows
    with t > `since` ('hist'; since<=0 → last 300 rows, all the client ever renders).
    'now' is the newest buffered t — the client echoes it back as the next since cursor."""
    snap = _latest_snap()
    rows = list(HIST)
    hist = [r for r in rows if r.get("t", 0) > since] if since > 0 else rows[-300:]
    now = rows[-1].get("t", 0) if rows else 0
    stream = read_stream()   # embeds live.json under "live" — reuse it instead of re-reading
    return {"live": stream["live"], "stream": stream, "server": snap["server"],
            "sys": snap["sys"], "activity": snap["activity"], "hist": hist, "now": now}

def read_log_tail(which, n=300):
    """Tail of a results log file (server|eval) — for the in-UI debug drawer."""
    fn = {"server": "server.log", "eval": "eval.log", "fetch": "fetch.log"}.get(which, "server.log")
    fp = os.path.join(RESULTS, fn)
    if not os.path.exists(fp): return {"file": fn, "text": "(no log yet)"}
    try:
        with open(fp, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 40000)); txt = f.read().decode("utf-8", "ignore")
        return {"file": fn, "text": "\n".join(txt.splitlines()[-n:])}
    except Exception as e:
        return {"file": fn, "text": "error reading log: %s" % e}

def _load_history():
    """Warm the in-memory ring buffer from metrics.jsonl so the System charts are full right after a restart."""
    if not os.path.exists(HISTORY): return
    try:
        with open(HISTORY, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 300000)); tail = f.read().decode("utf-8", "ignore")
        for line in tail.splitlines()[-HIST.maxlen:]:
            line = line.strip()
            if line:
                try: HIST.append(json.loads(line))
                except Exception: pass
    except Exception: pass

def _trim_history():
    try:
        lines = open(HISTORY, encoding="utf-8").readlines()
        if len(lines) > 5000: open(HISTORY, "w", encoding="utf-8", newline="\n").writelines(lines[-2500:])
    except Exception: pass

def _sample_metrics():
    """Background sampler: every 2s snapshot system + decode metrics into the ring buffer + metrics.jsonl.
    Single source for the System charts AND the only producer of the _SNAP sys/activity/server
    snapshot the polled GET endpoints serve; keeps live history across page refreshes/restarts. All real samples."""
    n = 0
    was_up = False
    while True:
        try:
            snap = _snapshot()
            _SNAP["cur"] = snap   # atomic swap — request handlers only ever read this
            m, a = snap["sys"], snap["activity"]; sd = m.get("sys") or {}
            up = bool(m.get("server_up"))
            if up and (not was_up or not DETECTED.get("model")):
                # (re-)learn the loaded model id on every down->up transition — a server swap
                # must never leave the UI showing the previous model
                DETECTED["model"] = detect_model()
            was_up = up
            decoding = a.get("phase") in ("generating", "thinking")
            rec = {"t": round(time.time(), 1),
                   "rss": m.get("rss_gb"), "cpu": m.get("cpu"),
                   "used": round(sd.get("wired_gb", 0) + sd.get("active_gb", 0) + sd.get("compressed_gb", 0), 1) if sd else None,
                   "free": sd.get("free_gb"), "wired": sd.get("wired_gb"), "swap": m.get("swap_gb"),
                   "tps": ((a.get("chunk_tps") if a.get("chunk_tps") is not None else m.get("decode_tps")) if decoding else None),
                   "phase": a.get("phase"), "gen": a.get("gen"), "up": bool(m.get("server_up"))}
            HIST.append(rec)
            try:
                with open(HISTORY, "a", encoding="utf-8", newline="\n") as f: f.write(json.dumps(rec) + "\n")
            except Exception: pass
            n += 1
            if n % 250 == 0: _trim_history()
        except Exception: pass
        time.sleep(2)

def make_report():
    meta = meta_payload(); cfg = load_config()
    runs = read_jsonl(RUNS); perf = read_jsonl(PERF)
    refs_all = merged_references(cfg)
    L = [f"# {meta['title']} — benchmark report", "",
         f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_", "",
         "## Environment", ""]
    for k, v in meta["env"].items(): L.append(f"- **{k}**: {v}")
    if not meta["env"]: L.append("_(model/host not detected — start the model server)_")
    L += ["", "## Accuracy", "", "| date | benchmark | tag | mode | N | accuracy | errors | s/q | notes |",
          "|---|---|---|---|--:|--:|--:|--:|---|"]
    # Honest export: runs the runner flagged as degraded must not read as clean numbers.
    # All flag fields are additive — legacy records (none of them) render exactly as before.
    any_invalid = any_suspect = any_unlocked = False
    for r in runs:
        errs = r.get("errors")
        acc = f"**{r.get('accuracy','')}%**"
        if r.get("invalid"):   # runner verdict: >5% of requests failed — accuracy not comparable
            acc += " ⚠ INVALID (%s err)" % (errs if errs is not None else "?")
            any_invalid = True
        mode = f"{r.get('mode','')}"
        if r.get("mode_suspect"):   # labelled thinking, but the server ignored the think flag
            mode += " ⚠"
            any_suspect = True
        bench = f"{r.get('benchmark','')}"
        if r.get("locked") is False:   # explicit False only — absent means legacy/unknown
            bench += " †"
            any_unlocked = True
        L.append(f"| {r.get('ts','')} | {bench} | {r.get('tag','')} | {mode} | {r.get('n','')} "
                 f"| {acc} | {'' if errs is None else errs} | {r.get('sec_per_q','')} | {r.get('notes','') or ''} |")
    foot = []
    if any_invalid:
        foot.append("- ⚠ INVALID: more than 5% of the run's requests failed; the accuracy covers only "
                    "the scored questions and is NOT comparable to a clean run.")
    if any_suspect:
        foot.append("- mode ⚠: run was labelled thinking but almost no answers contained think content — "
                    "the server likely ignored the 'think' flag.")
    if any_unlocked:
        foot.append("- †: dataset was not pinned/verified against benchmarks.lock.json at run time "
                    "(BYO file or unpinned set).")
    if foot: L += [""] + foot
    if refs_all:
        L += ["", "### Reference baselines (external; different size/class)", ""]
        for bench, refs in refs_all.items():
            L += [f"**{bench}**", "", "| model | accuracy | source |", "|---|--:|---|"]
            for ref in refs: L.append(f"| {ref.get('label','')} | {ref.get('accuracy','')}% | {ref.get('source','')} |")
            L.append("")
    if perf:
        L += ["", "## Inference performance", "",
              "| kind | config | prefetch | prefill t/s | gen t/s | hit-rate | notes |", "|---|---|---|--:|--:|--:|---|"]
        for p in perf:
            hr = p.get('hit_rate'); hr = f"{hr:.3f}" if isinstance(hr, (int, float)) else "—"
            L.append(f"| {p.get('kind','')} | {p.get('config','')} | {p.get('prefetch','')} | {p.get('prefill_tps','')} "
                     f"| **{p.get('gen_tps','')}** | {hr} | {p.get('notes','') or ''} |")
    L += ["", "_Caveats: small-N runs have wide Wilson CIs; non-thinking can show letter bias; "
          "decode t/s ±~10%. See README.md._"]
    return "\n".join(L)

# >>> PAGE loader (make_dist.py replaces this block with the inlined literal) >>>
# The SPA (markup + inline JS, formerly the PAGE literal here) ships alongside in
# dashboard.html; it is read once at startup and the per-launch __BENCHY_CSRF__
# placeholder is still substituted at serve time in do_GET, exactly as before.
PAGE_PATH = os.path.join(HERE, "dashboard.html")
try:
    with open(PAGE_PATH, encoding="utf-8") as f:
        PAGE = f.read()
except OSError as e:
    sys.exit("benchy: cannot read dashboard.html (%s) — it ships alongside dashboard.py; "
             "restore it from the repo, or build a single-file dashboard with make_dist.py." % e)
# <<< PAGE loader <<<

# Sent with "/" only: all page JS is inline ('unsafe-inline'); every third-party asset is
# vendored under static/vendor/ (served same-origin by /static/vendor/<name>) and fonts are
# the system stacks — no CDN, no Google Fonts, fully offline. connect-src 'self' keeps
# fetches on this origin.
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; "
       "style-src 'self' 'unsafe-inline'; font-src 'self'; "
       "connect-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; frame-ancestors 'none'")

# Vendored third-party assets: fetched at the exact pinned versions below from
# cdn.jsdelivr.net at vendoring time; their content digests, licenses and sources are
# recorded in NOTICE. Served ONLY by exact-name allowlist lookup: the URL tail is a
# dict key, never a filesystem path, so traversal cannot occur.
VENDOR_DIR = os.path.join(HERE, "static", "vendor")
VENDOR_FILES = {
    "chart.umd.min.js":    "application/javascript; charset=utf-8",   # Chart.js 4.4.1 (MIT)
    "marked.min.js":       "application/javascript; charset=utf-8",   # marked 12.0.2 (MIT)
    "purify.min.js":       "application/javascript; charset=utf-8",   # DOMPurify 3.4.9 (Apache-2.0 OR MPL-2.0)
    "highlight.min.js":    "application/javascript; charset=utf-8",   # highlight.js 11.11.1 (BSD-3-Clause)
    "github-dark.min.css": "text/css; charset=utf-8",                 # highlight.js theme (BSD-3-Clause)
}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        if isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def _json(self, obj, code=200): self._send(code, json.dumps(obj), "application/json")
    def _body(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n: return {}
        try: return json.loads(self.rfile.read(n).decode() or "{}")
        except Exception: return {}
    def _host_ok(self):
        """Same-origin sanity for EVERY request (GET included — results/config are private and
        the page embeds the CSRF token): the Host header must be PRESENT and a loopback:port we
        serve (fail-closed defeats DNS rebinding), and any Origin/Referer must be the FULL
        same origin — scheme http AND the exact host:port we serve (mirrors the Host
        allowlist), not merely a loopback hostname. A malformed header that urlparse cannot
        parse (e.g. 'http://[::1') counts as NOT same-origin: clean 403, never a 500."""
        allowed = {"127.0.0.1:%d" % DASH_PORT, "localhost:%d" % DASH_PORT}
        host = (self.headers.get("Host") or "").strip()
        if host not in allowed:
            return False
        origin = self.headers.get("Origin") or self.headers.get("Referer") or ""
        if origin:
            try:
                u = urlparse(origin)
            except ValueError:       # unparseable Origin/Referer — treat as cross-site
                return False
            if u.scheme != "http" or u.netloc not in allowed:
                return False
        return True
    def _guard(self):
        """Reject cross-site / DNS-rebound POSTs: Host/Origin sanity plus the per-launch CSRF
        token (only our own same-origin page can read it). The token is POST-only by design."""
        return self._host_ok() and self.headers.get("X-Benchy-CSRF", "") == CSRF_TOKEN
    def do_GET(self):
        if not self._host_ok():
            return self._send(403, "forbidden — bad or missing Host/Origin (DNS-rebinding guard)", "text/plain")
        p = self.path.split("?")[0]
        if p == "/" or p.startswith("/index"):
            self._send(200, PAGE.replace("__BENCHY_CSRF__", CSRF_TOKEN), extra={"Content-Security-Policy": CSP})
        elif p.startswith("/static/vendor/"):
            # strict allowlist: the tail must be an EXACT key in VENDOR_FILES — it is never
            # interpreted as a path, so "..", encodings and separators all fall to 404
            name = p[len("/static/vendor/"):]
            ctype = VENDOR_FILES.get(name)
            fp = os.path.join(VENDOR_DIR, name) if ctype else None
            if not ctype or not os.path.isfile(fp):
                return self._send(404, "not found", "text/plain")
            with open(fp, "rb") as f:
                self._send(200, f.read(), ctype, extra={"Cache-Control": "max-age=86400"})
        elif p == "/api/runs": self._json(read_jsonl(RUNS))
        elif p == "/api/perf": self._json(read_jsonl(PERF))
        elif p == "/api/meta": self._json(dict(meta_payload(), benchmarks=benchmarks()))
        elif p == "/api/config": self._json(load_config())
        elif p == "/api/benchmarks":
            meta = FB.registry_meta() if FB else {"available": [], "manual": []}
            locks = lock_states()   # no network — lockfile + cached content hashes only
            for b in meta.get("available", []):
                b["baselines"] = len(SHIPPED_REFS.get(b["key"], []))   # # of shipped frontier baselines
                b["lock"] = locks.get(b["key"], "unpinned")
            self._json(dict(meta, **fetch_outcome()))
        elif p == "/api/summary": self._json(summary())
        elif p == "/api/compare":
            q = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            g = lambda k: (q.get(k) or [""])[0]
            self._json(paired_compare(g("a"), g("b"), g("bench"), g("mode") or None))
        elif p == "/api/sys": self._json(_latest_snap()["sys"])           # sampler snapshot — no probes per request
        elif p == "/api/activity": self._json(_latest_snap()["activity"])
        elif p == "/api/history": self._json(list(HIST))
        elif p == "/api/pulse":
            q = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            try: since = float((q.get("since") or ["0"])[0])
            except ValueError: since = 0.0
            self._json(pulse_payload(since))
        elif p == "/api/live_details":
            try:
                dd = os.path.join(RESULTS, "details")
                fs = [f for f in os.listdir(dd) if f.endswith(".jsonl")] if os.path.isdir(dd) else []
                fs.sort(key=lambda f: os.path.getmtime(os.path.join(dd, f)))
                fn = fs[-1] if fs else None
                rows = [json.loads(l) for l in open(os.path.join(dd, fn), encoding="utf-8") if l.strip()] if fn else []
                self._json({"file": fn, "rows": rows})
            except Exception:
                self._json({"file": None, "rows": []})
        elif p == "/api/log":
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            which = next((kv[6:] for kv in q.split("&") if kv.startswith("which=")), "server")
            self._json(read_log_tail(re.sub(r"[^a-z]", "", which) or "server"))
        elif p == "/api/details":
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            fn = next((kv[5:] for kv in q.split("&") if kv.startswith("file=")), "")
            fn = re.sub(r"[^A-Za-z0-9_.\-]", "", fn)
            fp = os.path.join(RESULTS, "details", fn)
            if not fn or not os.path.isfile(fp):   # isfile: ".." survives the sanitizer and is a directory
                self._json({"error": "details file not found: %s" % (fn or "(none)")}, 404)
            else:
                try:
                    self._json([json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()])
                except Exception as e:
                    self._json({"error": "could not read details: %s" % e}, 400)
        elif p == "/api/live": self._json(read_live())
        elif p == "/api/stream": self._json(read_stream())
        elif p == "/api/server/status": self._json(dict(_latest_snap()["server"]))   # same {"up": bool} shape, now cached
        elif p == "/api/report":
            fn = "benchy-report-%s.md" % datetime.date.today().isoformat()
            self._send(200, make_report(), "text/markdown; charset=utf-8",
                       {"Content-Disposition": 'attachment; filename="%s"' % fn})
        else: self._send(404, "not found", "text/plain")
    def stream_chat(self, body):
        """Proxy a chat to the OpenAI-compatible server and relay tokens to the browser as SSE.
        Real streaming when the server supports it; otherwise a non-streaming fallback chunked for UX."""
        if not server_up():
            return self._json({"error": "model server not reachable on :%d — start it from Control" % SERVER_PORT}, 503)
        msgs = body.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return self._json({"error": "no messages"}, 400)
        msgs = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in msgs if m.get("content")]
        payload = {"model": model_id(), "messages": msgs,
                   "max_tokens": max(1, min(8192, int(body.get("max_tokens", 2048) or 2048))),
                   "temperature": float(body.get("temperature", 0.7) or 0),
                   "think": bool(body.get("think")), "stream": True}
        try:
            req = ureq.Request(CHAT_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            up = ureq.urlopen(req, timeout=600)
        except Exception as e:
            return self._json({"error": "upstream error: %s" % e}, 502)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        def w(obj):
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode()); self.wfile.flush()
        try:
            streamed = False
            if "text/event-stream" in up.headers.get("Content-Type", ""):
                for raw in up:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"): continue
                    data = line[5:].strip()
                    if data == "[DONE]": break
                    try: delta = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                    except Exception: delta = ""
                    if delta: w({"t": delta}); streamed = True
            if not streamed:
                try: content = json.loads(up.read().decode("utf-8", "ignore"))["choices"][0]["message"]["content"]
                except Exception: content = ""
                buf = ""
                for ch in content:
                    buf += ch
                    if len(buf) >= 18 or ch in " \n": w({"t": buf}); buf = ""
                if buf: w({"t": buf})
            w({"done": True})
        except Exception as e:
            try: w({"error": str(e)})
            except Exception: pass
        finally:
            try: up.close()
            except Exception: pass
    def save_settings(self, body):
        """Persist guided-setup choices to config.json (git-ignored). Everything optional
        and sanitized: display title, baseline tag, and per-benchmark reference baselines."""
        cfg = load_config()
        if "title" in body:
            cfg["title"] = (str(body.get("title") or "").strip()[:80]) or None
        if "subtitle" in body:
            cfg["subtitle"] = (str(body.get("subtitle") or "").strip()[:160]) or None
        if "baseline_tag" in body:
            cfg["baseline_tag"] = (re.sub(r"[^A-Za-z0-9_.\-]", "", str(body.get("baseline_tag") or ""))[:40]) or None
        if "primary_ref" in body:
            cfg["primary_ref"] = (str(body.get("primary_ref") or "").strip()[:60]) or None
        if "references" in body:
            refs = {}
            for bench, items in (body.get("references") or {}).items():
                b = re.sub(r"[^A-Za-z0-9_.\-]", "", str(bench))
                if not b or not isinstance(items, list): continue
                clean = []
                for it in items:
                    try:
                        lbl = str(it.get("label") or "").strip()[:60]; acc = round(float(it.get("accuracy")), 1)
                    except Exception:
                        continue
                    if not lbl or not (0 <= acc <= 100): continue
                    kind = it.get("kind") if it.get("kind") in ("frontier", "ref", "chance") else "ref"
                    clean.append({"label": lbl, "accuracy": acc, "kind": kind, "source": str(it.get("source") or "")[:160]})
                if clean: refs[b] = clean
            cfg["references"] = refs
        cfg = {k: v for k, v in cfg.items() if v not in (None, "", {}, [])}
        save_config(cfg)
        return {"ok": True, "config": cfg}
    def do_POST(self):
        if not self._guard():
            return self._send(403, "forbidden — cross-site request or missing/invalid CSRF token", "text/plain")
        p = self.path.split("?")[0]
        if p == "/api/chat": self.stream_chat(self._body())
        elif p == "/api/run": self._json(start_eval(self._body()))
        elif p == "/api/run/stop": self._json(stop_eval())
        elif p == "/api/server/start": self._json(start_server())
        elif p == "/api/stop-all": self._json(stop_all())
        elif p == "/api/server/stop": self._json(kill_server())
        elif p == "/api/config": self._json(self.save_settings(self._body()))
        elif p == "/api/benchmarks/fetch": self._json(fetch_benchmarks_async(self._body().get("keys") or []))
        else: self._send(404, "not found", "text/plain")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    DASH_PORT = port   # Host-header allowlist for the CSRF guard
    os.makedirs(RESULTS, exist_ok=True)
    _load_history()
    _SNAP["cur"] = _snapshot()   # seed before serving: request threads only ever READ _SNAP
    threading.Thread(target=_sample_metrics, daemon=True).start()
    print("benchy -> http://127.0.0.1:%d" % port, flush=True)
    if ALLOW_CODE_EXEC:
        print("WARNING: BENCHY_ALLOW_CODE_EXEC is set — code-generation benchmarks will EXECUTE model-written code on this host.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
