#!/usr/bin/env python3
"""benchy — local LLM benchmark explorer & live dashboard.

  python3 dashboard.py [port]      # default 8050  ->  http://127.0.0.1:8050

Stdlib backend (no pip deps), single-file SPA (Chart.js via CDN). Lets you:
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
except Exception:
    FB = None

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(RESULTS, "runs.jsonl")
PERF = os.path.join(RESULTS, "perf.jsonl")
LIVE = os.path.join(RESULTS, "live.json")
STREAM = os.path.join(RESULTS, "stream.jsonl")
CONFIG_PATH = os.path.join(HERE, "config.json")  # optional local settings (git-ignored); written by guided setup
SERVER_BASE = os.environ.get("BENCHY_SERVER", "http://127.0.0.1:8000").rstrip("/")  # OpenAI-compatible server
CHAT_URL = SERVER_BASE + "/v1/chat/completions"
MODELS_URL = SERVER_BASE + "/v1/models"
try: SERVER_PORT = int(re.search(r":(\d+)", SERVER_BASE).group(1))
except Exception: SERVER_PORT = 8000
DS4 = os.environ.get("DS4_DIR", os.path.expanduser("~/ds4"))  # optional ds4 checkout: fallback for the start-server button when config.json has no server.cmd
PROC = {"eval": None, "server": None, "fetch": None}
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
            with open(CONFIG_PATH) as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def save_config(cfg):
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)

def load_shipped_references():
    """Published frontier-model baselines shipped with the repo (references.json, git-tracked,
    cited). Keys starting with '_' are notes. Merged under any config.json references."""
    try:
        with open(os.path.join(HERE, "references.json")) as f:
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
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3).stdout.strip()
            if mem: info["mem_total_gb"] = round(int(mem) / 1e9, 1)
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3).stdout.strip()
            if cpu: info["cpu"] = cpu
            mac = platform.mac_ver()[0]
            if mac: info["os"] = "macOS " + mac
        else:
            try:
                with open("/proc/meminfo") as f:
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
        "references": merged_references(cfg),
        "options": cfg.get("options") or {},
        "baseline_tag": cfg.get("baseline_tag") or "baseline",
        "primary_ref": cfg.get("primary_ref"),
        "configured": bool(cfg),
    }

def read_jsonl(path):
    if not os.path.exists(path): return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def read_live():
    if not os.path.exists(LIVE): return {"running": False}
    try:
        d = json.load(open(LIVE))
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

def bench_kind(path):
    """Peek a benchmark file's first row to pick the runner: 'code' (has tests/entry_point)
    vs 'mcq' (has options). Works for bundled and user-supplied files alike."""
    try:
        with open(path) as f:
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
                             capture_output=True, text=True, timeout=3).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None

def start_server():
    """Start the model server from the UI. Reads server.cmd/server.cwd from config.json
    (any OpenAI-compatible server); falls back to a ds4 checkout if one is present."""
    if server_up(): return {"ok": True, "already": True}
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
    log = open(os.path.join(RESULTS, "server.log"), "a")
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
    try: n = max(1, int(p.get("n", 25)))
    except Exception: n = 25
    mode = "think" if p.get("mode") == "thinking" else "nothink"
    tag = (re.sub(r"[^A-Za-z0-9_.-]", "", str(p.get("tag", "run"))) or "run")[:40]
    if bench != "healthbench_hard" and bench_kind(path) == "code" and not ALLOW_CODE_EXEC:
        return {"ok": False, "error": "code-execution benchmarks run model-written code on this host and are "
                "disabled by default. Relaunch the dashboard with BENCHY_ALLOW_CODE_EXEC=1 to enable them."}
    if bench == "healthbench_hard":
        # rubric-graded, not MCQ — needs the HealthBench runner (grader via .apikey)
        args = [sys.executable, os.path.join(HERE, "healthbench.py"), str(n), tag, mode, "hard"]
    elif bench_kind(path) == "code":
        # code-generation tasks — generate, then EXECUTE against tests (pass@1)
        args = [sys.executable, os.path.join(HERE, "eval_code.py"), path, str(n), mode, tag]
    else:
        args = [sys.executable, os.path.join(HERE, "eval_mcq.py"), path, str(n), mode, tag]
    log = open(os.path.join(RESULTS, "eval.log"), "a")
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
    """Download one or more registry benchmarks into data/ via fetch_benchmarks.py (detached)."""
    if FB is None: return {"ok": False, "error": "fetch_benchmarks.py not importable"}
    valid = [k for k in keys if k in FB.REGISTRY]
    if not valid: return {"ok": False, "error": "no known benchmark keys in request"}
    if PROC.get("fetch") and PROC["fetch"].poll() is None:
        return {"ok": False, "error": "a fetch is already running"}
    os.makedirs(RESULTS, exist_ok=True)
    log = open(os.path.join(RESULTS, "fetch.log"), "w")
    PROC["fetch"] = subprocess.Popen([sys.executable, os.path.join(HERE, "fetch_benchmarks.py")] + valid,
                                     cwd=HERE, stdout=log, stderr=log, start_new_session=True)
    return {"ok": True, "fetching": valid}

def stop_all():
    """Forcefully stop ALL eval/chain processes — even those launched outside the dashboard."""
    killed = []
    for pat in ("eval_mcq.py", "healthbench.py", "run_sweep.sh", "cache_sweep.sh",
                "next_eval.sh", "healthbench_chain.sh"):
        try:
            if subprocess.run(["pkill", "-f", pat], capture_output=True).returncode == 0:
                killed.append(pat)
        except Exception: pass
    pr = PROC.get("eval")
    if pr and pr.poll() is None:
        try: pr.terminate()
        except Exception: pass
    try: json.dump({"running": False}, open(LIVE, "w"))
    except Exception: pass
    return {"ok": True, "killed": killed}

def kill_server():
    pid = server_pid()
    try:
        if pid:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True)
            return {"ok": True, "killed": pid}
        subprocess.run(["pkill", "-9", "-f", "ds4-server"], capture_output=True)   # ds4 dev fallback
        return {"ok": True, "killed": "server"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a proportion; returns (lo%, hi%)."""
    if not n: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    centre = (p + z*z/(2*n)) / d
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (round(max(0.0, centre-half)*100, 1), round(min(1.0, centre+half)*100, 1))

def _chi2_uniform(dist, opts):
    """χ² of a predicted-letter distribution vs uniform; returns (chi2, dof, total)."""
    cells = [dist.get(c, 0) for c in "ABCDE"[:opts]]
    tot = sum(cells)
    if not tot: return (0.0, opts-1, 0)
    exp = tot / opts
    return (round(sum((c-exp)**2/exp for c in cells), 2), opts-1, tot)

def is_rubric(b):
    """Rubric-graded (open-ended) benchmarks score a 0-100 rubric mean, not k/n correct,
    so they get no binomial CI / letter-bias and are excluded from the MCQ macro-average."""
    return bool(b) and b.startswith("healthbench")

def _details_for(tag, bench, mode=None):
    """The most-complete details file (largest n) for a (tag, benchmark[, mode]) run, parsed
    into {question -> correct?}. Joining on question text lets two runs of different N be
    compared on their overlapping questions."""
    best = None
    for r in read_jsonl(RUNS):
        if r.get("tag") != tag or r.get("benchmark") != bench: continue
        if mode and r.get("mode") != mode: continue
        if not r.get("details"): continue
        if best is None or (r.get("n") or 0) > (best.get("n") or 0):
            best = r
    if not best:
        return None, {}
    fp = os.path.join(RESULTS, "details", os.path.basename(best["details"]))
    by_q = {}
    try:
        for line in open(fp):
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            q = str(d.get("question", "")).strip()
            if q and "ok" in d:
                by_q[q] = bool(d["ok"])
    except Exception:
        return best, {}
    return best, by_q

def _mcnemar_p(b, c):
    """Two-sided McNemar p-value over the b+c discordant pairs: exact binomial (p=0.5) for
    moderate counts, continuity-corrected normal approximation for large ones."""
    n = b + c
    if n == 0:
        return 1.0
    if n <= 2000:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return min(1.0, 2.0 * tail)
    z = (abs(b - c) - 1) / math.sqrt(n)
    return math.erfc(z / math.sqrt(2))

def paired_compare(tag_a, tag_b, bench, mode=None):
    """McNemar paired comparison of two runs on the SAME benchmark over their COMMON questions —
    the statistically correct test for 'does quant A differ from quant B', far more powerful than
    eyeballing two independent Wilson intervals (it uses the per-question pairing)."""
    if not tag_a or not tag_b or not bench:
        return {"ok": False, "error": "pick two run tags and a benchmark"}
    if tag_a == tag_b:
        return {"ok": False, "error": "pick two different tags"}
    if is_rubric(bench):
        return {"ok": False, "error": "paired test applies to MCQ/code (pass/fail) benchmarks, not rubric-graded ones"}
    ra, qa = _details_for(tag_a, bench, mode)
    rb, qb = _details_for(tag_b, bench, mode)
    if not ra or not rb:
        return {"ok": False, "error": "need a completed run with per-question details for both tags on this benchmark"}
    common = sorted(set(qa) & set(qb))
    if not common:
        return {"ok": False, "error": "the two runs share no common questions (different benchmark file or no overlap)"}
    nc = len(common)
    both_right = sum(1 for q in common if qa[q] and qb[q])
    both_wrong = sum(1 for q in common if not qa[q] and not qb[q])
    a_better = sum(1 for q in common if qa[q] and not qb[q])     # A correct, B wrong
    b_better = sum(1 for q in common if not qa[q] and qb[q])     # B correct, A wrong
    acc_a = round(100 * sum(1 for q in common if qa[q]) / nc, 1)
    acc_b = round(100 * sum(1 for q in common if qb[q]) / nc, 1)
    p = _mcnemar_p(a_better, b_better)
    return {"ok": True, "tag_a": tag_a, "tag_b": tag_b, "benchmark": bench, "mode": mode,
            "n_common": nc, "n_a": ra.get("n"), "n_b": rb.get("n"),
            "acc_a": acc_a, "acc_b": acc_b, "delta": round(acc_a - acc_b, 1),
            "a_better": a_better, "b_better": b_better, "discordant": a_better + b_better,
            "both_right": both_right, "both_wrong": both_wrong,
            "p_value": round(p, 4), "significant": bool(p < 0.05)}

def summary():
    """Server-side stats over runs.jsonl x optional reference baselines: Wilson CIs,
    gaps, macro-avg, answer-bias. References & option counts come from config.json."""
    cfg = load_config()
    refs_all = merged_references(cfg)
    opts_map = cfg.get("options") or {}
    primary_label = cfg.get("primary_ref")
    runs = read_jsonl(RUNS)
    benches = sorted({r.get("benchmark") for r in runs if r.get("benchmark")} | set(refs_all.keys()))
    per_run, bias = [], []
    for r in runs:
        b = r.get("benchmark"); n = r.get("n") or 0; acc = r.get("accuracy")
        if acc is None or not b: continue
        if is_rubric(b):
            lo = hi = None   # rubric mean, not k/n successes — a binomial Wilson CI is invalid here
        else:
            k = r.get("correct");  k = round(acc/100.0*n) if k is None else k
            lo, hi = wilson(k, n)
        per_run.append({"ts": r.get("ts"), "benchmark": b, "mode": r.get("mode"), "tag": r.get("tag"),
                        "n": n, "accuracy": acc, "ci_lo": lo, "ci_hi": hi, "small_n": n < 50})
        ld = r.get("letter_dist")
        if ld:
            n_opts = r.get("n_options") or opts_map.get(b) or len([k for k in ld if ld[k]]) or 4
            chi2, dof, tot = _chi2_uniform(ld, n_opts)
            sev = "high" if chi2 > 2*dof else "some" if chi2 > dof else "ok"
            bias.append({"ts": r.get("ts"), "benchmark": b, "mode": r.get("mode"), "dist": ld,
                         "chi2": chi2, "dof": dof, "n": tot, "severity": sev})
    by = {}
    for b in benches:
        ours = [x for x in per_run if x["benchmark"] == b]
        def best(mode):
            c = [x for x in ours if x["mode"] == mode]; return max(c, key=lambda x: x["n"]) if c else None
        th, no = best("thinking"), best("nothink")
        best_ours = th or no
        refs = refs_all.get(b, [])
        non_chance = [x for x in refs if x.get("kind") != "chance"]
        primary = next((x for x in refs if x.get("label") == primary_label), None) if primary_label else None
        if not primary and non_chance:
            primary = max(non_chance, key=lambda x: x["accuracy"])   # default: strongest baseline
        best_ref = max(refs, key=lambda x: x["accuracy"]) if refs else None
        cell = {"thinking": th, "nothink": no, "best": best_ours,
                "refs": refs, "primary_ref": primary, "best_ref": best_ref}
        if best_ours and primary:
            cell["gap_primary"] = round(best_ours["accuracy"] - primary["accuracy"], 1)
            if best_ours["ci_lo"] is not None and best_ours["ci_hi"] is not None:
                cell["ref_in_ci"] = best_ours["ci_lo"] <= primary["accuracy"] <= best_ours["ci_hi"]
        if th and no: cell["think_delta"] = round(th["accuracy"] - no["accuracy"], 1)
        by[b] = cell
    # macro-avg: exclude rubric benchmarks (not % accuracy) and use ONE consistent tag per
    # mode (the widest-coverage tag) so different builds/configs are never blended.
    def macro_for(mode):
        cov = collections.Counter(x["tag"] for x in per_run
                                  if x["mode"] == mode and not is_rubric(x["benchmark"]))
        if not cov: return None, 0, None
        tag = cov.most_common(1)[0][0]
        accs = []
        for b in benches:
            if is_rubric(b): continue
            cands = [x for x in per_run if x["benchmark"] == b and x["mode"] == mode and x["tag"] == tag]
            if cands: accs.append(max(cands, key=lambda x: x["n"])["accuracy"])
        return (round(sum(accs)/len(accs), 1) if accs else None, len(accs), tag)
    tm_mean, tm_k, tm_tag = macro_for("thinking")
    nm_mean, nm_k, nm_tag = macro_for("nothink")
    macro = {"thinking_mean": tm_mean, "thinking_k": tm_k, "thinking_tag": tm_tag,
             "nothink_mean": nm_mean, "nothink_k": nm_k, "nothink_tag": nm_tag}
    return {"benchmarks": benches, "per_run": per_run, "by_benchmark": by,
            "macro": macro, "bias": bias, "primary_ref": primary_label, "opts": opts_map}

def sysmetrics():
    """Live system + model-server metrics: server RSS, CPU, system memory, decode t/s."""
    m = {"server_up": server_up()}
    pid = server_pid()
    m["pid"] = pid
    if pid:
        try:
            r = subprocess.run(["ps", "-o", "rss=,%cpu=", "-p", str(pid)], capture_output=True, text=True, timeout=3).stdout.split()
            if len(r) >= 2:
                m["rss_gb"] = round(int(r[0]) / 1048576, 1); m["cpu"] = float(r[1])
        except Exception: pass
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
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
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=3).stdout
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
        lines = open(HISTORY).readlines()
        if len(lines) > 5000: open(HISTORY, "w").writelines(lines[-2500:])
    except Exception: pass

def _sample_metrics():
    """Background sampler: every 2s snapshot system + decode metrics into the ring buffer + metrics.jsonl.
    Single source for the System charts; keeps live history across page refreshes/restarts. All real samples."""
    n = 0
    while True:
        try:
            m = sysmetrics(); a = activity(); sd = m.get("sys") or {}
            if m.get("server_up") and not DETECTED.get("model"):   # learn the loaded model id once the server is up
                DETECTED["model"] = detect_model()
            decoding = a.get("phase") in ("generating", "thinking")
            rec = {"t": round(time.time(), 1),
                   "rss": m.get("rss_gb"), "cpu": m.get("cpu"),
                   "used": round(sd.get("wired_gb", 0) + sd.get("active_gb", 0) + sd.get("compressed_gb", 0), 1) if sd else None,
                   "free": sd.get("free_gb"), "wired": sd.get("wired_gb"), "swap": m.get("swap_gb"),
                   "tps": ((a.get("chunk_tps") if a.get("chunk_tps") is not None else m.get("decode_tps")) if decoding else None),
                   "phase": a.get("phase"), "gen": a.get("gen"), "up": bool(m.get("server_up"))}
            HIST.append(rec)
            try:
                with open(HISTORY, "a") as f: f.write(json.dumps(rec) + "\n")
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
    L += ["", "## Accuracy", "", "| date | benchmark | tag | mode | N | accuracy | s/q | notes |",
          "|---|---|---|---|--:|--:|--:|---|"]
    for r in runs:
        L.append(f"| {r.get('ts','')} | {r.get('benchmark','')} | {r.get('tag','')} | {r.get('mode','')} | {r.get('n','')} "
                 f"| **{r.get('accuracy','')}%** | {r.get('sec_per_q','')} | {r.get('notes','') or ''} |")
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

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>benchy · Benchmark Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11/highlight.min.js"></script>
<style>
@property --ang{syntax:'<angle>';inherits:false;initial-value:0deg}
:root{--bg:#000;--card:#0c0c0d;--card2:#0f0f11;--bd:#1d1d20;--bd2:#2a2a2e;--ink:#ededed;--mut:#8a8a8a;--mut2:#5f5f63;
--acc:#0070f3;--acc2:#3291ff;--ok:#0cce6b;--teal:#50e3c2;--warn:#f5a623;--dang:#ff4d4f;--pur:#8a63d2;--pur2:#a855f7}
*{box-sizing:border-box}html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif;font-size:14px;-webkit-font-smoothing:antialiased;overflow-x:hidden}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace}
a{color:var(--acc2);text-decoration:none}a:hover{text-decoration:underline}
/* aurora backdrop */
.aurora{position:fixed;inset:-30vh -10vw;z-index:0;pointer-events:none;filter:blur(90px);opacity:.40}
.aurora i{position:absolute;border-radius:50%;mix-blend-mode:screen;animation:drift 22s ease-in-out infinite}
.aurora i:nth-child(1){width:46vw;height:46vw;left:-6vw;top:-8vh;background:radial-gradient(circle,#0b3a86,transparent 62%)}
.aurora i:nth-child(2){width:42vw;height:42vw;right:-6vw;top:-12vh;background:radial-gradient(circle,#5b2a8c,transparent 62%);animation-delay:-7s}
.aurora i:nth-child(3){width:40vw;height:40vw;left:30vw;top:8vh;background:radial-gradient(circle,#0c5a4a,transparent 64%);animation-delay:-13s}
@keyframes drift{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(5vw,4vh) scale(1.12)}66%{transform:translate(-4vw,2vh) scale(.92)}}
.nav{position:sticky;top:0;z-index:30;background:rgba(0,0,0,.66);backdrop-filter:blur(14px) saturate(1.2);-webkit-backdrop-filter:blur(14px) saturate(1.2);border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:18px;padding:0 22px;height:56px}
.brand{font-weight:800;letter-spacing:-.02em;display:flex;align-items:center;gap:9px}
.brand .logo{width:24px;height:24px;border-radius:7px;background:linear-gradient(135deg,var(--acc),var(--pur2));box-shadow:0 0 18px rgba(50,145,255,.45);position:relative}
.brand .logo::after{content:'';position:absolute;inset:5px;border-radius:4px;background:#000;opacity:.55}
.navlinks{display:flex;gap:2px}.navlinks a{color:var(--mut);padding:7px 12px;border-radius:8px;font-weight:500;font-size:13px;transition:.18s}
.navlinks a:hover{color:var(--ink);background:#141416}.navlinks a.on{color:var(--ink);background:#18181b}
.livebadge{margin-left:auto;display:none;align-items:center;gap:9px;font-size:12px;color:var(--ok);background:rgba(12,206,107,.08);border:1px solid rgba(12,206,107,.25);padding:5px 12px;border-radius:99px;font-weight:600}
.livebadge.run{display:flex}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 0 rgba(12,206,107,.5);animation:pp 1.4s infinite}
@keyframes pp{0%{box-shadow:0 0 0 0 rgba(12,206,107,.5)}70%{box-shadow:0 0 0 8px rgba(12,206,107,0)}100%{box-shadow:0 0 0 0 rgba(12,206,107,0)}}
.wrap{position:relative;z-index:1;max-width:1160px;margin:0 auto;padding:26px 22px 90px}
.hl{display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px}
h1{font-size:26px;font-weight:800;letter-spacing:-.03em;margin:0;background:linear-gradient(180deg,#fff,#b9b9c2);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:var(--mut);margin:5px 0 0}
.modelcard{display:flex;flex-wrap:wrap;gap:7px;margin:14px 0 4px}
.mcchip{font-size:11.5px;color:var(--mut);background:#0d0d0f;border:1px solid var(--bd);border-radius:7px;padding:5px 9px}
.mcchip b{color:var(--ink);font-weight:600}
.btn{background:var(--ink);color:#000;border:0;border-radius:9px;padding:8px 14px;font-weight:600;font-size:13px;cursor:pointer;font-family:inherit;transition:.16s}
.btn:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(255,255,255,.12)}
.btn.g{background:#161618;color:var(--ink);border:1px solid var(--bd2)}.btn.g:hover{box-shadow:0 4px 16px rgba(0,0,0,.4);border-color:#3a3a40}
.btn.dang{background:#3a1416;color:#ff8b8d;border:1px solid rgba(255,77,79,.35)}.btn.dang:hover{box-shadow:0 4px 16px rgba(255,77,79,.18)}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
.grid{display:grid;gap:14px}.g4{grid-template-columns:repeat(4,1fr)}.g3{grid-template-columns:repeat(3,1fr)}.g2{grid-template-columns:1fr 1fr}
@media(max-width:860px){.g4{grid-template-columns:1fr 1fr}.g3,.g2{grid-template-columns:1fr}}
.card{position:relative;background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--bd);border-radius:14px;padding:18px}
.card::after{content:'';position:absolute;inset:0;border-radius:inherit;pointer-events:none;opacity:0;transition:opacity .3s;background:radial-gradient(340px circle at var(--mx,50%) var(--my,-30%),rgba(120,160,255,.07),transparent 60%)}
.card:hover::after{opacity:1}
.card h3{margin:0 0 3px;font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.ch{display:flex;align-items:center;justify-content:space-between;gap:10px}
.kpi .v{font-size:30px;font-weight:800;letter-spacing:-.02em;margin-top:6px;line-height:1.05}.kpi .d{color:var(--mut);font-size:12px;margin-top:5px}
.kpi .spark{margin-top:10px;height:26px}
.sec{margin-top:30px}.sec>h2{font-size:16px;font-weight:700;margin:0 0 13px;display:flex;align-items:center;gap:10px;letter-spacing:-.01em}
.sec>h2 .hint{font-weight:500;font-size:12px;color:var(--mut2)}
.tag{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:600;background:#161618;border:1px solid var(--bd2);color:var(--mut)}
.tag.ok{color:var(--ok);border-color:rgba(12,206,107,.3);background:rgba(12,206,107,.07)}.tag.warn{color:var(--warn);border-color:rgba(245,166,35,.3);background:rgba(245,166,35,.07)}.tag.blue{color:var(--acc2);border-color:rgba(50,145,255,.3);background:rgba(50,145,255,.07)}.tag.dang{color:var(--dang);border-color:rgba(255,77,79,.3);background:rgba(255,77,79,.07)}.tag.pur{color:#c4a4ff;border-color:rgba(168,85,247,.3);background:rgba(168,85,247,.08)}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--bd);white-space:nowrap}th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em;cursor:pointer;user-select:none}th:hover{color:var(--ink)}td.n,th.n{text-align:right}tr:hover td{background:#0c0c0e}
.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{font-size:12px;color:var(--mut);background:#101012;border:1px solid var(--bd);border-radius:8px;padding:6px 10px;cursor:pointer;transition:.15s}.chip:hover{border-color:var(--bd2);color:var(--ink)}.chip b{color:var(--ink)}
.callout{border:1px solid rgba(245,166,35,.28);background:rgba(245,166,35,.05);border-radius:12px;padding:13px 16px;color:#e9d3a4;font-size:12.5px;line-height:1.55}.callout b{color:var(--warn)}
.barwrap{display:grid;grid-template-columns:210px 1fr 118px;align-items:center;gap:12px;margin:9px 0}
.barwrap .lab{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.barwrap .lab small{color:var(--mut)}
.track{position:relative;height:22px;background:#161618;border-radius:7px;overflow:hidden}
.track>.fill{position:absolute;top:0;left:0;height:100%;border-radius:7px;transition:width .7s cubic-bezier(.2,.8,.2,1)}
.track>.ci{position:absolute;top:0;bottom:0;background:repeating-linear-gradient(45deg,rgba(255,255,255,.16),rgba(255,255,255,.16) 5px,rgba(255,255,255,.06) 5px,rgba(255,255,255,.06) 10px);border-left:1px solid rgba(255,255,255,.4);border-right:1px solid rgba(255,255,255,.4)}
.barwrap .pv{text-align:right;font-weight:600;font-size:12.5px}.barwrap .pv small{color:var(--mut);font-weight:500}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input,select{background:#0b0b0d;border:1px solid var(--bd2);color:var(--ink);border-radius:9px;padding:8px 11px;font-size:13px;font-family:inherit;outline:none;transition:.15s}
input:focus,select:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(0,112,243,.18)}
input::placeholder{color:var(--mut2)}
.seg{display:flex;background:#0b0b0d;border:1px solid var(--bd2);border-radius:9px;overflow:hidden}
.seg button{background:transparent;border:0;color:var(--mut);padding:7px 13px;font-size:13px;cursor:pointer;font-family:inherit;transition:.15s}.seg button.on{background:#1c1c20;color:var(--ink)}
.seg.sm button{padding:4px 10px;font-size:12px}
.muted{color:var(--mut)}.foot{color:var(--mut2);font-size:12px;margin-top:34px;line-height:1.6}
canvas{max-height:280px}
.iconbtn{background:#141416;border:1px solid var(--bd2);color:var(--mut);border-radius:7px;padding:4px 9px;font-size:12px;cursor:pointer;font-family:inherit;transition:.15s;display:inline-flex;align-items:center;gap:5px}
.iconbtn:hover{color:var(--ink);border-color:#3a3a40;background:#1a1a1e}
.iconbtn.xs{padding:2px 6px;font-size:11px}
/* live section */
.glowcard{position:relative;border-radius:16px}
.glowcard::before{content:'';position:absolute;inset:-1.5px;border-radius:17px;padding:1.5px;background:conic-gradient(from var(--ang),transparent 0 58%,var(--acc2) 72%,var(--teal) 82%,transparent 92%);-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;animation:spin 4.5s linear infinite;opacity:.9}
@keyframes spin{to{--ang:360deg}}
.live-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.live-meta{font-size:13px;color:var(--mut)}.live-meta b{color:var(--ink)}
.phase{display:inline-flex;align-items:center;gap:8px;font-size:11.5px;font-weight:700;padding:5px 12px;border-radius:99px;border:1px solid var(--bd2);background:#0b0b0d;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.phase .phdot{width:8px;height:8px;border-radius:50%;background:var(--mut)}
.phase[data-p=generating]{color:var(--teal);border-color:rgba(80,227,194,.4);background:rgba(80,227,194,.08)}.phase[data-p=generating] .phdot{background:var(--teal);animation:pp 1.1s infinite}
.phase[data-p=thinking]{color:#c4a4ff;border-color:rgba(168,85,247,.4);background:rgba(168,85,247,.10)}.phase[data-p=thinking] .phdot{background:var(--pur2);animation:pp 1.1s infinite}
.phase[data-p=prefill]{color:var(--warn);border-color:rgba(245,166,35,.4);background:rgba(245,166,35,.08)}.phase[data-p=prefill] .phdot{background:var(--warn);animation:pp 1.1s infinite}
.phase[data-p=done]{color:var(--ok);border-color:rgba(12,206,107,.4);background:rgba(12,206,107,.08)}.phase[data-p=done] .phdot{background:var(--ok)}
.metric{background:#0a0a0c;border:1px solid var(--bd);border-radius:11px;padding:13px 14px}
.metric .mlabel{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.metric .mval{font-size:27px;font-weight:800;letter-spacing:-.02em;margin-top:3px;line-height:1}.metric .mval small{font-size:13px;font-weight:600;color:var(--mut);margin-left:4px}
.metric .mfoot{font-size:11.5px;color:var(--mut2);margin-top:5px}
.eq{display:flex;align-items:flex-end;gap:3px;height:66px;margin-top:16px}
.eq i{flex:1;min-width:2px;border-radius:3px 3px 0 0;background:linear-gradient(180deg,var(--teal),rgba(80,227,194,.12));height:6%;transition:height .4s cubic-bezier(.2,.8,.2,1)}
.eq i.think{background:linear-gradient(180deg,var(--pur2),rgba(168,85,247,.12))}
.eqcap{display:flex;justify-content:space-between;font-size:11px;color:var(--mut2);margin-top:6px}
.accbig{font-size:24px;font-weight:800;letter-spacing:-.02em}
.feedcard{overflow:hidden;display:flex;flex-direction:column;min-height:0}
.feed{display:flex;flex-direction:column;gap:7px;margin-top:10px;overflow:auto;flex:1;min-height:0;padding-right:3px}
.qadonut{display:flex;align-items:center;gap:18px;padding:12px 2px 14px;border-bottom:1px solid var(--bd)}
.qadonut .dchart{position:relative;width:92px;height:92px;flex:0 0 auto}
.qadonut .dctr{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none}
.qadonut .dctr b{font-size:21px;font-weight:800;letter-spacing:-.02em;line-height:1}
.qadonut .dctr small{font-size:9.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.qadonut .dleg{display:flex;flex-direction:column;gap:7px;font-size:12.5px}
.qadonut .dleg .li{display:flex;align-items:center;gap:8px}
.qadonut .dleg .sw{width:10px;height:10px;border-radius:3px;flex:0 0 auto}
.qrow{flex:0 0 auto;border:1px solid var(--bd);border-radius:10px;background:#0a0a0c;font-size:12.5px;overflow:hidden;transition:border-color .15s}
.qrow:hover{border-color:var(--bd2)}
.qrow.ok{border-left:3px solid var(--ok)}.qrow.no{border-left:3px solid var(--dang)}
.qrow .qh{display:flex;align-items:center;gap:8px;padding:9px 11px}
.qrow .qh-r{margin-left:auto;display:flex;align-items:center;gap:8px}
.qrow .qbody{display:none;padding:0 11px 11px;border-top:1px solid var(--bd)}
.qrow.open .qbody{display:block}
.qrow .qq{color:var(--ink);line-height:1.5;margin:9px 0 7px}
.qrow .qopt{color:var(--mut);line-height:1.5;font-size:12px}
.qrow pre.ans{margin:8px 0 0;background:#060607;border:1px solid var(--bd);border-radius:8px;padding:9px 10px;font-size:11.5px;color:#bdbdc4;white-space:pre-wrap;word-break:break-word;max-height:180px;overflow:auto}
.mk{width:18px;height:18px;border-radius:5px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;flex:0 0 auto}
.mk.ok{background:rgba(12,206,107,.15);color:var(--ok)}.mk.no{background:rgba(255,77,79,.15);color:var(--dang)}
/* reveal */
.sec,.hl{animation:rise .55s cubic-bezier(.2,.7,.2,1) both}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
/* toasts */
#toasts{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:120;display:flex;flex-direction:column;gap:8px;align-items:center}
.toast{background:#16161a;border:1px solid var(--bd2);color:var(--ink);padding:9px 15px;border-radius:10px;font-size:13px;font-weight:500;box-shadow:0 10px 30px rgba(0,0,0,.5);opacity:0;transform:translateY(10px);transition:.28s}
.toast.in{opacity:1;transform:none}.toast.bad{border-color:rgba(255,77,79,.5);color:#ff9b9d}
/* modal */
#detModal,#setupModal,#benchModal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.66);backdrop-filter:blur(4px);z-index:100;padding:40px 20px;overflow:auto}
#detModal .box,#setupModal .box,#benchModal .box{max-width:960px;margin:0 auto;background:#0d0d0f;border:1px solid #262629;border-radius:16px;padding:20px 22px}
.benchrow{display:flex;align-items:center;gap:10px;font-size:13px;padding:8px 10px;border:1px solid var(--bd);border-radius:9px;margin:6px 0;background:#0a0a0c}
.benchrow .bn{font-weight:600}.benchrow .br{margin-left:auto;display:flex;align-items:center;gap:8px}
.benchrow .dom{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut2)}
#refList .refrow{display:flex;align-items:center;gap:8px;font-size:12.5px;padding:6px 9px;border:1px solid var(--bd);border-radius:8px;margin:5px 0;background:#0a0a0c}
#refList .refrow .rm{margin-left:auto;cursor:pointer;color:var(--mut2);border:0;background:none;font-size:14px}#refList .refrow .rm:hover{color:var(--dang)}
#refList .refbench{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:12px 0 2px}
.dtoolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:10px 0 14px}
/* debug drawer */
.dbgbtn{position:fixed;right:18px;bottom:18px;z-index:90;background:#121214;border:1px solid var(--bd2);color:var(--mut);border-radius:10px;padding:9px 13px;font-size:12px;font-family:'JetBrains Mono',monospace;cursor:pointer;box-shadow:0 8px 24px rgba(0,0,0,.45);transition:.15s}
.dbgbtn:hover{color:var(--ink);border-color:#3a3a40}
.dbg{position:fixed;right:18px;bottom:62px;z-index:91;width:min(560px,calc(100vw - 36px));height:min(440px,60vh);background:#0a0a0c;border:1px solid var(--bd2);border-radius:13px;box-shadow:0 20px 60px rgba(0,0,0,.6);display:none;flex-direction:column;overflow:hidden}
.dbg.open{display:flex}
.dbgh{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--bd)}
.dbg pre{margin:0;flex:1;overflow:auto;padding:12px 14px;font-size:11.5px;line-height:1.5;color:#b6b6bd;white-space:pre-wrap;word-break:break-word}
/* experiment band */
.exphead{display:flex;justify-content:space-between;gap:22px;flex-wrap:wrap}
.expk{font-size:11px;letter-spacing:.12em;color:var(--mut2);font-weight:700}
.exptitle{display:flex;align-items:center;gap:10px;margin-top:7px;font-size:17px;font-weight:700}
.expmetrics{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}
.expmetric .expml{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
.expmetric .expmv{font-size:23px;font-weight:800;letter-spacing:-.02em;margin-top:3px;line-height:1}
.expprog{display:flex;align-items:center;gap:12px;margin-top:16px}
.expbar{flex:1;height:8px;background:#161618;border-radius:99px;overflow:hidden}
.expbar>i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--teal));border-radius:99px;transition:width .6s cubic-bezier(.2,.8,.2,1)}
/* latency bars + system + 2-col live */
.eq.yellow i{background:linear-gradient(180deg,var(--warn),rgba(245,166,35,.14))}
.livegrid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.06fr);gap:14px;align-items:start}
@media(max-width:860px){.livegrid{grid-template-columns:1fr}}
.livestack{display:flex;flex-direction:column;gap:14px}
.sysbig{font-size:16px;font-weight:700;letter-spacing:-.02em;white-space:nowrap}
.syschart{position:relative;height:120px;margin-top:8px}.syschart canvas{position:absolute;inset:0}
.syslegend{display:flex;gap:13px;font-size:10.5px;color:var(--mut2);margin-top:6px;flex-wrap:wrap}
.syslegend i{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:-1px}
.qrow .qtime{font-size:11px;color:var(--mut2);margin-top:7px}
.qrow .qh{cursor:pointer}
.blink{animation:blinkv 1.25s ease-in-out infinite}@keyframes blinkv{0%,100%{opacity:1}50%{opacity:.38}}
.livedot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-left:8px;vertical-align:middle;box-shadow:0 0 0 0 rgba(12,206,107,.5);animation:pp 1.25s infinite}
.envstrip{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:16px 0 2px}
.envstrip .es{font-size:11px;color:var(--mut2);background:#0b0b0d;border:1px solid var(--bd);border-radius:6px;padding:3px 8px;cursor:pointer;transition:.15s;white-space:nowrap}
.envstrip .es:hover{color:var(--ink);border-color:var(--bd2)}.envstrip .es b{color:var(--mut)}
/* chat console */
.chatbtn{position:fixed;left:18px;bottom:18px;z-index:90;background:#161618;color:var(--ink);border:1px solid var(--bd2);border-radius:10px;padding:8px 14px;font-size:13px;font-weight:500;cursor:pointer;box-shadow:0 6px 18px rgba(0,0,0,.4);transition:.15s;display:flex;align-items:center;gap:7px}
.chatbtn:hover{background:#1c1c20;border-color:#3a3a40}
.chatbtn svg{opacity:.7}
.chatwrap{position:fixed;inset:auto 0 0 0;z-index:95;display:none;justify-content:center;pointer-events:none}
.chatwrap.open{display:flex}
.chat{pointer-events:auto;width:min(980px,100%);height:min(78vh,820px);background:#0a0a0c;border:1px solid var(--bd2);border-bottom:0;border-radius:16px 16px 0 0;box-shadow:0 -24px 70px rgba(0,0,0,.65);display:flex;flex-direction:column;overflow:hidden;animation:slideup .28s cubic-bezier(.2,.8,.2,1)}
.chat.full{width:100%;height:100vh;border-radius:0}
@keyframes slideup{from{transform:translateY(40px);opacity:.3}to{transform:none;opacity:1}}
.chathead{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--bd);background:#0a0a0c}
.chathead .ttl{font-weight:700;display:flex;align-items:center;gap:8px;font-size:14px}
.chathead .ep{font-size:11px;color:var(--mut2);font-family:'JetBrains Mono',monospace}
.chatmsgs{flex:1;overflow:auto;padding:22px 0}
.cmsg{max-width:780px;margin:0 auto 18px;padding:0 22px;display:flex;gap:14px}
.cmsg .av{width:28px;height:28px;border-radius:7px;flex:0 0 auto;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}
.cmsg.user .av{background:#1c1c20;color:var(--ink)}
.cmsg.bot .av{background:#1a1a1c;color:var(--mut);border:1px solid var(--bd2)}
.cmsg .body{flex:1;min-width:0;line-height:1.62;font-size:14.5px;padding-top:3px}
.cmsg .body p{margin:0 0 10px}.cmsg .body>*:last-child{margin-bottom:0}
.cmsg .body pre{background:#060607;border:1px solid var(--bd);border-radius:10px;padding:12px 14px;overflow:auto;position:relative;font-size:12.5px;margin:0 0 10px}
.cmsg .body code{font-family:'JetBrains Mono',monospace}
.cmsg .body :not(pre)>code{background:#161618;border:1px solid var(--bd);border-radius:5px;padding:1px 5px;font-size:12.5px}
.cmsg .body ul,.cmsg .body ol{margin:0 0 10px;padding-left:22px}.cmsg .body li{margin:3px 0}
.cmsg .body h1,.cmsg .body h2,.cmsg .body h3{margin:14px 0 8px;font-size:16px}
.cmsg .body table{margin:0 0 10px}.cmsg .body a{color:var(--acc2)}
.reason{border:1px solid var(--bd);border-radius:10px;margin:0 0 12px;background:#0c0c0e;overflow:hidden}
.reason>summary{cursor:pointer;padding:8px 12px;font-size:12px;color:var(--mut);list-style:none;display:flex;align-items:center;gap:7px;user-select:none}
.reason>summary::-webkit-details-marker{display:none}
.reason .rbody{padding:0 12px 10px;font-size:12.5px;color:var(--mut);line-height:1.55;white-space:pre-wrap;max-height:280px;overflow:auto}
.cmeta{max-width:780px;margin:-10px auto 16px;padding:0 22px 0 58px;font-size:11px;color:var(--mut2);font-family:'JetBrains Mono',monospace}
.cursor{display:inline-block;width:7px;height:15px;background:var(--mut);vertical-align:text-bottom;animation:blinkv .9s steps(2) infinite;border-radius:1px}
.chatcompose{border-top:1px solid var(--bd);padding:10px 16px 13px;display:flex;flex-direction:column;gap:8px;background:#0c0c0e}
.ccontrols{display:flex;align-items:center;gap:12px;font-size:12px;color:var(--mut)}
.ccontrols label{display:flex;align-items:center;gap:6px;cursor:pointer}
.cinput{display:flex;gap:10px;align-items:flex-end}
.cinput textarea{flex:1;resize:none;max-height:170px;background:#0b0b0d;border:1px solid var(--bd2);border-radius:12px;padding:11px 13px;color:var(--ink);font-family:inherit;font-size:14px;line-height:1.5;outline:none}
.cinput textarea:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(0,112,243,.18)}
.csend{flex:0 0 auto;width:40px;height:40px;border-radius:11px;border:0;background:var(--ink);color:#000;font-size:17px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:.15s}
.csend:hover{opacity:.9}.csend:disabled{opacity:.4;cursor:not-allowed}.csend.stop{background:#3a1416;color:#ff8b8d}
.chatempty{max-width:580px;margin:36px auto;text-align:center;color:var(--mut)}
.chatempty h3{color:var(--ink);font-size:18px;margin:0 0 8px;text-transform:none;letter-spacing:-.01em}
.exq{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-top:18px}
.exq button{background:#0e0e10;border:1px solid var(--bd2);color:var(--mut);border-radius:9px;padding:9px 12px;font-size:12.5px;cursor:pointer;text-align:left;max-width:250px;line-height:1.4}
.exq button:hover{color:var(--ink);border-color:#3a3a40}
.codecopy{position:absolute;top:7px;right:7px;background:#161618;border:1px solid var(--bd2);color:var(--mut);border-radius:6px;padding:2px 8px;font-size:11px;cursor:pointer;opacity:0;transition:.15s}
.cmsg .body pre:hover .codecopy{opacity:1}.codecopy:hover{color:var(--ink)}
@media(prefers-reduced-motion:reduce){*{animation-duration:.001s!important}.aurora{display:none}}
</style></head><body>
<div class="aurora"><i></i><i></i><i></i></div>
<div class="wrap">
<div class="hl"><div>
  <h1 id="pageTitle">benchy</h1>
  <p class="sub" id="pageSub">Local LLM benchmark suite — each <b>run</b> (a model / quant / build) is scored across a panel of benchmarks on your own machine.</p>
</div>
<div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn g" onclick="openBenchmarks()">⬇ Benchmarks</button><button class="btn g" onclick="openSetup()">⚙ Setup</button><button class="btn g" onclick="loadAll();pulse()">↻ Refresh</button><button class="btn g" onclick="copyReport(this)">⧉ Copy report</button><button class="btn" onclick="location.href='/api/report'">⬇ Export</button></div></div>
<div class="envstrip" id="envstrip"></div>

<!-- EXPERIMENT -->
<div class="sec" id="experiment" style="margin-top:18px"><div class="card" id="expCard"></div></div>

<!-- PAIRED A/B (McNemar) -->
<div class="sec" id="paired" style="margin-top:18px"><div class="card">
  <h3>Paired A/B significance <span class="hint">McNemar on the same questions — the correct test for "does quant A differ from quant B"</span></h3>
  <div class="toolbar" style="margin:11px 0 0;flex-wrap:wrap;gap:8px;align-items:center">
    <select id="cmpA" title="run tag A"></select><span class="muted">vs</span><select id="cmpB" title="run tag B"></select>
    <select id="cmpBench" title="benchmark"></select>
    <select id="cmpMode" title="reasoning mode"><option value="">any mode</option><option value="thinking">thinking</option><option value="nothink">nothink</option></select>
    <button class="btn g" onclick="runCompare()">Compare</button>
  </div>
  <div id="cmpResult" class="muted" style="font-size:12.5px;margin:12px 0 0;line-height:1.6">Pick two run tags and a benchmark, then Compare — this pairs the two runs question-by-question (a far more powerful test than comparing two independent confidence intervals).</div>
</div></div>

<!-- CONTROL -->
<div class="sec" id="control"><div class="grid g2">
  <div class="card"><h3>Model server</h3>
    <div class="toolbar" style="margin:11px 0 0"><span class="tag" id="srvPill">checking…</span>
      <button class="btn g" id="srvBtn" onclick="startServer()">Start server</button>
      <button class="btn dang" onclick="stopAll()">⛔ Stop all tests</button>
      <button class="btn g" onclick="killServer()" title="force-kill the model server">kill server</button>
      <span class="muted" id="srvMsg" style="font-size:12px"></span></div>
    <p class="muted" style="font-size:12px;margin:11px 0 0;line-height:1.55">Tag each run (e.g. a model, quant, or build name) so runs compare against your <b id="baselineTagTxt" style="color:var(--pur2)">baseline</b> tag in Accuracy. Add optional reference baselines in <a href="#" onclick="openSetup();return false">Setup</a>.</p></div>
  <div class="card"><h3>Run a test</h3>
    <div class="toolbar" style="margin:11px 0 0">
      <select id="rBench" onchange="rBenchUser=true;updateRunDesc();syncAccToRun()"></select>
      <input id="rN" type="number" value="25" min="1" style="width:78px" title="N questions">
      <div class="seg"><button id="segT" class="on" onclick="setMode('thinking')">🧠 thinking</button><button id="segN" onclick="setMode('nothink')">⚡ nothink</button></div>
      <input id="rTag" type="text" value="" style="width:150px" placeholder="run tag (e.g. baseline)">
      <button class="btn" id="runBtn" onclick="runTest()">▶ Run</button>
      <button class="btn g" id="stopBtn" onclick="stopTest()" style="display:none">■ Stop</button>
      <span class="muted" id="runMsg" style="font-size:12px"></span></div>
    <p class="muted" id="runDesc" style="font-size:12px;margin:9px 0 0;min-height:16px;line-height:1.5"></p></div>
</div></div>

<!-- LIVE -->
<div class="sec" id="liveSec" style="display:none"><h2>Live run <span class="hint">real-time · ~1.5s</span></h2>
  <div class="glowcard"><div class="card">
    <div class="live-head">
      <span class="phase" id="phasePill" data-p="idle"><span class="phdot"></span><span id="phaseTxt">idle</span></span>
      <span class="live-meta" id="liveMeta"></span>
      <button class="iconbtn" style="margin-left:auto" onclick="copyDiag()" title="copy a JSON snapshot of live/sys/activity for debugging">{ } copy diagnostics</button>
    </div>
    <div class="grid g3">
      <div class="metric"><div class="mlabel">decode</div><div class="mval"><span id="tpsVal">—</span><small>t/s</small></div><div class="mfoot" id="tpsFoot">avg —</div></div>
      <div class="metric"><div class="mlabel">generated</div><div class="mval"><span id="genVal">0</span><small>tok</small></div><div class="mfoot" id="genFoot">current answer</div></div>
      <div class="metric"><div class="mlabel">progress</div><div class="mval"><span id="progVal">0/0</span></div><div class="mfoot" id="etaFoot">ETA —</div></div>
    </div>
    <div class="eq" id="actBars"></div>
    <div class="eqcap"><span>decode throughput · per-chunk (this answer)</span><span id="eqcapR"></span></div>
  </div></div>
  <div class="livegrid" style="margin-top:14px">
    <div class="livestack">
      <div class="card"><div class="ch"><h3>Running accuracy</h3><span class="accbig" id="accBig" style="color:var(--ok)">—</span></div>
        <div class="muted" id="accSub" style="font-size:12px;margin:2px 0 4px"></div><canvas id="accLive" height="118"></canvas></div>
      <div class="card"><div class="ch"><h3>Latency / question <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">· wall-clock s/q · hover to read</span></h3><span class="mono muted" id="latStat" style="font-size:11.5px"></span></div>
        <canvas id="latLive" height="104"></canvas>
        <div class="eqcap"><span>per question (oldest→newest)</span><span id="latCapR"></span></div></div>
    </div>
    <div class="card feedcard"><div class="ch"><h3>Questions &amp; answers <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500" id="feedCount"></span></h3>
      <div style="display:flex;gap:8px;align-items:center"><div class="seg sm"><button id="fAll" class="on" onclick="setFeedFilter('all')">all</button><button id="fOk" onclick="setFeedFilter('ok')">✓</button><button id="fNo" onclick="setFeedFilter('no')">✗</button></div><button class="iconbtn xs" title="open the full detail page (full question, all options, full model output)" onclick="openLiveDetails()">🔍 detail</button></div></div>
      <div class="qadonut"><div class="dchart"><canvas id="qaDonutChart"></canvas><div class="dctr"><b id="qaPct">—</b><small>correct</small></div></div><div class="dleg" id="qaLeg"></div></div>
      <div class="feed" id="feed" title="click a row for the full question & model output"></div></div>
  </div></div>

<!-- SYSTEM -->
<div class="sec" id="system"><h2>System resources <span class="hint">live · sampled every 2s · history kept server-side (full on refresh)</span></h2>
  <div class="grid g3">
    <div class="card"><div class="ch"><h3>Decode throughput</h3><span class="sysbig" id="sysTps" style="color:var(--warn)">—</span></div><div class="syschart"><canvas id="sysTpsChart"></canvas></div></div>
    <div class="card"><div class="ch"><h3>Memory <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">· model / system</span></h3><span class="sysbig" id="sysRam" style="color:var(--acc2)">—</span></div>
      <div class="syslegend" id="sysMemLeg"></div>
      <div class="syschart"><canvas id="sysRamChart"></canvas></div></div>
    <div class="card"><div class="ch"><h3>Server CPU</h3><span class="sysbig" id="sysCpu" style="color:var(--ok)">—</span></div><div class="syschart"><canvas id="sysCpuChart"></canvas></div></div>
  </div></div>

<!-- ACCURACY -->
<div class="sec" id="accuracy"><h2>Accuracy <select id="accBench" onchange="accBenchUser=true;accbars()" style="margin-left:6px;font-size:13px"></select><span class="hint" id="macroHint"></span></h2>
  <p class="muted" id="accDesc" style="font-size:12px;margin:-4px 0 12px;line-height:1.5"></p>
  <div class="grid g2">
    <div class="card"><h3>Your runs<span id="refLbl"></span> <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">· 95% Wilson CI shown on your bars</span></h3><div id="accbars" style="margin-top:12px"></div>
      <p class="muted" id="accNote" style="font-size:12px;margin:11px 0 0;line-height:1.5">Frontier baselines are <b>published numbers</b> (eval-setup-dependent — for context, not size-matched). Small-N runs have wide CIs; treat overlapping CIs as indistinguishable. Edit/add baselines in <a href="#" onclick="openSetup();return false">Setup</a>.</p></div>
    <div class="card"><h3>Thinking vs non-thinking</h3><canvas id="modeChart"></canvas></div>
  </div></div>

<!-- RUNS -->
<div class="sec" id="runs"><h2>Runs explorer <span class="hint">click a run to inspect every question</span></h2>
  <div class="toolbar" style="margin-bottom:12px">
    <input type="search" id="q" placeholder="filter runs…" oninput="renderRuns()" style="min-width:200px"><span class="muted" id="runcount" style="margin-left:auto"></span></div>
  <div class="card" style="padding:0;overflow:hidden"><table><thead id="runHead"></thead><tbody id="runBody"></tbody></table></div></div>

<div class="foot">Live stream, accuracy &amp; LLM activity update ~1.5s · tables 6s · live metrics &amp; benchmark numbers come from <span class="mono">results/</span> &amp; <span class="mono">server.log</span>; the model id &amp; host specs are auto-detected, reference baselines come from your <span class="mono">config.json</span> (Setup).<br>Methodology &amp; caveats in README.md — small-N Wilson CIs, page-cache/thermal noise, non-thinking letter bias, decode t/s ±~10%. Rubric-graded benchmarks (e.g. HealthBench) score 0–100, not % correct, and are excluded from the MCQ macro-average.</div>
</div>

<div id="detModal" onclick="if(event.target===this)closeDetails()"><div class="box">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px"><h2 id="detTitle" style="margin:0">Run detail</h2>
    <div style="display:flex;gap:8px"><button class="iconbtn" id="detCopyWrong" onclick="copyWrong()">⧉ copy wrong as JSON</button><button class="btn g" onclick="closeDetails()">✕ Close</button></div></div>
  <div id="detSummary" class="muted" style="font-size:13px;margin-top:4px"></div>
  <div class="dtoolbar"><div class="seg sm"><button id="dfAll" class="on" onclick="setDetFilter('all')">all</button><button id="dfOk" onclick="setDetFilter('ok')">✓ correct</button><button id="dfNo" onclick="setDetFilter('no')">✗ wrong</button></div>
    <input type="search" id="detQ" placeholder="search questions…" oninput="renderDet()" style="flex:1;min-width:160px"></div>
  <div id="detBody"></div>
</div></div>

<div id="setupModal" onclick="if(event.target===this)closeSetup()"><div class="box" style="max-width:680px">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px"><h2 style="margin:0">⚙ Setup</h2>
    <button class="btn g" onclick="closeSetup()">✕ Close</button></div>
  <p class="muted" style="font-size:12.5px;margin:6px 0 14px;line-height:1.5">All optional. Model id and host specs are auto-detected from your running server and OS — nothing here is required to run benchmarks. Saved to a git-ignored <code class="mono">config.json</code>.</p>

  <h3 style="font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 8px">Auto-detected</h3>
  <div class="chips" id="setupDetected" style="margin-bottom:18px"></div>

  <h3 style="font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 8px">Display</h3>
  <div class="toolbar" style="margin-bottom:6px"><label class="muted" style="font-size:12px;width:96px">Title</label><input id="setTitle" type="text" placeholder="benchy" style="flex:1"></div>
  <div class="toolbar" style="margin-bottom:18px"><label class="muted" style="font-size:12px;width:96px">Baseline tag</label><input id="setBaseline" type="text" placeholder="baseline" style="flex:1" title="the run tag everything else is compared against in Accuracy"></div>

  <h3 style="font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 4px">Reference baselines <span class="muted" style="text-transform:none;letter-spacing:0">· optional external scores to compare against, per benchmark</span></h3>
  <div class="toolbar" style="margin:10px 0">
    <select id="refBench" style="min-width:150px"></select>
    <input id="refLabel" type="text" placeholder="model / source label" style="flex:1;min-width:140px">
    <input id="refAcc" type="number" step="0.1" min="0" max="100" placeholder="score %" style="width:96px">
    <select id="refKind" title="frontier = large general model · reference = comparable/specialized"><option value="ref">reference</option><option value="frontier">frontier</option><option value="chance">chance</option></select>
    <button class="btn g" onclick="setupAddRef()">+ add</button></div>
  <div id="refList" style="margin-bottom:16px"></div>

  <div style="display:flex;gap:8px;justify-content:flex-end"><button class="btn g" onclick="closeSetup()">Cancel</button><button class="btn" onclick="setupSave()">Save settings</button></div>
</div></div>

<div id="benchModal" onclick="if(event.target===this)closeBenchmarks()"><div class="box" style="max-width:680px">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px"><h2 style="margin:0">⬇ Benchmarks</h2>
    <div style="display:flex;gap:8px"><button class="btn" id="benchAll" onclick="fetchBench('__current__')">⬇ Fetch current set</button><button class="btn g" onclick="closeBenchmarks()">✕ Close</button></div></div>
  <p class="muted" style="font-size:12.5px;margin:6px 0 14px;line-height:1.5">Download benchmarks into <code class="mono">data/</code>. <b>Current</b> = still discriminates mid-2026 models; <b>legacy</b> = saturated (small-model regression only). <b>Code</b> sets (HumanEval/MBPP) <b>generate &amp; execute code</b> for pass@1 — ⚠ runs model-written code locally (subprocess + timeout). Each set keeps its own license/source — see <code class="mono">DATA.md</code>.</p>
  <div id="benchStatus" class="muted" style="font-size:12px;margin-bottom:8px"></div>
  <div id="benchList"></div>
  <div id="benchManual" style="margin-top:14px"></div>
</div></div>

<button class="chatbtn" onclick="chatToggle()"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>Chat</button>
<div class="chatwrap" id="chatwrap"><div class="chat" id="chatpanel">
  <div class="chathead"><div class="ttl" id="chatTtl">chat <span class="ep" id="chatEp">/v1</span></div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
      <label class="ccontrols" title="chain-of-thought reasoning"><input type="checkbox" id="chatThink" style="width:auto"> 🧠 thinking</label>
      <button class="iconbtn" onclick="chatClear()" title="clear conversation">⌫ clear</button>
      <button class="iconbtn" onclick="chatFull()" title="expand / restore">⤢</button>
      <button class="iconbtn" onclick="chatToggle()">✕</button></div></div>
  <div class="chatmsgs" id="chatmsgs"></div>
  <div class="chatcompose">
    <div class="ccontrols"><span id="chatStatus">checking server…</span><span style="margin-left:auto" class="muted">Enter to send · Shift+Enter newline</span></div>
    <div class="cinput"><textarea id="chatInput" rows="1" placeholder="Ask the model something…" oninput="chatAutogrow(this)" onkeydown="chatKey(event)"></textarea><button class="csend" id="chatSend" onclick="chatSendOrStop()">↑</button></div></div>
</div></div>

<button class="dbgbtn" onclick="toggleDbg()">&lt;/&gt; debug</button>
<div class="dbg" id="dbg"><div class="dbgh"><b style="font-size:13px">Debug</b>
  <div class="seg sm" style="margin-left:6px"><button class="on" data-d="timeline" onclick="dbgTab('timeline')">timeline</button><button data-d="snap" onclick="dbgTab('snap')">snapshot</button><button data-d="server" onclick="dbgTab('server')">server.log</button><button data-d="eval" onclick="dbgTab('eval')">eval.log</button></div>
  <button class="iconbtn" style="margin-left:auto" onclick="dbgCopy()">⧉ copy</button><button class="iconbtn" onclick="toggleDbg()">✕</button></div>
  <pre id="dbgBody" class="mono">…</pre></div>

<div id="toasts"></div>
<script>
const CSRF="__BENCHY_CSRF__";
/* Inject the per-launch CSRF token on every state-changing request. A cross-site page can't
   read this token (same-origin policy hides our HTML), so its forged POSTs are rejected. */
(function(){const _f=window.fetch;window.fetch=function(u,o){o=o||{};const m=(o.method||'GET').toUpperCase();
  if(m!=='GET'&&m!=='HEAD'){o.headers=Object.assign({'X-Benchy-CSRF':CSRF},o.headers||{});}return _f.call(this,u,o);};})();
let RUNS=[],PERF=[],META={},SUMMARY={},LIVE={},STREAM={},SYS={},ACT={},HISTM=[];
let VIEW='acc',SORT={k:'ts',d:-1},RMODE='thinking',FEEDF='all',DETF='all',DET=[],DETFILE='',charts={},_sid=0,_dataSig='',_streamSig='',_sparkSig='',_expSig='',accBenchUser=false,rBenchUser=false;
function setText(id,t){const e=$(id);if(e&&e.textContent!==t)e.textContent=t;}
let DBG={open:false,tab:'timeline'};
let REGMAP={};   // key -> {name,tier,fit,desc,domain} built from /api/benchmarks (registry + manual)
const NICE={medqa_test:'MedQA-USMLE',medmcqa:'MedMCQA',mmlu_medical:'MMLU Medical',pubmedqa:'PubMedQA',medxpertqa:'MedXpertQA',healthbench_hard:'HealthBench Hard',mmlu_pro:'MMLU-Pro',supergpqa:'SuperGPQA',humaneval:'HumanEval',mbpp:'MBPP',gpqa:'GPQA',hle:'HLE'};
function regName(b){return (REGMAP[b]&&REGMAP[b].name)||NICE[b]||b;}
function regDesc(b){return (REGMAP[b]&&REGMAP[b].desc)||'';}
function regTier(b){return (REGMAP[b]&&REGMAP[b].tier)||'other';}
const nice=b=>regName(b);
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const num=v=>(v==null||v==='')?'—':v;
function chart(id,cfg){const ex=charts[id];
  if(ex&&ex.config.type===cfg.type){const nd=(cfg.data&&cfg.data.datasets)||[];
    ex.data.labels=(cfg.data&&cfg.data.labels)||[];
    if(nd.length!==ex.data.datasets.length)ex.data.datasets=nd;else nd.forEach((d,i)=>Object.assign(ex.data.datasets[i],d));
    if(cfg.options)ex.options=cfg.options;ex.update('none');return;}
  if(ex)ex.destroy();const c=$(id);if(c&&window.Chart)charts[id]=new Chart(c,cfg);}
const grid='#1b1b1e';
/* Chart.js is a CDN dependency; if it failed to load (offline/air-gapped), guard every global
   access so the rest of the dashboard still boots instead of dying on a ReferenceError. */
if(window.Chart){Chart.defaults.color='#8a8a8a';Chart.defaults.font.family='Inter';Chart.defaults.borderColor=grid;Chart.defaults.animation=false;
Chart.defaults.interaction={mode:'index',intersect:false};Chart.defaults.plugins.tooltip.displayColors=false;Chart.defaults.plugins.tooltip.padding=8;Chart.defaults.plugins.tooltip.titleColor='#ededed';Chart.defaults.plugins.tooltip.bodyColor='#ededed';Chart.defaults.plugins.tooltip.backgroundColor='#16161a';Chart.defaults.plugins.tooltip.borderColor='#2a2a2e';Chart.defaults.plugins.tooltip.borderWidth=1;Chart.defaults.elements.point.hoverRadius=5;}
else{console.warn('benchy: Chart.js failed to load (offline?) — charts disabled, the rest of the dashboard still works.');}
if(window.marked)marked.setOptions({breaks:true,gfm:true});

/* ---------- clipboard + toast ---------- */
function toast(msg,bad){const t=document.createElement('div');t.className='toast'+(bad?' bad':'');t.textContent=msg;$('toasts').appendChild(t);requestAnimationFrame(()=>t.classList.add('in'));setTimeout(()=>{t.classList.remove('in');setTimeout(()=>t.remove(),320)},1800);}
async function copy(text,label){try{await navigator.clipboard.writeText(text);toast((label||'Copied')+' ✓');return}catch(e){}
  const ta=document.createElement('textarea');ta.value=text;ta.style.position='fixed';ta.style.opacity='0';document.body.appendChild(ta);ta.select();
  try{document.execCommand('copy');toast((label||'Copied')+' ✓')}catch(_){toast('Copy failed',1)}ta.remove();}
async function copyReport(btn){btn&&(btn.disabled=true);try{const md=await fetch('/api/report').then(r=>r.text());await copy(md,'Report markdown copied')}catch(e){toast('Could not fetch report',1)}btn&&(btn.disabled=false);}
function copyDiag(){copy(JSON.stringify({ts:new Date().toISOString(),live:LIVE,stream_live:STREAM.live,sys:SYS,activity:ACT},null,2),'Diagnostics snapshot copied');}

/* ---------- count-up ---------- */
function countUp(el,to,opts){if(!el)return;opts=opts||{};const dec=opts.dec||0,suf=opts.suf||'';
  const from=parseFloat(el.dataset.v||'');const f=isNaN(from)?to:from;
  el.dataset.v=to;if(f===to){el.textContent=to.toFixed(dec)+suf;return}
  const t0=performance.now(),dur=opts.dur||560;
  function step(t){let k=Math.min(1,(t-t0)/dur);k=1-Math.pow(1-k,3);el.textContent=(f+(to-f)*k).toFixed(dec)+suf;if(k<1)requestAnimationFrame(step)}
  requestAnimationFrame(step);}
function liveNum(el,val,dec,suf){if(!el)return;if(val==null){el.textContent='—';el.dataset.v='';return}countUp(el,val,{dec:dec,suf:suf||''});}
function fmtDur(s){if(s==null||isNaN(s))return '—';s=Math.round(s);const m=Math.floor(s/60);return m?m+'m '+(s%60)+'s':s+'s';}

/* ---------- sparkline ---------- */
function spark(vals,c,w,h){c=c||'#3291ff';w=w||120;h=h||26;const v=vals.filter(x=>x!=null&&!isNaN(x));if(v.length<2)return '';
  const mn=Math.min(...v),mx=Math.max(...v),rng=(mx-mn)||1;
  const pts=v.map((x,i)=>[i/(v.length-1)*w,h-2-((x-mn)/rng)*(h-4)]);
  const d=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');
  const id='sg'+(_sid++);const last=pts[pts.length-1];
  return `<svg width="100%" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" style="display:block;overflow:visible"><defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${c}" stop-opacity=".34"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></linearGradient></defs><path d="${d} L ${w} ${h} L 0 ${h} Z" fill="url(#${id})"/><path d="${d}" fill="none" stroke="${c}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/><circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="2.1" fill="${c}"/></svg>`;}

/* ---------- discreet environment strip ---------- */
function envStrip(){
  const el=$('envstrip');if(!el)return;
  const env=META.env||{};el.dataset.sig=el.dataset.sig||'';
  const sig=JSON.stringify(env);if(el.dataset.sig===sig)return;el.dataset.sig=sig;
  if(!Object.keys(env).length){el.innerHTML='<span class="es muted">model &amp; host auto-detect when the server is up · live footprint in System</span>';return;}
  el.innerHTML=Object.entries(env).map(([k,v])=>`<span class="es" title="click to copy" onclick="copy('${esc(k)}: ${esc(String(v)).replace(/'/g,'')}','${esc(k)} copied')"><b>${esc(k)}</b> ${esc(String(v))}</span>`).join('');
}

/* ---------- experiment band ---------- */
function experimentBand(){
  const el=$('expCard');if(!el)return;
  const running=!!STREAM.running,lv=STREAM.live||{};
  const sorted=[...RUNS].sort((a,b)=>(a.ts>b.ts?1:(a.ts<b.ts?-1:0)));
  const activeTag=(running&&lv.tag)?lv.tag:(sorted.length?sorted[sorted.length-1].tag:'—');
  const baseline=(META.baseline_tag||'baseline');
  const lastRun=sorted.length?sorted[sorted.length-1]:null;
  const bench=(running&&lv.benchmark)?lv.benchmark:(lastRun?lastRun.benchmark:'');
  const ourBest=RUNS.filter(r=>r.tag===activeTag&&r.benchmark===bench).sort((a,b)=>b.accuracy-a.accuracy)[0];
  const baseBest=RUNS.filter(r=>r.tag===baseline&&r.benchmark===bench).sort((a,b)=>b.accuracy-a.accuracy)[0];
  const refs=(META.references||{})[bench]||[];const primaryRef=SUMMARY.primary_ref;
  let refPick=primaryRef?refs.find(r=>r.label===primaryRef):null;   // configured primary baseline if it exists for this benchmark
  if(!refPick&&refs.length)refPick=refs.filter(r=>r.kind!=='chance').sort((a,b)=>b.accuracy-a.accuracy)[0]||refs[0];  // else strongest existing reference
  const liveActive=running&&lv.benchmark===bench&&lv.accuracy!=null;   // currently recording this benchmark
  const ourAcc=liveActive?lv.accuracy:(ourBest?ourBest.accuracy:null);
  const delta=(ourAcc!=null&&baseBest)?+(ourAcc-baseBest.accuracy).toFixed(1):null;
  const sig=[running,activeTag,bench,lv.i,lv.accuracy,ourBest&&ourBest.accuracy,baseBest&&baseBest.accuracy,refPick&&refPick.label].join('|');
  if(sig===_expSig)return;_expSig=sig;
  const isBase=activeTag===baseline;
  const desc=isBase?('This is the baseline tag (<b>'+esc(baseline)+'</b>) — the reference every other run is measured against.'):('Comparing run <b>'+esc(activeTag)+'</b> against the baseline tag <b>'+esc(baseline)+'</b> on the same benchmark.');
  const status=running?'<span class="tag ok" style="display:inline-flex;align-items:center;gap:6px"><span class="dot"></span>running</span>':'<span class="tag">idle · last run</span>';
  const mc=(l,v,c,cls)=>`<div class="expmetric"><div class="expml">${l}</div><div class="expmv ${cls||''}" style="color:${c||'var(--ink)'}">${v}</div></div>`;
  const dV=delta==null?'—':(delta>=0?'+':'')+delta+' pts';
  const dC=delta==null?'var(--mut)':delta>0?'var(--ok)':delta<0?'var(--dang)':'var(--mut)';
  const ourV=ourAcc!=null?(ourAcc+'%'+(liveActive?'<span class="livedot"></span>':'')):'—';
  const prog=running?`<div class="expprog"><div class="expbar"><i style="width:${lv.n?Math.round((lv.i||0)/lv.n*100):0}%"></i></div><span class="mono muted" style="font-size:12px;white-space:nowrap">${lv.i||0}/${lv.n||0} · ${esc(nice(lv.benchmark))} · ${esc(lv.mode||'')} · acc ${lv.accuracy!=null?lv.accuracy:'?'}%</span></div>`:'';
  el.innerHTML=`<div class="exphead"><div style="min-width:240px;flex:1">
      <div class="expk">CURRENT RUN</div>
      <div class="exptitle"><span class="tag pur">${esc(activeTag)}</span> ${status}</div>
      <p class="muted" style="margin:9px 0 0;font-size:12.5px;max-width:680px;line-height:1.55">${desc}</p></div>
    <div class="expmetrics">
      ${mc(esc(nice(bench))+' · this build'+(liveActive?' · live':''),ourV,'var(--ok)',liveActive?'blink':'')}
      ${mc('vs '+baseline,baseBest?baseBest.accuracy+'%':'—')}
      ${mc('Δ vs baseline'+(liveActive?' · live':''),dV,dC,liveActive&&delta!=null?'blink':'')}
      ${mc(refPick?esc(refPick.label):'reference',refPick?refPick.accuracy+'%':'—','var(--teal)')}
    </div></div>${prog}`;
}
/* ---------- system resources ---------- */
function sysOpts(unit,max){return {animation:false,responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' '+(c.parsed.y==null?'—':c.parsed.y)+' '+unit}}},scales:{y:{min:0,max:max,grid:{color:grid},ticks:{maxTicksLimit:4}},x:{display:false}}};}
function systemUI(){
  const h=(HISTM||[]).slice(-300);if(!h.length)return;
  const labels=h.map((_,i)=>i),last=h[h.length-1];
  const lastTps=[...h].reverse().find(r=>r.tps!=null);
  setText('sysTps',lastTps&&lastTps.tps!=null?lastTps.tps.toFixed(1)+' t/s':(last.up?'idle':'offline'));
  // decode: 0 when not decoding (so the line is anchored at the baseline, not floating gaps)
  const tps=h.map(r=>r.up?(r.tps!=null?r.tps:0):null),tpsMax=Math.max(10,...tps.filter(x=>x!=null));
  chart('sysTpsChart',{type:'line',data:{labels,datasets:[{data:tps,borderColor:'#f5a623',backgroundColor:'rgba(245,166,35,.12)',fill:true,tension:.3,pointRadius:0,borderWidth:2,spanGaps:false}]},options:sysOpts('t/s',Math.ceil(tpsMax*1.1))});
  const memTot=(META.host&&META.host.mem_total_gb)||null;
  setText('sysRam',last.rss!=null?last.rss+' GB':'—');   // big readout = model RSS only (short, never wraps)
  const hasSwap=h.some(r=>r.swap!=null&&r.swap>0);
  const leg=$('sysMemLeg');
  if(leg){const ps=['<span><i style="background:#3291ff"></i>model '+(last.rss!=null?last.rss:'—')+' GB</span>',
    '<span><i style="background:#8a63d2"></i>system '+(last.used!=null?last.used:'—')+(memTot?'/'+memTot:'')+' GB</span>'];
    if(hasSwap)ps.push('<span><i style="background:#e5484d"></i>swap '+(last.swap!=null?last.swap:0)+' GB</span>');
    leg.innerHTML=ps.join('');}
  const memDs=[{label:'model RSS',data:h.map(r=>r.rss),borderColor:'#3291ff',backgroundColor:'rgba(50,145,255,.12)',fill:true,tension:.3,pointRadius:0,borderWidth:2,spanGaps:true},
    {label:'system used',data:h.map(r=>r.used),borderColor:'#8a63d2',tension:.3,pointRadius:0,borderWidth:1.5,spanGaps:true}];
  if(hasSwap)memDs.push({label:'swap',data:h.map(r=>r.swap),borderColor:'#e5484d',backgroundColor:'rgba(229,72,77,.08)',tension:.3,pointRadius:0,borderWidth:1.5,spanGaps:true});
  chart('sysRamChart',{type:'line',data:{labels,datasets:memDs},options:sysOpts('GB',memTot?Math.ceil(memTot*1.05):null)});
  setText('sysCpu',last.cpu!=null?Math.round(last.cpu)+'%':'—');
  const cpu=h.map(r=>r.up?(r.cpu!=null?r.cpu:0):null),cpuMax=Math.max(100,...cpu.filter(x=>x!=null));
  chart('sysCpuChart',{type:'line',data:{labels,datasets:[{data:cpu,borderColor:'#0cce6b',backgroundColor:'rgba(12,206,107,.12)',fill:true,tension:.3,pointRadius:0,borderWidth:2,spanGaps:false}]},options:sysOpts('%',Math.ceil(cpuMax/10)*10)});
}
/* ---------- paired A/B (McNemar) ---------- */
function populateCompare(){
  const tags=[...new Set((RUNS||[]).map(r=>r.tag).filter(Boolean))];
  const benches=[...new Set((RUNS||[]).filter(r=>!String(r.benchmark||'').startsWith('healthbench')).map(r=>r.benchmark).filter(Boolean))];
  const fill=(id,vals,label)=>{const s=$(id);if(!s)return;const cur=s.value;s.innerHTML=vals.map(v=>`<option value="${esc(v)}">${esc(label?label(v):v)}</option>`).join('');if(vals.includes(cur))s.value=cur;};
  fill('cmpA',tags);fill('cmpB',tags);fill('cmpBench',benches,nice);
  const A=$('cmpA'),B=$('cmpB');
  if(A&&B&&tags.length>=2&&A.value===B.value){B.value=tags.find(t=>t!==A.value)||tags[1];}
}
function fmtCompare(d){
  if(!d||!d.ok)return '<span class="muted">'+esc((d&&d.error)||'no result')+'</span>';
  const verdict=d.significant
    ? `<b style="color:${d.delta>0?'var(--ok)':'var(--dang)'}">${esc(d.tag_a)} ${d.delta>0?'>':'<'} ${esc(d.tag_b)}</b> — <b>significant</b> (p=${d.p_value})`
    : `<b style="color:var(--warn)">no significant difference</b> (p=${d.p_value})`;
  return `<div style="font-size:13.5px">${verdict}</div>
    <div class="mono" style="margin-top:7px">${esc(d.tag_a)} <b>${d.acc_a}%</b> · ${esc(d.tag_b)} <b>${d.acc_b}%</b> · Δ <b>${d.delta>0?'+':''}${d.delta} pts</b> · N=${d.n_common} common</div>
    <div class="mono muted" style="margin-top:4px">discordant ${d.discordant} ( ${esc(d.tag_a)}-only ${d.a_better} · ${esc(d.tag_b)}-only ${d.b_better} ) · both ✓ ${d.both_right} · both ✗ ${d.both_wrong}</div>
    <div class="muted" style="margin-top:7px;font-size:11.5px">McNemar exact two-sided over the ${d.discordant} discordant pairs. A small p means the two builds genuinely disagree on these questions — overlapping Wilson bars can't detect a paired difference this small.</div>`;
}
async function runCompare(){
  const a=$('cmpA').value,b=$('cmpB').value,bench=$('cmpBench').value,mode=$('cmpMode').value,el=$('cmpResult');
  if(!el)return;
  if(!a||!b||!bench){el.innerHTML='<span class="muted">pick two run tags and a benchmark</span>';return;}
  el.textContent='comparing…';
  try{const qs='/api/compare?a='+encodeURIComponent(a)+'&b='+encodeURIComponent(b)+'&bench='+encodeURIComponent(bench)+(mode?'&mode='+encodeURIComponent(mode):'');
    el.innerHTML=fmtCompare(await fetch(qs).then(r=>r.json()));}
  catch(e){el.innerHTML='<span class="muted">compare failed</span>';}
}
/* ---------- accuracy ---------- */
function accbars(){
  const bench=($('accBench')&&$('accBench').value)||'';
  const installed=(META.benchmarks||[]).includes(bench);
  const ours=(SUMMARY.per_run||[]).filter(r=>r.benchmark===bench).map(r=>({label:r.tag+' · '+r.mode+' (N='+r.n+')',acc:r.accuracy,lo:r.ci_lo,hi:r.ci_hi,small:r.small_n,cls:'ours'}));
  const refs=((META.references||{})[bench]||[]).map(r=>({label:r.label,acc:r.accuracy,cls:r.kind||'ref'}));
  const all=ours.concat(refs).sort((a,b)=>b.acc-a.acc);
  const refLbl=$('refLbl');if(refLbl)refLbl.textContent=refs.length?' vs reference baselines':'';
  const ad=$('accDesc');if(ad)ad.textContent=regDesc(bench)||'';
  const CLS={ours:['linear-gradient(90deg,#0070f3,#3291ff)','Your runs'],frontier:['#a855f7','Frontier'],ref:['#14b8a6','Reference'],chance:['#3a3a3a','Chance']};
  const present=[...new Set(all.map(x=>x.cls))];
  const legend='<div style="display:flex;gap:16px;flex-wrap:wrap;margin:2px 0 14px;font-size:12px;color:#9a9aa2">'+['ours','frontier','ref','chance'].filter(k=>present.includes(k)).map(k=>`<span style="display:inline-flex;align-items:center;gap:6px"><i style="width:11px;height:11px;border-radius:3px;display:inline-block;background:${CLS[k][0]}"></i>${CLS[k][1]}</span>`).join('')+'</div>';
  $('accbars').innerHTML=all.length?(legend+all.map(x=>{const s=CLS[x.cls]||CLS.ref;const lbl=x.cls==='ours'?`<b>${esc(x.label)}</b>`:esc(x.label);
    const ci=(x.cls==='ours'&&x.lo!=null)?`<div class="ci" style="left:${x.lo}%;width:${Math.max(0,x.hi-x.lo)}%"></div>`:'';
    const pv=(x.cls==='ours'&&x.lo!=null)?`${x.acc}% <small>[${x.lo}–${x.hi}]${x.small?' ⚠':''}</small>`:`${x.acc}%`;
    return `<div class="barwrap"><div class="lab" title="${esc(x.label)}">${lbl}</div><div class="track"><div class="fill" style="width:${Math.max(2,x.acc)}%;background:${s[0]}"></div>${ci}</div><div class="pv mono">${pv}</div></div>`}).join('')):('<div class="muted">No data for '+(nice(bench)||'this benchmark')+' yet — '+(installed?'launch a run from <b>Run a test</b> above.':'<a href="#" onclick="openBenchmarks();return false">fetch it from ⬇ Benchmarks</a> first.')+'</div>');
  // coherence: tell the user when the selected benchmark has baselines but isn't installed to run
  const note=$('accNote');
  if(note)note.innerHTML=((bench&&!installed&&refs.length)?'⚠ <b>'+esc(nice(bench))+'</b> isn\'t installed — <a href="#" onclick="openBenchmarks();return false">fetch it from ⬇ Benchmarks</a> to score your model against these baselines. ':'')+'Frontier baselines are <b>published numbers</b> (eval-setup-dependent — for context, not size-matched). Edit/add in <a href="#" onclick="openSetup();return false">Setup</a>.';
  // mode chart from raw runs
  const runs=RUNS.filter(r=>r.benchmark===bench);
  const byMode={};runs.forEach(r=>{(byMode[r.mode]=byMode[r.mode]||[]).push(r.accuracy)});
  const labels=Object.keys(byMode),data=labels.map(m=>byMode[m].reduce((a,b)=>a+b,0)/byMode[m].length);
  chart('modeChart',{type:'bar',data:{labels:labels.map(m=>m==='thinking'?'🧠 thinking':'⚡ non-thinking'),datasets:[{data,backgroundColor:labels.map(m=>m==='thinking'?'#0cce6b':'#8a63d2'),borderRadius:8,barThickness:46}]},options:{indexAxis:'y',plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' '+c.parsed.x.toFixed(1)+'%'}}},scales:{x:{beginAtZero:true,max:100,grid:{color:grid},ticks:{callback:v=>v+'%'}},y:{grid:{display:false},ticks:{font:{size:14}}}}}});
  const mac=SUMMARY.macro||{};
  $('macroHint').textContent=(mac.thinking_mean!=null)?('MCQ macro-avg '+mac.thinking_mean+'% thinking (k='+mac.thinking_k+(mac.thinking_tag?', '+mac.thinking_tag:'')+')'+(mac.nothink_mean!=null?' · '+mac.nothink_mean+'% nothink (k='+mac.nothink_k+(mac.nothink_tag?', '+mac.nothink_tag:'')+')':'')+' · excl. HealthBench'):'';
}

/* ---------- performance ---------- */
/* ---------- runs explorer ---------- */
function renderRuns(){
  const q=($('q').value||'').toLowerCase();
  const cols=[['ts','date'],['benchmark','benchmark'],['tag','tag'],['mode','mode'],['n','N',1],['accuracy','accuracy',1],['sec_per_q','s/q',1],['notes','notes']];
  let rows=RUNS.slice().filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q));
  rows.sort((a,b)=>{const x=a[SORT.k],y=b[SORT.k];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*SORT.d});
  $('runHead').innerHTML='<tr>'+cols.map(c=>`<th class="${c[2]?'n':''}" onclick="sortBy('${c[0]}')">${c[1]}${SORT.k===c[0]?(SORT.d>0?' ↑':' ↓'):''}</th>`).join('')+'<th></th></tr>';
  $('runBody').innerHTML=rows.map((r,ri)=>{const clk=r.details?` style="cursor:pointer" title="click to inspect every question" onclick="openDetails('${esc(r.details)}')"`:'';
    const cells=cols.map(c=>{let v=r[c[0]];
      if(c[0]==='accuracy')return `<td class="n mono"><b>${esc(v)}%</b>${r.details?' <span style="color:var(--acc2)">🔍</span>':''}</td>`;
      if(c[0]==='tag')return `<td><span class="tag">${esc(v)}</span></td>`;
      if(c[0]==='mode')return `<td><span class="tag ${v==='thinking'?'pur':''}">${esc(v||'—')}</span></td>`;
      return `<td class="${c[2]?'n mono':''} ${c[0]==='notes'?'muted':''}">${esc(v==null?'—':v)}</td>`}).join('');
    return `<tr${clk}>${cells}<td class="n"><button class="iconbtn xs" title="copy run JSON" onclick="event.stopPropagation();copyRun(${ri})">⧉</button></td></tr>`;}).join('');
  $('runcount').textContent=rows.length+' run'+(rows.length!==1?'s':'');
  window._rrows=rows;
}
function copyRun(ri){const r=(window._rrows||[])[ri];if(r)copy(JSON.stringify(r,null,2),'Run JSON copied');}
function sortBy(k){if(SORT.k===k)SORT.d*=-1;else{SORT.k=k;SORT.d=1}renderRuns()}

/* ---------- live stream + activity ---------- */
function setMode(m){RMODE=m;$('segT').classList.toggle('on',m==='thinking');$('segN').classList.toggle('on',m==='nothink')}
function syncBench(){
  if(!(STREAM.running&&STREAM.live&&STREAM.live.benchmark))return;const b=STREAM.live.benchmark;
  if(!accBenchUser){const ab=$('accBench');if(ab&&ab.value!==b&&[...ab.options].some(o=>o.value===b)){ab.value=b;accbars();}}
  if(!rBenchUser){const rb=$('rBench');if(rb&&rb.value!==b&&[...rb.options].some(o=>o.value===b))rb.value=b;}
}
function setFeedFilter(f){FEEDF=f;['all','ok','no'].forEach(k=>$('f'+k[0].toUpperCase()+k.slice(1)).classList.toggle('on',k===f));streamUI();}
function timingParts(e){return [e.prefill_s!=null?'prefill '+e.prefill_s+'s':'',e.think_tokens?'think '+e.think_tokens+' tok':'',e.gen_tokens!=null?'gen '+e.gen_tokens+' tok'+(e.gen_tps!=null?' @ '+e.gen_tps+' t/s':''):'',e.gen_s!=null?e.gen_s+'s decode':''].filter(Boolean);}
function timingLine(e){const p=timingParts(e);return p.length?`<div class="qtime mono">⏱ ${p.join(' · ')}</div>`:'';}
function feedText(i){const e=(STREAM.events||[]).find(x=>x.i===i)||{};const tm=timingParts(e).join(' · ');
  if(e.gold==='rubric')  // HealthBench: rubric score, not pred/gold
    return `Q${e.i} — HealthBench rubric score ${e.pred} (${e.pred_txt})  running-avg ${e.acc}%  per-q ${e.lat_q!=null?e.lat_q:e.t}s\nprompt: ${e.q}\nmodel output (tail): ${e.ans||''}`;
  return `Q${e.i} [${e.ok?'CORRECT':'WRONG'}]  pred=${e.pred}  gold=${e.gold}  acc=${e.acc}%  per-q ${e.lat_q!=null?e.lat_q:e.t}s${tm?'\ntiming: '+tm:''}\nQ: ${e.q}\npred: ${e.pred}. ${e.pred_txt}\ngold: ${e.gold}. ${e.gold_txt}\nmodel output (tail): ${e.ans||''}`;}
function feedRow(e){
  if(e.gold==='rubric'){  // HealthBench: rubric-scored, not MCQ — no pred/gold, no ✓/✗, color by score band
    const sc=parseFloat(e.pred)||0,col=sc>=50?'#0cce6b':(sc>=25?'#f5a623':'#ff4d4f');
    return `<div class="qrow" data-qi="${e.i}"><div class="qh" onclick="openLiveDetails(${e.i})" title="open the conversation, response & rubric">
      <span style="display:inline-block;min-width:36px;text-align:center;padding:1px 5px;border-radius:5px;font-weight:700;font-size:11px;background:${col};color:#0a0a0c">${sc.toFixed(0)}%</span><b class="mono">Q${e.i}</b>
      <span class="muted">rubric score</span> <b>${esc(e.pred)}</b> <span class="muted">${esc(e.gold_txt||'')}</span>
      <span class="qh-r"><span class="mono muted">${e.lat_q!=null?e.lat_q:e.t}s · ${e.acc}% avg</span><span class="muted" style="font-size:11px">🔍</span><button class="iconbtn xs" title="copy full Q&amp;A" onclick="event.stopPropagation();copyLiveQ(${e.i})">⧉</button></span></div></div>`;}
  const cls=e.ok?'ok':'no';
  return `<div class="qrow ${cls}" data-qi="${e.i}"><div class="qh" onclick="openLiveDetails(${e.i})" title="open the full question & model output">
    <span class="mk ${cls}">${e.ok?'✓':'✗'}</span><b class="mono">Q${e.i}</b>
    <span class="muted">pred</span> <b>${esc(e.pred)}</b><span class="muted">gold</span> <b>${esc(e.gold)}</b>
    <span class="qh-r"><span class="mono muted">${e.lat_q!=null?e.lat_q:e.t}s · ${e.acc}%</span><span class="muted" style="font-size:11px">🔍</span><button class="iconbtn xs" title="copy full Q&amp;A" onclick="event.stopPropagation();copyLiveQ(${e.i})">⧉</button></span></div></div>`;}
function renderFeed(evs){
  const open=new Set([...document.querySelectorAll('#feed .qrow.open')].map(n=>n.dataset.qi));
  const rows=evs.filter(e=>FEEDF==='all'||(FEEDF==='ok'&&e.ok)||(FEEDF==='no'&&!e.ok)).slice().reverse();
  $('feed').innerHTML=rows.map(feedRow).join('')||'<div class="muted" style="padding:10px">waiting for first answer…</div>';
  open.forEach(qi=>{const n=document.querySelector('#feed .qrow[data-qi="'+qi+'"]');if(n)n.classList.add('open');});
  const ok=evs.filter(e=>e.ok).length;setText('feedCount',evs.length?('· '+ok+'/'+evs.length+' ✓'):'');
  renderQaDonut();
}
function renderQaDonut(){
  const lv=STREAM.live||{},evs=STREAM.events||[];
  const correct=(lv.correct!=null?lv.correct:evs.filter(e=>e.ok).length);
  const total=((lv.i!=null&&lv.i>0)?lv.i:evs.length);
  const wrong=Math.max(0,total-correct);
  setText('qaPct',total?Math.round(100*correct/total)+'%':'—');
  chart('qaDonutChart',{type:'doughnut',data:{labels:['correct','wrong'],datasets:[{data:[correct,wrong],backgroundColor:['#0cce6b','#ff4d4f'],borderColor:'#0a0a0c',borderWidth:2,hoverOffset:3}]},options:{maintainAspectRatio:false,cutout:'72%',plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' '+c.label+': '+c.parsed}}}}});
  const lg=$('qaLeg');if(lg)lg.innerHTML=`<div class="li"><span class="sw" style="background:#0cce6b"></span><b>${correct}</b>&nbsp;correct</div><div class="li"><span class="sw" style="background:#ff4d4f"></span><b>${wrong}</b>&nbsp;wrong</div><div class="li muted" style="margin-top:1px">${total} answered</div>`;
}
function syncFeedHeight(){const ls=document.querySelector('#liveSec .livestack'),fc=document.querySelector('#liveSec .feedcard');if(ls&&fc&&ls.offsetHeight>40)fc.style.height=ls.offsetHeight+'px';}
function renderLatBars(lat){
  const labels=lat.map(p=>p[0]),vals=lat.map(p=>p[1]);
  const present=vals.filter(x=>x!=null);const mx=Math.max(1,...present);
  chart('latLive',{type:'bar',data:{labels,datasets:[{data:vals,backgroundColor:'#f5a623',hoverBackgroundColor:'#ffc857',borderRadius:3,borderSkipped:false,categoryPercentage:0.9,barPercentage:0.95}]},options:{plugins:{legend:{display:false},tooltip:{callbacks:{title:items=>'Q'+items[0].label,label:c=>' '+c.parsed.y+' s'}}},scales:{y:{min:0,grid:{color:grid},ticks:{maxTicksLimit:4,callback:v=>v+'s'}},x:{grid:{display:false},ticks:{maxTicksLimit:12,autoSkip:true}}}}});
  const s=present.slice().sort((a,b)=>a-b);
  if(s.length){const md=s[Math.floor(s.length/2)],p90=s[Math.min(s.length-1,Math.floor(s.length*0.9))];
    setText('latStat','med '+md.toFixed(1)+'s · p90 '+p90.toFixed(1)+'s');setText('latCapR','peak '+mx.toFixed(0)+'s · n='+present.length);}
  else{setText('latStat','');setText('latCapR','');}
}
function renderLiveCharts(S){
  const ser=S.series||[],lat=S.lat||[];
  chart('accLive',{type:'line',data:{labels:ser.map(p=>p[0]),datasets:[{label:'accuracy',data:ser.map(p=>p[1]),borderColor:'#0cce6b',backgroundColor:'rgba(12,206,107,.10)',fill:true,tension:.25,pointRadius:0,borderWidth:2}]},options:{plugins:{legend:{display:false},tooltip:{callbacks:{title:items=>'Q'+items[0].label,label:c=>' acc '+c.parsed.y+'%'}}},scales:{y:{min:0,max:100,grid:{color:grid},ticks:{callback:v=>v+'%'}},x:{grid:{display:false},title:{display:true,text:'question #'}}}}});
  renderLatBars(lat);
  requestAnimationFrame(syncFeedHeight);
}
window.addEventListener('resize',()=>syncFeedHeight());
function streamUI(){
  const running=STREAM.running,evs=STREAM.events||[],lv=STREAM.live||{};
  $('liveSec').style.display=(running||evs.length)?'block':'none';
  $('liveMeta').innerHTML=lv.benchmark?`<b>${esc(nice(lv.benchmark))}</b> · ${esc(lv.mode)} · <span class="mono">${esc(lv.tag||'')}</span>`:'';
  liveNum($('accBig'),lv.accuracy!=null?lv.accuracy:null,1,'%');
  setText('accSub',(lv.correct!=null&&lv.i)?(lv.correct+'/'+lv.i+' correct · elapsed '+fmtDur(lv.elapsed_s)):'');
  $('runBtn').disabled=running;$('stopBtn').style.display=running?'inline-block':'none';
  const last=evs.length?evs[evs.length-1]:null;
  const sig=FEEDF+'|'+evs.length+'|'+(last?last.i+'_'+last.acc:'')+'|'+(running?1:0);
  if((running||evs.length)&&sig!==_streamSig){_streamSig=sig;renderFeed(evs);renderLiveCharts(STREAM);}
}
function activityUI(){
  const a=ACT||{},lv=STREAM.live||LIVE||{};const running=!!STREAM.running;
  let ph=a.phase||'idle';if(!a.server_up)ph='idle';
  $('phasePill').dataset.p=ph;$('phaseTxt').textContent=ph;
  const active=a.server_up&&(ph==='generating'||ph==='thinking'||ph==='prefill');
  liveNum($('tpsVal'),active?(a.chunk_tps!=null?a.chunk_tps:null):null,1);
  $('tpsFoot').textContent=a.avg_tps!=null?('avg '+a.avg_tps.toFixed(1)+' t/s'):(a.last_finish?'last: '+a.last_finish.tokens+' tok in '+a.last_finish.secs+'s':'avg —');
  liveNum($('genVal'),active?(a.gen||0):(a.last_finish?a.last_finish.tokens:0),0);
  $('genFoot').textContent=active?'current answer':(a.last_finish?'last answer':'current answer');
  const ser=a.series||[];
  if($('actBars').children.length!==42){let h='';for(let i=0;i<42;i++)h+='<i style="height:6%"></i>';$('actBars').innerHTML=h;}
  const bars=$('actBars').children,N=bars.length,slice=ser.slice(-N),mx=Math.max(1,...slice.map(s=>s.tps||0));
  for(let i=0;i<N;i++){const s=slice[i-(N-slice.length)],b=bars[i];
    if(s&&active){b.style.height=Math.max(6,Math.round((s.tps/mx)*100))+'%';b.className=s.thinking?'think':'';b.title=(s.thinking?'thinking ':'')+s.tps+' t/s';}
    else{b.style.height='6%';b.className='';b.removeAttribute('title');}}
  setText('eqcapR',active?(ph==='thinking'?'🧠 reasoning':'⚡ answering')+' · peak '+mx.toFixed(1)+' t/s':(a.server_up?'idle between questions':'server offline'));
  const i=lv.i||0,n=lv.n||0,el=lv.elapsed_s;
  $('progVal').textContent=i+'/'+n;
  if(running&&i>0&&n>i&&el)$('etaFoot').textContent='ETA '+fmtDur((el/i)*(n-i));
  else $('etaFoot').textContent=running?'ETA —':'idle';
}

/* ---------- controls ---------- */
async function runTest(){
  $('runMsg').textContent='starting…';$('runMsg').style.color='var(--mut)';
  const body={benchmark:$('rBench').value,n:$('rN').value,mode:RMODE,tag:$('rTag').value};
  const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
  $('runMsg').textContent=r.ok?('▶ running '+r.started.tag+' '+r.started.mode+' N='+r.started.n):('⚠ '+r.error);
  $('runMsg').style.color=r.ok?'var(--ok)':'var(--dang)';pulse();
}
async function stopTest(){await fetch('/api/run/stop',{method:'POST'});$('runMsg').textContent='stopped';toast('Run stop requested');pulse()}
async function startServer(){$('srvMsg').textContent='starting (~1 min)…';
  const r=await fetch('/api/server/start',{method:'POST'}).then(r=>r.json());
  $('srvMsg').textContent=r.ok?(r.already?'already up':'loading model…'):('⚠ '+r.error);}
function stopAll(){if(!confirm('Stop ALL running tests/chains?'))return;fetch('/api/stop-all',{method:'POST'}).then(r=>r.json()).then(d=>{$('srvMsg').textContent='stopped: '+((d.killed||[]).join(', ')||'nothing was running');toast('Stopped all tests');pulse();});}
function killServer(){if(!confirm('Force-kill the model server? Next run reloads the model (~1 min).'))return;fetch('/api/server/stop',{method:'POST'}).then(()=>{$('srvMsg').textContent='server killed';toast('Server killed');setTimeout(pulse,800);});}

/* ---------- details modal ---------- */
function closeDetails(){$('detModal').style.display='none';}
function setDetFilter(f){DETF=f;$('dfAll').classList.toggle('on',f==='all');$('dfOk').classList.toggle('on',f==='ok');$('dfNo').classList.toggle('on',f==='no');renderDet();}
function openDetails(file){
  DETFILE=file;DETF='all';DET=[];$('detModal').style.display='block';$('detTitle').textContent='Run detail';$('detSummary').textContent='';$('detQ').value='';
  setDetFilter('all');$('detBody').innerHTML='<div class="muted">loading…</div>';
  fetch('/api/details?file='+encodeURIComponent(file)).then(r=>r.json()).then(rows=>{
    DET=rows||[];
    if(!DET.length){$('detBody').innerHTML='<div class="muted">No saved per-question details for this run — it predates detail logging. Re-run it to populate.</div>';$('detCopyWrong').style.display='none';return;}
    const isHB=DET[0].criteria!==undefined;DET._hb=isHB;
    $('detTitle').textContent=(file.split('__')[0]||'Run')+(isHB?' · HealthBench scoring':' · questions');
    $('detCopyWrong').style.display=isHB?'none':'inline-flex';
    if(isHB){const mean=(DET.reduce((a,r)=>a+(r.score||0),0)/DET.length).toFixed(1);$('detSummary').textContent=DET.length+' questions · mean '+mean+'%';}
    else{const ok=DET.filter(r=>r.ok).length;$('detSummary').textContent=DET.length+' questions · '+ok+'/'+DET.length+' correct ('+(100*ok/DET.length).toFixed(1)+'%)';}
    renderDet();
  }).catch(()=>{$('detBody').innerHTML='<div class="muted">error loading details</div>';});
}
function renderDet(){
  if(!DET.length)return;const isHB=DET._hb;const q=($('detQ').value||'').toLowerCase();
  let rows=DET.filter(r=>DETF==='all'||(!isHB&&DETF==='ok'&&r.ok)||(!isHB&&DETF==='no'&&!r.ok));
  if(q)rows=rows.filter(r=>JSON.stringify(r).toLowerCase().includes(q));
  $('detBody').innerHTML=rows.length?rows.map(r=>isHB?hbRow(r):mcqRow(r)).join(''):'<div class="muted">no questions match</div>';
}
function copyWrong(){const wrong=DET.filter(r=>!r.ok&&!DET._hb);if(!wrong.length){toast('No wrong answers 🎉');return}copy(JSON.stringify(wrong,null,2),wrong.length+' wrong answers copied');}
function showDetailRows(title,rows,focusI){
  DET=rows||[];DET._hb=DET[0]&&DET[0].criteria!==undefined;
  $('detTitle').textContent=title+(DET._hb?' · HealthBench scoring':' · questions');
  $('detCopyWrong').style.display=DET._hb?'none':'inline-flex';
  if(!DET.length){$('detBody').innerHTML='<div class="muted">No per-question details yet for this run.</div>';$('detSummary').textContent='';return;}
  if(DET._hb){const mean=(DET.reduce((a,r)=>a+(r.score||0),0)/DET.length).toFixed(1);$('detSummary').textContent=DET.length+' questions · mean '+mean+'%';}
  else{const ok=DET.filter(r=>r.ok).length;$('detSummary').textContent=DET.length+' answered · '+ok+'/'+DET.length+' correct ('+(100*ok/DET.length).toFixed(1)+'%)';}
  renderDet();
  if(focusI!=null)setTimeout(()=>{const el=document.querySelector('#detBody details[data-i="'+focusI+'"]');if(el){el.open=true;el.scrollIntoView({block:'center'});}},30);
}
function openLiveDetails(focusI){
  $('detModal').style.display='block';$('detTitle').textContent='Live run · loading…';$('detSummary').textContent='';$('detQ').value='';DETF='all';setDetFilter('all');$('detBody').innerHTML='<div class="muted">loading full detail…</div>';
  fetch('/api/live_details').then(r=>r.json()).then(d=>{DETFILE=d.file||'';showDetailRows((d.file?d.file.split('__')[0]:'Run')+' · live',d.rows||[],focusI);}).catch(()=>{$('detBody').innerHTML='<div class="muted">error loading live detail</div>';});
}
async function copyLiveQ(i){try{const d=await fetch('/api/live_details').then(r=>r.json());const r=(d.rows||[]).find(x=>x.i===i);if(r){copy(r.criteria!==undefined?hbCopyText(r):mcqText(r),'Q'+i+' (full) copied');return;}}catch(e){}copy(feedText(i),'Q'+i+' copied (preview only)');}
function hbCopyText(r){const conv=(r.conversation||[]).map(m=>m.role+': '+m.content).join('\n');const cr=(r.criteria||[]).map(c=>(c.met?'✓':'✗')+' ['+c.points+'] '+c.criterion).join('\n');return `Q${r.i} — HealthBench rubric score ${r.score}% (${r.achieved}/${r.total} pts)\n\nConversation:\n${conv}\n\nModel response:\n${r.response||''}\n\nRubric (${(r.criteria||[]).length} criteria):\n${cr}`;}
function mcqText(r){const opts=r.options||{};const ol=Object.keys(opts).map(k=>k+'. '+opts[k]).join('\n');const tm=timingParts(r).join(' · ');
  return `Q${r.i} [${r.ok?'CORRECT':'WRONG'}]  model=${r.pred}  gold=${r.gold}${tm?'\ntiming: '+tm:''}\n\n${r.question}\n${ol}\n\nmodel output:\n${r.answer||''}`;}
function mcqRow(r){
  const opts=r.options||{};
  const ol=Object.keys(opts).map(k=>{const hl=k===r.gold?'background:rgba(12,206,107,.12);border-color:rgba(12,206,107,.45)':(k===r.pred?'background:rgba(255,77,79,.10);border-color:rgba(255,77,79,.45)':'');
    return `<div style="border:1px solid #232326;border-radius:6px;padding:5px 9px;margin:3px 0;font-size:12px;${hl}"><b>${esc(k)}.</b> ${esc(opts[k])}${k===r.gold?' ✅':''}${(k===r.pred&&k!==r.gold)?' ⟵ model':''}</div>`;}).join('');
  return `<details data-i="${r.i}" style="border:1px solid #1d1d20;border-radius:10px;padding:10px 12px;margin:8px 0"><summary style="cursor:pointer;display:flex;align-items:center;gap:8px"><span class="mk ${r.ok?'ok':'no'}">${r.ok?'✓':'✗'}</span> <b>Q${r.i}</b> — model <b>${esc(r.pred)}</b> · gold <b>${esc(r.gold)}</b><button class="iconbtn xs" style="margin-left:auto" onclick="event.preventDefault();event.stopPropagation();copy(mcqText(DET[${DET.indexOf(r)}]),'Q${r.i} copied')">⧉ copy full</button></summary><div style="margin:8px 0;font-size:13px;line-height:1.55">${esc(r.question)}</div>${ol}${timingLine(r)}<div style="margin-top:8px"><div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px">model output (full)</div><pre class="ans" style="max-height:300px">${esc(r.answer||'')}</pre></div></details>`;
}
function hbRow(r){
  const cr=(r.criteria||[]).map(c=>`<div style="display:flex;gap:8px;padding:5px 0;font-size:12px;border-top:1px solid #1a1a1c"><span class="mk ${c.met?'ok':'no'}">${c.met?'✓':'✗'}</span><span class="mono" style="min-width:38px;color:${c.points>0?'var(--ok)':'var(--dang)'}">${c.points>0?'+':''}${c.points}</span><span>${esc(c.criterion)}<div class="muted" style="margin-top:2px">${esc(c.explanation||'')}</div></span></div>`).join('');
  const conv=esc((r.conversation||[]).map(m=>m.role+': '+m.content).join('  /  ')).slice(0,900);
  return `<details style="border:1px solid #1d1d20;border-radius:10px;padding:10px 12px;margin:8px 0"><summary style="cursor:pointer"><b>Q${r.i}</b> — score <b style="color:${r.score>=25?'var(--ok)':'var(--warn)'}">${r.score}%</b> (${r.achieved}/${r.total} pts)</summary><div style="margin:8px 0;font-size:12px;line-height:1.5;color:#aaa">${conv}</div><div style="margin:8px 0;font-size:13px;line-height:1.55;border-left:2px solid var(--acc);padding-left:10px"><b>model's response:</b><br>${esc(r.response||'')}</div><div style="margin-top:8px"><b style="font-size:12px">rubric (${(r.criteria||[]).length} criteria):</b>${cr}</div></details>`;
}

/* ---------- debug drawer ---------- */
function toggleDbg(){DBG.open=!DBG.open;$('dbg').classList.toggle('open',DBG.open);if(DBG.open)dbgRender();}
function dbgTab(t){DBG.tab=t;document.querySelectorAll('.dbgh [data-d]').forEach(b=>b.classList.toggle('on',b.dataset.d===t));dbgRender();}
async function dbgRender(){const b=$('dbgBody');
  if(DBG.tab==='timeline'){const evs=STREAM.events||[],a=ACT||{},lv=STREAM.live||{};
    const head='# correlated event timeline'+(lv.tag?' · '+lv.tag+' · '+nice(lv.benchmark||'')+' · '+(lv.mode||''):'')+'\n';
    const rows=evs.map(e=>{const t=timingParts(e).join(' · ');return 'Q'+String(e.i).padStart(2)+' '+(e.ok?'OK':'XX')+' pred='+e.pred+' gold='+e.gold+' acc='+e.acc+'% | wall '+e.t+'s'+(t?' | '+t:'');});
    b.textContent=head+(rows.join('\n')||'(no events yet)')+'\n\n# live now: phase='+a.phase+' gen='+a.gen+' chunk='+a.chunk_tps+' avg='+a.avg_tps+' t/s · log age '+a.age+'s';return;}
  if(DBG.tab==='snap'){b.textContent=JSON.stringify({live:LIVE,stream_live:STREAM.live,sys:SYS,activity:ACT,history_pts:(HISTM||[]).length},null,2);return;}
  try{const r=await fetch('/api/log?which='+DBG.tab).then(r=>r.json());b.textContent=r.text||'(empty)';b.scrollTop=b.scrollHeight;}catch(e){b.textContent='error loading log';}}
function dbgCopy(){copy($('dbgBody').textContent,'Debug '+DBG.tab+' copied');}

/* ---------- chat console ---------- */
let CHAT={msgs:[],busy:false,abort:null};
function chatToggle(){const w=$('chatwrap');const open=!w.classList.contains('open');w.classList.toggle('open',open);if(open){chatRender();chatStatus();setTimeout(()=>$('chatInput').focus(),60);}}
function chatFull(){$('chatpanel').classList.toggle('full');}
function chatClear(){if(CHAT.busy)chatStop();CHAT.msgs=[];chatRender();}
function chatAutogrow(t){t.style.height='auto';t.style.height=Math.min(170,t.scrollHeight)+'px';}
function chatKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();chatSend();}}
function chatSetStatus(up){const e=$('chatStatus');if(!e||CHAT.busy)return;e.textContent=up?'● model online':'● model offline — start it from Control';e.style.color=up?'var(--ok)':'var(--dang)';$('chatSend').disabled=!up;}
async function chatStatus(){try{const s=await fetch('/api/server/status').then(r=>r.json());chatSetStatus(s.up);}catch(e){}}
function chatExample(b){const i=$('chatInput');i.value=b.textContent;chatAutogrow(i);i.focus();}
function chatEmpty(){const ex=['Explain how a hash map works, with a worked example.','Write a function that returns the nth Fibonacci number.','What trade-offs separate TCP from UDP?'];
  const m=(META&&META.model)?esc(META.model):'the model';
  return `<div class="chatempty"><h3>Chat with ${m}</h3><p class="muted">Your local model, served on the OpenAI-compatible endpoint. Toggle 🧠 thinking for chain-of-thought reasoning.</p><div class="exq">${ex.map(q=>`<button onclick="chatExample(this)">${esc(q)}</button>`).join('')}</div></div>`;}
function mdRender(t){try{return DOMPurify.sanitize(marked.parse(t||''));}catch(e){return esc(t||'').replace(/\n/g,'<br>');}}
function splitThink(raw){const o=raw.search(/<think>/i);if(o<0)return {think:'',answer:raw};
  const open=raw.match(/<think>/i)[0];const after=raw.slice(o+open.length);const c=after.search(/<\/think>/i);
  if(c<0)return {think:after,answer:''};const close=after.match(/<\/think>/i)[0];
  return {think:after.slice(0,c),answer:(raw.slice(0,o)+after.slice(c+close.length))};}
function chatBotInner(m){const sp=m.streaming?splitThink(m.raw):{think:m.think,answer:m.answer};
  let h='';const ans=(sp.answer||'').trim();
  if(sp.think&&sp.think.trim())h+=`<details class="reason"${(m.streaming&&!ans)?' open':''}><summary>🧠 reasoning${(m.streaming&&!ans)?' · thinking…':''}</summary><div class="rbody">${esc(sp.think.trim())}</div></details>`;
  h+=ans?mdRender(ans):(m.streaming?'<span class="cursor"></span>':'');
  return h;}
function chatBubble(m){if(m.role==='user')return `<div class="cmsg user"><div class="av">you</div><div class="body">${esc(m.content).replace(/\n/g,'<br>')}</div></div>`;
  const sid=m.streaming?' id="cmsg-stream"':'';
  return `<div class="cmsg bot"${sid}><div class="av">ai</div><div class="body">${chatBotInner(m)}</div></div>`+chatMeta(m);}
function chatMeta(m){if(m.streaming||!m.dt)return '';return `<div class="cmeta">${m.toks||0} tok · ${m.dt.toFixed(1)}s · ${(m.toks/Math.max(m.dt,0.1)).toFixed(1)} t/s${(m.think&&m.think.trim())?' · 🧠 reasoned':''}</div>`;}
function chatRender(){const c=$('chatmsgs');if(!CHAT.msgs.length){c.innerHTML=chatEmpty();return;}c.innerHTML=CHAT.msgs.map(chatBubble).join('');chatDecorate();chatScroll();}
function chatUpdateStreaming(bot){const el=document.querySelector('#cmsg-stream .body');if(!el)return;el.innerHTML=chatBotInner(bot);chatDecorate();const dt=(performance.now()-bot.t0)/1000;const e=$('chatStatus');if(e){e.textContent='generating · '+bot.toks+' tok · '+(bot.toks/Math.max(dt,0.1)).toFixed(1)+' t/s';e.style.color='var(--mut)';}chatScroll(true);}
function chatDecorate(){document.querySelectorAll('#chatmsgs pre code').forEach(c=>{if(c.dataset.hl)return;c.dataset.hl=1;try{if(window.hljs)hljs.highlightElement(c);}catch(e){}});
  document.querySelectorAll('#chatmsgs pre').forEach(pre=>{if(pre.dataset.cc)return;pre.dataset.cc=1;const code=pre.querySelector('code');const txt=(code||pre).innerText;const b=document.createElement('button');b.className='codecopy';b.textContent='copy';b.onclick=()=>copy(txt,'Code copied');pre.appendChild(b);});}
function chatScroll(soft){const c=$('chatmsgs');if(!c)return;if(soft&&c.scrollHeight-c.scrollTop-c.clientHeight>140)return;c.scrollTop=c.scrollHeight;}
function chatSendOrStop(){if(CHAT.busy)chatStop();else chatSend();}
function chatStop(){if(CHAT.abort)try{CHAT.abort.abort();}catch(e){}}
async function chatSend(){
  const inp=$('chatInput');const text=(inp.value||'').trim();if(!text||CHAT.busy)return;
  CHAT.msgs.push({role:'user',content:text});inp.value='';chatAutogrow(inp);
  const bot={role:'assistant',raw:'',think:'',answer:'',streaming:true,toks:0,t0:performance.now()};
  CHAT.msgs.push(bot);CHAT.busy=true;const sb=$('chatSend');sb.disabled=false;sb.textContent='■';sb.classList.add('stop');
  chatRender();chatScroll();setText('chatStatus','generating…');$('chatStatus').style.color='var(--mut)';
  const payload={think:$('chatThink').checked,max_tokens:1024,
    messages:CHAT.msgs.slice(0,-1).map(m=>({role:m.role,content:m.role==='assistant'?(m.answer||''):m.content}))};
  CHAT.abort=new AbortController();
  try{
    const resp=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:CHAT.abort.signal});
    if(!resp.ok||!resp.body){let err='request failed';try{err=(await resp.json()).error||err;}catch(e){}bot.raw='⚠ '+err;}
    else{const reader=resp.body.getReader();const dec=new TextDecoder();let buf='',last=0;
      while(true){const {value,done}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});
        let idx;while((idx=buf.indexOf('\n\n'))>=0){const ln=buf.slice(0,idx).trim();buf=buf.slice(idx+2);
          if(!ln.startsWith('data:'))continue;try{const o=JSON.parse(ln.slice(5).trim());if(o.t){bot.raw+=o.t;bot.toks++;}if(o.error)bot.raw+='\n\n⚠ '+o.error;}catch(e){}}
        const now=performance.now();if(now-last>55){last=now;chatUpdateStreaming(bot);}}}
  }catch(e){if(e.name!=='AbortError')bot.raw+='\n\n⚠ '+e;}
  const sp=splitThink(bot.raw);bot.think=sp.think;bot.answer=sp.answer?sp.answer:(sp.think?'':bot.raw);
  if(!(bot.answer||'').trim()&&!(bot.think||'').trim())bot.answer='_(no output)_';
  bot.streaming=false;bot.dt=(performance.now()-bot.t0)/1000;
  CHAT.busy=false;CHAT.abort=null;sb.textContent='↑';sb.classList.remove('stop');sb.disabled=false;
  chatRender();chatScroll();chatStatus();inp.focus();
}

/* ---------- meta wiring (title/model/host — all auto-detected) ---------- */
function applyMeta(){
  setText('pageTitle',(META.title||'benchy')+(META.model?(' · '+META.model):''));
  if(META.subtitle)setText('pageSub',META.subtitle);
  setText('baselineTagTxt',META.baseline_tag||'baseline');
  const ct=$('chatTtl');if(ct)ct.innerHTML=(META.model?esc(META.model)+' · ':'')+'chat <span class="ep">'+esc((META.server||'')+'/v1')+'</span>';
}

/* ---------- guided setup ---------- */
let SETUP={refs:{}};
function openSetup(){
  $('setupModal').style.display='block';
  fetch('/api/config').then(r=>r.json()).then(cfg=>{
    SETUP.refs={};const cr=cfg.references||{};
    for(const b in cr)SETUP.refs[b]=(cr[b]||[]).map(r=>({label:r.label,acc:r.accuracy,kind:r.kind||'ref',source:r.source||''}));
    $('setTitle').value=cfg.title||'';$('setBaseline').value=cfg.baseline_tag||'';
    const host=META.host||{};
    const det=[['Model',META.model||'— (start the server)'],['Host',[host.cpu||host.machine,host.mem_total_gb?host.mem_total_gb+' GB RAM':'',host.os].filter(Boolean).join(' · ')||'—'],['Server',META.server||'—']];
    $('setupDetected').innerHTML=det.map(([k,v])=>`<span class="chip"><b>${esc(k)}</b> ${esc(v)}</span>`).join('');
    const benches=[...new Set([...(META.benchmarks||[]),...Object.keys(SETUP.refs)])].filter(Boolean);
    $('refBench').innerHTML=benches.length?benches.map(b=>`<option value="${esc(b)}">${esc(nice(b))}</option>`).join(''):'<option value="">— no benchmarks yet —</option>';
    renderRefList();
  }).catch(()=>toast('Could not load config',1));
}
function closeSetup(){$('setupModal').style.display='none';}
function renderRefList(){
  const benches=Object.keys(SETUP.refs).filter(b=>(SETUP.refs[b]||[]).length);
  if(!benches.length){$('refList').innerHTML='<div class="muted" style="font-size:12px">No reference baselines yet — add one above, or leave empty to just compare your own runs.</div>';return;}
  $('refList').innerHTML=benches.map(b=>`<div class="refbench">${esc(nice(b))}</div>`+SETUP.refs[b].map((r,i)=>`<div class="refrow"><span class="tag ${r.kind==='frontier'?'pur':(r.kind==='chance'?'':'blue')}">${esc(r.kind)}</span><b>${esc(r.label)}</b><span class="mono muted">${r.acc}%</span><button class="rm" title="remove" onclick="setupDelRef('${esc(b)}',${i})">✕</button></div>`).join('')).join('');
}
function setupAddRef(){
  const b=$('refBench').value,lbl=($('refLabel').value||'').trim(),acc=parseFloat($('refAcc').value),kind=$('refKind').value;
  if(!b){toast('Pick a benchmark first',1);return;}
  if(!lbl){toast('Add a label',1);return;}
  if(isNaN(acc)||acc<0||acc>100){toast('Score must be 0–100',1);return;}
  (SETUP.refs[b]=SETUP.refs[b]||[]).push({label:lbl,acc:acc,kind:kind});
  $('refLabel').value='';$('refAcc').value='';renderRefList();
}
function setupDelRef(b,i){SETUP.refs[b].splice(i,1);if(!SETUP.refs[b].length)delete SETUP.refs[b];renderRefList();}
async function setupSave(){
  const references={};
  for(const b in SETUP.refs)references[b]=(SETUP.refs[b]||[]).map(r=>({label:r.label,accuracy:r.acc,kind:r.kind,source:r.source||''}));
  const body={title:$('setTitle').value||'',baseline_tag:$('setBaseline').value||'',references:references};
  try{const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
    if(r.ok){toast('Settings saved ✓');closeSetup();loadAll();}else toast('Save failed',1);
  }catch(e){toast('Save failed',1);}
}

/* ---------- benchmark browser ---------- */
let BENCH={data:null,timer:null};
function openBenchmarks(){$('benchModal').style.display='block';loadBench();}
function closeBenchmarks(){$('benchModal').style.display='none';if(BENCH.timer){clearInterval(BENCH.timer);BENCH.timer=null;}}
async function loadBench(){
  try{BENCH.data=await fetch('/api/benchmarks').then(r=>r.json());}catch(e){$('benchList').innerHTML='<div class="muted">could not load registry</div>';return;}
  renderBench();
  if(BENCH.data.fetching&&!BENCH.timer)BENCH.timer=setInterval(loadBench,2500);
  if(!BENCH.data.fetching&&BENCH.timer){clearInterval(BENCH.timer);BENCH.timer=null;loadAll();}
}
function benchItem(b,fetching){
  const fit=b.fit==='code'?'<span class="tag" title="generates code, executed against tests (pass@1)">⚙ code</span>':'';
  const base=b.baselines?`<span class="tag pur" title="${b.baselines} published frontier baselines ship for this benchmark">★ baselines</span>`:'';
  return `<div class="benchrow"><div style="min-width:0;flex:1"><div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap"><span class="bn">${esc(b.name)}</span><span class="dom">${esc(b.domain)}</span>${fit}${base}</div>
    <div class="muted" style="font-size:11.5px;margin-top:2px;line-height:1.45">${esc(b.desc||'')}</div></div>
    <span class="br">${b.present?'<span class="tag ok">✓ installed</span>':`<button class="btn g" ${fetching?'disabled':''} onclick="fetchBench('${esc(b.key)}')">fetch</button>`}</span></div>`;
}
function renderBench(){
  const d=BENCH.data||{},av=d.available||[],fetching=d.fetching;
  const cur=av.filter(b=>b.tier==='current'),leg=av.filter(b=>b.tier!=='current');
  const curMissing=cur.filter(b=>!b.present).length;
  $('benchStatus').innerHTML=fetching?'<span class="livedot"></span> fetching… (files appear as they finish)':(av.filter(b=>b.present).length+' of '+av.length+' installed · '+curMissing+' current not yet fetched');
  $('benchAll').disabled=fetching||!curMissing;$('benchAll').textContent=curMissing?('⬇ Fetch current set ('+curMissing+')'):'Current set installed';
  const sec=(label,arr)=>arr.length?`<div class="refbench" style="margin-top:12px">${label}</div>`+arr.map(b=>benchItem(b,fetching)).join(''):'';
  $('benchList').innerHTML=sec('Current — discriminating for mid-2026 models',cur)+sec('Legacy / saturated — small-model regression only',leg);
  const man=d.manual||[];
  $('benchManual').innerHTML=man.length?('<div class="refbench">Manual / gated — see DATA.md</div>'+man.map(m=>`<div class="benchrow" style="opacity:.75"><div style="flex:1;min-width:0"><span class="bn mono" style="font-size:12px">${esc(m.key)}</span><div class="muted" style="font-size:11.5px;margin-top:2px;line-height:1.45">${esc(m.desc||m.note||'')}</div></div></div>`).join('')):'';
}
async function fetchBench(key){
  let keys;
  if(key==='__current__')keys=(BENCH.data.available||[]).filter(b=>b.tier==='current'&&!b.present).map(b=>b.key);
  else if(key==='__missing__')keys=(BENCH.data.available||[]).filter(b=>!b.present).map(b=>b.key);
  else keys=[key];
  if(!keys.length){toast('Nothing to fetch');return;}
  try{const r=await fetch('/api/benchmarks/fetch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys})}).then(r=>r.json());
    if(r.ok){toast('Fetching '+keys.length+' benchmark'+(keys.length>1?'s':'')+'…');loadBench();}
    else toast(r.error||'fetch failed',1);
  }catch(e){toast('fetch failed',1);}
}

/* ---------- spotlight ---------- */
document.addEventListener('mousemove',e=>{const c=e.target.closest&&e.target.closest('.card');if(!c)return;const r=c.getBoundingClientRect();c.style.setProperty('--mx',(e.clientX-r.left)+'px');c.style.setProperty('--my',(e.clientY-r.top)+'px');});

/* ---------- polling ---------- */
async function pulse(){try{
  [LIVE,STREAM,ACT,HISTM]=await Promise.all([fetch('/api/live').then(r=>r.json()),fetch('/api/stream').then(r=>r.json()),fetch('/api/activity').then(r=>r.json()),fetch('/api/history').then(r=>r.json())]);
  streamUI();activityUI();systemUI();experimentBand();syncBench();
  const s=await fetch('/api/server/status').then(r=>r.json());
  const up=s.up;$('srvPill').textContent=up?'online':'offline';$('srvPill').className='tag '+(up?'ok':'dang');
  $('srvBtn').style.display=up?'none':'inline-block';
  if($('chatwrap').classList.contains('open'))chatSetStatus(up);
  try{SYS=await fetch('/api/sys').then(r=>r.json());}catch(e){}
  if(DBG.open)dbgRender();
}catch(e){}}
// shared option builder: group keys by tier (current → legacy → third), registry order within each
function tieredOptions(keys,cur,thirdLabel){
  const g={current:[],legacy:[],other:[]};
  keys.forEach(b=>{const t=regTier(b);g[(t==='current')?'current':(t==='legacy'?'legacy':'other')].push(b);});
  const ord=b=>(REGMAP[b]&&REGMAP[b].ord!=null)?REGMAP[b].ord:999;
  const og=(lab,arr)=>{if(!arr.length)return '';arr.sort((a,b)=>ord(a)-ord(b));
    return '<optgroup label="'+lab+'">'+arr.map(b=>'<option value="'+esc(b)+'"'+(b===cur?' selected':'')+'>'+esc(regName(b))+((REGMAP[b]&&REGMAP[b].fit==='code')?' ⚙':'')+'</option>').join('')+'</optgroup>';};
  return og('current (recommended)',g.current)+og('legacy / saturated',g.legacy)+og(thirdLabel||'other',g.other);
}
function fillRunMenu(){
  const rb=$('rBench');if(!rb)return;const cur=rb.value,opts=(META.benchmarks||[]),sig=opts.join(',');
  if(rb.dataset.sig===sig){if(cur&&opts.includes(cur))rb.value=cur;return;}
  rb.dataset.sig=sig;
  rb.innerHTML=opts.length?tieredOptions(opts,cur,'other / custom'):'<option value="">no benchmarks — fetch from ⬇ Benchmarks</option>';
  if(cur&&opts.includes(cur))rb.value=cur;updateRunDesc();
}
function updateRunDesc(){const d=$('runDesc'),b=($('rBench')&&$('rBench').value);if(d)d.textContent=b?regDesc(b):'';}
function syncAccToRun(){const rv=($('rBench')&&$('rBench').value),ab=$('accBench');
  if(rv&&ab&&[...ab.options].some(o=>o.value===rv)){ab.value=rv;accBenchUser=true;accbars();}}
async function loadAll(){try{
  let BENCH;
  [RUNS,PERF,META,SUMMARY,BENCH]=await Promise.all([fetch('/api/runs').then(r=>r.json()),fetch('/api/perf').then(r=>r.json()),fetch('/api/meta').then(r=>r.json()),fetch('/api/summary').then(r=>r.json()),fetch('/api/benchmarks').then(r=>r.json())]);
  REGMAP={};(BENCH.available||[]).concat(BENCH.manual||[]).forEach((b,i)=>{REGMAP[b.key]={name:b.name||NICE[b.key]||b.key,tier:b.tier||'manual',fit:b.fit||'',desc:b.desc||'',domain:b.domain||'',ord:i};});
  applyMeta();
  fillRunMenu();
  const benches=[...new Set([...RUNS.map(r=>r.benchmark),...Object.keys(META.references||{}),...(META.benchmarks||[])])].filter(Boolean);
  const ab=$('accBench'),cur=ab.value,asig=benches.join(',');if(ab.dataset.sig!==asig){ab.innerHTML=tieredOptions(benches,cur,'reference / gated');if(cur)ab.value=cur;ab.dataset.sig=asig;}
  envStrip();experimentBand();syncBench();
  const sig=RUNS.length+'|'+PERF.length+'|'+(RUNS.length?RUNS[RUNS.length-1].ts:'')+'|'+JSON.stringify(SUMMARY.macro||{});
  if(sig!==_dataSig){_dataSig=sig;accbars();renderRuns();populateCompare();}
}catch(e){console.error(e)}}
loadAll();setInterval(loadAll,6000);pulse();setInterval(pulse,1500);
</script></body></html>"""

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
    def _guard(self):
        """Reject cross-site / DNS-rebound POSTs: the Host must be a loopback:port we serve
        (defeats DNS rebinding), any Origin/Referer present must be same-origin, and the
        per-launch CSRF token must match (only our own same-origin page can read it)."""
        allowed = {"127.0.0.1:%d" % DASH_PORT, "localhost:%d" % DASH_PORT}
        host = (self.headers.get("Host") or "").strip()
        if host and host not in allowed:
            return False
        origin = self.headers.get("Origin") or self.headers.get("Referer") or ""
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost"):
            return False
        return self.headers.get("X-Benchy-CSRF", "") == CSRF_TOKEN
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/" or p.startswith("/index"): self._send(200, PAGE.replace("__BENCHY_CSRF__", CSRF_TOKEN))
        elif p == "/api/runs": self._json(read_jsonl(RUNS))
        elif p == "/api/perf": self._json(read_jsonl(PERF))
        elif p == "/api/meta": self._json(dict(meta_payload(), benchmarks=benchmarks()))
        elif p == "/api/config": self._json(load_config())
        elif p == "/api/benchmarks":
            fetching = bool(PROC.get("fetch") and PROC["fetch"].poll() is None)
            meta = FB.registry_meta() if FB else {"available": [], "manual": []}
            for b in meta.get("available", []):
                b["baselines"] = len(SHIPPED_REFS.get(b["key"], []))   # # of shipped frontier baselines
            self._json(dict(meta, fetching=fetching))
        elif p == "/api/summary": self._json(summary())
        elif p == "/api/compare":
            q = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
            g = lambda k: (q.get(k) or [""])[0]
            self._json(paired_compare(g("a"), g("b"), g("bench"), g("mode") or None))
        elif p == "/api/sys": self._json(sysmetrics())
        elif p == "/api/activity": self._json(activity())
        elif p == "/api/history": self._json(list(HIST))
        elif p == "/api/live_details":
            try:
                dd = os.path.join(RESULTS, "details")
                fs = [f for f in os.listdir(dd) if f.endswith(".jsonl")] if os.path.isdir(dd) else []
                fs.sort(key=lambda f: os.path.getmtime(os.path.join(dd, f)))
                fn = fs[-1] if fs else None
                rows = [json.loads(l) for l in open(os.path.join(dd, fn)) if l.strip()] if fn else []
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
            self._json([json.loads(l) for l in open(fp) if l.strip()] if fn and os.path.exists(fp) else [])
        elif p == "/api/live": self._json(read_live())
        elif p == "/api/stream": self._json(read_stream())
        elif p == "/api/server/status": self._json({"up": server_up()})
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
    threading.Thread(target=_sample_metrics, daemon=True).start()
    print("benchy -> http://127.0.0.1:%d" % port, flush=True)
    if ALLOW_CODE_EXEC:
        print("⚠ BENCHY_ALLOW_CODE_EXEC is set — code-generation benchmarks will EXECUTE model-written code on this host.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
