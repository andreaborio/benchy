#!/usr/bin/env python3
"""BeepMed Benchmark Explorer — local control center for ds4 medical evals.

  python3 dashboard.py [port]      # default 8050  ->  http://127.0.0.1:8050

Stdlib backend (no pip deps), polished SPA (Chart.js via CDN). Lets you:
 - watch a live Q/A stream + real-time accuracy chart,
 - watch the model's real-time decode activity (tokens/s, phase, throughput),
 - launch the model server and an eval run from the UI,
 - explore accuracy + performance benchmarks and export a Markdown report.
"""
import json, os, sys, re, math, datetime, subprocess, threading, time, collections
import urllib.request as ureq
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(RESULTS, "runs.jsonl")
PERF = os.path.join(RESULTS, "perf.jsonl")
LIVE = os.path.join(RESULTS, "live.json")
STREAM = os.path.join(RESULTS, "stream.jsonl")
DS4 = os.environ.get("DS4_DIR", os.path.expanduser("~/ds4"))  # ds4 checkout (for the start-server button); override with DS4_DIR
CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"
PROC = {"eval": None, "server": None}
HISTORY = os.path.join(RESULTS, "metrics.jsonl")
HIST = collections.deque(maxlen=900)   # ~30 min @ 2s — survives browser refresh; persisted to metrics.jsonl

META = {
    "env": {
        "Engine": "ds4 fork feat/onedge-imatrix @9afb525",
        "Model": "DeepSeek-V4-Flash-IQ2 · medical imatrix (iq2-medimatrix) · 81 GB",
        "Quant": "asymmetric 2-bit: experts IQ2_XXS/Q2_K, rest Q8",
        "Class": "284B total / 13B active · 43 layers · 256 experts/layer",
        "Machine": "Apple M5 Pro · 64 GB · macOS 26.5.1",
        "Backend": "Metal-4 tensor API (M5 neural accelerators)",
    },
    "ram": {"footprint_gb": 44, "rss_gb": 40, "cache_gb": 40, "experts": 6068, "model_disk_gb": 81},
    "correctness": {"test": "ds4_test --streaming-decode-prefill-correctness", "result": "PASS",
                    "runs": 3, "cases": 5,
                    "detail": "bit-identical (cold_warm_max_abs=0, warm_repeat_max_abs=0, top1==canonical)"},
    "references": {
        "medqa_test": [
            {"label": "GPT-5", "accuracy": 95.8, "kind": "frontier", "source": "frontier report 2026"},
            {"label": "o3 (≈ saturated)", "accuracy": 96.0, "kind": "frontier", "source": "approx (MedQA saturated ~96%)"},
            {"label": "Gemini-2.5-Pro", "accuracy": 92.6, "kind": "frontier", "source": "comparative eval 2025-26"},
            {"label": "DeepSeek-R1 (671B)", "accuracy": 90.0, "kind": "frontier", "source": "comparative eval 2025-26 (88-94 by setup)"},
            {"label": "GPT-4o", "accuracy": 89.2, "kind": "frontier", "source": "comparative eval 2025-26"},
            {"label": "MedGemma-27B (TTS)", "accuracy": 87.7, "kind": "ref", "source": "MedGemma report arXiv:2507.05201"},
            {"label": "MedGemma-4B", "accuracy": 64.4, "kind": "ref", "source": "MedGemma report"},
            {"label": "random chance", "accuracy": 25.0, "kind": "chance", "source": "chance"},
        ],
        "medmcqa": [
            {"label": "Gemini-2.5-Pro", "accuracy": 82.7, "kind": "frontier", "source": "comparative eval 2025"},
            {"label": "DeepSeek-R1", "accuracy": 79.9, "kind": "frontier", "source": "comparative eval 2025"},
            {"label": "GPT-4o", "accuracy": 76.9, "kind": "frontier", "source": "comparative eval 2025"},
            {"label": "MedGemma-27B (TTS)", "accuracy": 74.2, "kind": "ref", "source": "MedGemma report"},
            {"label": "MedGemma-4B", "accuracy": 55.7, "kind": "ref", "source": "MedGemma report"},
            {"label": "random chance", "accuracy": 25.0, "kind": "chance", "source": "chance"},
        ],
        "pubmedqa": [
            {"label": "MedGemma-27B (TTS)", "accuracy": 76.8, "kind": "ref", "source": "MedGemma report"},
            {"label": "Gemini-2.5-Pro", "accuracy": 76.4, "kind": "frontier", "source": "comparative eval 2025"},
            {"label": "MedGemma-4B", "accuracy": 73.4, "kind": "ref", "source": "MedGemma report"},
            {"label": "DeepSeek-R1", "accuracy": 73.2, "kind": "frontier", "source": "comparative eval 2025"},
            {"label": "GPT-4o", "accuracy": 71.8, "kind": "frontier", "source": "comparative eval 2025"},
            {"label": "random chance (3-opt)", "accuracy": 33.3, "kind": "chance", "source": "chance"},
        ],
        "mmlu_medical": [
            {"label": "MedGemma-27B (TTS) (MMLU-Med ~avg)", "accuracy": 87.0, "kind": "ref", "source": "MedGemma report (approx avg)"},
            {"label": "MedGemma-4B (MMLU-Med ~avg)", "accuracy": 66.0, "kind": "ref", "source": "MedGemma report (approx avg)"},
            {"label": "random chance", "accuracy": 25.0, "kind": "chance", "source": "chance"},
        ],
        "medxpertqa": [
            {"label": "o1", "accuracy": 44.7, "kind": "frontier", "source": "MedXpertQA leaderboard (Text avg)"},
            {"label": "DeepSeek-R1 (same family)", "accuracy": 37.8, "kind": "frontier", "source": "MedXpertQA leaderboard"},
            {"label": "o3-mini", "accuracy": 37.3, "kind": "frontier", "source": "MedXpertQA leaderboard"},
            {"label": "Claude-3.5-Sonnet", "accuracy": 21.3, "kind": "frontier", "source": "MedXpertQA leaderboard"},
            {"label": "Gemini-2.0-Flash", "accuracy": 20.6, "kind": "frontier", "source": "MedXpertQA leaderboard"},
            {"label": "random chance (10-opt)", "accuracy": 10.0, "kind": "chance", "source": "chance"},
        ],
        "healthbench_hard": [
            {"label": "GPT-5 (thinking)", "accuracy": 46.2, "kind": "frontier", "source": "OpenAI GPT-5 System Card"},
            {"label": "GPT-5-mini (thinking)", "accuracy": 40.3, "kind": "frontier", "source": "OpenAI GPT-5 System Card"},
            {"label": "o3", "accuracy": 31.6, "kind": "frontier", "source": "OpenAI HealthBench paper + System Card"},
            {"label": "GPT-5 (main, non-thinking)", "accuracy": 25.5, "kind": "frontier", "source": "OpenAI GPT-5 System Card"},
            {"label": "GPT-4o", "accuracy": 0.0, "kind": "frontier", "source": "OpenAI GPT-5 System Card (0 on Hard)"},
        ],
    },
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
    return sorted(f[:-6] for f in os.listdir(DATA) if f.endswith(".jsonl"))

def server_up():
    try:
        with ureq.urlopen("http://127.0.0.1:8000/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def start_server():
    if server_up(): return {"ok": True, "already": True}
    if not os.path.exists(os.path.join(DS4, "ds4-server")):
        return {"ok": False, "error": "ds4-server not found in " + DS4 + " — set the DS4_DIR env var to your ds4 checkout"}
    args = ["./ds4-server", "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB",
            "--ctx", "8192", "--port", "8000"]
    log = open(os.path.join(RESULTS, "server.log"), "a")
    PROC["server"] = subprocess.Popen(args, cwd=DS4, stdout=log, stderr=log, start_new_session=True)
    return {"ok": True, "starting": True}

def start_eval(p):
    if read_live().get("running"): return {"ok": False, "error": "a run is already in progress"}
    if not server_up(): return {"ok": False, "error": "model server not reachable on :8000 — start it first"}
    bench = re.sub(r"[^A-Za-z0-9_.-]", "", str(p.get("benchmark", "medqa_test")))
    path = os.path.join(DATA, bench + ".jsonl")
    if not os.path.exists(path): return {"ok": False, "error": "benchmark not found: " + bench}
    try: n = max(1, int(p.get("n", 25)))
    except Exception: n = 25
    mode = "think" if p.get("mode") == "thinking" else "nothink"
    tag = (re.sub(r"[^A-Za-z0-9_.-]", "", str(p.get("tag", "iq2"))) or "iq2")[:40]
    if bench == "healthbench_hard":
        # rubric-graded, not MCQ — needs the HealthBench runner (OpenAI grader via .apikey)
        args = [sys.executable, os.path.join(HERE, "healthbench.py"), str(n), tag, mode, "hard"]
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
    try:
        subprocess.run(["pkill", "-9", "-f", "ds4-server"], capture_output=True)
        return {"ok": True, "killed": "ds4-server"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

OPTS = {"medqa_test": 4, "medmcqa": 4, "mmlu_medical": 4, "pubmedqa": 3, "medxpertqa": 10}
PRIMARY_REF = "MedGemma-27B (TTS)"   # closest apples-to-apples external baseline

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

def summary():
    """Server-side stats over runs.jsonl x references: Wilson CIs, gaps, macro-avg, bias."""
    runs = read_jsonl(RUNS); refs_all = META["references"]
    benches = sorted({r.get("benchmark") for r in runs if r.get("benchmark")} | set(refs_all.keys()))
    per_run, bias = [], []
    for r in runs:
        b = r.get("benchmark"); n = r.get("n") or 0; acc = r.get("accuracy")
        if acc is None or not b: continue
        if b == "healthbench_hard":
            lo = hi = None   # rubric mean, not k/n successes — a binomial Wilson CI is invalid here
        else:
            k = r.get("correct");  k = round(acc/100.0*n) if k is None else k
            lo, hi = wilson(k, n)
        per_run.append({"ts": r.get("ts"), "benchmark": b, "mode": r.get("mode"), "tag": r.get("tag"),
                        "n": n, "accuracy": acc, "ci_lo": lo, "ci_hi": hi, "small_n": n < 50})
        ld = r.get("letter_dist")
        if ld:
            chi2, dof, tot = _chi2_uniform(ld, OPTS.get(b, 4))
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
        primary = next((x for x in refs if x["label"] == PRIMARY_REF), None)
        best_ref = max(refs, key=lambda x: x["accuracy"]) if refs else None
        cell = {"thinking": th, "nothink": no, "best": best_ours,
                "refs": refs, "primary_ref": primary, "best_ref": best_ref}
        if best_ours and primary:
            cell["gap_primary"] = round(best_ours["accuracy"] - primary["accuracy"], 1)
            if best_ours["ci_lo"] is not None and best_ours["ci_hi"] is not None:
                cell["ref_in_ci"] = best_ours["ci_lo"] <= primary["accuracy"] <= best_ours["ci_hi"]
        if th and no: cell["think_delta"] = round(th["accuracy"] - no["accuracy"], 1)
        by[b] = cell
    # macro-avg: exclude healthbench_hard (rubric score, not % accuracy) and use ONE consistent
    # build tag per mode (the widest-coverage tag) so different builds are never blended.
    def macro_for(mode):
        cov = collections.Counter(x["tag"] for x in per_run
                                  if x["mode"] == mode and x["benchmark"] != "healthbench_hard")
        if not cov: return None, 0, None
        tag = cov.most_common(1)[0][0]
        accs = []
        for b in benches:
            if b == "healthbench_hard": continue
            cands = [x for x in per_run if x["benchmark"] == b and x["mode"] == mode and x["tag"] == tag]
            if cands: accs.append(max(cands, key=lambda x: x["n"])["accuracy"])
        return (round(sum(accs)/len(accs), 1) if accs else None, len(accs), tag)
    tm_mean, tm_k, tm_tag = macro_for("thinking")
    nm_mean, nm_k, nm_tag = macro_for("nothink")
    macro = {"thinking_mean": tm_mean, "thinking_k": tm_k, "thinking_tag": tm_tag,
             "nothink_mean": nm_mean, "nothink_k": nm_k, "nothink_tag": nm_tag}
    return {"benchmarks": benches, "per_run": per_run, "by_benchmark": by,
            "macro": macro, "bias": bias, "primary_ref": PRIMARY_REF, "opts": OPTS}

def sysmetrics():
    """Live system + model-server metrics: server RSS, CPU, system memory, decode t/s."""
    m = {"server_up": server_up()}
    pid = None
    try:
        o = subprocess.run(["pgrep", "-f", "ds4-server"], capture_output=True, text=True, timeout=3).stdout.split()
        pid = int(o[0]) if o else None
    except Exception: pass
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
    fn = {"server": "server.log", "eval": "eval.log"}.get(which, "server.log")
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
    runs = read_jsonl(RUNS); perf = read_jsonl(PERF)
    L = ["# BeepMed Benchmark Report", "", f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')}_", "",
         "## Environment", ""]
    for k, v in META["env"].items(): L.append(f"- **{k}**: {v}")
    L += ["", "## Accuracy — MedQA-USMLE", "", "| date | tag | mode | N | accuracy | s/q | notes |",
          "|---|---|---|--:|--:|--:|---|"]
    for r in runs:
        L.append(f"| {r.get('ts','')} | {r.get('tag','')} | {r.get('mode','')} | {r.get('n','')} "
                 f"| **{r.get('accuracy','')}%** | {r.get('sec_per_q','')} | {r.get('notes','') or ''} |")
    L += ["", "### External reference baselines (per benchmark; different size/class)", ""]
    for bench, refs in META["references"].items():
        L += [f"**{bench}**", "", "| model | accuracy | source |", "|---|--:|---|"]
        for ref in refs: L.append(f"| {ref['label']} | {ref['accuracy']}% | {ref['source']} |")
        L.append("")
    L += ["", "## Inference performance", "",
          "| kind | config | prefetch | prefill t/s | gen t/s | hit-rate | notes |", "|---|---|---|--:|--:|--:|---|"]
    for p in perf:
        hr = p.get('hit_rate'); hr = f"{hr:.3f}" if isinstance(hr, (int, float)) else "—"
        L.append(f"| {p.get('kind','')} | {p.get('config','')} | {p.get('prefetch','')} | {p.get('prefill_tps','')} "
                 f"| **{p.get('gen_tps','')}** | {hr} | {p.get('notes','') or ''} |")
    c = META["correctness"]
    L += ["", "## Correctness", "", f"- {c['test']} → **{c['result']}** ({c['runs']} runs, {c['cases']} cases): {c['detail']}",
          "", "## RAM", "", f"- 81 GB model runs in **~{META['ram']['footprint_gb']} GB RAM** "
          f"({META['ram']['cache_gb']} GB cache = {META['ram']['experts']} experts; rest streamed)",
          "", "_Caveats: small-N runs have wide CIs; non-thinking has letter bias; t/s ±~10%. See README.md._"]
    return "\n".join(L)

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>BeepMed · Benchmark Explorer</title>
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
#detModal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.66);backdrop-filter:blur(4px);z-index:100;padding:40px 20px;overflow:auto}
#detModal .box{max-width:960px;margin:0 auto;background:#0d0d0f;border:1px solid #262629;border-radius:16px;padding:20px 22px}
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
.sysbig{font-size:19px;font-weight:800;letter-spacing:-.02em}
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
  <h1>BeepMed · DeepSeek-V4-Flash IQ2</h1>
  <p class="sub">Medical-eval lab — each <b>experiment</b> (a quant / imatrix build) is run across medical benchmarks on one local rig.</p>
</div>
<div style="display:flex;gap:8px;flex-wrap:wrap"><button class="btn g" onclick="loadAll();pulse()">↻ Refresh</button><button class="btn g" onclick="copyReport(this)">⧉ Copy report</button><button class="btn" onclick="location.href='/api/report'">⬇ Export</button></div></div>
<div class="envstrip" id="envstrip"></div>

<!-- EXPERIMENT -->
<div class="sec" id="experiment" style="margin-top:18px"><div class="card" id="expCard"></div></div>

<!-- CONTROL -->
<div class="sec" id="control"><div class="grid g2">
  <div class="card"><h3>Model server (:8000)</h3>
    <div class="toolbar" style="margin:11px 0 0"><span class="tag" id="srvPill">checking…</span>
      <button class="btn g" id="srvBtn" onclick="startServer()">Start server</button>
      <button class="btn dang" onclick="stopAll()">⛔ Stop all tests</button>
      <button class="btn g" onclick="killServer()" title="force-kill the model server">kill server</button>
      <span class="muted" id="srvMsg" style="font-size:12px"></span></div>
    <p class="muted" style="font-size:12px;margin:11px 0 0;line-height:1.55">Active build: <b style="color:var(--pur2)">iq2-medimatrix</b> (medical-recalibrated imatrix). Tag your runs so they compare against <code class="mono">iq2-baseline</code> in Accuracy — specs in the strip up top.</p></div>
  <div class="card"><h3>Run a test</h3>
    <div class="toolbar" style="margin:11px 0 0">
      <select id="rBench" onchange="rBenchUser=true"></select>
      <input id="rN" type="number" value="25" min="1" style="width:78px" title="N questions">
      <div class="seg"><button id="segT" class="on" onclick="setMode('thinking')">🧠 thinking</button><button id="segN" onclick="setMode('nothink')">⚡ nothink</button></div>
      <input id="rTag" type="text" value="iq2-medimatrix" style="width:150px" placeholder="version tag">
      <button class="btn" id="runBtn" onclick="runTest()">▶ Run</button>
      <button class="btn g" id="stopBtn" onclick="stopTest()" style="display:none">■ Stop</button>
      <span class="muted" id="runMsg" style="font-size:12px"></span></div></div>
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
    <div class="card"><div class="ch"><h3>Decode throughput</h3><span class="sysbig" id="sysTps" style="color:var(--warn)">—</span></div><canvas id="sysTpsChart" height="116"></canvas></div>
    <div class="card"><div class="ch"><h3>Memory <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">· server RSS / system</span></h3><span class="sysbig" id="sysRam" style="color:var(--acc2)">—</span></div><canvas id="sysRamChart" height="116"></canvas></div>
    <div class="card"><div class="ch"><h3>Server CPU</h3><span class="sysbig" id="sysCpu" style="color:var(--ok)">—</span></div><canvas id="sysCpuChart" height="116"></canvas></div>
  </div></div>

<!-- ACCURACY -->
<div class="sec" id="accuracy"><h2>Medical accuracy <select id="accBench" onchange="accBenchUser=true;accbars()" style="margin-left:6px;font-size:13px"></select><span class="hint" id="macroHint"></span></h2>
  <div class="grid g2">
    <div class="card"><h3>Our runs vs reference baselines <span class="muted" style="text-transform:none;letter-spacing:0;font-weight:500">· 95% Wilson CI shown on our bars</span></h3><div id="accbars" style="margin-top:12px"></div>
      <p class="muted" style="font-size:12px;margin:11px 0 0;line-height:1.5">⚠ Frontier on MedQA is <b style="color:var(--warn)">saturated (~95–96% in 2026)</b> — newest models report HealthBench / Medmarks, not MCQ. Our thinking run already sits in this top band; small-N CIs are wide, read them.</p></div>
    <div class="card"><h3>Thinking vs non-thinking</h3><canvas id="modeChart"></canvas></div>
  </div></div>

<!-- PERFORMANCE -->
<div class="sec" id="performance"><h2>Inference performance</h2>
  <div class="callout" style="margin-bottom:14px">Decode on 64 GB is <b>bandwidth-bound</b>: speed scales with the <b>cache hit-rate</b> — how much of the model is resident in RAM. The curve below maps footprint → speed (the model's real operational size).</div>
  <div class="card"><h3>Cache size → hit-rate &amp; decode speed <span class="muted" style="font-weight:500;font-size:12px;text-transform:none;letter-spacing:0">· operational size</span></h3><canvas id="hrChart"></canvas></div>
  <div class="card" style="margin-top:14px"><h3>All performance runs</h3>
    <table><thead><tr><th>kind</th><th>config</th><th class="n">prefill t/s</th><th class="n">gen t/s</th><th class="n">hit-rate</th><th>notes</th></tr></thead><tbody id="perfBody"></tbody></table></div></div>

<!-- RUNS -->
<div class="sec" id="runs"><h2>Runs explorer <span class="hint">click an accuracy run to inspect every question</span></h2>
  <div class="toolbar" style="margin-bottom:12px"><div class="seg"><button id="vAcc" class="on" onclick="setView('acc')">Accuracy</button><button id="vPerf" onclick="setView('perf')">Performance</button></div>
    <input type="search" id="q" placeholder="filter…" oninput="renderRuns()"><span class="muted" id="runcount" style="margin-left:auto"></span></div>
  <div class="card" style="padding:0;overflow:hidden"><table><thead id="runHead"></thead><tbody id="runBody"></tbody></table></div></div>

<div class="foot">Live stream, accuracy &amp; LLM activity update ~1.5s · tables 6s · live metrics &amp; benchmark numbers come from <span class="mono">results/</span> &amp; <span class="mono">server.log</span>; the Environment panel shows static spec/reference values, not live readings.<br>Methodology &amp; caveats in README.md — small-N Wilson CIs, page-cache/thermal noise, non-thinking letter bias, t/s ±~10%. HealthBench is a rubric score (not % correct), small-N, partly multilingual.</div>
</div>

<div id="detModal" onclick="if(event.target===this)closeDetails()"><div class="box">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:10px"><h2 id="detTitle" style="margin:0">Run detail</h2>
    <div style="display:flex;gap:8px"><button class="iconbtn" id="detCopyWrong" onclick="copyWrong()">⧉ copy wrong as JSON</button><button class="btn g" onclick="closeDetails()">✕ Close</button></div></div>
  <div id="detSummary" class="muted" style="font-size:13px;margin-top:4px"></div>
  <div class="dtoolbar"><div class="seg sm"><button id="dfAll" class="on" onclick="setDetFilter('all')">all</button><button id="dfOk" onclick="setDetFilter('ok')">✓ correct</button><button id="dfNo" onclick="setDetFilter('no')">✗ wrong</button></div>
    <input type="search" id="detQ" placeholder="search questions…" oninput="renderDet()" style="flex:1;min-width:160px"></div>
  <div id="detBody"></div>
</div></div>

<button class="chatbtn" onclick="chatToggle()"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>Chat</button>
<div class="chatwrap" id="chatwrap"><div class="chat" id="chatpanel">
  <div class="chathead"><div class="ttl">ds4 · chat <span class="ep">127.0.0.1:8000/v1</span></div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:8px">
      <label class="ccontrols" title="chain-of-thought reasoning"><input type="checkbox" id="chatThink" style="width:auto"> 🧠 thinking</label>
      <button class="iconbtn" onclick="chatClear()" title="clear conversation">⌫ clear</button>
      <button class="iconbtn" onclick="chatFull()" title="expand / restore">⤢</button>
      <button class="iconbtn" onclick="chatToggle()">✕</button></div></div>
  <div class="chatmsgs" id="chatmsgs"></div>
  <div class="chatcompose">
    <div class="ccontrols"><span id="chatStatus">checking server…</span><span style="margin-left:auto" class="muted">Enter to send · Shift+Enter newline</span></div>
    <div class="cinput"><textarea id="chatInput" rows="1" placeholder="Ask the model a medical question…" oninput="chatAutogrow(this)" onkeydown="chatKey(event)"></textarea><button class="csend" id="chatSend" onclick="chatSendOrStop()">↑</button></div></div>
</div></div>

<button class="dbgbtn" onclick="toggleDbg()">&lt;/&gt; debug</button>
<div class="dbg" id="dbg"><div class="dbgh"><b style="font-size:13px">Debug</b>
  <div class="seg sm" style="margin-left:6px"><button class="on" data-d="timeline" onclick="dbgTab('timeline')">timeline</button><button data-d="snap" onclick="dbgTab('snap')">snapshot</button><button data-d="server" onclick="dbgTab('server')">server.log</button><button data-d="eval" onclick="dbgTab('eval')">eval.log</button></div>
  <button class="iconbtn" style="margin-left:auto" onclick="dbgCopy()">⧉ copy</button><button class="iconbtn" onclick="toggleDbg()">✕</button></div>
  <pre id="dbgBody" class="mono">…</pre></div>

<div id="toasts"></div>
<script>
let RUNS=[],PERF=[],META={},SUMMARY={},LIVE={},STREAM={},SYS={},ACT={},HISTM=[];
let VIEW='acc',SORT={k:'ts',d:-1},RMODE='thinking',FEEDF='all',DETF='all',DET=[],DETFILE='',charts={},_sid=0,_dataSig='',_streamSig='',_sparkSig='',_expSig='',accBenchUser=false,rBenchUser=false;
function setText(id,t){const e=$(id);if(e&&e.textContent!==t)e.textContent=t;}
let DBG={open:false,tab:'timeline'};
const NICE={medqa_test:'MedQA-USMLE',medmcqa:'MedMCQA',mmlu_medical:'MMLU Medical',pubmedqa:'PubMedQA',medxpertqa:'MedXpertQA',healthbench_hard:'HealthBench Hard'},nice=b=>NICE[b]||b;
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const num=v=>(v==null||v==='')?'—':v;
function chart(id,cfg){const ex=charts[id];
  if(ex&&ex.config.type===cfg.type){const nd=(cfg.data&&cfg.data.datasets)||[];
    ex.data.labels=(cfg.data&&cfg.data.labels)||[];
    if(nd.length!==ex.data.datasets.length)ex.data.datasets=nd;else nd.forEach((d,i)=>Object.assign(ex.data.datasets[i],d));
    if(cfg.options)ex.options=cfg.options;ex.update('none');return;}
  if(ex)ex.destroy();const c=$(id);if(c)charts[id]=new Chart(c,cfg);}
const grid='#1b1b1e';Chart.defaults.color='#8a8a8a';Chart.defaults.font.family='Inter';Chart.defaults.borderColor=grid;Chart.defaults.animation=false;
Chart.defaults.interaction={mode:'index',intersect:false};Chart.defaults.plugins.tooltip.displayColors=false;Chart.defaults.plugins.tooltip.padding=8;Chart.defaults.plugins.tooltip.titleColor='#ededed';Chart.defaults.plugins.tooltip.bodyColor='#ededed';Chart.defaults.plugins.tooltip.backgroundColor='#16161a';Chart.defaults.plugins.tooltip.borderColor='#2a2a2e';Chart.defaults.plugins.tooltip.borderWidth=1;Chart.defaults.elements.point.hoverRadius=5;
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
  if(el.children.length||!Object.keys(META.env||{}).length)return;
  const env=META.env||{},rm=META.ram||{};
  const all=Object.assign({},env,{'Footprint (spec)':'~'+(rm.footprint_gb||'?')+' GB RAM · '+(rm.model_disk_gb||'?')+' GB disk · live RSS in System card'});
  el.innerHTML=Object.entries(all).map(([k,v])=>`<span class="es" title="click to copy" onclick="copy('${esc(k)}: ${esc(v).replace(/'/g,'')}','${esc(k)} copied')"><b>${esc(k)}</b> ${esc(v)}</span>`).join('');
}

/* ---------- experiment band ---------- */
function experimentBand(){
  const el=$('expCard');if(!el)return;
  const running=!!STREAM.running,lv=STREAM.live||{};
  const sorted=[...RUNS].sort((a,b)=>(a.ts>b.ts?1:(a.ts<b.ts?-1:0)));
  const activeTag=(running&&lv.tag)?lv.tag:(sorted.length?sorted[sorted.length-1].tag:'—');
  const baseline='iq2-baseline';
  const lastRun=sorted.length?sorted[sorted.length-1]:null;
  const bench=(running&&lv.benchmark)?lv.benchmark:(lastRun?lastRun.benchmark:'medqa_test');
  const ourBest=RUNS.filter(r=>r.tag===activeTag&&r.benchmark===bench).sort((a,b)=>b.accuracy-a.accuracy)[0];
  const baseBest=RUNS.filter(r=>r.tag===baseline&&r.benchmark===bench).sort((a,b)=>b.accuracy-a.accuracy)[0];
  const refs=(META.references||{})[bench]||[];const primaryRef=SUMMARY.primary_ref||'MedGemma-27B (TTS)';
  let refPick=refs.find(r=>r.label===primaryRef);   // preferred apples-to-apples baseline if it exists for this benchmark
  if(!refPick&&refs.length)refPick=refs.filter(r=>r.kind!=='chance').sort((a,b)=>b.accuracy-a.accuracy)[0]||refs[0];  // else best existing reference
  const liveActive=running&&lv.benchmark===bench&&lv.accuracy!=null;   // currently recording this benchmark
  const ourAcc=liveActive?lv.accuracy:(ourBest?ourBest.accuracy:null);
  const delta=(ourAcc!=null&&baseBest)?+(ourAcc-baseBest.accuracy).toFixed(1):null;
  const sig=[running,activeTag,bench,lv.i,lv.accuracy,ourBest&&ourBest.accuracy,baseBest&&baseBest.accuracy,refPick&&refPick.label].join('|');
  if(sig===_expSig)return;_expSig=sig;
  const isBase=activeTag===baseline;
  const desc=isBase?'General-imatrix 2-bit build — the control baseline every other build is measured against.':'Medical-recalibrated 2-bit imatrix — testing whether a medical importance matrix lifts accuracy vs the general-imatrix baseline, at the same footprint.';
  const status=running?'<span class="tag ok" style="display:inline-flex;align-items:center;gap:6px"><span class="dot"></span>running</span>':'<span class="tag">idle · last run</span>';
  const mc=(l,v,c,cls)=>`<div class="expmetric"><div class="expml">${l}</div><div class="expmv ${cls||''}" style="color:${c||'var(--ink)'}">${v}</div></div>`;
  const dV=delta==null?'—':(delta>=0?'+':'')+delta+' pts';
  const dC=delta==null?'var(--mut)':delta>0?'var(--ok)':delta<0?'var(--dang)':'var(--mut)';
  const ourV=ourAcc!=null?(ourAcc+'%'+(liveActive?'<span class="livedot"></span>':'')):'—';
  const prog=running?`<div class="expprog"><div class="expbar"><i style="width:${lv.n?Math.round((lv.i||0)/lv.n*100):0}%"></i></div><span class="mono muted" style="font-size:12px;white-space:nowrap">${lv.i||0}/${lv.n||0} · ${esc(nice(lv.benchmark))} · ${esc(lv.mode||'')} · acc ${lv.accuracy!=null?lv.accuracy:'?'}%</span></div>`:'';
  el.innerHTML=`<div class="exphead"><div style="min-width:240px;flex:1">
      <div class="expk">EXPERIMENT</div>
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
function sysOpts(unit,max){return {animation:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' '+c.parsed.y+' '+unit}}},scales:{y:{min:0,max:max,grid:{color:grid},ticks:{maxTicksLimit:4}},x:{display:false}}};}
function systemUI(){
  const h=(HISTM||[]).slice(-300);if(!h.length)return;
  const labels=h.map((_,i)=>i),last=h[h.length-1];
  const lastTps=[...h].reverse().find(r=>r.tps!=null);
  setText('sysTps',lastTps&&lastTps.tps!=null?lastTps.tps.toFixed(1)+' t/s':(last.up?'idle':'offline'));
  chart('sysTpsChart',{type:'line',data:{labels,datasets:[{data:h.map(r=>r.tps),borderColor:'#f5a623',backgroundColor:'rgba(245,166,35,.10)',fill:true,tension:.3,pointRadius:0,borderWidth:2,spanGaps:true}]},options:sysOpts('t/s')});
  setText('sysRam',(last.rss!=null?'model '+last.rss+' GB':'—')+(last.used!=null?' · sys '+last.used+'/68.7':'')+(last.swap!=null&&last.swap>0?' · sys-swap '+last.swap+'G':''));
  chart('sysRamChart',{type:'line',data:{labels,datasets:[{label:'model RSS',data:h.map(r=>r.rss),borderColor:'#3291ff',backgroundColor:'rgba(50,145,255,.10)',fill:true,tension:.3,pointRadius:0,borderWidth:2,spanGaps:true},{label:'system used',data:h.map(r=>r.used),borderColor:'#8a63d2',tension:.3,pointRadius:0,borderWidth:1.5,spanGaps:true},{label:'system swap',data:h.map(r=>r.swap),borderColor:'#e5484d',backgroundColor:'rgba(229,72,77,.08)',tension:.3,pointRadius:0,borderWidth:1.5,spanGaps:true}]},options:sysOpts('GB',70)});
  setText('sysCpu',last.cpu!=null?Math.round(last.cpu)+'%':'—');
  chart('sysCpuChart',{type:'line',data:{labels,datasets:[{data:h.map(r=>r.cpu),borderColor:'#0cce6b',backgroundColor:'rgba(12,206,107,.10)',fill:true,tension:.3,pointRadius:0,borderWidth:2,spanGaps:true}]},options:sysOpts('%')});
}
/* ---------- accuracy ---------- */
function accbars(){
  const bench=($('accBench')&&$('accBench').value)||'medqa_test';
  const ours=(SUMMARY.per_run||[]).filter(r=>r.benchmark===bench).map(r=>({label:r.tag+' · '+r.mode+' (N='+r.n+')',acc:r.accuracy,lo:r.ci_lo,hi:r.ci_hi,small:r.small_n,cls:'ours'}));
  const refs=((META.references||{})[bench]||[]).map(r=>({label:r.label,acc:r.accuracy,cls:r.kind||'ref'}));
  const all=ours.concat(refs).sort((a,b)=>b.acc-a.acc);
  const CLS={ours:['linear-gradient(90deg,#0070f3,#3291ff)','Our IQ2'],frontier:['#a855f7','Frontier'],ref:['#14b8a6','Medical-spec.'],chance:['#3a3a3a','Chance']};
  const legend='<div style="display:flex;gap:16px;flex-wrap:wrap;margin:2px 0 14px;font-size:12px;color:#9a9aa2">'+['ours','frontier','ref','chance'].map(k=>`<span style="display:inline-flex;align-items:center;gap:6px"><i style="width:11px;height:11px;border-radius:3px;display:inline-block;background:${CLS[k][0]}"></i>${CLS[k][1]}</span>`).join('')+'</div>';
  $('accbars').innerHTML=all.length?(legend+all.map(x=>{const s=CLS[x.cls]||CLS.ref;const lbl=x.cls==='ours'?`<b>${esc(x.label)}</b>`:esc(x.label);
    const ci=(x.cls==='ours'&&x.lo!=null)?`<div class="ci" style="left:${x.lo}%;width:${Math.max(0,x.hi-x.lo)}%"></div>`:'';
    const pv=(x.cls==='ours'&&x.lo!=null)?`${x.acc}% <small>[${x.lo}–${x.hi}]${x.small?' ⚠':''}</small>`:`${x.acc}%`;
    return `<div class="barwrap"><div class="lab" title="${esc(x.label)}">${lbl}</div><div class="track"><div class="fill" style="width:${Math.max(2,x.acc)}%;background:${s[0]}"></div>${ci}</div><div class="pv mono">${pv}</div></div>`}).join('')):('<div class="muted">No runs for '+nice(bench)+' yet — launch one from Control.</div>');
  // mode chart from raw runs
  const runs=RUNS.filter(r=>r.benchmark===bench);
  const byMode={};runs.forEach(r=>{(byMode[r.mode]=byMode[r.mode]||[]).push(r.accuracy)});
  const labels=Object.keys(byMode),data=labels.map(m=>byMode[m].reduce((a,b)=>a+b,0)/byMode[m].length);
  chart('modeChart',{type:'bar',data:{labels:labels.map(m=>m==='thinking'?'🧠 thinking':'⚡ non-thinking'),datasets:[{data,backgroundColor:labels.map(m=>m==='thinking'?'#0cce6b':'#8a63d2'),borderRadius:8,barThickness:46}]},options:{indexAxis:'y',plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>' '+c.parsed.x.toFixed(1)+'%'}}},scales:{x:{beginAtZero:true,max:100,grid:{color:grid},ticks:{callback:v=>v+'%'}},y:{grid:{display:false},ticks:{font:{size:14}}}}}});
  const mac=SUMMARY.macro||{};
  $('macroHint').textContent=(mac.thinking_mean!=null)?('MCQ macro-avg '+mac.thinking_mean+'% thinking (k='+mac.thinking_k+(mac.thinking_tag?', '+mac.thinking_tag:'')+')'+(mac.nothink_mean!=null?' · '+mac.nothink_mean+'% nothink (k='+mac.nothink_k+(mac.nothink_tag?', '+mac.nothink_tag:'')+')':'')+' · excl. HealthBench'):'';
}

/* ---------- performance ---------- */
function perfCharts(){
  const pts=PERF.filter(p=>p.hit_rate!=null).sort((a,b)=>a.cache_gb-b.cache_gb);
  const labels=[...new Set(pts.map(p=>p.cache_gb+' GB'))];
  const hr=labels.map(l=>{const g=parseInt(l),r=pts.find(p=>p.cache_gb===g);return r?r.hit_rate*100:null});
  const gt=labels.map(l=>{const g=parseInt(l),r=pts.find(p=>p.cache_gb===g);return r?r.gen_tps:null});
  chart('hrChart',{type:'line',data:{labels,datasets:[{label:'hit-rate %',data:hr,borderColor:'#0cce6b',backgroundColor:'rgba(12,206,107,.1)',tension:.3,yAxisID:'y',fill:true,pointRadius:4},{label:'gen tok/s',data:gt,borderColor:'#f5a623',tension:.3,yAxisID:'y1',pointRadius:4}]},options:{plugins:{legend:{position:'top',labels:{boxWidth:12}}},scales:{y:{position:'left',grid:{color:grid},min:0,max:100,title:{display:true,text:'hit-rate %'}},y1:{position:'right',grid:{display:false},min:0,title:{display:true,text:'gen t/s'}},x:{grid:{display:false}}}}});
  $('perfBody').innerHTML=PERF.map(p=>`<tr><td><span class="tag ${p.cold?'warn':'blue'}">${esc(p.kind)}</span></td><td class="mono">${esc(p.config)}</td><td class="n mono">${num(p.prefill_tps)}</td><td class="n mono"><b>${num(p.gen_tps)}</b></td><td class="n mono">${p.hit_rate!=null?p.hit_rate.toFixed(3):'—'}</td><td class="muted">${esc(p.notes||'')}</td></tr>`).join('');
}

/* ---------- runs explorer ---------- */
function setView(v){VIEW=v;$('vAcc').classList.toggle('on',v==='acc');$('vPerf').classList.toggle('on',v==='perf');renderRuns()}
function renderRuns(){
  const q=($('q').value||'').toLowerCase();let cols,rows;
  if(VIEW==='acc'){cols=[['ts','date'],['benchmark','benchmark'],['tag','tag'],['mode','mode'],['n','N',1],['accuracy','accuracy',1],['sec_per_q','s/q',1],['notes','notes']];rows=RUNS.slice();}
  else{cols=[['kind','kind'],['config','config'],['prefill_tps','prefill t/s',1],['gen_tps','gen t/s',1],['hit_rate','hit-rate',1],['notes','notes']];rows=PERF.slice();}
  rows=rows.filter(r=>!q||JSON.stringify(r).toLowerCase().includes(q));
  rows.sort((a,b)=>{const x=a[SORT.k],y=b[SORT.k];if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:0)*SORT.d});
  $('runHead').innerHTML='<tr>'+cols.map(c=>`<th class="${c[2]?'n':''}" onclick="sortBy('${c[0]}')">${c[1]}${SORT.k===c[0]?(SORT.d>0?' ↑':' ↓'):''}</th>`).join('')+'<th></th></tr>';
  $('runBody').innerHTML=rows.map((r,ri)=>{const clk=(VIEW==='acc'&&r.details)?` style="cursor:pointer" title="click to inspect every question" onclick="openDetails('${esc(r.details)}')"`:'';
    const cells=cols.map(c=>{let v=r[c[0]];
      if(c[0]==='accuracy')return `<td class="n mono"><b>${esc(v)}%</b>${(VIEW==='acc'&&r.details)?' <span style="color:var(--acc2)">🔍</span>':''}</td>`;
      if(c[0]==='hit_rate')return `<td class="n mono">${v!=null?v.toFixed(3):'—'}</td>`;
      if(c[0]==='tag'||c[0]==='kind')return `<td><span class="tag">${esc(v)}</span></td>`;
      if(c[0]==='mode')return `<td><span class="tag ${v==='thinking'?'pur':''}">${esc(v||'—')}</span></td>`;
      return `<td class="${c[2]?'n mono':''} ${c[0]==='notes'?'muted':''}">${esc(v==null?'—':v)}</td>`}).join('');
    return `<tr${clk}>${cells}<td class="n"><button class="iconbtn xs" title="copy run JSON" onclick="event.stopPropagation();copyRun('${VIEW}',${ri})">⧉</button></td></tr>`;}).join('');
  $('runcount').textContent=rows.length+' run'+(rows.length!==1?'s':'');
  window._rrows=rows;
}
function copyRun(view,ri){const r=(window._rrows||[])[ri];if(r)copy(JSON.stringify(r,null,2),'Run JSON copied');}
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
function chatEmpty(){const ex=['A 45-year-old with crushing chest pain radiating to the left arm — differential and first steps?','Explain the mechanism of action of metformin.','Labs: Na 128, K 5.8, glucose 480, pH 7.1 — interpret and manage.'];
  return `<div class="chatempty"><h3>Chat with DeepSeek-V4-Flash IQ2</h3><p class="muted">Your local medical model on <span class="mono">:8000</span>. Toggle 🧠 thinking for chain-of-thought reasoning.</p><div class="exq">${ex.map(q=>`<button onclick="chatExample(this)">${esc(q)}</button>`).join('')}</div></div>`;}
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
  return `<div class="cmsg bot"${sid}><div class="av">ds4</div><div class="body">${chatBotInner(m)}</div></div>`+chatMeta(m);}
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
async function loadAll(){try{
  [RUNS,PERF,META,SUMMARY]=await Promise.all([fetch('/api/runs').then(r=>r.json()),fetch('/api/perf').then(r=>r.json()),fetch('/api/meta').then(r=>r.json()),fetch('/api/summary').then(r=>r.json())]);
  {const rb=$('rBench'),cur=rb.value,opts=(META.benchmarks||['medqa_test']),sig=opts.join(',');
   if(rb.dataset.sig!==sig){rb.innerHTML=opts.map(b=>`<option value="${esc(b)}" ${b===cur?'selected':''}>${esc(nice(b))}</option>`).join('');if(cur)rb.value=cur;rb.dataset.sig=sig;}}
  const benches=[...new Set([...RUNS.map(r=>r.benchmark),...Object.keys(META.references||{})])].filter(Boolean);
  const ab=$('accBench'),cur=ab.value,asig=benches.join(',');if(ab.dataset.sig!==asig){ab.innerHTML=benches.map(b=>`<option value="${esc(b)}" ${b===cur?'selected':''}>${esc(nice(b))}</option>`).join('');if(cur)ab.value=cur;ab.dataset.sig=asig;}
  envStrip();experimentBand();syncBench();
  const sig=RUNS.length+'|'+PERF.length+'|'+(RUNS.length?RUNS[RUNS.length-1].ts:'')+'|'+JSON.stringify(SUMMARY.macro||{});
  if(sig!==_dataSig){_dataSig=sig;accbars();perfCharts();renderRuns();}
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
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/" or p.startswith("/index"): self._send(200, PAGE)
        elif p == "/api/runs": self._json(read_jsonl(RUNS))
        elif p == "/api/perf": self._json(read_jsonl(PERF))
        elif p == "/api/meta": self._json(dict(META, benchmarks=benchmarks()))
        elif p == "/api/summary": self._json(summary())
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
            fn = "beepmed-report-%s.md" % datetime.date.today().isoformat()
            self._send(200, make_report(), "text/markdown; charset=utf-8",
                       {"Content-Disposition": 'attachment; filename="%s"' % fn})
        else: self._send(404, "not found", "text/plain")
    def stream_chat(self, body):
        """Proxy a chat to the ds4 OpenAI-compatible server and relay tokens to the browser as SSE.
        Real streaming when the server supports it; otherwise a non-streaming fallback chunked for UX."""
        if not server_up():
            return self._json({"error": "model server not reachable on :8000 — start it from Control"}, 503)
        msgs = body.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return self._json({"error": "no messages"}, 400)
        msgs = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in msgs if m.get("content")]
        payload = {"model": "ds4", "messages": msgs,
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
    def do_POST(self):
        p = self.path.split("?")[0]
        if p == "/api/chat": self.stream_chat(self._body())
        elif p == "/api/run": self._json(start_eval(self._body()))
        elif p == "/api/run/stop": self._json(stop_eval())
        elif p == "/api/server/start": self._json(start_server())
        elif p == "/api/stop-all": self._json(stop_all())
        elif p == "/api/server/stop": self._json(kill_server())
        else: self._send(404, "not found", "text/plain")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    os.makedirs(RESULTS, exist_ok=True)
    _load_history()
    threading.Thread(target=_sample_metrics, daemon=True).start()
    print("BeepMed Benchmark Explorer -> http://127.0.0.1:%d" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
