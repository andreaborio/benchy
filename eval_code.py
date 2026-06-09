#!/usr/bin/env python3
"""Code-generation benchmark runner: the model writes a Python function, we EXECUTE it
against the task's tests and score pass@1. For HumanEval / MBPP-style sets (data shape
{task_id, prompt, tests, entry_point}), fetched via fetch_benchmarks.py.

⚠ SECURITY: this RUNS model-generated code on your machine (each candidate in a separate
process with a timeout, in a temp dir). Only run it locally, with models/benchmarks you
trust. This is the standard way HumanEval/MBPP are evaluated, but the risk is real.

Usage: eval_code.py <jsonl> <N> [think|nothink] [tag] [notes...]
Env: BENCHY_SERVER (default http://127.0.0.1:8000), BENCHY_MODEL (auto-detected),
     BENCHY_CODE_TIMEOUT (per-task execution timeout, seconds; default 12).
Writes results/{runs,live,stream}.jsonl + details/ — same format as eval_mcq.py.
Stdlib only. Greedy, deterministic.
"""
import json, sys, re, random, urllib.request, time, datetime, os, tempfile, subprocess
import benchy_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(RESULTS, "runs.jsonl")
LIVE = os.path.join(RESULTS, "live.json")
STREAM = os.path.join(RESULTS, "stream.jsonl")
DETAILS = os.path.join(RESULTS, "details")
SERVER_BASE = os.environ.get("BENCHY_SERVER", "http://127.0.0.1:8000").rstrip("/")
SERVER = SERVER_BASE + "/v1/chat/completions"
TIMEOUT = float(os.environ.get("BENCHY_CODE_TIMEOUT", "12"))
SEED = 1234
# Code-execution is OFF unless the operator explicitly opts in: this runner executes
# model-written AND benchmark-supplied Python on the host with no real isolation (subprocess
# + tempdir + timeout only). Require BENCHY_ALLOW_CODE_EXEC so it can never run by surprise
# (e.g. via a cross-site request to the dashboard's /api/run).
ALLOW = os.environ.get("BENCHY_ALLOW_CODE_EXEC", "").lower() in ("1", "true", "yes", "on")

MODEL = bc.resolve_model(SERVER_BASE)

def load(path, n):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(SEED).shuffle(rows)
    return rows if n == 0 else rows[:n]

def build_prompt(task):
    return ("Complete the following Python task. Respond with ONLY the function "
            "implementation inside a single ```python code block — no prose, no examples.\n\n"
            + task["prompt"])

def ask(prompt, think):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096 if think else 1536, "temperature": 0.0, "think": bool(think)}
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def extract_code(text):
    """Pull the Python out of a chat reply: prefer a fenced block, else use the body."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, flags=re.S)
    if m: return m.group(1).strip()
    return text.strip()

def run_tests(candidate, task):
    """Build the full program (candidate + tests) and execute it in a fresh subprocess.
    pass = exit code 0 within the timeout. Returns (passed, error_summary)."""
    ep = task.get("entry_point") or ""
    code = candidate
    if ep and ("def " + ep) not in code:        # model returned only a body → prepend the signature
        code = task.get("prompt", "") + "\n" + code
    program = code + "\n\n" + task.get("tests", "")
    d = tempfile.mkdtemp(prefix="benchy_code_")
    fp = os.path.join(d, "cand.py")
    try:
        with open(fp, "w") as f: f.write(program)
        r = subprocess.run([sys.executable, fp], capture_output=True, text=True,
                           timeout=TIMEOUT, cwd=d)
        if r.returncode == 0:
            return True, ""
        err = (r.stderr or r.stdout or "").strip().splitlines()
        return False, (err[-1] if err else "non-zero exit")[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout (%ss)" % TIMEOUT
    except Exception as e:
        return False, ("harness error: %s" % e)[:200]
    finally:
        try:
            os.remove(fp); os.rmdir(d)
        except Exception:
            pass

def write_live(d):
    os.makedirs(RESULTS, exist_ok=True)
    d["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = LIVE + ".tmp"; open(tmp, "w").write(json.dumps(d)); os.replace(tmp, LIVE)

def main():
    if not ALLOW:
        sys.exit("⛔ code-execution benchmarks are disabled. This runner executes model-generated "
                 "AND benchmark-supplied Python on your machine with no sandbox. Enable it deliberately "
                 "and only for files you trust:\n    BENCHY_ALLOW_CODE_EXEC=1 python3 eval_code.py ...\n"
                 "(set the same var in the dashboard's environment to allow code runs from the UI).")
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    think = len(sys.argv) > 3 and sys.argv[3] == "think"
    tag = sys.argv[4] if len(sys.argv) > 4 else "baseline"
    notes = " ".join(sys.argv[5:]) if len(sys.argv) > 5 else ""
    mode = "thinking" if think else "nothink"
    bench = os.path.basename(path).replace(".jsonl", "")
    rows = load(path, n)
    os.makedirs(RESULTS, exist_ok=True)
    open(STREAM, "w").close()
    os.makedirs(DETAILS, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    detfile = os.path.join(DETAILS, f"{bench}__{mode}__{run_id}.jsonl")
    open(detfile, "w").close()
    print(f"⚠ executing model-generated code in subprocesses (timeout {TIMEOUT}s each). bench={bench} N={len(rows)} model={MODEL}", flush=True)
    passed = 0; t0 = time.time()
    for i, task in enumerate(rows):
        q0 = time.time()
        try:
            out = ask(build_prompt(task), think)
        except Exception as e:
            out = "ERR:%s" % e
        code = extract_code(out)
        ok, err = run_tests(code, task)
        passed += ok
        acc = 100 * passed / (i + 1)
        ev = {"i": i + 1, "n": len(rows), "ok": bool(ok), "pred": "PASS" if ok else "FAIL",
              "gold": "tests", "acc": round(acc, 1), "t": round(time.time() - q0, 1),
              "q": str(task.get("prompt", ""))[:260], "pred_txt": "" if ok else err,
              "gold_txt": str(task.get("task_id", "")), "ans": code[:180]}
        with open(STREAM, "a") as f: f.write(json.dumps(ev) + "\n")
        with open(detfile, "a") as f:
            f.write(json.dumps({"i": i + 1, "question": task.get("prompt", ""), "options": {},
                                "pred": "PASS" if ok else "FAIL", "gold": "tests", "ok": bool(ok),
                                "answer": code + ("\n\n# test result: PASS" if ok else "\n\n# test result: FAIL — " + err)}) + "\n")
        write_live({"running": True, "tag": tag, "benchmark": bench, "mode": mode,
                    "i": i + 1, "n": len(rows), "correct": passed,
                    "accuracy": round(acc, 1), "elapsed_s": round(time.time() - t0)})
        print(f"{i+1:>4}/{len(rows)} {'PASS' if ok else 'FAIL'} pass@1={acc:5.1f}%" + ("" if ok else "  ("+err+")"), flush=True)
    dt = time.time() - t0; acc = 100 * passed / len(rows) if rows else 0.0
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag,
           "benchmark": bench, "mode": mode, "n": len(rows), "correct": passed,
           "accuracy": round(acc, 1), "seed": SEED, "duration_s": round(dt),
           "sec_per_q": round(dt / max(1, len(rows)), 1), "kind": "code", "notes": notes,
           "details": os.path.basename(detfile), **bc.run_meta(MODEL, SERVER_BASE, path)}
    open(RUNS, "a").write(json.dumps(rec) + "\n")
    write_live({"running": False, "tag": tag, "benchmark": bench, "mode": mode,
                "i": len(rows), "n": len(rows), "correct": passed,
                "accuracy": round(acc, 1), "elapsed_s": round(dt)})
    print(f"\n=== {bench} [{mode}] tag={tag} N={len(rows)} pass@1 = {passed}/{len(rows)} = {acc:.1f}%  ({dt:.0f}s) ===")

if __name__ == "__main__":
    main()
