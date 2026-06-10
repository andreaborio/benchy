#!/usr/bin/env python3
"""Code-generation benchmark runner: the model writes a Python function, we EXECUTE it
against the task's tests and score pass@1. For HumanEval / MBPP-style sets (data shape
{task_id, prompt, tests, entry_point}), fetched via fetch_benchmarks.py.

⚠ SECURITY: this RUNS model-generated code on your machine (each candidate in its own
process group with a wall timeout, CPU/file-size/fd rlimits and a temp dir — mitigations,
not a sandbox). Only run it locally, with models/benchmarks you trust. This is the
standard way HumanEval/MBPP are evaluated, but the risk is real.

Usage: eval_code.py <jsonl> <N> <think|nothink> <tag> [--seed INT]
Env: BENCHY_SERVER (overrides config.json server_base / the built-in default),
     BENCHY_MODEL (auto-detected),
     BENCHY_CODE_TIMEOUT (per-task execution timeout, seconds; default 12).
Writes results/{runs,live,stream}.jsonl + details/ — same format as eval_mcq.py.
Stdlib only. Greedy, deterministic.
"""
import json, sys, re, random, time, os, signal, tempfile, subprocess
try:
    import resource                 # POSIX only — rlimits are skipped where unavailable
except ImportError:
    resource = None
import benchy_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_BASE = bc.settings()["server_base"]
TIMEOUT = float(os.environ.get("BENCHY_CODE_TIMEOUT", "12"))
SEED = 1234
# Code-execution is OFF unless the operator explicitly opts in: this runner executes
# model-written AND benchmark-supplied Python on the host with no real isolation (subprocess
# + tempdir + timeout only). Require BENCHY_ALLOW_CODE_EXEC so it can never run by surprise
# (e.g. via a cross-site request to the dashboard's /api/run).
ALLOW = os.environ.get("BENCHY_ALLOW_CODE_EXEC", "").lower() in ("1", "true", "yes", "on")

_MODEL = None
def get_model():
    # resolved lazily (and cached) so importing this module never hits the network
    global _MODEL
    if _MODEL is None:
        _MODEL = bc.resolve_model(SERVER_BASE)
    return _MODEL

def load(path, n):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    random.Random(SEED).shuffle(rows)
    return rows if n == 0 else rows[:n]

def build_prompt(task):
    return ("Complete the following Python task. Respond with ONLY the function "
            "implementation inside a single ```python code block — no prose, no examples.\n\n"
            + task["prompt"])

def ask(prompt, think):
    return bc.chat(prompt, think=think, max_tokens=4096 if think else 1536,
                   get_model=get_model, seed=SEED, server_base=SERVER_BASE, timeout=600)

def extract_code(text):
    """Pull the Python out of a chat reply: prefer a fenced block, else use the body."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, flags=re.S)
    if m: return m.group(1).strip()
    return text.strip()

POSIX = os.name == "posix"

# The currently-executing candidate (a Popen leading its own process group), or None.
# Module-global so the SIGTERM/SIGINT handlers can reach it: without this, every stop path
# other than the wall-clock timeout (dashboard stop_all SIGTERM, Ctrl-C) would orphan the
# detached group and leave model-written code running with no wall clock.
ACTIVE_PROC = None

def _kill_group(p):
    """SIGKILL the candidate's whole process group (it leads its own session); best-effort
    fallback to killing just the direct child if the group is already gone / non-POSIX."""
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass

def _handle_stop(signum, frame):
    """SIGTERM/SIGINT: take the active candidate group down with us, then re-raise the
    signal with default semantics so the exit status still says 'killed by <signum>'."""
    p = ACTIVE_PROC
    if p is not None:
        _kill_group(p)
        try:
            p.wait(timeout=5)           # reap — never leave a zombie behind
        except Exception:
            pass
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)

def install_signal_handlers():
    """Called by main() (and by anything embedding run_tests in a long loop): ensures a
    stopped harness never orphans a running candidate."""
    if not POSIX:
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handle_stop)

def _child_limits(timeout):
    """preexec_fn for the candidate process (POSIX only): resource ceilings, NOT a sandbox.
    - RLIMIT_CPU ≈ 2x the wall timeout: a spin loop / fork bomb that dodges the wall-clock
      kill still gets SIGKILLed by the kernel once it has burned that much CPU.
    - RLIMIT_FSIZE 64MB: a runaway writer gets SIGXFSZ instead of filling the disk.
    - RLIMIT_NOFILE 256: caps fd exhaustion.
    Deliberately omitted: RLIMIT_AS (not enforced on macOS, so it would be false comfort
    here) and RLIMIT_NPROC (per-UID, not per-process — a low cap would strangle the
    operator's own session, not just the candidate). This narrows the blast radius of
    accidents; it does not make untrusted code safe."""
    def fn():
        cpu = max(1, int(timeout * 2))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    return fn

def run_tests(candidate, task, timeout=None):
    """Build the full program (candidate + tests) and execute it in a fresh subprocess.
    pass = exit code 0 within the timeout. Returns (passed, error_summary).
    start_new_session puts the candidate in its own process group (same pattern as the
    dashboard's child processes) so the timeout path can SIGKILL the whole group — a
    candidate that fork()s can no longer outlive the harness."""
    global ACTIVE_PROC
    if timeout is None: timeout = TIMEOUT
    ep = task.get("entry_point") or ""
    code = candidate
    if ep and ("def " + ep) not in code:        # model returned only a body → prepend the signature
        code = task.get("prompt", "") + "\n" + code
    program = code + "\n\n" + task.get("tests", "")
    d = tempfile.mkdtemp(prefix="benchy_code_")
    fp = os.path.join(d, "cand.py")
    p = None
    try:
        with open(fp, "w", encoding="utf-8") as f: f.write(program)
        # errors="replace": model-written code may print arbitrary bytes — keep the real
        # error line instead of dying on a UnicodeDecodeError in the harness
        p = subprocess.Popen([sys.executable, fp], cwd=d, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
                             start_new_session=POSIX,
                             preexec_fn=_child_limits(timeout) if (POSIX and resource) else None)
        ACTIVE_PROC = p                         # visible to the SIGTERM/SIGINT handlers
        try:
            out, err_out = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(p)                      # kill the whole group, not just the direct child
            p.communicate()                     # reap — never leave a zombie behind
            return False, "timeout (%ss)" % timeout
        if p.returncode == 0:
            return True, ""
        err = (err_out or out or "").strip().splitlines()
        return False, (err[-1] if err else "non-zero exit")[:200]
    except Exception as e:
        return False, ("harness error: %s" % e)[:200]
    finally:
        # ANY exit path (incl. KeyboardInterrupt / harness bugs, not just TimeoutExpired):
        # if the candidate group is still alive, take it down before leaving run_tests
        if p is not None:
            if p.poll() is None:
                _kill_group(p)
                try:
                    p.wait(timeout=5)
                except Exception:
                    pass
            ACTIVE_PROC = None
        try:
            os.remove(fp); os.rmdir(d)
        except Exception:
            pass

def main():
    global SEED
    if not ALLOW:
        sys.exit("⛔ code-execution benchmarks are disabled. This runner executes model-generated "
                 "AND benchmark-supplied Python on your machine with no sandbox. Enable it deliberately "
                 "and only for files you trust:\n    BENCHY_ALLOW_CODE_EXEC=1 python3 eval_code.py ...\n"
                 "(set the same var in the dashboard's environment to allow code runs from the UI).")
    install_signal_handlers()       # SIGTERM/Ctrl-C must never orphan a running candidate
    args = bc.parse_run_args(prog="eval_code.py")
    SEED = args.seed
    path, think, tag, mode = args.bench, args.think, args.tag, args.mode
    max_tokens = 4096 if think else 1536
    bench = os.path.basename(path).replace(".jsonl", "")
    # integrity gate BEFORE anything runs (see eval_mcq.py): abort on lockfile drift,
    # BENCHY_SKIP_LOCK_CHECK=1 to warn instead; locked is recorded in the run record.
    locked = bc.check_dataset_lock(path)
    rows = load(path, args.n)
    w = bc.RunWriter(bench, mode, tag, bc.KIND_CODE)
    print(f"⚠ executing model-generated code in subprocesses (timeout {TIMEOUT}s each). bench={bench} N={len(rows)} model={get_model()}", flush=True)
    passed = 0; scored = 0; errors = 0; t0 = time.time()
    for i, task in enumerate(rows):
        q0 = time.time()
        try:
            out = ask(build_prompt(task), think)
        except Exception as e:
            # request still failed after bc.chat's retries: excluded from the pass@1
            # numerator AND denominator, counted in the run record's "errors" field
            errors += 1
            acc = 100 * passed / scored if scored else 0.0
            w.stream({"i": i + 1, "n": len(rows), "ok": False, "pred": "ERR", "gold": "tests",
                      "acc": round(acc, 1), "t": round(time.time() - q0, 1),
                      "q": str(task.get("prompt", ""))[:260], "error": True,
                      "gold_txt": str(task.get("task_id", "")), "ans": str(e)[:180]})
            w.detail({"i": i + 1, "question": task.get("prompt", ""), "options": {},
                      "gold": "tests", "error": True, "error_msg": str(e)})
            w.live({"running": True, "i": i + 1, "n": len(rows), "correct": passed,
                    "accuracy": round(acc, 1), "errors": errors,
                    "elapsed_s": round(time.time() - t0)})
            print(f"{i+1:>4}/{len(rows)} ERROR (excluded from scoring): {e}", flush=True)
            continue
        scored += 1
        code = extract_code(out)
        ok, err = run_tests(code, task)
        passed += ok
        acc = 100 * passed / scored
        w.stream({"i": i + 1, "n": len(rows), "ok": bool(ok), "pred": "PASS" if ok else "FAIL",
                  "gold": "tests", "acc": round(acc, 1), "t": round(time.time() - q0, 1),
                  "q": str(task.get("prompt", ""))[:260], "pred_txt": "" if ok else err,
                  "gold_txt": str(task.get("task_id", "")), "ans": code[:180]})
        w.detail({"i": i + 1, "question": task.get("prompt", ""), "options": {},
                  "pred": "PASS" if ok else "FAIL", "gold": "tests", "ok": bool(ok),
                  "answer": code + ("\n\n# test result: PASS" if ok else "\n\n# test result: FAIL — " + err)})
        w.live({"running": True, "i": i + 1, "n": len(rows), "correct": passed,
                "accuracy": round(acc, 1), "errors": errors,
                "elapsed_s": round(time.time() - t0)})
        print(f"{i+1:>4}/{len(rows)} {'PASS' if ok else 'FAIL'} pass@1={acc:5.1f}%" + ("" if ok else "  ("+err+")"), flush=True)
    dt = time.time() - t0
    acc = 100 * passed / scored if scored else 0.0
    fields = {"n": scored, "correct": passed, "accuracy": round(acc, 1), "seed": SEED,
              "duration_s": round(dt), "sec_per_q": round(dt / max(1, len(rows)), 1),
              "errors": errors, "max_tokens": max_tokens, "locked": locked, "notes": ""}
    if rows and errors / len(rows) > 0.05:
        fields["invalid"] = True
        print(f"\n⚠⚠⚠ INVALID RUN: {errors}/{len(rows)} requests failed (>5%) — "
              f"the pass@1 below covers only the {scored} scored tasks and is NOT "
              f"comparable to a clean run.", file=sys.stderr)
    w.finish(fields, get_model(), SERVER_BASE, path)
    w.live({"running": False, "i": len(rows), "n": len(rows), "correct": passed,
            "accuracy": round(acc, 1), "errors": errors, "elapsed_s": round(dt)})
    print(f"\n=== {bench} [{mode}] tag={tag} N={scored} pass@1 = {passed}/{scored} = {acc:.1f}%  ({dt:.0f}s, {errors} errors) ===")

if __name__ == "__main__":
    main()
