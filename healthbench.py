#!/usr/bin/env python3
"""HealthBench eval with a faithful API judge (OpenAI simple-evals methodology).

The local model under test answers the open-ended conversations; an API grader scores
each rubric criterion as criteria_met (true/false) using OpenAI's exact GRADER_TEMPLATE.
Per-example score = sum(points of met criteria) / sum(positive points), clamped to [0,1].
Mean over examples = HealthBench score. (This is one of the bundled rubric-graded
benchmarks; MCQ sets go through eval_mcq.py instead.)

Usage: healthbench.py <hard|consensus> <N> <think|nothink> <tag> [--seed INT]
Env: BENCHY_SERVER (overrides config.json server_base / the built-in default),
     BENCHY_MODEL (default: auto-detected),
     GRADER_MODEL (override the per-provider grader; default gpt-4.1 / claude-sonnet-4-6 / gemini-2.5-pro),
     BENCHY_ALLOW_UNPINNED=1 (accept a dataset whose sha256 is not pinned in EXPECTED_SHA256).
Reads the grader API key from a .apikey file next to this script (provider auto-detected; never logged).
"""
import json, os, re, sys, time, random, hashlib, urllib.request, concurrent.futures
import benchy_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SERVER_BASE = bc.settings()["server_base"]

URLS = {"hard": "https://huggingface.co/datasets/openai/healthbench/resolve/main/hard_2025-05-08-21-00-10.jsonl",
        "consensus": "https://huggingface.co/datasets/openai/healthbench/resolve/main/consensus_2025-05-09-20-00-46.jsonl"}
# The HealthBench download is OUTSIDE benchmarks.lock.json, so pin it here: full sha256 of
# the dataset snapshot we benchmarked against, verified after download and on every load.
# A mismatch (or a missing pin) refuses to run unless BENCHY_ALLOW_UNPINNED=1.
EXPECTED_SHA256 = {
    "hard": "b0320430e5cd974e746585594c1dd10b5a3fc2aff9c72b26106c2c4a069d74e9",
    "consensus": None,  # no pinned local snapshot yet — requires BENCHY_ALLOW_UNPINNED=1
}
# default grader per provider; OpenAI gpt-4.1 = the official HealthBench judge
GRADER = {"openai": "gpt-4.1", "anthropic": "claude-sonnet-4-6", "google": "gemini-2.5-pro"}

GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that the patient should drink fluids for hydration, such as water, juice, or sports drinks", and the response just says "the patient should drink water", the criteria should be met.

# Example response
{{"explanation": "The response correctly identifies ...", "criteria_met": true}}

Return just the json object."""

_MODEL = None
def get_model():
    # resolved lazily (and cached) so importing this module never hits the network
    global _MODEL
    if _MODEL is None:
        _MODEL = bc.resolve_model(SERVER_BASE)
    return _MODEL

def dataset_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(path, subset):
    """Refuse to score an unverified dataset: the download is not covered by
    benchmarks.lock.json, so the pin lives in EXPECTED_SHA256 instead."""
    want = EXPECTED_SHA256.get(subset)
    got = dataset_sha256(path)
    if want and got == want:
        return
    if os.environ.get("BENCHY_ALLOW_UNPINNED") == "1":
        print(f"  WARNING: running with UNPINNED dataset {os.path.basename(path)} "
              f"(sha256 {got[:12]}…, expected {want[:12] + '…' if want else 'no pin'}) — "
              f"results are not comparable across snapshots.", file=sys.stderr)
        return
    if want:
        sys.exit(f"⛔ {path} sha256 mismatch: expected {want}, got {got}. The upstream file "
                 f"changed (or the download is corrupt) — scores would not be comparable. "
                 f"Delete the file to re-download, or set BENCHY_ALLOW_UNPINNED=1 to run anyway.")
    sys.exit(f"⛔ no pinned sha256 for HealthBench subset '{subset}' (got {got}). Add it to "
             f"EXPECTED_SHA256 in healthbench.py, or set BENCHY_ALLOW_UNPINNED=1 to run anyway.")


def load_key():
    p = os.path.join(HERE, ".apikey")
    if not os.path.exists(p):
        sys.exit("No .apikey found next to healthbench.py — create it:  echo 'sk-...' > .apikey  (then chmod 600 .apikey)")
    k = open(p, encoding="utf-8").read().strip()
    prov = "anthropic" if k.startswith("sk-ant-") else "google" if k.startswith("AIza") else "openai"
    model = os.environ.get("GRADER_MODEL", GRADER[prov])
    return k, prov, model

def grader_call(key, prov, model, prompt):
    if prov == "anthropic":
        req = urllib.request.Request("https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": model, "max_tokens": 700, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["content"][0]["text"]
    if prov == "google":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        req = urllib.request.Request(url, data=json.dumps({"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0}}).encode(),
            headers={"content-type": "application/json", "x-goog-api-key": key})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["candidates"][0]["content"]["parts"][0]["text"]
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]

def grade_criterion(key, prov, model, conv, idx, r):
    """Grade one rubric criterion; retries with backoff. Final failure returns met=None —
    the criterion is EXCLUDED from achieved AND achievable points (and counted in
    grader_errors) instead of deflating the score as a phantom met=False."""
    pr = GRADER_TEMPLATE.format(conversation=conv, rubric_item=f"[{r['points']}] {r['criterion']}")
    err = None
    for delay in (2, 8, None):  # 2 retries with exponential backoff, then give up
        try:
            gtext = grader_call(key, prov, model, pr); met = parse_met(gtext)
            me = re.search(r'"explanation"\s*:\s*"(.*?)"', gtext, re.S)
            expl = (me.group(1) if me else gtext)[:240]
            return {"criterion": r["criterion"], "points": r["points"], "met": bool(met), "explanation": expl}
        except Exception as e:
            err = e
            if delay is None:
                print(f"  WARNING: grader failed on criterion {idx} after retries ({e}); excluding it from scoring",
                      file=sys.stderr)
            else:
                time.sleep(delay)
    return {"criterion": r["criterion"], "points": r["points"], "met": None,
            "error": str(err)[:240], "explanation": ""}

def parse_met(text):
    m = re.search(r'"criteria_met"\s*:\s*(true|false)', text, re.I)
    if m: return m.group(1).lower() == "true"
    try:
        return bool(json.loads(re.search(r"\{.*\}", text, re.S).group(0)).get("criteria_met"))
    except Exception:
        return False

def ask_model(messages, think, seed):
    txt = bc.chat(messages, think=think, max_tokens=4000 if think else 1200,
                  get_model=get_model, seed=seed, server_base=SERVER_BASE, timeout=1200)
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S | re.I).strip()

def main():
    args = bc.parse_run_args(prog="healthbench.py", bench_help="HealthBench subset: hard|consensus")
    subset = args.bench.replace("healthbench_", "")
    if subset not in URLS:
        sys.exit(f"unknown HealthBench subset {args.bench!r} (expected hard|consensus)")
    think, mode, tag, seed = args.think, args.mode, args.tag, args.seed
    key, prov, model = load_key()
    bench = "healthbench_" + subset
    path = os.path.join(DATA, f"healthbench_{subset}.jsonl")
    if not os.path.exists(path):
        print("downloading", subset, "..."); urllib.request.urlretrieve(URLS[subset], path)
    verify_dataset(path, subset)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    random.seed(seed); random.shuffle(rows)
    rows = rows if args.n == 0 else rows[:args.n]
    n = len(rows)
    print(f"grader={model} ({prov}) · {bench} · N={n} · {mode}")
    w = bc.RunWriter(bench, mode, tag, bc.KIND_RUBRIC)
    scores, t0 = [], time.time()
    gen_errors = grader_errors = ungradable = 0
    for i, ex in enumerate(rows, 1):
        q0 = time.time()
        msgs, rubrics = ex["prompt"], ex.get("rubrics", [])
        try: resp = ask_model(msgs, think, seed)
        except Exception as e: print("  gen error", e, file=sys.stderr); resp = ""
        if not resp.strip():
            # empty/errored generation: NOT graded (an API judge scoring silence wastes
            # grader calls and deflates) — excluded from the score denominator
            gen_errors += 1
            acc = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
            dt = round(time.time() - q0, 1)
            w.stream({"i": i, "n": n, "ok": False, "pred": "ERR", "gold": "rubric", "acc": acc,
                      "t": dt, "q": (msgs[-1]["content"] if msgs else "")[:260], "error": True,
                      "pred_txt": "generation failed/empty — not graded",
                      "gold_txt": f"{len(rubrics)} criteria", "ans": ""})
            w.detail({"i": i, "conversation": msgs, "response": resp, "error": True,
                      "error_msg": "generation failed or empty; example not graded"})
            w.live({"running": True, "i": i, "n": n, "accuracy": acc,
                    "errors": gen_errors + grader_errors,
                    "gen_errors": gen_errors, "grader_errors": grader_errors})
            print(f"  [{i}/{n}] GEN ERROR — example excluded from scoring")
            continue
        conv = "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + f"\n\nassistant: {resp}"
        # grade the rubric criteria in parallel; pool.map keeps original criterion order
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            crit = list(pool.map(lambda ir: grade_criterion(key, prov, model, conv, ir[0], ir[1]),
                                 enumerate(rubrics)))
        # criteria the grader could not score (met=None) are excluded from BOTH the
        # achieved and the achievable points, and counted in grader_errors
        failed = sum(1 for c in crit if c["met"] is None)
        grader_errors += failed
        total = sum(c["points"] for c in crit if c["met"] is not None and c["points"] > 0)
        achieved = sum(c["points"] for c in crit if c["met"])
        if failed and total == 0:
            # every achievable point sat on criteria the grader could not score —
            # the example is ungradable, exclude it from the denominator entirely
            ungradable += 1
            acc = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
            w.detail({"i": i, "conversation": msgs, "response": resp, "error": True,
                      "error_msg": f"grader failed on all {failed} scorable criteria; example not scored",
                      "criteria": crit})
            w.live({"running": True, "i": i, "n": n, "accuracy": acc,
                    "errors": gen_errors + grader_errors,
                    "gen_errors": gen_errors, "grader_errors": grader_errors})
            print(f"  [{i}/{n}] GRADER ERROR on every scorable criterion — example excluded")
            continue
        score = max(0.0, min(1.0, achieved / total)) if total else 0.0
        scores.append(score); acc = round(sum(scores) / len(scores) * 100, 1); dt = round(time.time() - q0, 1)
        w.stream({"i": i, "n": n, "ok": score >= 0.5, "pred": f"{score*100:.0f}%", "gold": "rubric", "acc": acc,
                  "t": dt, "q": (msgs[-1]["content"] if msgs else "")[:260],
                  "pred_txt": f"score {score*100:.0f}% ({achieved}/{total}pts)",
                  "gold_txt": f"{len(rubrics)} criteria", "ans": resp[:180]})
        w.detail({"i": i, "conversation": msgs, "response": resp,
                  "score": round(score * 100, 1), "achieved": achieved, "total": total, "criteria": crit})
        w.live({"running": True, "i": i, "n": n, "accuracy": acc,
                "errors": gen_errors + grader_errors,
                "gen_errors": gen_errors, "grader_errors": grader_errors})
        print(f"  [{i}/{n}] score {score*100:.0f}%  running {acc}%  ({dt}s)")
    final = round(sum(scores) / len(scores) * 100, 1) if scores else 0.0
    fields = {"n": len(scores), "accuracy": final, "seed": seed,
              "duration_s": round(time.time() - t0, 1),
              "sec_per_q": round((time.time() - t0) / max(1, len(scores)), 1),
              "grader": model, "max_tokens": 4000 if think else 1200,
              "errors": gen_errors + grader_errors,
              "gen_errors": gen_errors, "grader_errors": grader_errors,
              "notes": f"HealthBench {subset}, faithful API judge ({model})"}
    # mirror the MCQ/code runners: too many examples lost to generation failures or
    # fully-ungradable rubrics (>5%) means the mean covers a biased subset -> mark invalid
    excluded = gen_errors + ungradable
    if rows and excluded / len(rows) > 0.05:
        fields["invalid"] = True
        print(f"\n⚠⚠⚠ INVALID RUN: {excluded}/{len(rows)} examples failed or were ungradable (>5%) — "
              f"the score below covers only the {len(scores)} scored examples and is NOT "
              f"comparable to a clean run.", file=sys.stderr)
    w.finish(fields, get_model(), SERVER_BASE, path)
    w.live({"running": False})
    print(f"DONE — HealthBench {subset}: {final}%  (N={len(scores)}, grader={model}, "
          f"gen_errors={gen_errors}, grader_errors={grader_errors})")

if __name__ == "__main__":
    main()
