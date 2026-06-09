#!/usr/bin/env python3
"""Fetch well-known LLM benchmarks and normalise them to the unified MCQ format the eval
harness expects: {question, options{A..}, answer_idx}. Writes data/<name>.jsonl.

Uses the HuggingFace datasets-server JSON API (stdlib only — no pyarrow/datasets). Every
dataset id/config/split below is verified against that API. Each set keeps its OWN license
and source (see DATA.md); nothing is redistributed here.

Usage:
  python3 fetch_benchmarks.py              # list the registry + what's already in data/
  python3 fetch_benchmarks.py list
  python3 fetch_benchmarks.py all          # fetch everything (large)
  python3 fetch_benchmarks.py mmlu_pro arc_challenge medmcqa   # fetch specific sets

Note: true code-execution benchmarks (HumanEval, MBPP, LiveCodeBench, SWE-bench) are NOT
here — they need a code runner (generate → execute tests), which this MCQ/rubric harness
does not do. The "coding" coverage below is code/CS *knowledge & reasoning* in MCQ form.
Gated datasets (e.g. GPQA, HLE) require a HF login/token and are listed as `manual`.
"""
import json, os, sys, time, urllib.request, urllib.parse

import registry as _registry

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
BENCH_DIR = os.path.join(HERE, "benchmarks")
LETTERS = "ABCDEFGHIJ"

def api_rows(dataset, config, split, cap, revision=None):
    """Page rows out of the datasets-server API (100 at a time) up to `cap`.

    `revision` (a dataset commit sha or branch) is passed through for best-effort
    pinning; the datasets-server accepts it without erroring on unknown values, so
    integrity is additionally enforced by content hashing in api.py."""
    out, off = [], 0
    while off < cap:
        q = {"dataset": dataset, "config": config, "split": split, "offset": off, "length": 100}
        if revision: q["revision"] = revision
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(q)
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                d = json.load(r)
        except Exception as e:
            print("  ! fetch error", dataset, config, split, off, e); break
        rs = d.get("rows", [])
        if not rs: break
        out += [x["row"] for x in rs]
        off += 100
        if off >= d.get("num_rows_total", 0): break
        time.sleep(0.12)
    return out[:cap]

# ---- hooks: normalizers for the few sets the declarative `map` can't express — a code
# shape, or a context-join. Everything else is declarative in benchmarks/*.json, resolved
# by registry.make_norm(). Hooks are referenced by name from a benchmark's `hook` field.

def norm_pubmedqa(r):
    M = {"yes": "A", "no": "B", "maybe": "C"}
    fd = r.get("final_decision")
    if fd not in M: return None
    ctx = ""; c = r.get("context")
    if isinstance(c, dict) and c.get("contexts"): ctx = " ".join(c["contexts"])
    q = (ctx[:1500].rstrip() + "\n\nQuestion: " + r["question"]) if ctx else r["question"]
    return {"question": q, "answer_idx": M[fd], "options": {"A": "yes", "B": "no", "C": "maybe"}}

# code-generation tasks: a different shape — {task_id, prompt, tests, entry_point} — run by
# eval_code.py (generate code, execute against tests, score pass@1). NOT multiple-choice.

def norm_humaneval(r):
    ep = r.get("entry_point")
    if not r.get("prompt") or not r.get("test") or not ep: return None
    return {"task_id": r.get("task_id", ""), "prompt": r["prompt"],
            "tests": r["test"] + ("\n\ncheck(%s)\n" % ep), "entry_point": ep}

def norm_mbpp(r):
    tl = r.get("test_list") or []
    if not r.get("prompt") or not tl: return None
    imports = r.get("test_imports") or []
    tests = "\n".join(list(imports) + list(tl))
    shown = "\n".join(tl[:3])
    return {"task_id": str(r.get("task_id", "")),
            "prompt": r["prompt"].strip() + "\n\nThe function is tested with assertions like:\n" + shown,
            "tests": tests, "entry_point": ""}

# ---- registry: declarative, loaded from benchmarks/*.json. Adding a benchmark is a data
# PR (a JSON file); only the code-shape / context-join sets keep a Python hook here. ----
HOOKS = {"humaneval": norm_humaneval, "mbpp": norm_mbpp, "pubmedqa": norm_pubmedqa}
REGISTRY = _registry.load_benchmarks(BENCH_DIR, HOOKS)
# Gated / upstream-only / rubric sets — not fetched via the datasets-server API (see DATA.md):
MANUAL = {
    "gpqa":      "GPQA Diamond — graduate-level science MCQ; THE 2026 frontier science discriminator. Gated (Idavidrein/gpqa needs a HF token).",
    "hle":       "Humanity's Last Exam (2025) — hardest broad academic set; gated + mostly free-form/multimodal (poor fit, reference only).",
    "healthbench_hard": "HealthBench — open-ended, rubric-graded by an external judge; fetched & run by healthbench.py.",
}

def dataset_of(key):
    """The single upstream HF dataset id a benchmark key draws from (parts share it)."""
    return REGISTRY[key]["parts"][0][0]

def fetch(key, revision=None):
    spec = REGISTRY[key]; recs = []
    for dataset, config, split in spec["parts"]:
        part = api_rows(dataset, config, split, spec["cap"], revision=revision)
        kept = [rec for rec in (spec["norm"](r) for r in part) if rec]
        if len(spec["parts"]) > 1: print(f"    {config}: +{len(kept)}/{len(part)}")
        recs += kept
    if not recs:   # never leave a 0-byte file behind on a failed/empty fetch
        print(f"  ! {key} ({spec['name']}): 0 records (network error or empty response) — not writing a file")
        return 0
    os.makedirs(DATA, exist_ok=True)
    tmp = os.path.join(DATA, key + ".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    os.replace(tmp, os.path.join(DATA, key + ".jsonl"))   # atomic: no half-written file
    print(f"  {key} ({spec['name']}): {len(recs)} questions -> data/{key}.jsonl")
    return len(recs)

# manual / gated sets have no spec file; their one-line descriptions live here
MANUAL_DESC = {
    "gpqa": "Google-proof GRADUATE-level science (bio/chem/physics) MCQ — very hard, still discriminative. Gated on HF.",
    "hle": "Humanity's Last Exam (2025): expert questions at the frontier of human knowledge. Gated; partly multimodal.",
    "healthbench_hard": "Open-ended medical conversations graded against physician rubrics (run by healthbench.py).",
}

def present(key):
    return os.path.exists(os.path.join(DATA, key + ".jsonl"))

def fit_of(s):
    return s.get("fit", "mcq")

def current_keys():
    return [k for k, s in REGISTRY.items() if s.get("tier") == "current"]

def registry_meta():
    """Machine-readable registry for the dashboard's benchmark browser."""
    return {"available": [{"key": k, "name": s["name"], "domain": s["domain"], "desc": s.get("desc", ""),
                           "tier": s.get("tier", "current"), "fit": fit_of(s),
                           "license": s.get("license", ""), "present": present(k)}
                          for k, s in REGISTRY.items()],
            "manual": [{"key": k, "note": v, "desc": MANUAL_DESC.get(k, "")} for k, v in MANUAL.items()]}

def list_registry():
    print("Benchmarks — '✓' already in data/. Fit: [mcq] letter, [code] executed pass@1.\n")
    for tier, head in (("current", "CURRENT (still discriminating mid-2026)"),
                       ("legacy", "LEGACY / saturated (small-model regression only)")):
        print(f"  == {head} ==")
        for k, s in sorted(REGISTRY.items()):
            if s.get("tier", "current") != tier: continue
            print(f"    {'✓' if present(k) else ' '} {k:<14} [{fit_of(s)}] {s['name']:<22} {s.get('desc','')}")
        print()
    print("  == manual / gated (see DATA.md) ==")
    for k, note in MANUAL.items():
        print(f"      {k:<14} {note}")
    print("\nFetch with:  python3 fetch_benchmarks.py <key> [<key> ...]   |   all   |   current")

def main():
    args = sys.argv[1:]
    if not args or args == ["list"]:
        list_registry(); return
    keys = list(REGISTRY) if args == ["all"] else current_keys() if args == ["current"] else args
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print("unknown benchmark(s):", ", ".join(unknown), "\n"); list_registry(); sys.exit(2)
    for k in keys:
        try: fetch(k)
        except Exception as e: print(f"  ! {k} failed: {e}")
    print("done.")

if __name__ == "__main__":
    main()
