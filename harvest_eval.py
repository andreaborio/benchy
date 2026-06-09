#!/usr/bin/env python3
"""Harvest the eval runs into reusable training/calibration assets.

The eval harness already persists per-question detail under results/details/*.jsonl
(question, options, model prediction, gold, ok, answer). This script turns that
accumulated signal into:

  1. results/lora_hard_examples.jsonl  — the questions the model got WRONG, paired
     with the correct answer. Hard examples are the most valuable to fine-tune on.
  2. results/eval_corpus.txt (optional) — every evaluated question as plain text, ready
     to use as a calibration/importance-matrix corpus drawn from the exact target workload.

Run any time; it re-reads all detail files and dedups by question. No model needed.
Usage: harvest_eval.py
"""
import json, os, glob, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(HERE, "results", "details")
OUT_LORA = os.path.join(HERE, "results", "lora_hard_examples.jsonl")
OUT_CORPUS = os.path.join(HERE, "results", "eval_corpus.txt")

def bench_of(fn):  # details file name is "<benchmark>__<mode>__<ts>.jsonl"
    return os.path.basename(fn).split("__")[0]

def main():
    files = sorted(glob.glob(os.path.join(DET, "*.jsonl")))
    if not files:
        print("no detail files yet under results/details/ — run an eval first."); return
    seen, hard, corpus = set(), [], []
    per_bench = {}
    for fn in files:
        b = bench_of(fn)
        for line in open(fn):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            q = r.get("question", "")
            if not q: continue
            key = hashlib.md5((b + "|" + q).encode()).hexdigest()
            if key in seen: continue
            seen.add(key)
            corpus.append(q)
            # MCQ detail -> hard example when wrong
            if "options" in r and r.get("ok") is False:
                gold = r.get("gold"); opts = r.get("options", {})
                hard.append({"source": b, "question": q, "options": opts,
                             "correct": gold, "correct_text": opts.get(gold, ""),
                             "model_predicted": r.get("pred")})
                per_bench[b] = per_bench.get(b, 0) + 1

    with open(OUT_LORA, "w") as f:
        for h in hard: f.write(json.dumps(h) + "\n")
    with open(OUT_CORPUS, "w") as f:
        f.write("\n\n".join(corpus))

    print(f"detail files read: {len(files)} · unique questions: {len(seen)}")
    print(f"HARD (wrong) examples -> {OUT_LORA}: {len(hard)}")
    for b, n in sorted(per_bench.items(), key=lambda x: -x[1]):
        print(f"    {b}: {n} wrong")
    print(f"eval corpus (all questions) -> {OUT_CORPUS}: {len(corpus)} passages")
    print("\nThese feed: (a) fine-tuning on hard examples, and")
    print("(b) a calibration/importance-matrix corpus drawn from the eval workload.")

if __name__ == "__main__":
    main()
