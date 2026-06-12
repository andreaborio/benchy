#!/usr/bin/env python3
"""Paired analysis of the 2026 A/B coding suite: each coder model vs the plain2bit baseline.

Evals are paired by construction (SEED=1234 -> same rows, order, option shuffle), so
per-question correctness compares directly. Significance: exact McNemar (two-sided binomial
on the discordant pairs). Writes a markdown verdict to stdout.

Usage: analyze.py [results_dir]   (default: ./results)
"""
import json, math, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
BASELINE = "plain2bit"
CODERS = ["coder-iq2", "coder-q4boost", "coder-q4boost-v2"]
# only the 2026 suite — keeps legacy/aborted runs and the void 16% LCB out of the analysis
# bigcodebench_heldout: pre-registered confirmatory set (REPORT.md §Pre-registration) —
# in SUITE so it gets its own paired row, deliberately NOT in CODE_BENCH/KNOWLEDGE_BENCH
# so no pooled statistic ever mixes held-out with seen rows.
SUITE = {"lcb_v6_func", "supergpqa_cs", "mmlu_pro_cs", "mbppplus", "bigcodebench",
         "lcb_v6_ext", "bigcodebench_ext", "bigcodebench_heldout"}
CODE_BENCH = {"lcb_v6_func", "bigcodebench", "mbppplus",
              "lcb_v6_ext", "bigcodebench_ext"}               # capability legs
KNOWLEDGE_BENCH = {"supergpqa_cs", "mmlu_pro_cs"}             # "imatrix tax" legs
PRIMARY = ("lcb_v6_func", "nothink")


def binom_two_sided(k, n):
    """Exact two-sided sign-test p-value on discordant pairs; X~Bin(n, 1/2)."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


def err_rate(rec):
    """Fraction of rows recorded as transport/server errors. Two formats coexist:
    this harness writes the error text into `answer` ("ERR:..."); benchy-dash details
    mark the row `error: true` (and carry no `ok`). Count both — for campaign records
    `error` never appears, so their err_rate is byte-identical to the old heuristic."""
    try:
        path = os.path.join(RESULTS, "details", rec.get("details", ""))
        rows = [json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            return 1.0
        return sum(1 for r in rows
                   if r.get("error") is True
                   or str(r.get("answer", "")).lstrip().startswith("ERR:")) / len(rows)
    except Exception:
        return 1.0


def load_runs():
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, "runs.jsonl")) if l.strip()]
    last = {}  # keep LAST CLEAN record per (tag,benchmark,mode) — a retry/fix supersedes a broken one
    for r in recs:
        if r.get("benchmark") not in SUITE or r.get("tag") not in [BASELINE] + CODERS:
            continue
        if err_rate(r) > 0.30:   # a server-broken run (e.g. all HTTP 500) never counts as a result
            continue
        last[(r.get("tag"), r.get("benchmark"), r.get("mode"))] = r
    return last


def load_details(rec):
    # benchy-dash error rows (`error: true`, no `ok` key) are skipped: the common-index
    # intersection below then drops that row from pairing on BOTH legs.
    path = os.path.join(RESULTS, "details", rec.get("details", ""))
    return {r["i"]: bool(r["ok"]) for r in (json.loads(l) for l in open(path) if l.strip())
            if r.get("error") is not True}


def main():
    runs = load_runs()
    benches = sorted({(b, m) for (t, b, m) in runs})
    out = ["# 2026 A/B coding suite — paired analysis\n"]

    out.append("## Accuracy per run\n")
    out.append("| benchmark | mode | " + " | ".join([BASELINE] + CODERS) + " |")
    out.append("|---|---|" + "---|" * (1 + len(CODERS)))
    for b, m in benches:
        cells = []
        for tag in [BASELINE] + CODERS:
            r = runs.get((tag, b, m))
            cells.append("%.1f%% (n=%d)" % (r["accuracy"], r["n"]) if r else "—")
        out.append("| %s | %s | %s |" % (b, m, " | ".join(cells)))

    out.append("\n## Paired McNemar vs %s (nothink)\n" % BASELINE)
    out.append("| benchmark | model | both ok | both fail | base only | model only | delta | p (exact) | sig |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    pooled_code = defaultdict(lambda: [0, 0])      # tag -> [base_only, model_only]
    pooled_know = defaultdict(lambda: [0, 0])
    primary_delta = {}
    for b, m in benches:
        if m != "nothink":
            continue
        base = runs.get((BASELINE, b, m))
        if not base:
            continue
        dbase = load_details(base)
        for tag in CODERS:
            r = runs.get((tag, b, m))
            if not r:
                continue
            d = load_details(r)
            common = sorted(set(dbase) & set(d))
            b_only = sum(1 for i in common if dbase[i] and not d[i])
            m_only = sum(1 for i in common if not dbase[i] and d[i])
            both = sum(1 for i in common if dbase[i] and d[i])
            both_fail = len(common) - both - b_only - m_only   # reporting only
            delta = 100.0 * (m_only - b_only) / max(1, len(common))
            p = binom_two_sided(m_only, b_only + m_only)
            sig = "**yes**" if p < 0.05 else "no"
            if b in CODE_BENCH:
                pooled_code[tag][0] += b_only; pooled_code[tag][1] += m_only
            if b in KNOWLEDGE_BENCH:
                pooled_know[tag][0] += b_only; pooled_know[tag][1] += m_only
            if (b, m) == PRIMARY:
                primary_delta[tag] = (delta, p)
            out.append("| %s | %s | %d | %d | %d | %d | %+.1fpt | %.3f | %s |"
                       % (b, tag, both, both_fail, b_only, m_only, delta, p, sig))

    def pooled_line(name, pooled):
        lines = []
        for tag, (bo, mo) in sorted(pooled.items()):
            p = binom_two_sided(mo, bo + mo)
            lines.append("- **%s**: discordant %d (model wins %d, baseline wins %d), p=%.4f %s"
                         % (tag, bo + mo, mo, bo, p, "(**sig**)" if p < 0.05 else ""))
        return ["\n## %s\n" % name] + (lines or ["- (no data yet)"])

    out += pooled_line("Pooled CODE capability (LCB+BCB+MBPP+)", pooled_code)
    out += pooled_line("Pooled KNOWLEDGE / imatrix-tax (SuperGPQA-CS+MMLU-Pro-CS)", pooled_know)

    out.append("\n## Verdict (decision rule)\n")
    out.append("Win = positive & significant LiveCodeBench delta AND no significant knowledge-tax.\n")
    for tag in CODERS:
        if tag not in primary_delta:
            out.append("- **%s**: pending (LiveCodeBench not yet run)" % tag)
            continue
        cd, cp = primary_delta[tag]
        bo, mo = pooled_know[tag]
        kp = binom_two_sided(mo, bo + mo)
        tax = (kp < 0.05 and mo < bo)
        win = (cd > 0 and cp < 0.05 and not tax)
        out.append("- **%s vs %s**: LCB %+.1fpt (p=%.3f); knowledge-tax=%s → **%s**"
                   % (tag, BASELINE, cd, cp, "YES" if tax else "no",
                      "SOLID WIN" if win else ("directional win, not significant" if cd > 0 else "no win")))

    print("\n".join(out))


if __name__ == "__main__":
    main()
