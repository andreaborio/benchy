#!/usr/bin/env python3
"""benchy_stats — the statistics core behind the dashboard's /api/summary and
/api/compare routes, extracted so it is importable headless: importing this module
starts no servers, reads no files and touches no network. Pure helpers (wilson,
_chi2_uniform, _mcnemar_p, is_rubric, read_jsonl) are module-level; everything that
touches run artifacts lives on Stats(results_dir), so the whole module is testable
against a tempdir. Stdlib only, no third-party deps."""
import os, json, math, collections
import benchy_common as bc   # canonical MODE_THINKING / MODE_NOTHINK tokens (import-time safe)


def read_jsonl(path):
    if not os.path.exists(path): return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a proportion; returns (lo%, hi%)."""
    if not n: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    centre = (p + z*z/(2*n)) / d
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (round(max(0.0, centre-half)*100, 1), round(min(1.0, centre+half)*100, 1))

def _chi2_uniform(dist, opts):
    """χ² of a predicted-letter distribution vs uniform; returns (chi2, dof, total).
    Only valid when every question offers the same number of options — mixed-option
    benchmarks (truthfulqa, arc_challenge) must use _chi2_hist instead."""
    cells = [dist.get(c, 0) for c in "ABCDEFGHIJ"[:opts]]   # up to 10 options (mmlu_pro, medxpertqa)
    tot = sum(cells)
    if not tot: return (0.0, opts-1, 0)
    exp = tot / opts
    return (round(sum((c-exp)**2/exp for c in cells), 2), opts-1, tot)

def _chi2_hist(dist, opts_hist):
    """χ² of a predicted-letter distribution vs the NO-BIAS expectation for a benchmark whose
    questions offer a MIXED number of options (opts_hist = {str(k): n_rows offering k options}).
    A row with k options contributes 1/k of an expected answer to each of its first k letters,
    so the expected count for letter L (index i) is sum over k >= i+1 of opts_hist[k]/k —
    later letters, offered only by long questions, correctly expect fewer picks than A-D
    (the uniform formula mis-scores an unbiased model on truthfulqa as chi2≈673 'high').
    Expectations are rescaled to the observed parsed-letter total (mirrors _chi2_uniform,
    whose tot/opts expectation is likewise built from the parsed total, excluding '?').
    dof = (number of letters with expected > 0) - 1.
    Returns (chi2, dof, total), or None when the histogram is unusable -> caller falls back
    to the uniform formula."""
    hist = {}
    for k, n in (opts_hist or {}).items():
        try:
            k, n = int(k), int(n)
        except (TypeError, ValueError):
            return None
        if k > 0 and n > 0:
            hist[k] = hist.get(k, 0) + n
    if not hist:
        return None
    letters = "ABCDEFGHIJ"[:max(hist)]                       # capped at 10 (LETTERS)
    raw = [sum(n / k for k, n in hist.items() if k >= i + 1) for i in range(len(letters))]
    raw_tot = sum(raw)                                       # == number of scored rows
    dof = max(1, sum(1 for e in raw if e > 0) - 1)
    cells = [dist.get(c, 0) for c in letters]
    tot = sum(cells)
    if not tot or not raw_tot:
        return (0.0, dof, tot)
    chi2 = sum((c - tot * r / raw_tot) ** 2 / (tot * r / raw_tot)
               for c, r in zip(cells, raw) if r > 0)
    return (round(chi2, 2), dof, tot)

def is_rubric(b):
    """Rubric-graded (open-ended) benchmarks score a 0-100 rubric mean, not k/n correct,
    so they get no binomial CI / letter-bias and are excluded from the MCQ macro-average."""
    return bool(b) and b.startswith("healthbench")

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
    z = max(0.0, abs(b - c) - 1) / math.sqrt(n)
    return min(1.0, math.erfc(z / math.sqrt(2)))


class Stats:
    """Run statistics over ONE results directory (runs.jsonl + details/*.jsonl).
    The dashboard binds Stats(RESULTS); tests bind a tempdir. Holds no open files,
    no caches — every call re-reads, exactly like the pre-split dashboard did."""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.runs_path = os.path.join(results_dir, "runs.jsonl")
        self.details_dir = os.path.join(results_dir, "details")

    def runs(self):
        """All summary records from runs.jsonl (missing file -> [])."""
        return read_jsonl(self.runs_path)

    def _details_for(self, tag, bench, mode=None):
        """The most-complete details file (largest n) for a (tag, benchmark[, mode]) run, parsed
        into {question -> correct?}. Joining on question text lets two runs of different N be
        compared on their overlapping questions."""
        best = None
        for r in self.runs():
            if r.get("tag") != tag or r.get("benchmark") != bench: continue
            if mode and r.get("mode") != mode: continue
            if not r.get("details"): continue
            if best is None or (r.get("n") or 0) > (best.get("n") or 0):
                best = r
        if not best:
            return None, {}
        fp = os.path.join(self.details_dir, os.path.basename(best["details"]))
        by_q = {}
        try:
            for line in open(fp, encoding="utf-8"):
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                q = str(d.get("question", "")).strip()
                if q and "ok" in d:
                    by_q[q] = bool(d["ok"])
        except Exception:
            return best, {}
        return best, by_q

    def paired_compare(self, tag_a, tag_b, bench, mode=None):
        """McNemar paired comparison of two runs on the SAME benchmark over their COMMON questions —
        the statistically correct test for 'does quant A differ from quant B', far more powerful than
        eyeballing two independent Wilson intervals (it uses the per-question pairing)."""
        if not tag_a or not tag_b or not bench:
            return {"ok": False, "error": "pick two run tags and a benchmark"}
        if tag_a == tag_b:
            return {"ok": False, "error": "pick two different tags"}
        if is_rubric(bench):
            return {"ok": False, "error": "paired test applies to MCQ/code (pass/fail) benchmarks, not rubric-graded ones"}
        ra, qa = self._details_for(tag_a, bench, mode)
        rb, qb = self._details_for(tag_b, bench, mode)
        if not ra or not rb:
            return {"ok": False, "error": "need a completed run with per-question details for both tags on this benchmark"}
        # comparability invariants — hard mismatches make the question-by-question pairing invalid
        sh_a, sh_b = ra.get("shuffle_options"), rb.get("shuffle_options")
        if sh_a is not None and sh_b is not None and bool(sh_a) != bool(sh_b):
            return {"ok": False, "error": "runs not comparable: shuffle_options differs (%s shuffled=%s, %s shuffled=%s)"
                    % (tag_a, bool(sh_a), tag_b, bool(sh_b))}
        ds_a, ds_b = ra.get("data_sha"), rb.get("data_sha")
        if ds_a and ds_b and ds_a != ds_b:
            return {"ok": False, "error": "runs not comparable: different dataset snapshots (data_sha %s vs %s)" % (ds_a, ds_b)}
        # the run seed drives BOTH the row subset (load() shuffles rows with it) and — with
        # option-shuffle on — the per-question option permutation (seeded by "seed|question").
        # Shuffled runs with different seeds presented DIFFERENT option orders, so the
        # question-by-question pairing is invalid: hard mismatch. With shuffle off, different
        # seeds only change the row subsets, and the question-text join below handles that ->
        # soft warning. A missing seed is legacy provenance -> soft warning (below).
        seed_a, seed_b = ra.get("seed"), rb.get("seed")
        if seed_a is not None and seed_b is not None and seed_a != seed_b \
                and bool(sh_a) and bool(sh_b):
            return {"ok": False, "error": "runs not comparable: option-shuffle seeds differ "
                    "(seed %s vs %s) — the two runs presented different option orders" % (seed_a, seed_b)}
        # soft mismatches — pairing still works, but flag them so the verdict is read with care
        warnings = []
        if seed_a is not None and seed_b is not None and seed_a != seed_b:
            warnings.append("seeds differ (%s vs %s): the runs sampled different question subsets; "
                            "the comparison covers only their overlap" % (seed_a, seed_b))
        if ra.get("mode") != rb.get("mode"):
            warnings.append("modes differ (%s vs %s)" % (ra.get("mode") or "?", rb.get("mode") or "?"))
        if (ra.get("model") or rb.get("model")) and ra.get("model") != rb.get("model"):
            warnings.append("model ids differ (%s vs %s)" % (ra.get("model") or "?", rb.get("model") or "?"))
        for tag, rec in ((tag_a, ra), (tag_b, rb)):
            if rec.get("shuffle_options") is None or not rec.get("data_sha"):
                warnings.append("%s is a legacy run missing shuffle_options/data_sha provenance" % tag)
            if rec.get("seed") is None:
                warnings.append("%s is a legacy run missing seed provenance — option-shuffle "
                                "comparability cannot be verified" % tag)
            if rec.get("invalid"):
                warnings.append("%s is marked INVALID (>5%% of its requests failed) — its accuracy is not comparable" % tag)
            elif rec.get("errors"):
                warnings.append("%s had %s failed request(s), excluded from its scoring" % (tag, rec.get("errors")))
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
        out = {"ok": True, "tag_a": tag_a, "tag_b": tag_b, "benchmark": bench, "mode": mode,
               "n_common": nc, "n_a": ra.get("n"), "n_b": rb.get("n"),
               "n_min": min(ra.get("n") or 0, rb.get("n") or 0),   # coverage: n_common vs the smaller run
               "acc_a": acc_a, "acc_b": acc_b, "delta": round(acc_a - acc_b, 1),
               "a_better": a_better, "b_better": b_better, "discordant": a_better + b_better,
               "both_right": both_right, "both_wrong": both_wrong,
               "p_value": round(p, 4), "significant": bool(p < 0.05)}
        if warnings: out["warnings"] = warnings
        return out

    def summary(self, refs_all=None, opts_map=None, primary_label=None):
        """Server-side stats over runs.jsonl x optional reference baselines: Wilson CIs,
        gaps, macro-avg, answer-bias. `refs_all` ({benchmark -> [baseline, ...]}), `opts_map`
        ({benchmark -> n_options}) and `primary_label` come from the caller's config."""
        refs_all = refs_all or {}
        opts_map = opts_map or {}
        runs = self.runs()
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
            try:
                errs = int(r.get("errors") or 0)
            except (TypeError, ValueError):
                errs = 0
            # additive run-health fields (default False/0 for legacy records): the UI marks
            # invalid (>5% failed requests) and mode-suspect runs, and best()/macro below
            # never prefer an invalid run over a valid one
            per_run.append({"ts": r.get("ts"), "benchmark": b, "mode": r.get("mode"), "tag": r.get("tag"),
                            "n": n, "accuracy": acc, "ci_lo": lo, "ci_hi": hi, "small_n": n < 50,
                            "invalid": bool(r.get("invalid")), "mode_suspect": bool(r.get("mode_suspect")),
                            "errors": errs})
            ld = r.get("letter_dist")
            if ld:
                # mixed-option benchmarks record opts_hist ({str(k): rows offering k options});
                # use the histogram expectation, falling back to uniform for legacy records
                res = _chi2_hist(ld, r.get("opts_hist")) if r.get("opts_hist") else None
                if res is None:
                    n_opts = r.get("n_options") or opts_map.get(b) or len([k for k in ld if ld[k] and k != "?"]) or 4   # "?" = unparseable, not an option
                    res = _chi2_uniform(ld, n_opts)
                chi2, dof, tot = res
                sev = "high" if chi2 > 2*dof else "some" if chi2 > dof else "ok"
                bias.append({"ts": r.get("ts"), "benchmark": b, "mode": r.get("mode"), "dist": ld,
                             "chi2": chi2, "dof": dof, "n": tot, "severity": sev})
        by = {}
        for b in benches:
            ours = [x for x in per_run if x["benchmark"] == b]
            def best(mode):
                # largest VALID run for this benchmark/mode; an invalid run (>5% failed
                # requests) is only surfaced when NO valid run exists — and then it keeps
                # its "invalid": True flag so the UI can mark it
                c = [x for x in ours if x["mode"] == mode]
                if not c: return None
                valid = [x for x in c if not x.get("invalid")]
                return max(valid or c, key=lambda x: x["n"])
            th, no = best(bc.MODE_THINKING), best(bc.MODE_NOTHINK)
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
        # A tag's coverage counts DISTINCT benchmarks where it has at least one VALID run —
        # breadth built out of invalid (>5% failed requests) runs must not win the macro
        # slot over a narrower tag whose runs are actually scoreable. Ties break as before:
        # most_common is stable, so the tag seen first in runs.jsonl order wins.
        def macro_for(mode):
            rows = [x for x in per_run if x["mode"] == mode and not is_rubric(x["benchmark"])]
            if not rows: return None, 0, None
            cov, counted = collections.Counter(), set()
            for x in rows:
                if x.get("invalid"): continue
                key = (x["tag"], x["benchmark"])
                if key not in counted:
                    counted.add(key)
                    cov[x["tag"]] += 1
            if not cov:
                # NO tag has a single valid run in this mode: degrade to the legacy
                # run-count heuristic over the invalid runs so the macro panel still
                # shows something (every surfaced row keeps its "invalid" flag) rather
                # than going empty
                cov = collections.Counter(x["tag"] for x in rows)
            tag = cov.most_common(1)[0][0]
            accs = []
            for b in benches:
                if is_rubric(b): continue
                cands = [x for x in rows if x["benchmark"] == b and x["tag"] == tag]
                if cands:
                    # same validity preference as best(): never blend an invalid run into the
                    # macro average while a valid one exists for this benchmark/mode/tag
                    valid = [x for x in cands if not x.get("invalid")]
                    accs.append(max(valid or cands, key=lambda x: x["n"])["accuracy"])
            return (round(sum(accs)/len(accs), 1) if accs else None, len(accs), tag)
        tm_mean, tm_k, tm_tag = macro_for(bc.MODE_THINKING)
        nm_mean, nm_k, nm_tag = macro_for(bc.MODE_NOTHINK)
        macro = {"thinking_mean": tm_mean, "thinking_k": tm_k, "thinking_tag": tm_tag,
                 "nothink_mean": nm_mean, "nothink_k": nm_k, "nothink_tag": nm_tag}
        return {"benchmarks": benches, "per_run": per_run, "by_benchmark": by,
                "macro": macro, "bias": bias, "primary_ref": primary_label, "opts": opts_map}
