#!/usr/bin/env python3
"""Tests pinning the code that produces every published number — fully offline.

Covers the extracted stats core (benchy_stats: wilson / _mcnemar_p / _chi2_uniform /
Stats.paired_compare / Stats.summary), the MCQ scorer's pure pieces (eval_mcq: extract /
norm_gold / prepare option-shuffle) and the shared harness (benchy_common: parse_run_args /
settings / RunWriter / write_live). Golden values were computed independently (Wilson 95%
CI for 50/100, exact binomial McNemar tails) so a regression in any of them changes a
number someone may have published.

Run:  python3 test_scoring.py
"""
import contextlib, io, json, os, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchy_common as bc
import benchy_stats
import eval_mcq
from benchy_stats import Stats, wilson, _chi2_uniform, _chi2_hist, _mcnemar_p, is_rubric


def write_runs(results_dir, *recs):
    with open(os.path.join(results_dir, "runs.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")


def write_details(results_dir, fname, oks):
    """One details row per (question, ok) pair, in the runner's format."""
    dd = os.path.join(results_dir, "details")
    os.makedirs(dd, exist_ok=True)
    with open(os.path.join(dd, fname), "w", encoding="utf-8", newline="\n") as f:
        for q, ok in oks:
            f.write(json.dumps({"question": q, "ok": bool(ok), "pred": "A", "gold": "A"}) + "\n")


class TestWilson(unittest.TestCase):
    def test_golden_50_of_100(self):
        # textbook Wilson 95% CI for p̂=0.5, n=100: (40.38%, 59.62%)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 40.4, delta=0.1)
        self.assertAlmostEqual(hi, 59.6, delta=0.1)

    def test_edges(self):
        self.assertEqual(wilson(0, 0), (0.0, 0.0))      # no data -> degenerate, not a crash
        lo, hi = wilson(0, 50)                           # k=0: interval starts at exactly 0
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.0 < hi < 100.0)
        lo, hi = wilson(50, 50)                          # k=n: interval ends at exactly 100
        self.assertEqual(hi, 100.0)
        self.assertTrue(0.0 < lo < 100.0)

    def test_bounds_contain_point_estimate(self):
        for k, n in ((0, 1), (1, 1), (1, 2), (3, 7), (5, 50), (50, 100), (10, 1000), (999, 1000)):
            lo, hi = wilson(k, n)
            pt = 100.0 * k / n
            self.assertTrue(0.0 <= lo <= pt <= hi <= 100.0, "k=%d n=%d -> (%s, %s)" % (k, n, lo, hi))


class TestMcNemar(unittest.TestCase):
    def test_exact_binomial_golden(self):
        # b=5, c=15: two-sided exact binomial p = 2 * sum_{i<=5} C(20,i)/2^20 = 0.041389...
        self.assertAlmostEqual(_mcnemar_p(5, 15), 0.0414, delta=1e-3)

    def test_symmetry(self):
        self.assertEqual(_mcnemar_p(5, 15), _mcnemar_p(15, 5))
        self.assertEqual(_mcnemar_p(1, 6), _mcnemar_p(6, 1))

    def test_equal_discordants_p_is_one(self):
        self.assertEqual(_mcnemar_p(7, 7), 1.0)   # perfectly balanced -> no evidence at all
        self.assertEqual(_mcnemar_p(1, 1), 1.0)

    def test_no_discordants(self):
        self.assertEqual(_mcnemar_p(0, 0), 1.0)

    def test_large_sample_normal_branch(self):
        # b+c>2000 switches to the continuity-corrected normal approximation
        import math
        p = _mcnemar_p(1100, 1000)
        expected = math.erfc(((abs(1100 - 1000) - 1) / math.sqrt(2100)) / math.sqrt(2))
        self.assertAlmostEqual(p, expected, places=12)
        self.assertTrue(0.02 < p < 0.04)   # ≈0.0307 — agrees with the exact tail to ~1e-3

    def test_large_sample_clamped_to_one(self):
        # b≈c with b+c>2000: the continuity correction would push z negative and
        # erfc above 1.0 — the p-value must stay clamped to [0, 1]
        self.assertLessEqual(_mcnemar_p(1500, 1500), 1.0)
        self.assertLessEqual(_mcnemar_p(1500, 1501), 1.0)


class TestChi2Uniform(unittest.TestCase):
    def test_uniform_10_options_no_bias(self):
        # QW1 pin: 10-option benchmarks (mmlu_pro, medxpertqa) must use all 10 cells.
        # The old `"ABCDE"[:opts]` code silently dropped F..J: this exact uniform input
        # scored chi2=25 over 50 answers ("high" bias) instead of 0 over 100.
        dist = {c: 10 for c in "ABCDEFGHIJ"}
        self.assertEqual(_chi2_uniform(dist, 10), (0.0, 9, 100))

    def test_four_option_case_unchanged(self):
        self.assertEqual(_chi2_uniform({"A": 5, "B": 5, "C": 5, "D": 5}, 4), (0.0, 3, 20))

    def test_skewed_distribution_flags(self):
        chi2, dof, tot = _chi2_uniform({"A": 30, "B": 2, "C": 2, "D": 2}, 4)
        self.assertEqual((chi2, dof, tot), (65.33, 3, 36))
        self.assertGreater(chi2, 2 * dof)   # summary() severity threshold for "high"

    def test_empty_distribution(self):
        self.assertEqual(_chi2_uniform({}, 4), (0.0, 3, 0))


class TestChi2Hist(unittest.TestCase):
    """F1 pin: mixed-option benchmarks (truthfulqa, arc_challenge) take their no-bias
    expectation from the per-question option-count histogram, not a uniform 1/n_options —
    a row offering k options contributes 1/k expected count to each of its first k letters."""
    # 40 four-option + 20 five-option + 12 six-option rows (72 total):
    # expected counts A-D = 40/4 + 20/5 + 12/6 = 16 each, E = 20/5 + 12/6 = 6, F = 12/6 = 2
    HIST = {"4": 40, "5": 20, "6": 12}
    DIST = {"A": 16, "B": 16, "C": 16, "D": 16, "E": 6, "F": 2}   # exactly the expectation

    def test_unbiased_mixed_distribution_scores_zero(self):
        # letter counts drawn exactly proportional to the histogram expectation -> chi2 0,
        # dof = (letters with expected>0) - 1 = 5, n = 72 parsed answers
        self.assertEqual(_chi2_hist(self.DIST, self.HIST), (0.0, 5, 72))

    def test_uniform_formula_would_have_false_flagged(self):
        # the bug being fixed: the SAME unbiased distribution under the uniform formula
        # exceeds the summary() "high" threshold (truthfulqa-style guaranteed false positive)
        chi2, dof, tot = _chi2_uniform(self.DIST, 6)
        self.assertGreater(chi2, 2 * dof)

    def test_real_skew_still_flags(self):
        chi2, dof, tot = _chi2_hist({"A": 60, "B": 4, "C": 4, "D": 2, "E": 1, "F": 1}, self.HIST)
        self.assertEqual((dof, tot), (5, 72))
        self.assertGreater(chi2, 2 * dof)

    def test_unusable_hist_returns_none_for_fallback(self):
        self.assertIsNone(_chi2_hist(self.DIST, {}))
        self.assertIsNone(_chi2_hist(self.DIST, None))
        self.assertIsNone(_chi2_hist(self.DIST, {"x": 3}))        # corrupt key -> fall back
        self.assertIsNone(_chi2_hist(self.DIST, {"0": 5}))        # no usable bins

    def test_empty_distribution(self):
        self.assertEqual(_chi2_hist({}, self.HIST), (0.0, 5, 0))


class TestSummaryBias(unittest.TestCase):
    """Stats.summary over a synthetic runs.jsonl: Wilson CIs and the answer-bias table."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        write_runs(self.tmp.name,
            {"ts": "2026-06-10T10:00:00", "benchmark": "b10", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 100, "correct": 50, "accuracy": 50.0, "n_options": 10,
             "letter_dist": {c: 10 for c in "ABCDEFGHIJ"}},
            {"ts": "2026-06-10T10:01:00", "benchmark": "b4", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 36, "correct": 18, "accuracy": 50.0, "n_options": 4,
             "letter_dist": {"A": 30, "B": 2, "C": 2, "D": 2}},
            # legacy record: no n_options -> the option count is inferred from letter_dist
            {"ts": "2026-06-10T10:02:00", "benchmark": "binf", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 12, "correct": 6, "accuracy": 50.0,
             "letter_dist": {"A": 3, "B": 4, "C": 3, "D": 2, "?": 2}},
            {"ts": "2026-06-10T10:03:00", "benchmark": "healthbench_hard", "mode": bc.MODE_NOTHINK,
             "tag": "t1", "n": 10, "accuracy": 61.0},
            # mixed-option benchmark with opts_hist provenance (see TestChi2Hist for the math)
            {"ts": "2026-06-10T10:04:00", "benchmark": "bmix", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 72, "correct": 36, "accuracy": 50.0, "n_options": 6,
             "opts_hist": {"4": 40, "5": 20, "6": 12},
             "letter_dist": {"A": 16, "B": 16, "C": 16, "D": 16, "E": 6, "F": 2}})
        self.summary = Stats(self.tmp.name).summary()
        self.bias = {b["benchmark"]: b for b in self.summary["bias"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_ten_option_uniform_severity_ok(self):
        # QW1 pin at the published-table level: a perfectly uniform 10-letter spread is
        # NOT flagged as bias (the old 5-cell code reported chi2=25 over n=50, "high").
        b = self.bias["b10"]
        self.assertEqual((b["chi2"], b["dof"], b["n"], b["severity"]), (0.0, 9, 100, "ok"))

    def test_skew_flags_high(self):
        b = self.bias["b4"]
        self.assertEqual((b["dof"], b["severity"]), (3, "high"))

    def test_mixed_option_run_uses_hist_expectation(self):
        # F1 pin at the published-table level: with opts_hist present, an unbiased
        # mixed-option spread is NOT flagged (the uniform formula on the same dist gives
        # chi2 16.67 > 2*dof -> a guaranteed false "high" on truthfulqa-like benchmarks)
        b = self.bias["bmix"]
        self.assertEqual((b["chi2"], b["dof"], b["n"], b["severity"]), (0.0, 5, 72, "ok"))

    def test_question_mark_excluded_from_option_inference(self):
        # '?' marks unparseable answers, not an answer option: inferring the option count
        # from letter_dist must see 4 options (dof 3), not 5.
        b = self.bias["binf"]
        self.assertEqual(b["dof"], 3)
        self.assertEqual(b["n"], 12)   # the 2 '?' answers are not chi2 cells either

    def test_wilson_ci_in_per_run(self):
        row = next(x for x in self.summary["per_run"] if x["benchmark"] == "b10")
        self.assertEqual((row["ci_lo"], row["ci_hi"]), (40.4, 59.6))
        self.assertFalse(row["small_n"])

    def test_rubric_runs_get_no_binomial_ci(self):
        row = next(x for x in self.summary["per_run"] if x["benchmark"] == "healthbench_hard")
        self.assertIsNone(row["ci_lo"])
        self.assertIsNone(row["ci_hi"])
        self.assertNotIn("healthbench_hard", self.bias)   # no letter bias for rubric scores

    def test_is_rubric(self):
        self.assertTrue(is_rubric("healthbench_hard"))
        self.assertFalse(is_rubric("mmlu_pro"))
        self.assertFalse(is_rubric(""))
        self.assertFalse(is_rubric(None))


class TestSummaryValidity(unittest.TestCase):
    """F2 pins: summary() per_run rows carry the additive run-health fields ("invalid" bool,
    "mode_suspect" bool, "errors" int default 0), and best()/the macro average never prefer
    an invalid run over a valid one for the same benchmark/mode."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        write_runs(self.tmp.name,
            # larger but INVALID run vs smaller valid run — same benchmark/mode/tag
            {"ts": "t1", "benchmark": "vb", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 100, "correct": 90, "accuracy": 90.0, "invalid": True, "errors": 12},
            {"ts": "t2", "benchmark": "vb", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 50, "correct": 35, "accuracy": 70.0},
            # benchmark whose ONLY run is invalid -> still surfaced, flag kept for the UI
            {"ts": "t3", "benchmark": "oi", "mode": bc.MODE_NOTHINK, "tag": "t1",
             "n": 40, "correct": 20, "accuracy": 50.0, "invalid": True, "errors": 9},
            # a mode-suspect run (think flag ignored by the server)
            {"ts": "t4", "benchmark": "ms", "mode": bc.MODE_THINKING, "tag": "t1",
             "n": 30, "correct": 15, "accuracy": 50.0, "mode_suspect": True})
        self.summary = Stats(self.tmp.name).summary()

    def tearDown(self):
        self.tmp.cleanup()

    def test_per_run_flags_present_with_defaults(self):
        for row in self.summary["per_run"]:
            for key in ("invalid", "mode_suspect", "errors"):
                self.assertIn(key, row)
        clean = next(x for x in self.summary["per_run"] if x["benchmark"] == "vb" and x["n"] == 50)
        self.assertEqual((clean["invalid"], clean["mode_suspect"], clean["errors"]), (False, False, 0))
        bad = next(x for x in self.summary["per_run"] if x["benchmark"] == "vb" and x["n"] == 100)
        self.assertEqual((bad["invalid"], bad["errors"]), (True, 12))
        sus = next(x for x in self.summary["per_run"] if x["benchmark"] == "ms")
        self.assertTrue(sus["mode_suspect"])
        self.assertFalse(sus["invalid"])

    def test_best_prefers_smaller_valid_over_larger_invalid(self):
        cell = self.summary["by_benchmark"]["vb"]
        self.assertEqual((cell["best"]["n"], cell["best"]["accuracy"]), (50, 70.0))
        self.assertFalse(cell["best"]["invalid"])

    def test_best_falls_back_to_invalid_and_keeps_flag(self):
        cell = self.summary["by_benchmark"]["oi"]
        self.assertEqual(cell["best"]["n"], 40)
        self.assertTrue(cell["best"]["invalid"])   # flag survives so the UI can mark it

    def test_macro_prefers_valid_run(self):
        macro = self.summary["macro"]
        # nothink macro = mean(vb valid 70.0 — NOT the invalid 90.0 — and oi fallback 50.0)
        self.assertEqual(macro["nothink_mean"], 60.0)
        self.assertEqual(macro["nothink_k"], 2)


class TestMacroTagCoverage(unittest.TestCase):
    """Macro-average tag selection: a tag's coverage counts only DISTINCT benchmarks where
    it has at least one VALID run — breadth built out of invalid (>5% failed requests) runs
    must not win the macro slot. When NO tag has any valid run, the macro still degrades to
    the invalid runs (which stay flagged, as today) instead of emptying the panel."""

    def _macro(self, *recs):
        with tempfile.TemporaryDirectory() as tmp:
            write_runs(tmp, *recs)
            return Stats(tmp).summary()

    @staticmethod
    def _run(bench, tag, acc, **over):
        rec = {"ts": "t", "benchmark": bench, "mode": bc.MODE_NOTHINK, "tag": tag,
               "n": 50, "correct": int(round(acc / 2)), "accuracy": acc}
        rec.update(over)
        return rec

    def test_valid_coverage_beats_invalid_breadth(self):
        # tag B comes first in runs.jsonl with 4 runs on 4 benchmarks (the old run-count
        # heuristic scored it coverage 4 and picked it), but 2 of them are INVALID ->
        # real coverage 2. Tag A has 3 valid runs -> coverage 3 and must win the slot.
        s = self._macro(
            self._run("b1", "B", 80.0),
            self._run("b2", "B", 80.0),
            self._run("b3", "B", 90.0, invalid=True, errors=9),
            self._run("b4", "B", 90.0, invalid=True, errors=9),
            self._run("b1", "A", 60.0),
            self._run("b2", "A", 70.0),
            self._run("b3", "A", 80.0))
        macro = s["macro"]
        self.assertEqual(macro["nothink_tag"], "A")
        self.assertEqual(macro["nothink_k"], 3)
        self.assertEqual(macro["nothink_mean"], 70.0)   # mean(60, 70, 80) — no B run blended in

    def test_repeat_runs_on_same_benchmark_do_not_inflate_coverage(self):
        # coverage counts distinct benchmarks, not runs: 4 valid runs over 2 benchmarks
        # is coverage 2 and loses to 3 single-run benchmarks (old code: 4 > 3 -> R won)
        s = self._macro(
            self._run("b1", "R", 80.0), self._run("b1", "R", 82.0),
            self._run("b2", "R", 80.0), self._run("b2", "R", 84.0),
            self._run("b1", "W", 50.0), self._run("b2", "W", 60.0),
            self._run("b3", "W", 70.0))
        macro = s["macro"]
        self.assertEqual(macro["nothink_tag"], "W")
        self.assertEqual((macro["nothink_k"], macro["nothink_mean"]), (3, 60.0))

    def test_all_invalid_still_returns_macro_flagged(self):
        # NO tag has a valid nothink run: the panel degrades gracefully — the legacy
        # run-count heuristic picks a tag among the invalid runs and a macro IS returned,
        # while every surfaced row keeps its "invalid" flag exactly as today
        s = self._macro(
            self._run("b1", "X", 40.0, invalid=True, errors=5),
            self._run("b2", "X", 60.0, invalid=True, errors=5),
            self._run("b1", "Y", 50.0, invalid=True, errors=4))
        macro = s["macro"]
        self.assertEqual(macro["nothink_tag"], "X")     # 2 runs beat Y's 1 (legacy fallback)
        self.assertEqual(macro["nothink_k"], 2)
        self.assertEqual(macro["nothink_mean"], 50.0)   # mean(40, 60) — degraded, not empty
        self.assertTrue(all(r["invalid"] for r in s["per_run"]))      # nothing laundered
        self.assertTrue(s["by_benchmark"]["b1"]["best"]["invalid"])   # UI can still mark it


class TestPairedCompare(unittest.TestCase):
    """End-to-end McNemar verdicts over a synthetic runs.jsonl + details tree."""

    def _rec(self, tag, bench, details, **over):
        rec = {"ts": "2026-06-10T11:00:00", "tag": tag, "benchmark": bench, "mode": bc.MODE_NOTHINK,
               "n": 20, "accuracy": 50.0, "details": details, "shuffle_options": True,
               "data_sha": "feedfacecafe", "model": "m-q4", "errors": 0, "seed": 1234}
        rec.update(over)
        return rec

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.stats = Stats(d)
        qs = ["q%02d" % i for i in range(1, 21)]
        # known discordants: q01 A-only-right (b=1), q02..q07 B-only-right (c=6),
        # q08..q14 both right, q15..q20 both wrong -> exact p = 2*(C(7,0)+C(7,1))/2^7 = 0.125
        a_ok = {q: (q == "q01" or "q08" <= q <= "q14") for q in qs}
        b_ok = {q: ("q02" <= q <= "q14") for q in qs}
        write_details(d, "synth__nothink__a.jsonl", [(q, a_ok[q]) for q in qs])
        write_details(d, "synth__nothink__b.jsonl", [(q, b_ok[q]) for q in qs])
        four = [("p%d" % i, True) for i in range(4)]
        for fn in ("sh_a", "sh_b", "ds_a", "ds_b", "lg_a", "lg_b", "er_a", "er_b",
                   "se_a", "se_b", "su_a", "su_b"):
            write_details(d, fn + ".jsonl", four)
        write_runs(d,
            self._rec("qA", "synth", "synth__nothink__a.jsonl", accuracy=40.0),
            self._rec("qB", "synth", "synth__nothink__b.jsonl", accuracy=65.0),
            # hard mismatch: shuffle_options differs -> pairing by question is invalid
            self._rec("sA", "shuf", "sh_a.jsonl", shuffle_options=True),
            self._rec("sB", "shuf", "sh_b.jsonl", shuffle_options=False),
            # hard mismatch: different dataset snapshots
            self._rec("dA", "snap", "ds_a.jsonl", data_sha="aaaaaaaaaaaa"),
            self._rec("dB", "snap", "ds_b.jsonl", data_sha="bbbbbbbbbbbb"),
            # hard mismatch: both runs shuffled options but with DIFFERENT seeds -> the
            # per-question permutation (seeded by "seed|question") differed between them
            self._rec("zA", "seedshuf", "se_a.jsonl", seed=1234),
            self._rec("zB", "seedshuf", "se_b.jsonl", seed=7),
            # shuffle OFF with different seeds: only the row subsets differ, and the
            # question-text join handles the overlap -> soft warning, not a refusal
            self._rec("uA", "seedplain", "su_a.jsonl", shuffle_options=False, seed=1),
            self._rec("uB", "seedplain", "su_b.jsonl", shuffle_options=False, seed=2),
            # legacy records: no shuffle_options / data_sha provenance at all
            {"ts": "t", "tag": "lA", "benchmark": "lega", "mode": bc.MODE_NOTHINK, "n": 4,
             "accuracy": 100.0, "details": "lg_a.jsonl"},
            {"ts": "t", "tag": "lB", "benchmark": "lega", "mode": bc.MODE_NOTHINK, "n": 4,
             "accuracy": 100.0, "details": "lg_b.jsonl"},
            # one run had failed requests (excluded from its scoring)
            self._rec("eA", "errs", "er_a.jsonl", errors=2),
            self._rec("eB", "errs", "er_b.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_known_discordants_exact_verdict(self):
        v = self.stats.paired_compare("qA", "qB", "synth")
        self.assertTrue(v["ok"])
        self.assertEqual(v["n_common"], 20)
        self.assertEqual((v["a_better"], v["b_better"], v["discordant"]), (1, 6, 7))
        self.assertEqual((v["both_right"], v["both_wrong"]), (7, 6))
        self.assertEqual(v["p_value"], 0.125)
        self.assertFalse(v["significant"])
        self.assertEqual((v["acc_a"], v["acc_b"], v["delta"]), (40.0, 65.0, -25.0))
        self.assertNotIn("warnings", v)   # clean provenance -> no caveats

    def test_shuffle_mismatch_not_comparable(self):
        v = self.stats.paired_compare("sA", "sB", "shuf")
        self.assertFalse(v["ok"])
        self.assertIn("shuffle_options", v["error"])

    def test_seed_mismatch_with_shuffle_not_comparable(self):
        # F0 pin: two shuffled runs with different seeds presented different option orders
        # AND different row subsets — the question-by-question pairing is invalid
        v = self.stats.paired_compare("zA", "zB", "seedshuf")
        self.assertFalse(v["ok"])
        self.assertIn("seeds differ", v["error"])

    def test_seed_mismatch_without_shuffle_warns(self):
        # with shuffle off the option order is fixed, so differing seeds only change which
        # rows were sampled — comparable over the overlap, but flagged
        v = self.stats.paired_compare("uA", "uB", "seedplain")
        self.assertTrue(v["ok"])
        warns = " ".join(v.get("warnings", []))
        self.assertIn("seeds differ", warns)

    def test_seed_missing_is_legacy_warning(self):
        v = self.stats.paired_compare("lA", "lB", "lega")
        self.assertTrue(v["ok"])
        warns = " ".join(v.get("warnings", []))
        self.assertIn("seed provenance", warns)

    def test_data_sha_mismatch_not_comparable(self):
        v = self.stats.paired_compare("dA", "dB", "snap")
        self.assertFalse(v["ok"])
        self.assertIn("data_sha", v["error"])

    def test_legacy_runs_warn(self):
        v = self.stats.paired_compare("lA", "lB", "lega")
        self.assertTrue(v["ok"])
        warns = " ".join(v.get("warnings", []))
        self.assertIn("legacy", warns)
        self.assertIn("lA", warns)
        self.assertIn("lB", warns)

    def test_errors_warn(self):
        v = self.stats.paired_compare("eA", "eB", "errs")
        self.assertTrue(v["ok"])
        warns = " ".join(v.get("warnings", []))
        self.assertIn("eA", warns)
        self.assertIn("2 failed request(s)", warns)

    def test_refuses_rubric_same_tag_and_missing_runs(self):
        self.assertFalse(self.stats.paired_compare("qA", "qB", "healthbench_hard")["ok"])
        self.assertFalse(self.stats.paired_compare("qA", "qA", "synth")["ok"])
        self.assertFalse(self.stats.paired_compare("qA", "nosuch", "synth")["ok"])


ROW = {"question": "What is 2+2?", "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
       "answer_idx": "D"}


class TestExtract(unittest.TestCase):
    KEYS = ["A", "B", "C", "D"]

    def test_table(self):
        cases = [
            ("B", "B"),                                   # bare letter
            ("B.", "B"),                                  # trailing punctuation
            ("(B)", "B"),                                 # parenthesised
            ("**B**", "B"),                               # markdown-bold
            ("b", "B"),                                   # whole-reply lowercase is accepted
            ("Answer: B", "B"),                           # answer cue
            ("answer: b", "?"),                           # lowercase after a cue is NOT (anchoring)
            ("answer: B (not C)", "B"),                   # explicit cue beats a later isolated letter
            ("answer: A ... answer: B", "B"),             # last cue wins
            ("Let me think.\nFinal answer:\nB", "B"),     # final letter-only line
            ("<think>maybe C</think>B", "B"),             # think block stripped before parsing
            ("<think>B is right</think>", "?"),           # nothing outside the think block
            ("I would not pick any of these.", "?"),      # 'I' is prose, not an option letter
            ("The patient most likely has influenza.", "?"),  # full option TEXT alone never matches
            ("", "?"),
        ]
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(eval_mcq.extract(text, self.KEYS), want)

    def test_prose_articles_never_match(self):
        # the regression in the docstring: case-sensitive anchoring means prose like
        # "a banana ..." must NOT be scored as option A (the old \b[A-J]\b over
        # text.upper() scored every indefinite article as 'A').
        self.assertEqual(eval_mcq.extract("a banana is yellow.", self.KEYS), "?")
        self.assertEqual(eval_mcq.extract("a common cause is low sodium.", self.KEYS), "?")

    def test_letter_outside_keys_is_unparseable(self):
        self.assertEqual(eval_mcq.extract("C", ["A", "B"]), "?")


class TestNormGold(unittest.TestCase):
    def test_letter_and_integer_equivalence(self):
        self.assertEqual(eval_mcq.norm_gold({"answer_idx": "B"}, ["A", "B"]), "B")
        self.assertEqual(eval_mcq.norm_gold({"answer_idx": 1}, ["A", "B"]), "B")     # 0-based int
        self.assertEqual(eval_mcq.norm_gold({"answer_idx": "1"}, ["A", "B"]), "B")   # digit string
        self.assertEqual(eval_mcq.norm_gold({"answer_idx": "b"}, ["A", "B"]), "B")   # case-folded
        self.assertEqual(eval_mcq.norm_gold({"answer_idx": "B"}, []),
                         eval_mcq.norm_gold({"answer_idx": 1}, []))


class TestPrepare(unittest.TestCase):
    """Option-shuffle determinism: the permutation is hash-seeded from the question text,
    so every model/quant (and every process) sees the SAME order — the precondition for a
    valid paired comparison."""

    def setUp(self):
        self._shuffle, self._seed = eval_mcq.SHUFFLE, eval_mcq.SEED
        eval_mcq.SHUFFLE, eval_mcq.SEED = True, 1234

    def tearDown(self):
        eval_mcq.SHUFFLE, eval_mcq.SEED = self._shuffle, self._seed

    def test_import_is_offline_safe(self):
        self.assertIsNone(eval_mcq._MODEL)   # model resolution is lazy, never at import

    def test_deterministic_and_gold_tracked(self):
        q1 = eval_mcq.prepare(dict(ROW))
        q2 = eval_mcq.prepare(dict(ROW))
        self.assertEqual(q1, q2)                                   # same text -> same permutation
        question, opts, keys, gold, perm = q1
        self.assertEqual(perm, {"A": "C", "B": "D", "C": "A", "D": "B"})  # pinned (sha256 of seed|question)
        self.assertEqual(opts[gold], ROW["options"]["D"])          # gold remapped with its option
        for nl, ol in perm.items():                                # perm audits presented -> original
            self.assertEqual(opts[nl], ROW["options"][ol])
        self.assertEqual(sorted(perm.values()), keys)              # a real permutation of the keys

    def test_deterministic_across_processes(self):
        local = json.loads(json.dumps(list(eval_mcq.prepare(dict(ROW)))))
        script = ("import sys, json; sys.path.insert(0, %r); import eval_mcq; "
                  "eval_mcq.SHUFFLE = True; eval_mcq.SEED = 1234; "
                  "print(json.dumps(list(eval_mcq.prepare(json.loads(sys.stdin.read())))))" % HERE)
        out = subprocess.run([sys.executable, "-c", script], input=json.dumps(ROW),
                             capture_output=True, encoding="utf-8", timeout=60, cwd=HERE)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), local)

    def test_shuffle_off_is_identity(self):
        eval_mcq.SHUFFLE = False   # what BENCHY_SHUFFLE_OPTIONS=0 sets at import time
        question, opts, keys, gold, perm = eval_mcq.prepare(dict(ROW))
        self.assertEqual(perm, {})
        self.assertEqual(gold, "D")
        self.assertEqual(opts, ROW["options"])
        self.assertEqual(list(opts), ["A", "B", "C", "D"])   # original order preserved


class TestIterResultsCancellation(unittest.TestCase):
    """F13 pin: aborting a BENCHY_CONCURRENCY>1 eval (Ctrl-C / consumer exception closes the
    result generator) must CANCEL the queued questions — before the fix every submitted
    request still executed to completion. Fully offline: run_one is replaced by a counter."""

    def setUp(self):
        self._run_one, self._conc = eval_mcq.run_one, eval_mcq.CONCURRENCY

    def tearDown(self):
        eval_mcq.run_one, eval_mcq.CONCURRENCY = self._run_one, self._conc

    def test_close_cancels_queued_futures(self):
        import threading, time
        started, lock, release = [], threading.Lock(), threading.Event()

        def fake_run_one(i, r, think):
            with lock:
                started.append(i)
            if i != 0:
                release.wait(10)    # park every other in-flight worker; queue stays pending
            return {"i": i}

        eval_mcq.run_one = fake_run_one
        eval_mcq.CONCURRENCY = 2
        gen = eval_mcq.iter_results(list(range(50)), think=False)
        first = next(gen)           # consume exactly one result...
        self.assertEqual(first["i"], 0)
        gen.close()                 # ...then abort: GeneratorExit -> shutdown(cancel_futures=True)
        release.set()               # un-park the in-flight workers so they drain
        time.sleep(0.3)             # workers skip the cancelled queue and exit
        # only task bodies already in flight at close time ever ran: i=0, the parked worker's
        # task, and at most one task the freed worker grabbed before the cancel — never all 50
        self.assertLessEqual(len(started), 4, "queued futures were not cancelled: %d ran" % len(started))

    def test_full_consumption_unaffected(self):
        eval_mcq.run_one = lambda i, r, think: {"i": i}
        eval_mcq.CONCURRENCY = 3
        out = sorted(res["i"] for res in eval_mcq.iter_results(list(range(20)), think=False))
        self.assertEqual(out, list(range(20)))


class TestHarness(unittest.TestCase):
    def test_parse_run_args_maps_think_to_canonical_mode(self):
        a = bc.parse_run_args(["bench.jsonl", "10", "think", "tagx"])
        self.assertEqual(a.mode, bc.MODE_THINKING)
        self.assertTrue(a.think)
        self.assertEqual(a.seed, 1234)
        b = bc.parse_run_args(["bench.jsonl", "0", "nothink", "tagy", "--seed", "7"])
        self.assertEqual(b.mode, bc.MODE_NOTHINK)
        self.assertFalse(b.think)
        self.assertEqual(b.seed, 7)

    def test_parse_run_args_rejects_fifth_positional(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bc.parse_run_args(["bench.jsonl", "10", "think", "tag", "extra"])

    def test_settings_precedence(self):
        # env BENCHY_SERVER > config.json server_base > DEFAULT_SERVER
        tmp = tempfile.TemporaryDirectory()
        old_here, old_env = bc.HERE, os.environ.get("BENCHY_SERVER")
        bc.HERE = tmp.name
        try:
            os.environ["BENCHY_SERVER"] = "http://10.9.8.7:1234/"
            self.assertEqual(bc.settings()["server_base"], "http://10.9.8.7:1234")
            os.environ.pop("BENCHY_SERVER")
            with open(os.path.join(tmp.name, "config.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"server_base": "http://cfg-host:81/"}, f)
            self.assertEqual(bc.settings()["server_base"], "http://cfg-host:81")
            os.remove(os.path.join(tmp.name, "config.json"))
            self.assertEqual(bc.settings()["server_base"], bc.DEFAULT_SERVER)
        finally:
            bc.HERE = old_here
            if old_env is not None:
                os.environ["BENCHY_SERVER"] = old_env
            else:
                os.environ.pop("BENCHY_SERVER", None)
            tmp.cleanup()

    def test_runwriter_finish_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = os.path.join(tmp, "set.jsonl")
            with open(ds, "w", encoding="utf-8", newline="\n") as f:
                f.write('{"question":"q","options":{"A":"x","B":"y"},"answer_idx":"A"}\n')
            w = bc.RunWriter("synthb", bc.MODE_NOTHINK, "t1", bc.KIND_MCQ, results_dir=tmp)
            rec = w.finish({"n": 4, "correct": 2, "accuracy": 50.0}, "model-x", "http://s:1", ds)
            self.assertRegex(rec["ts"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")  # isoformat, seconds
            self.assertEqual(rec["kind"], bc.KIND_MCQ)
            self.assertEqual(rec["errors"], 0)                       # defaulted, not omitted
            self.assertEqual(rec["details"], os.path.basename(w.details_path))
            self.assertNotIn(os.sep, rec["details"])                 # basename only
            self.assertEqual(rec["model"], "model-x")
            self.assertEqual(rec["benchy_version"], bc.__version__)
            self.assertEqual(rec["data_sha"], bc.sha256_file(ds)[:12])  # lock-prefix-comparable
            rows = [json.loads(l) for l in open(os.path.join(tmp, "runs.jsonl"), encoding="utf-8")]
            self.assertEqual(rows, [rec])                            # appended verbatim

    def test_write_live_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "live.json")
            bc.write_live(p, {"running": True, "i": 3})
            self.assertEqual(json.load(open(p, encoding="utf-8")), {"running": True, "i": 3})
            self.assertFalse(os.path.exists(p + ".tmp"))             # tmp file consumed by os.replace


if __name__ == "__main__":
    unittest.main(verbosity=2)
