#!/usr/bin/env python3
"""Equivalence + resolver tests for the declarative registry (no network).

The key guarantee: the declarative `map` resolver produces output BYTE-IDENTICAL to the
hand-written normalizers it replaced — so existing content-locked sets don't drift.

Run:  python3 test_registry.py
"""
import json, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry, fetch_benchmarks as fb

L = "ABCDEFGHIJ"


# ---- reference implementations (the pre-refactor normalizers) ----
def _opts(vs):
    vs = [str(v).strip() for v in vs]
    return {L[i]: v for i, v in enumerate(vs)}

def ref_list_idx(q, c, a):
    def f(r):
        ch = r.get(c) or []
        try: ai = int(r.get(a))
        except Exception: return None
        if len(ch) < 2 or len(ch) > 10 or not (0 <= ai < len(ch)): return None
        return {"question": str(r[q]).strip(), "options": _opts(ch), "answer_idx": L[ai]}
    return f

def ref_label_text(q):
    def f(r):
        ch = r.get("choices") or {}; t = ch.get("text") or []; lb = ch.get("label") or []
        ak = r.get("answerKey")
        if not t or len(t) != len(lb) or len(t) > 10 or ak not in lb: return None
        return {"question": str(r[q]).strip(), "options": _opts(t), "answer_idx": L[lb.index(ak)]}
    return f

def ref_mmlu_pro(r):
    ops = r.get("options") or []
    try: ai = int(r.get("answer_index"))
    except Exception: return None
    if len(ops) < 2 or len(ops) > 10 or not (0 <= ai < len(ops)): return None
    return {"question": str(r["question"]).strip(), "options": _opts(ops), "answer_idx": L[ai]}

def ref_supergpqa(r):
    ops = r.get("options") or []; al = r.get("answer_letter")
    if len(ops) < 2 or len(ops) > 10: return None
    d = _opts(ops)
    if al not in d: return None
    return {"question": str(r["question"]).strip(), "options": d, "answer_idx": al}

def ref_dict_label(q, lk):
    def f(r):
        opts = r.get("options"); lab = r.get(lk)
        if not isinstance(opts, dict) or lab not in opts or len(opts) > 10: return None
        return {"question": str(r[q]).strip(), "options": {k: str(v) for k, v in opts.items()},
                "answer_idx": lab}
    return f

def ref_truthfulqa(r):
    mc = r.get("mc1_targets") or {}; ch = mc.get("choices") or []; lab = mc.get("labels") or []
    if not ch or len(ch) != len(lab) or len(ch) > 10 or 1 not in lab: return None
    return {"question": str(r["question"]).strip(), "options": _opts(ch), "answer_idx": L[lab.index(1)]}


# ---- synthetic rows matching each dataset's real shape ----
ROWS = {
    "logic":          ({"question": " Is p→q valid? ", "choices": ["yes", "no", "maybe", "n/a"], "answer": 2},
                       ref_list_idx("question", "choices", "answer")),
    "mmlu_cs":        ({"question": "Big-O of binary search?", "choices": ["O(1)", "O(log n)", "O(n)", "O(n^2)"], "answer": 1},
                       ref_list_idx("question", "choices", "answer")),
    "mmlu_medical":   ({"question": "The femur is a?", "choices": ["bone", "muscle", "nerve", "vein"], "answer": 0},
                       ref_list_idx("question", "choices", "answer")),
    "mmlu_pro":       ({"question": "Pick one", "options": ["a", "b", "c", "d", "e"], "answer_index": 3},
                       ref_mmlu_pro),
    "supergpqa":      ({"question": "Q", "options": ["w", "x", "y", "z"], "answer_letter": "C"},
                       ref_supergpqa),
    "medxpertqa":     ({"question": "Dx?", "options": {"A": "flu", "B": "cold", "C": "covid"}, "label": "B"},
                       ref_dict_label("question", "label")),
    "medqa_test":     ({"question": "Tx?", "options": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer_idx": "D"},
                       ref_dict_label("question", "answer_idx")),
    "arc_challenge":  ({"question": "Sky color?", "choices": {"text": ["blue", "red", "green"], "label": ["A", "B", "C"]}, "answerKey": "A"},
                       ref_label_text("question")),
    "commonsense_qa": ({"question": "Where keys?", "choices": {"text": ["pocket", "moon"], "label": ["A", "B"]}, "answerKey": "A"},
                       ref_label_text("question")),
    "openbookqa":     ({"question_stem": "Plants need?", "choices": {"text": ["sun", "gold"], "label": ["A", "B"]}, "answerKey": "A"},
                       ref_label_text("question_stem")),
    "truthfulqa":     ({"question": "Myth?", "mc1_targets": {"choices": ["true thing", "false thing"], "labels": [1, 0]}},
                       ref_truthfulqa),
}


class TestEquivalence(unittest.TestCase):
    def test_declarative_matches_reference(self):
        for key, (row, ref) in ROWS.items():
            norm = fb.REGISTRY[key]["norm"]
            got, want = norm(dict(row)), ref(dict(row))
            self.assertEqual(got, want, f"{key}: dict mismatch")
            # byte-identity: the on-disk jsonl must be reproducible bit-for-bit
            self.assertEqual(json.dumps(got), json.dumps(want), f"{key}: json bytes differ")

    def test_special_shapes(self):
        # winogrande: template fill-blank + pair options + 1/2 -> A/B
        w = fb.REGISTRY["winogrande"]["norm"]({"sentence": "The _ ran.", "option1": "cat", "option2": "dog", "answer": "2"})
        self.assertEqual(w, {"question": "Fill the blank ( _ ):\nThe _ ran.",
                             "options": {"A": "cat", "B": "dog"}, "answer_idx": "B"})
        # hellaswag: context + list endings + index
        h = fb.REGISTRY["hellaswag"]["norm"]({"ctx": "A man cooks.", "endings": ["He eats.", "He flies."], "label": 0})
        self.assertEqual(h["answer_idx"], "A")
        self.assertTrue(h["question"].endswith("A man cooks."))
        # medmcqa: opa..opd keys + cop index
        m = fb.REGISTRY["medmcqa"]["norm"]({"question": "q", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 2})
        self.assertEqual(m["answer_idx"], "C")
        self.assertEqual(m["options"], {"A": "a", "B": "b", "C": "c", "D": "d"})


class TestResolver(unittest.TestCase):
    def test_invalid_answer_index(self):
        f = registry.make_norm({"question": "q", "options": {"from": "list", "key": "c"},
                                "answer": {"from": "index", "key": "a"}})
        self.assertIsNone(f({"q": "x", "c": ["a", "b"], "a": 5}))   # out of range
        self.assertIsNone(f({"q": "x", "c": ["a"], "a": 0}))        # <2 options

    def test_letter_must_be_valid_option(self):
        f = registry.make_norm({"question": "q", "options": {"from": "list", "key": "c"},
                                "answer": {"from": "letter", "key": "a"}})
        self.assertIsNone(f({"q": "x", "c": ["a", "b"], "a": "Z"}))

    def test_dotted_path(self):
        self.assertEqual(registry._get({"x": {"y": 7}}, "x.y"), 7)
        self.assertIsNone(registry._get({"x": 1}, "x.y"))

    def test_fixed_and_map(self):
        f = registry.make_norm({"question": "q", "options": {"from": "fixed", "values": ["yes", "no", "maybe"]},
                                "answer": {"from": "map", "key": "fd", "map": {"yes": "A", "no": "B", "maybe": "C"}}})
        self.assertEqual(f({"q": "x", "fd": "maybe"})["answer_idx"], "C")
        self.assertIsNone(f({"q": "x", "fd": "huh"}))


class TestLoader(unittest.TestCase):
    def test_all_17_load_with_callable_norm(self):
        self.assertEqual(len(fb.REGISTRY), 17)
        for k, s in fb.REGISTRY.items():
            self.assertTrue(callable(s["norm"]), k)
            self.assertIn(s["fit"], ("mcq", "code"))
            self.assertTrue(s["parts"])

    def test_hooks_wired(self):
        for k in ("humaneval", "mbpp", "pubmedqa"):
            self.assertIs(fb.REGISTRY[k]["norm"], fb.HOOKS[k])


if __name__ == "__main__":
    unittest.main(verbosity=2)
