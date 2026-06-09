#!/usr/bin/env python3
"""benchy.registry — declarative benchmark registry.

A benchmark is a JSON file under `benchmarks/` — no Python needed to add one. Each
declares where its rows come from and a `map` describing how to turn one source row
into benchy's normalized record `{question, options{A..}, answer_idx}` (MCQ) or, for
code sets, names a Python `hook`. A generic resolver turns the `map` into the same
normalizer the hand-written code used, so adding a benchmark is a data PR.

Schema (benchmarks/<key>.json):
{
  "name": "MMLU-Pro", "domain": "reasoning", "tier": "current",
  "license": "MIT (verify)", "desc": "one line shown in the UI", "cap": 1000,
  "source": [{"dataset": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "test"}],
  "map": {                       # OR  "hook": "humaneval"  (a Python normalizer)
    "question": "question",      # str key, or {key, context, template, context_join, context_limit}
    "options": {"from": "list", "key": "options"},
    "answer":  {"from": "index", "key": "answer_index"}
  }
}

options.from:  list (key->[..]) · dict (key->{A:..}) · labeled (key->{text:[],label:[]})
               · keys (["opa","opb",..]) · pair (["option1","option2"]) · fixed (["yes",..])
answer.from:   index (int->letter) · letter (already a letter) · answerKey (value in .label)
               · map (value->letter via {map}) · match (index where key==value)
Dotted keys ("mc1_targets.choices") index into nested objects.
"""
import json, os

LETTERS = "ABCDEFGHIJ"


def _get(row, path):
    """Dotted lookup: _get(r, 'mc1_targets.choices')."""
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _opts(values):
    vals = [str(v).strip() for v in values]
    if len(vals) < 2 or len(vals) > 10:
        return None
    return {LETTERS[i]: v for i, v in enumerate(vals)}


def _question(spec, row):
    q = spec.get("question", "question")
    if isinstance(q, str):
        v = _get(row, q)
        return str(v).strip() if v is not None else None
    # dict form: {key, context, template, context_join, context_limit}
    key = q.get("key")
    val = str(_get(row, key)).strip() if key and _get(row, key) is not None else ""
    ctx = ""
    for ck in ([q["context"]] if isinstance(q.get("context"), str) else q.get("context", [])):
        c = _get(row, ck)
        if c:
            ctx = (q.get("context_join", " ").join(c) if isinstance(c, list) else str(c)).strip()
            break
    if "context_limit" in q and ctx:
        ctx = ctx[:q["context_limit"]].rstrip()
    tmpl = q.get("template") if (ctx or "context" not in q) else q.get("template_no_context", "{question}")
    if not tmpl:
        return val or None
    out = tmpl.replace("{context}", ctx).replace("{question}", val)
    return out.strip() or None


def _options(spec, row):
    o = spec["options"]
    frm = o["from"]
    if frm == "list":
        return _opts(_get(row, o["key"]) or [])
    if frm == "dict":
        d = _get(row, o["key"])
        if not isinstance(d, dict) or not (2 <= len(d) <= 10):
            return None
        return {k: str(v) for k, v in d.items()}
    if frm == "labeled":
        ch = _get(row, o["key"]) or {}
        texts, labels = ch.get("text") or [], ch.get("label") or []
        if not texts or len(texts) != len(labels):
            return None
        d = _opts(texts)
        return (d, labels) if d else None      # labels returned for answerKey resolution
    if frm == "keys":
        vals = [_get(row, k) for k in o["keys"]]
        if any(v is None or v == "" for v in vals):
            return None
        return _opts(vals)
    if frm == "pair":
        vals = [_get(row, k) for k in o["keys"]]
        if any(v is None for v in vals):
            return None
        return _opts(vals)
    if frm == "fixed":
        return _opts(o["values"])
    return None


def _answer(spec, row, opts, labels):
    a = spec["answer"]
    frm = a["from"]
    n = len(opts)
    if frm == "index":
        try:
            ai = int(_get(row, a["key"]))
        except (TypeError, ValueError):
            return None
        return LETTERS[ai] if 0 <= ai < n else None
    if frm == "letter":
        v = _get(row, a["key"])
        return v if v in opts else None
    if frm == "answerKey":
        ak = _get(row, a["key"])
        return LETTERS[labels.index(ak)] if labels and ak in labels else None
    if frm == "map":
        return a["map"].get(str(_get(row, a["key"])))
    if frm == "match":
        seq = _get(row, a["key"]) or []
        want = a.get("value", 1)
        for i, v in enumerate(seq):
            if v == want:
                return LETTERS[i] if i < n else None
        return None
    return None


def make_norm(spec):
    """Build a row -> normalized-record function from a declarative `map`."""
    fit = spec.get("fit", "mcq")

    def f(row):
        opts = _options(spec, row)
        labels = []
        if isinstance(opts, tuple):       # labeled returns (opts, labels)
            opts, labels = opts
        if not opts:
            return None
        ans = _answer(spec, row, opts, labels)
        if ans not in opts:
            return None
        q = _question(spec, row)
        if not q:
            return None
        return {"question": q, "options": opts, "answer_idx": ans}

    f.fit = fit
    return f


def load_benchmarks(bench_dir, hooks):
    """Read benchmarks/*.json -> {key: spec_dict_with_norm}. `hooks` maps a hook name
    to a Python normalizer (for code/oddball sets). Raises on a malformed entry."""
    reg = {}
    if not os.path.isdir(bench_dir):
        return reg
    for fn in sorted(os.listdir(bench_dir)):
        if not fn.endswith(".json"):
            continue
        key = fn[:-5]
        spec = json.load(open(os.path.join(bench_dir, fn)))
        for req in ("name", "domain", "tier", "source"):
            if req not in spec:
                raise ValueError(f"benchmarks/{fn}: missing '{req}'")
        spec["key"] = key
        spec["parts"] = [(p["dataset"], p.get("config", "default"), p["split"]) for p in spec["source"]]
        spec["cap"] = spec.get("cap", 1000)
        if "hook" in spec:
            if spec["hook"] not in hooks:
                raise ValueError(f"benchmarks/{fn}: unknown hook '{spec['hook']}'")
            spec["norm"] = hooks[spec["hook"]]
            spec["fit"] = spec.get("fit", "code")
        elif "map" in spec:
            spec["norm"] = make_norm(spec["map"])
            spec["fit"] = spec.get("fit", "mcq")
        else:
            raise ValueError(f"benchmarks/{fn}: needs a 'map' or a 'hook'")
        reg[key] = spec
    return reg
