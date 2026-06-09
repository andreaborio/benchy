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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
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

# ---- normalisers: each maps one source row -> {question, options{A..}, answer_idx} or None ----

def _opts(values):
    """Build an options dict {A: v0, B: v1, ...} from an ordered list (max 10)."""
    vals = [str(v).strip() for v in values]
    return {LETTERS[i]: v for i, v in enumerate(vals)}

def norm_list_idx(qkey, ckey, akey):
    """choices = list, answer = int index (e.g. cais/mmlu, hellaswag-style)."""
    def f(r):
        ch = r.get(ckey) or []
        try: ai = int(r.get(akey))
        except Exception: return None
        if len(ch) < 2 or len(ch) > 10 or not (0 <= ai < len(ch)): return None
        return {"question": str(r[qkey]).strip(), "options": _opts(ch), "answer_idx": LETTERS[ai]}
    return f

def norm_label_text(qkey):
    """choices = {text:[...], label:[...]}, answerKey = a label (ARC / OpenBookQA / CSQA)."""
    def f(r):
        ch = r.get("choices") or {}; texts = ch.get("text") or []; labels = ch.get("label") or []
        ak = r.get("answerKey")
        if not texts or len(texts) != len(labels) or len(texts) > 10 or ak not in labels: return None
        return {"question": str(r[qkey]).strip(), "options": _opts(texts),
                "answer_idx": LETTERS[labels.index(ak)]}
    return f

def norm_mmlu_pro(r):
    ops = r.get("options") or []
    try: ai = int(r.get("answer_index"))
    except Exception: return None
    if len(ops) < 2 or len(ops) > 10 or not (0 <= ai < len(ops)): return None
    return {"question": str(r["question"]).strip(), "options": _opts(ops), "answer_idx": LETTERS[ai]}

def norm_hellaswag(r):
    ends = r.get("endings") or []
    try: ai = int(r.get("label"))
    except Exception: return None
    if len(ends) < 2 or len(ends) > 10 or not (0 <= ai < len(ends)): return None
    ctx = (r.get("ctx") or r.get("ctx_a") or "").strip()
    return {"question": "Choose the most plausible continuation:\n" + ctx,
            "options": _opts(ends), "answer_idx": LETTERS[ai]}

def norm_truthfulqa(r):
    mc = r.get("mc1_targets") or {}; ch = mc.get("choices") or []; lab = mc.get("labels") or []
    if not ch or len(ch) != len(lab) or len(ch) > 10 or 1 not in lab: return None
    return {"question": str(r["question"]).strip(), "options": _opts(ch),
            "answer_idx": LETTERS[lab.index(1)]}

def norm_winogrande(r):
    a = str(r.get("answer") or "")
    if a not in ("1", "2"): return None
    return {"question": "Fill the blank ( _ ):\n" + str(r["sentence"]).strip(),
            "options": {"A": str(r["option1"]), "B": str(r["option2"])},
            "answer_idx": "A" if a == "1" else "B"}

def norm_medmcqa(r):
    if r.get("cop") is None or not all(r.get(k) for k in ("opa", "opb", "opc", "opd")): return None
    return {"question": str(r["question"]).strip(), "answer_idx": "ABCD"[int(r["cop"])],
            "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}}

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

def norm_dict_label(qkey, lkey):
    """options is already an A.. dict, answer is a letter (MedXpertQA / MedQA)."""
    def f(r):
        opts = r.get("options"); lab = r.get(lkey)
        if not isinstance(opts, dict) or lab not in opts or len(opts) > 10: return None
        return {"question": str(r[qkey]).strip(), "options": {k: str(v) for k, v in opts.items()},
                "answer_idx": lab}
    return f

def norm_supergpqa(r):
    ops = r.get("options") or []; al = r.get("answer_letter")
    if len(ops) < 2 or len(ops) > 10: return None
    d = _opts(ops)
    if al not in d: return None
    return {"question": str(r["question"]).strip(), "options": d, "answer_idx": al}

# ---- registry: verified dataset/config/split. parts can merge several sources into one set.
# tier "current" = still discriminates mid-2026 models; "legacy" = saturated (small-model
# regression/sanity only). All verified live on the HF datasets-server (2026-06). ----
REGISTRY = {
    # ===== current (still discriminating) =====
    "mmlu_pro":       {"name": "MMLU-Pro", "domain": "reasoning", "tier": "current", "license": "MIT (verify)",
                       "parts": [("TIGER-Lab/MMLU-Pro", "default", "test")], "cap": 1000, "norm": norm_mmlu_pro},
    "supergpqa":      {"name": "SuperGPQA", "domain": "reasoning", "tier": "current", "license": "ODC-BY (verify)",
                       "parts": [("m-a-p/SuperGPQA", "default", "train")], "cap": 1500, "norm": norm_supergpqa},
    "logic":          {"name": "MMLU — formal logic", "domain": "reasoning", "tier": "current", "license": "MIT",
                       "parts": [("cais/mmlu", "formal_logic", "test")], "cap": 200, "norm": norm_list_idx("question", "choices", "answer")},
    "truthfulqa":     {"name": "TruthfulQA (MC1)", "domain": "truthfulness", "tier": "current", "license": "Apache-2.0 (verify)",
                       "parts": [("truthfulqa/truthful_qa", "multiple_choice", "validation")], "cap": 817, "norm": norm_truthfulqa},
    # code generation (executed against tests — run by eval_code.py, pass@1)
    "humaneval":      {"name": "HumanEval", "domain": "code", "tier": "current", "license": "MIT (verify)",
                       "parts": [("openai/openai_humaneval", "openai_humaneval", "test")], "cap": 164, "norm": norm_humaneval},
    "mbpp":           {"name": "MBPP (sanitized)", "domain": "code", "tier": "current", "license": "CC-BY-4.0 (verify)",
                       "parts": [("google-research-datasets/mbpp", "sanitized", "test")], "cap": 257, "norm": norm_mbpp},
    # medical — the unsaturated, harness-fitting ones
    "medxpertqa":     {"name": "MedXpertQA (Text)", "domain": "medical", "tier": "current", "license": "see source (verify; may be non-commercial)",
                       "parts": [("TsinghuaC3I/MedXpertQA", "Text", "test")], "cap": 1000, "norm": norm_dict_label("question", "label")},
    "medmcqa":        {"name": "MedMCQA", "domain": "medical", "tier": "current", "license": "MIT (verify)",
                       "parts": [("openlifescienceai/medmcqa", "default", "validation")], "cap": 800, "norm": norm_medmcqa},
    "medqa_test":     {"name": "MedQA (USMLE)", "domain": "medical", "tier": "current", "license": "research use (verify)",
                       "parts": [("GBaker/MedQA-USMLE-4-options", "default", "test")], "cap": 1273, "norm": norm_dict_label("question", "answer_idx")},

    # ===== legacy (saturated at the frontier — small/quantized-model regression only) =====
    "arc_challenge":  {"name": "ARC-Challenge", "domain": "reasoning", "tier": "legacy", "license": "CC-BY-SA-4.0 (verify)",
                       "parts": [("allenai/ai2_arc", "ARC-Challenge", "test")], "cap": 1172, "norm": norm_label_text("question")},
    "hellaswag":      {"name": "HellaSwag", "domain": "commonsense", "tier": "legacy", "license": "MIT (verify)",
                       "parts": [("Rowan/hellaswag", "default", "validation")], "cap": 1000, "norm": norm_hellaswag},
    "commonsense_qa": {"name": "CommonsenseQA", "domain": "commonsense", "tier": "legacy", "license": "MIT (verify)",
                       "parts": [("tau/commonsense_qa", "default", "validation")], "cap": 1221, "norm": norm_label_text("question")},
    "openbookqa":     {"name": "OpenBookQA", "domain": "knowledge", "tier": "legacy", "license": "Apache-2.0 (verify)",
                       "parts": [("allenai/openbookqa", "main", "test")], "cap": 500, "norm": norm_label_text("question_stem")},
    "winogrande":     {"name": "WinoGrande", "domain": "commonsense", "tier": "legacy", "license": "CC-BY (verify)",
                       "parts": [("allenai/winogrande", "winogrande_xl", "validation")], "cap": 1000, "norm": norm_winogrande},
    "mmlu_cs":        {"name": "MMLU — CS cluster", "domain": "knowledge", "tier": "legacy", "license": "MIT",
                       "parts": [("cais/mmlu", s, "test") for s in
                                 ("college_computer_science", "high_school_computer_science", "machine_learning")],
                       "cap": 400, "norm": norm_list_idx("question", "choices", "answer")},
    "pubmedqa":       {"name": "PubMedQA", "domain": "medical", "tier": "legacy", "license": "MIT (verify)",
                       "parts": [("qiaojin/PubMedQA", "pqa_labeled", "train")], "cap": 1000, "norm": norm_pubmedqa},
    "mmlu_medical":   {"name": "MMLU — medical cluster", "domain": "medical", "tier": "legacy", "license": "MIT",
                       "parts": [("cais/mmlu", s, "test") for s in
                                 ("anatomy", "clinical_knowledge", "college_biology", "college_medicine",
                                  "medical_genetics", "professional_medicine")],
                       "cap": 300, "norm": norm_list_idx("question", "choices", "answer")},
}
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

# one-line "what it tests" descriptions (registry + manual), surfaced in the UI
DESC = {
    "mmlu_pro": "Harder 10-option MMLU across 14 subjects — reasoning-heavy, far less saturated than MMLU.",
    "supergpqa": "26.5k graduate-level questions across 285 disciplines — broad, hard, well below ceiling.",
    "arc_challenge": "Grade-school science questions filtered to defeat retrieval/word-association shortcuts.",
    "hellaswag": "Commonsense sentence completion: pick the plausible continuation of a described scene.",
    "commonsense_qa": "Commonsense reasoning MCQ built from ConceptNet relations (5 options).",
    "winogrande": "Winograd-schema pronoun/coreference resolution at scale — fill the blank with the right noun.",
    "openbookqa": "Elementary-science Q&A needing a known science fact plus a step of reasoning.",
    "truthfulqa": "Do answers avoid imitating common human misconceptions/falsehoods (MC1 single-true).",
    "mmlu_cs": "MMLU computer-science & machine-learning subsets — CS knowledge in MCQ form.",
    "logic": "MMLU formal-logic subset — symbolic/logical reasoning.",
    "humaneval": "Write Python functions from a docstring; scored by EXECUTING unit tests (pass@1).",
    "mbpp": "Short Python programming problems; scored by EXECUTING the provided assertions (pass@1).",
    "medmcqa": "Indian medical-entrance exam (AIIMS/NEET-PG) multiple-choice questions.",
    "pubmedqa": "Biomedical yes/no/maybe research questions answered from a PubMed abstract.",
    "mmlu_medical": "MMLU clinical/biomedical subsets (anatomy, clinical knowledge, medicine, genetics…).",
    # manual / gated
    "gpqa": "Google-proof GRADUATE-level science (bio/chem/physics) MCQ — very hard, still discriminative. Gated on HF.",
    "hle": "Humanity's Last Exam (2025): expert questions at the frontier of human knowledge. Gated; partly multimodal.",
    "medqa_test": "USMLE-style US medical-licensing exam questions (4-option, GBaker/MedQA-USMLE-4-options).",
    "medxpertqa": "Expert-level medical reasoning MCQ (up to 10 options) — the unsaturated medical set.",
    "healthbench_hard": "Open-ended medical conversations graded against physician rubrics (run by healthbench.py).",
}

def present(key):
    return os.path.exists(os.path.join(DATA, key + ".jsonl"))

def fit_of(s):
    return "code" if s["domain"] == "code" else "mcq"

def current_keys():
    return [k for k, s in REGISTRY.items() if s.get("tier") == "current"]

def registry_meta():
    """Machine-readable registry for the dashboard's benchmark browser."""
    return {"available": [{"key": k, "name": s["name"], "domain": s["domain"], "desc": DESC.get(k, ""),
                           "tier": s.get("tier", "current"), "fit": fit_of(s),
                           "license": s.get("license", ""), "present": present(k)}
                          for k, s in REGISTRY.items()],
            "manual": [{"key": k, "note": v, "desc": DESC.get(k, "")} for k, v in MANUAL.items()]}

def list_registry():
    print("Benchmarks — '✓' already in data/. Fit: [mcq] letter, [code] executed pass@1.\n")
    for tier, head in (("current", "CURRENT (still discriminating mid-2026)"),
                       ("legacy", "LEGACY / saturated (small-model regression only)")):
        print(f"  == {head} ==")
        for k, s in REGISTRY.items():
            if s.get("tier", "current") != tier: continue
            print(f"    {'✓' if present(k) else ' '} {k:<14} [{fit_of(s)}] {s['name']:<22} {DESC.get(k,'')}")
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
