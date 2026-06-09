#!/usr/bin/env python3
"""Fetch medical MCQ benchmarks via the HuggingFace datasets-server API (JSON, no
pyarrow/datasets needed) and convert them to the unified format the eval harness
expects: {question, options{A..}, answer_idx}. Writes data/<name>.jsonl.
"""
import json, os, time, urllib.request, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

def rows(dataset, config, split, cap):
    out, off = [], 0
    while off < cap:
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": off, "length": 100})
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
        time.sleep(0.15)
    return out[:cap]

def write(name, recs):
    with open(os.path.join(DATA, name + ".jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"  {name}: {len(recs)} questions")

# --- MedMCQA (Indian PG entrance; 4-opt) ---
mm = []
for r in rows("openlifescienceai/medmcqa", "default", "validation", 800):
    if r.get("cop") is None or not all(r.get(k) for k in ("opa", "opb", "opc", "opd")): continue
    mm.append({"question": r["question"], "answer_idx": "ABCD"[r["cop"]],
               "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}})
write("medmcqa", mm)

# --- PubMedQA (yes/no/maybe over an abstract; context prepended) ---
M = {"yes": "A", "no": "B", "maybe": "C"}
pq = []
for r in rows("qiaojin/PubMedQA", "pqa_labeled", "train", 1000):
    fd = r.get("final_decision")
    if fd not in M: continue
    ctx = ""
    c = r.get("context")
    if isinstance(c, dict) and c.get("contexts"):
        ctx = " ".join(c["contexts"])
    q = (ctx[:1500].rstrip() + "\n\nQuestion: " + r["question"]) if ctx else r["question"]
    pq.append({"question": q, "answer_idx": M[fd],
               "options": {"A": "yes", "B": "no", "C": "maybe"}})
write("pubmedqa", pq)

# --- MMLU medical cluster (6 subjects; 4-opt) ---
subs = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine",
        "medical_genetics", "professional_medicine"]
mlu = []
for s in subs:
    sub = rows("cais/mmlu", s, "test", 300)
    for r in sub:
        ch = r.get("choices") or []
        if len(ch) < 4: continue
        mlu.append({"question": r["question"], "answer_idx": "ABCD"[r["answer"]],
                    "options": {"A": ch[0], "B": ch[1], "C": ch[2], "D": ch[3]},
                    "subject": s})
    print(f"    mmlu/{s}: +{len(sub)}")
write("mmlu_medical", mlu)

print("done.")
