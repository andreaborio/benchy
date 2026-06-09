#!/usr/bin/env python3
"""Multiple-choice benchmark scorer against any OpenAI-compatible chat API, with
structured + live per-question streaming for the dashboard.

Works on any dataset in the unified MCQ format — {question, options{A..}, answer_idx} —
so you can drop in your own JSONL (knowledge, reasoning, coding-MCQ, …), not just the
bundled sets.

Usage: eval_mcq.py <jsonl> <N> [think|nothink] [tag] [notes...]

Env: BENCHY_SERVER (default http://127.0.0.1:8000), BENCHY_MODEL (default: auto-detected
from the server's /v1/models).

Writes (under results/):
  runs.jsonl   one summary record per completed run
  live.json    aggregate live progress
  stream.jsonl one line per question (cleared at run start) — the live Q/A feed
Stdlib only. Greedy, deterministic.
"""
import json, sys, re, random, urllib.request, time, datetime, os, hashlib
import benchy_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(RESULTS, "runs.jsonl")
LIVE = os.path.join(RESULTS, "live.json")
STREAM = os.path.join(RESULTS, "stream.jsonl")
DETAILS = os.path.join(RESULTS, "details")
SERVER_BASE = os.environ.get("BENCHY_SERVER", "http://127.0.0.1:8000").rstrip("/")
SERVER = SERVER_BASE + "/v1/chat/completions"
SEED = 1234
LETTERS = "ABCDEFGHIJ"
# Option-order randomisation (default ON): each question's options are permuted with a
# deterministic per-question seed so position/letter bias is averaged out instead of
# confounding a quantization A/B delta. The seed is derived from the question text only, so
# every model/quant sees the SAME order — a precondition for a valid paired comparison.
SHUFFLE = os.environ.get("BENCHY_SHUFFLE_OPTIONS", "1").lower() not in ("0", "false", "no", "")

MODEL = bc.resolve_model(SERVER_BASE)

def norm_gold(r, keys):
    """Return the gold option LETTER. Accepts answer_idx as a letter ('B') or an integer index
    (0-based, e.g. 1 -> 'B') — the field is named *_idx, so bring-your-own datasets routinely
    put an int there; without this they would silently score 0% with no diagnostic."""
    g = r.get("answer_idx")
    if isinstance(g, bool):
        return str(g)
    if isinstance(g, int):
        return LETTERS[g] if 0 <= g < len(LETTERS) else str(g)
    g = str(g).strip()
    if g.isdigit():
        i = int(g)
        return LETTERS[i] if 0 <= i < len(LETTERS) else g
    return g.upper() if len(g) == 1 else g

def load(path, n):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    random.Random(SEED).shuffle(rows)
    rows = rows if n == 0 else rows[:n]
    bad = sum(1 for r in rows if norm_gold(r, [k for k in LETTERS if k in (r.get("options") or {})])
              not in [k for k in LETTERS if k in (r.get("options") or {})])
    if rows and bad == len(rows):
        print(f"⚠ every row's answer_idx is outside its options — check the dataset's gold format "
              f"(expected a letter A.. or a 0-based int). Scores will be meaningless.", flush=True)
    return rows

def prepare(r):
    """The question as PRESENTED to the model: (question, opts, keys, gold, perm).
    With SHUFFLE on, option order is permuted deterministically (seeded by the question text)
    and gold is remapped; perm maps each presented letter -> original letter for auditing."""
    opts = r.get("options") or {}
    keys = [k for k in LETTERS if k in opts]
    gold = norm_gold(r, keys)
    question = str(r.get("question", "")).strip()
    if not SHUFFLE or gold not in keys or len(keys) < 2:
        return question, {k: opts[k] for k in keys}, keys, gold, {}
    qseed = int(hashlib.sha256(("%s|%s" % (SEED, question)).encode()).hexdigest()[:8], 16)
    order = list(range(len(keys)))
    random.Random(qseed).shuffle(order)
    new_opts, perm, new_gold = {}, {}, gold
    for pos, oi in enumerate(order):
        nl, ol = keys[pos], keys[oi]
        new_opts[nl] = opts[ol]; perm[nl] = ol
        if ol == gold: new_gold = nl
    return question, new_opts, keys, new_gold, perm

def build_prompt(question, opts, keys):
    lines = [question, ""]
    for k in keys:
        lines.append(f"{k}. {opts[k]}")
    lines += ["", "Answer with only the single letter of the correct option. No explanation."]
    return "\n".join(lines)

def ask(prompt, think):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 3072 if think else 24, "temperature": 0.0, "think": bool(think)}
    req = urllib.request.Request(SERVER, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def extract(text, keys):
    """Find the chosen option letter. Anchored and CASE-SENSITIVE for the option letters: the
    text is never upper-cased, so prose words like 'a' / 'I' are not misread as answers (the
    old `\\b[A-J]\\b` over text.upper() scored every indefinite article as 'A'). Returns a
    present key, or '?' when nothing parses — '?' is tracked as the unparseable rate per run."""
    present = set(keys)
    kk = "".join(keys)
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I).strip()
    if not t:
        return "?"
    # 0) the whole reply is essentially just the letter (accept any case here)
    m = re.fullmatch(r"\**\s*\(?\s*([A-Za-z])\s*\)?\**[.):]*\s*", t)
    if m and m.group(1).upper() in present:
        return m.group(1).upper()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    # 1) a line that is exactly an (optionally markdown/paren-wrapped) UPPERCASE option letter
    for ln in reversed(lines):
        m = re.fullmatch(r"[-*>#\s]*\**\(?\s*([A-J])\s*\)?\**[.):]*\s*", ln)
        if m and m.group(1) in present:
            return m.group(1)
    # 2) an explicit answer cue followed by an UPPERCASE option letter — last such wins
    if kk:
        cues = [c for c in re.findall(
            r"(?:answer|option|choice|correct|select)\b[^A-Za-z0-9]{0,15}\(?([%s])\)?(?![A-Za-z])" % kk, t)
            if c in present]
        if cues:
            return cues[-1]
        # 3) last isolated UPPERCASE option letter anywhere (case-sensitive: 'a'/'i' never match)
        iso = [c for c in re.findall(r"(?<![A-Za-z])([%s])(?![A-Za-z])" % kk, t) if c in present]
        if iso:
            return iso[-1]
    return "?"

def server_timing():
    """Best-effort per-question timing from server.log: prefill seconds, total seconds, gen tokens, avg t/s.
    The eval is sequential (one request at a time), so the most recent finished block is this question's."""
    log = os.path.join(RESULTS, "server.log")
    try:
        with open(log, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - 9000)); tail = f.read().decode("utf-8", "ignore")
        prefill = total = gen = avg = None
        for ln in tail.splitlines():
            m = re.search(r"prompt done ([\d.]+)s", ln)
            if m: prefill = round(float(m.group(1)), 1)
            m = re.search(r"gen=(\d+) finish=\S+ ([\d.]+)s", ln)
            if m: gen = int(m.group(1)); total = float(m.group(2))
            m = re.search(r"avg=([\d.]+) t/s", ln)
            if m: avg = float(m.group(1))
        gen_s = round(total - prefill, 1) if (total is not None and prefill is not None) else (round(total, 1) if total is not None else None)
        return {"prefill_s": prefill, "gen_tokens": gen, "gen_tps": avg, "gen_s": gen_s}
    except Exception:
        return {}

def write_live(d):
    os.makedirs(RESULTS, exist_ok=True)
    d["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = LIVE + ".tmp"; open(tmp, "w").write(json.dumps(d)); os.replace(tmp, LIVE)

def main():
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    think = len(sys.argv) > 3 and sys.argv[3] == "think"
    tag = sys.argv[4] if len(sys.argv) > 4 else "baseline"
    notes = " ".join(sys.argv[5:]) if len(sys.argv) > 5 else ""
    mode = "thinking" if think else "nothink"
    bench = os.path.basename(path).replace(".jsonl", "")
    rows = load(path, n)
    n_options = max((len([k for k in LETTERS if k in (r.get("options") or {})]) for r in rows), default=0)  # #answer options (for chance line + bias χ²)
    os.makedirs(RESULTS, exist_ok=True)
    open(STREAM, "w").close()  # clear the live feed
    os.makedirs(DETAILS, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    detfile = os.path.join(DETAILS, f"{bench}__{mode}__{run_id}.jsonl")
    open(detfile, "w").close()
    correct = 0; dist = {}; unparsed = 0; t0 = time.time()
    for i, r in enumerate(rows):
        question, opts, keys, gold, perm = prepare(r)
        prompt = build_prompt(question, opts, keys)
        q0 = time.time()
        try:
            out = ask(prompt, think)
        except Exception as e:
            out = f"ERR:{e}"
        pred = extract(out, keys)
        if pred == "?": unparsed += 1
        ok = pred == gold; correct += ok
        dist[pred] = dist.get(pred, 0) + 1
        acc = 100 * correct / (i + 1)
        ans = re.sub(r"<think>.*?</think>", "", out, flags=re.S | re.I).strip()
        think_txt = " ".join(re.findall(r"<think>(.*?)</think>", out, flags=re.S | re.I))
        think_tokens = len(think_txt.split()) if think_txt else 0
        tm = server_timing()
        ev = {"i": i + 1, "n": len(rows), "ok": bool(ok), "pred": pred, "gold": gold,
              "acc": round(acc, 1), "t": round(time.time() - q0, 1),
              "q": question[:260],
              "pred_txt": opts.get(pred, "")[:90], "gold_txt": opts.get(gold, "")[:90],
              "ans": ans[:180], "think_tokens": think_tokens,
              "prefill_s": tm.get("prefill_s"), "gen_s": tm.get("gen_s"),
              "gen_tokens": tm.get("gen_tokens"), "gen_tps": tm.get("gen_tps")}
        with open(STREAM, "a") as f: f.write(json.dumps(ev) + "\n")
        with open(detfile, "a") as f:
            f.write(json.dumps({"i": i + 1, "question": question, "options": opts,
                                "pred": pred, "gold": gold, "ok": bool(ok), "answer": ans,
                                "perm": perm or None, "think_tokens": think_tokens,
                                "prefill_s": tm.get("prefill_s"),
                                "gen_s": tm.get("gen_s"), "gen_tokens": tm.get("gen_tokens"),
                                "gen_tps": tm.get("gen_tps")}) + "\n")
        write_live({"running": True, "tag": tag, "benchmark": bench, "mode": mode,
                    "i": i + 1, "n": len(rows), "correct": correct,
                    "accuracy": round(acc, 1), "elapsed_s": round(time.time() - t0)})
        print(f"{i+1:>4}/{len(rows)} pred={pred} gold={gold} {'OK ' if ok else ' x '} acc={acc:5.1f}%", flush=True)
    dt = time.time() - t0; acc = 100 * correct / len(rows)
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "tag": tag,
           "benchmark": bench, "mode": mode, "n": len(rows), "correct": correct,
           "accuracy": round(acc, 1), "seed": SEED, "duration_s": round(dt),
           "sec_per_q": round(dt / len(rows), 1), "letter_dist": dist, "n_options": n_options,
           "unparsed": unparsed, "shuffle_options": SHUFFLE,
           "notes": notes, "details": os.path.basename(detfile),
           **bc.run_meta(MODEL, SERVER_BASE, path)}
    open(RUNS, "a").write(json.dumps(rec) + "\n")
    write_live({"running": False, "tag": tag, "benchmark": bench, "mode": mode,
                "i": len(rows), "n": len(rows), "correct": correct,
                "accuracy": round(acc, 1), "elapsed_s": round(dt)})
    print(f"\n=== {bench} [{mode}] tag={tag} N={len(rows)} "
          f"accuracy = {correct}/{len(rows)} = {acc:.1f}%  ({dt:.0f}s, {dt/len(rows):.1f}s/q) ===")

if __name__ == "__main__":
    main()
