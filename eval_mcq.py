#!/usr/bin/env python3
"""Multiple-choice benchmark scorer against any OpenAI-compatible chat API, with
structured + live per-question streaming for the dashboard.

Works on any dataset in the unified MCQ format — {question, options{A..}, answer_idx} —
so you can drop in your own JSONL (knowledge, reasoning, coding-MCQ, …), not just the
bundled sets.

Usage: eval_mcq.py <jsonl> <N> <think|nothink> <tag> [--seed INT]

Env: BENCHY_SERVER (overrides config.json server_base / the built-in default),
BENCHY_MODEL (default: auto-detected from the server's /v1/models),
BENCHY_CONCURRENCY (int, default 1 = strictly sequential; N>1 asks N questions at a
time and skips the sequential-only server.log timing scrape).

Writes (under results/):
  runs.jsonl   one summary record per completed run
  live.json    aggregate live progress
  stream.jsonl one line per question (cleared at run start) — the live Q/A feed
Stdlib only. Greedy, deterministic.
"""
import json, sys, re, random, time, os, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import benchy_common as bc

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SERVER_BASE = bc.settings()["server_base"]
SEED = 1234
LETTERS = "ABCDEFGHIJ"
# Option-order randomisation (default ON): each question's options are permuted with a
# deterministic per-question seed so position/letter bias is averaged out instead of
# confounding a quantization A/B delta. The seed is derived from the question text only, so
# every model/quant sees the SAME order — a precondition for a valid paired comparison.
SHUFFLE = os.environ.get("BENCHY_SHUFFLE_OPTIONS", "1").lower() not in ("0", "false", "no", "")
# Opt-in request concurrency (BENCHY_CONCURRENCY=N, default 1). 1 keeps the historical
# strictly-sequential behavior, including the ds4 server.log timing scrape. With N>1 the
# prepare/ask/extract work runs on N threads while ALL accounting and file writes stay on
# the main thread (results are funnelled through as_completed, so stream/details/live are
# never written concurrently), and server_timing() is skipped entirely — its log
# tail-scrape assumes exactly one request in flight. Scoring is unaffected: the per-question
# option permutation is seeded by the question text alone (see prepare), never by request
# order, so every concurrency level presents identical questions.
# Opt-in live token streaming (BENCHY_LIVE_STREAM=1): the current question's reasoning +
# answer stream into results/gen.json for the dashboard's generation box. Off by default so
# the scoring path stays the plain blocking chat(); only honored at CONCURRENCY==1 (with N
# workers there is no single "current" generation to show). Display only — never feeds scoring.
LIVE_STREAM = os.environ.get("BENCHY_LIVE_STREAM", "").lower() in ("1", "true", "yes", "on")
try:
    CONCURRENCY = max(1, int(os.environ.get("BENCHY_CONCURRENCY", "1")))
except ValueError:
    CONCURRENCY = 1

_MODEL = None
def get_model():
    # resolved lazily (and cached) so importing this module never hits the network
    global _MODEL
    if _MODEL is None:
        _MODEL = bc.resolve_model(SERVER_BASE)
    return _MODEL

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
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
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
    return bc.chat(prompt, think=think, max_tokens=3072 if think else 24,
                   get_model=get_model, seed=SEED, server_base=SERVER_BASE, timeout=600)

def ask_streaming(prompt, think, w, qnum, n, question):
    """Streaming ask for the live generation box (CONCURRENCY==1 only): bc.stream_into feeds the
    reasoning/answer deltas into gen.json as they arrive, and returns the SAME assembled text
    ask() would, so scoring is unchanged. qnum is the 1-based question-in-flight number."""
    return bc.stream_into(w, qnum, n, question, messages=prompt, think=think,
                          max_tokens=3072 if think else 24, get_model=get_model,
                          seed=SEED, server_base=SERVER_BASE, timeout=600)

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

def run_one(i, r, think, w=None, n=0):
    """prepare/ask/extract for one question — the unit of work a worker thread runs (bc.chat
    does its own retries; everything else here is pure). Returns a result dict; a request
    that still failed after retries comes back with an "error" key instead of raising, so
    the main thread counts every question exactly once. When `w` is given and live streaming
    is on, the generation streams into gen.json (only used at CONCURRENCY==1)."""
    question, opts, keys, gold, perm = prepare(r)
    prompt = build_prompt(question, opts, keys)
    q0 = time.time()
    try:
        if LIVE_STREAM and w is not None:
            out = ask_streaming(prompt, think, w, i + 1, n, question)
        else:
            out = ask(prompt, think)
    except Exception as e:
        return {"i": i, "q": question, "opts": opts, "gold": gold, "perm": perm,
                "t": time.time() - q0, "error": e}
    return {"i": i, "q": question, "opts": opts, "gold": gold, "perm": perm,
            "t": time.time() - q0, "out": out, "pred": extract(out, keys)}

def iter_results(rows, think, w=None):
    """Yield one run_one() result per question. CONCURRENCY=1 (default): dataset order, one
    request in flight at a time — the historical sequential behavior, and the only mode that
    streams the live generation box (`w` passed through). CONCURRENCY>1: results arrive in
    COMPLETION order from a thread pool; the caller does all file writes, so they stay
    serialized on the main thread (A/B pairing is by question text, not row order)."""
    if CONCURRENCY == 1:
        for i, r in enumerate(rows):
            yield run_one(i, r, think, w, len(rows))
        return
    # NOT a `with` block: an early exit in the consumer (Ctrl-C / exception / generator
    # close raising GeneratorExit at the yield) must CANCEL the queued questions, not run
    # them all to completion — executor.__exit__ would block on shutdown(wait=True) with
    # every pending future still queued. cancel_futures exists since Python 3.9.
    ex = ThreadPoolExecutor(max_workers=CONCURRENCY)
    try:
        futs = [ex.submit(run_one, i, r, think) for i, r in enumerate(rows)]
        for fut in as_completed(futs):
            yield fut.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

def main():
    global SEED
    args = bc.parse_run_args(prog="eval_mcq.py")
    SEED = args.seed
    path, think, tag, mode = args.bench, args.think, args.tag, args.mode
    max_tokens = 3072 if think else 24
    bench = os.path.basename(path).replace(".jsonl", "")
    # integrity gate BEFORE anything runs: a registered benchmark must still match its
    # benchmarks.lock.json content hash. Aborts on drift (BENCHY_SKIP_LOCK_CHECK=1 to warn
    # instead); locked=False for BYO files / unpinned sets — recorded in the run record.
    locked = bc.check_dataset_lock(path)
    rows = load(path, args.n)
    n_options = max((len([k for k in LETTERS if k in (r.get("options") or {})]) for r in rows), default=0)  # #answer options (for chance line + bias χ²)
    w = bc.RunWriter(bench, mode, tag, bc.KIND_MCQ)
    # extra live.json fields when concurrent: flag the concurrency and that per-question
    # timings are wall-clock only (no sec_per_q / prefill breakdown from server_timing)
    live_extra = {} if CONCURRENCY == 1 else {
        "concurrency": CONCURRENCY,
        "timing_note": "server_timing skipped at concurrency>1; timings are wall-clock"}
    correct = 0; scored = 0; errors = 0; done = 0; think_count = 0; dist = {}; unparsed = 0; t0 = time.time()
    opts_hist = {}   # {str(option_count): scored rows presenting that many options} — mixed-option χ² expectation
    # initial live.json so the dashboard shows the run immediately (question 1 in flight)
    # instead of only after Q1 completes; i = completed count, the UI shows i+1 while running
    w.live(dict({"running": True, "i": 0, "n": len(rows), "correct": 0,
                 "accuracy": 0.0, "errors": 0, "elapsed_s": 0}, **live_extra))
    if LIVE_STREAM and CONCURRENCY > 1:   # no single "current" generation to show at N>1
        w.gen({"running": False, "disabled": True, "reason": "live streaming is shown at concurrency=1"})
    for res in iter_results(rows, think, w):
        done += 1
        i, question, opts, gold, perm = res["i"], res["q"], res["opts"], res["gold"], res["perm"]
        if "error" in res:
            # request still failed after bc.chat's retries: excluded from the accuracy
            # numerator AND denominator, counted in the run record's "errors" field
            e = res["error"]
            errors += 1
            acc = 100 * correct / scored if scored else 0.0
            w.stream({"i": i + 1, "n": len(rows), "ok": False, "pred": "ERR", "gold": gold,
                      "acc": round(acc, 1), "t": round(res["t"], 1), "q": question[:260],
                      "error": True, "ans": str(e)[:180]})
            w.detail({"i": i + 1, "question": question, "options": opts, "gold": gold,
                      "perm": perm or None, "error": True, "error_msg": str(e)})
            w.live(dict({"running": True, "i": done, "n": len(rows), "correct": correct,
                         "accuracy": round(acc, 1), "errors": errors,
                         "elapsed_s": round(time.time() - t0)}, **live_extra))
            print(f"{i+1:>4}/{len(rows)} ERROR (excluded from scoring): {e}", flush=True)
            continue
        scored += 1
        opts_hist[str(len(opts))] = opts_hist.get(str(len(opts)), 0) + 1
        out, pred = res["out"], res["pred"]
        if pred == "?": unparsed += 1
        ok = pred == gold; correct += ok
        dist[pred] = dist.get(pred, 0) + 1
        acc = 100 * correct / scored
        ans = re.sub(r"<think>.*?</think>", "", out, flags=re.S | re.I).strip()
        think_txt = " ".join(re.findall(r"<think>(.*?)</think>", out, flags=re.S | re.I))
        think_tokens = len(think_txt.split()) if think_txt else 0
        if think_tokens > 0: think_count += 1
        tm = server_timing() if CONCURRENCY == 1 else {}
        w.stream({"i": i + 1, "n": len(rows), "ok": bool(ok), "pred": pred, "gold": gold,
                  "acc": round(acc, 1), "t": round(res["t"], 1),
                  "q": question[:260],
                  "pred_txt": opts.get(pred, "")[:90], "gold_txt": opts.get(gold, "")[:90],
                  "ans": ans[:180], "think_tokens": think_tokens,
                  "prefill_s": tm.get("prefill_s"), "gen_s": tm.get("gen_s"),
                  "gen_tokens": tm.get("gen_tokens"), "gen_tps": tm.get("gen_tps")})
        w.detail({"i": i + 1, "question": question, "options": opts,
                  "pred": pred, "gold": gold, "ok": bool(ok), "answer": ans,
                  "perm": perm or None, "think_tokens": think_tokens,
                  "prefill_s": tm.get("prefill_s"),
                  "gen_s": tm.get("gen_s"), "gen_tokens": tm.get("gen_tokens"),
                  "gen_tps": tm.get("gen_tps")})
        w.live(dict({"running": True, "i": done, "n": len(rows), "correct": correct,
                     "accuracy": round(acc, 1), "errors": errors,
                     "elapsed_s": round(time.time() - t0)}, **live_extra))
        print(f"{i+1:>4}/{len(rows)} pred={pred} gold={gold} {'OK ' if ok else ' x '} acc={acc:5.1f}%", flush=True)
    dt = time.time() - t0
    acc = 100 * correct / scored if scored else 0.0
    think_frac = round(think_count / scored, 3) if scored else 0.0
    fields = {"n": scored, "correct": correct, "accuracy": round(acc, 1), "seed": SEED,
              "duration_s": round(dt), "sec_per_q": round(dt / max(1, len(rows)), 1),
              "letter_dist": dist, "n_options": n_options, "opts_hist": opts_hist,
              "unparsed": unparsed, "shuffle_options": SHUFFLE, "errors": errors,
              "max_tokens": max_tokens, "think_frac": think_frac, "locked": locked, "notes": ""}
    if CONCURRENCY > 1:
        # additive provenance: this run's requests overlapped, and sec_per_q above is
        # wall-clock per question slot, not per-request latency
        fields["concurrency"] = CONCURRENCY
    if rows and errors / len(rows) > 0.05:
        fields["invalid"] = True
        print(f"\n⚠⚠⚠ INVALID RUN: {errors}/{len(rows)} requests failed (>5%) — "
              f"the accuracy below covers only the {scored} scored questions and is NOT "
              f"comparable to a clean run.", file=sys.stderr)
    if mode == bc.MODE_THINKING and think_frac < 0.05:
        # the "think" request field is ignored by non-ds4 servers — don't let the label lie
        fields["mode_suspect"] = True
        print(f"\n⚠ MODE SUSPECT: run labelled '{bc.MODE_THINKING}' but only "
              f"{think_frac*100:.1f}% of scored answers contain <think> content — the server "
              f"likely ignores the 'think' flag.", file=sys.stderr)
    w.finish(fields, get_model(), SERVER_BASE, path)
    w.live(dict({"running": False, "i": len(rows), "n": len(rows), "correct": correct,
                 "accuracy": round(acc, 1), "errors": errors, "elapsed_s": round(dt)}, **live_extra))
    w.gen_finish()   # freeze the generation box on the last answer (running:false)
    print(f"\n=== {bench} [{mode}] tag={tag} N={scored} "
          f"accuracy = {correct}/{scored} = {acc:.1f}%  ({dt:.0f}s, {dt/max(1,len(rows)):.1f}s/q, {errors} errors) ===")

if __name__ == "__main__":
    main()
