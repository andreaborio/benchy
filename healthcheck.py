#!/usr/bin/env python3
"""benchy healthcheck — probe every benchmark so breakage is caught proactively.

For each registry entry it fetches ONE row from the datasets-server and runs the
normalizer, confirming the source still exists and still has the expected schema. It
also reports upstream drift vs the lockfile. Designed for CI/cron: exits non-zero if
anything is broken, so a renamed/changed dataset is flagged before it breaks a real run.

  python3 healthcheck.py            # probe all
  python3 healthcheck.py current    # probe only the current tier
  python3 healthcheck.py <key> ...  # probe specific sets
"""
import sys, time

import api, fetch_benchmarks as fb


def probe(key):
    """Return (status, detail). status: OK | EMPTY | NORM | SHAPE | ERROR."""
    spec = fb.REGISTRY[key]
    ds, cfg, sp = spec["parts"][0]
    try:
        rows = fb.api_rows(ds, cfg, sp, 1)
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {e}"
    if not rows:
        return "EMPTY", f"no rows from {ds}/{cfg}/{sp}"
    try:
        rec = spec["norm"](rows[0])
    except Exception as e:
        return "ERROR", f"normalizer raised {type(e).__name__}: {e}"
    if rec is None:
        return "NORM", "normalizer rejected the sample row (schema likely changed)"
    if spec.get("fit", "mcq") == "mcq":
        if not rec.get("question") or not rec.get("options") \
                or rec.get("answer_idx") not in (rec.get("options") or {}):
            return "SHAPE", "record missing question/options/answer_idx"
    else:
        if not rec.get("prompt") or not rec.get("tests"):
            return "SHAPE", "code record missing prompt/tests"
    return "OK", ""


def main(argv):
    sel = argv or ["all"]
    keys = (list(fb.REGISTRY) if sel == ["all"]
            else fb.current_keys() if sel == ["current"]
            else [k for k in sel if k in fb.REGISTRY])
    if not keys:
        print("no known benchmarks selected"); sys.exit(2)
    drift = {r["key"]: r.get("drift") for r in api.lock_status()["benchmarks"]}
    print(f"benchy healthcheck — probing {len(keys)} benchmark(s)\n")
    broken = []
    for k in sorted(keys):
        status, detail = probe(k)
        d = " ·DRIFT" if drift.get(k) else ""
        mark = "ok " if status == "OK" else status
        print(f"  [{mark:^5}] {k:<15}{d}  {detail}")
        if status != "OK":
            broken.append((k, status, detail))
        time.sleep(0.1)
    drifted = [k for k in keys if drift.get(k)]
    print(f"\n{len(keys) - len(broken)}/{len(keys)} ok"
          + (f" · {len(broken)} BROKEN: {', '.join(k for k, _, _ in broken)}" if broken else "")
          + (f" · {len(drifted)} drifted (relock to accept): {', '.join(drifted)}" if drifted else ""))
    sys.exit(1 if broken else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
