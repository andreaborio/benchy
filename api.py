#!/usr/bin/env python3
"""benchy.api — the STABLE contract for tools that consume benchy's benchmarks.

Other projects (forgequant, …) build on benchy as their benchmark source. They must
import ONLY this module, never `fetch_benchmarks` internals (REGISTRY, norm_*, …),
which are free to change. `API_VERSION` bumps on a breaking change to the functions
below; consumers check it for compatibility.

What this adds over the raw fetcher:

  • Reproducible calibration via a LOCKFILE (`benchmarks.lock.json`, tracked in the
    repo). Each benchmark pins the upstream HF dataset commit it was locked at plus
    the content SHA-256 + row count of the normalized rows. `fetch()` pins to the
    locked revision and verifies the content hash, so a calibration is always tied
    to an exact benchmark snapshot — and any upstream drift is detected, not silently
    absorbed.

  • A small, documented surface: registry(), meta(), keys(), fetch(), lock_status().

CLI:
  python3 api.py status                 lock state + drift vs upstream for every set
  python3 api.py prelock [<sel>]        pin upstream revisions WITHOUT downloading data
  python3 api.py lock [<key|all>]       fetch + lock (pin current upstream) — first time
  python3 api.py relock <key|all>       bump the lock to the latest upstream + re-hash
  python3 api.py verify [<key|all>]     re-fetch pinned revision, check the content hash
"""
import datetime, hashlib, json, os, sys, urllib.request

import fetch_benchmarks as _fb

API_VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(HERE, "benchmarks.lock.json")


# ---------- lockfile ----------
def _load_lock():
    if os.path.exists(LOCK_PATH):
        try:
            return json.load(open(LOCK_PATH))
        except Exception:
            pass
    return {"_api_version": API_VERSION, "benchmarks": {}}


def _save_lock(lock):
    lock["_api_version"] = API_VERSION
    tmp = LOCK_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
    os.replace(tmp, LOCK_PATH)


def upstream_sha(dataset, timeout=20):
    """Current `main` commit of an HF dataset repo, via the Hub refs API (no data download)."""
    url = f"https://huggingface.co/api/datasets/{dataset}/refs"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            refs = json.load(r)
        for b in refs.get("branches", []):
            if b.get("name") == "main":
                return b.get("targetCommit")
    except Exception:
        return None
    return None


def content_sha(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def _data_path(key):
    return os.path.join(_fb.DATA, key + ".jsonl")


def data_path(key):
    """Public: where benchmark `key`'s rows live on disk (may not exist until fetch())."""
    return _data_path(key)


def _rows(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for ln in f if ln.strip())


# ---------- stable surface ----------
def registry():
    """All non-gated benchmarks: [{key,name,domain,tier,fit,license,present,locked}]."""
    lock = _load_lock()["benchmarks"]
    out = _fb.registry_meta()["available"]
    for b in out:
        b["locked"] = b["key"] in lock
    return out


def manual():
    """Gated / manual sets that need a HF token (not fetchable here)."""
    return _fb.registry_meta()["manual"]


def meta(key):
    for b in registry():
        if b["key"] == key:
            return b
    return None


def keys(tier=None, domain=None):
    return [b["key"] for b in registry()
            if (tier is None or b["tier"] == tier) and (domain is None or b["domain"] == domain)]


def current_keys():
    return keys(tier="current")


def _write_lock_entry(lock, key, csha, path, keep_rev):
    dataset = _fb.dataset_of(key)
    prev = lock["benchmarks"].get(key, {})
    rev = prev.get("upstream_sha") if (keep_rev and prev.get("upstream_sha")) else upstream_sha(dataset)
    lock["benchmarks"][key] = {
        "dataset": dataset, "upstream_sha": rev, "content_sha": csha, "rows": _rows(path),
        "locked_at": datetime.datetime.now().isoformat(timespec="seconds")}
    _save_lock(lock)


def fetch(key, update=False, verify=True):
    """Return the path to benchmark `key`'s rows, fetching if needed.

    Pinned by the lockfile: if `key` is locked, fetch its pinned upstream revision and
    verify the content hash (raises LockError on a mismatch unless update=True). If it
    is not locked, fetch latest and lock it (lock-on-first-fetch). A revision-only
    prelock is completed (content hash filled) the first time the data is present.
    `update=True` re-locks to the current upstream. Returns the path (None on failure).
    """
    if key not in _fb.REGISTRY:
        raise KeyError(f"unknown benchmark '{key}'")
    lock = _load_lock()
    ent = lock["benchmarks"].get(key)
    path = _data_path(key)

    if os.path.exists(path) and not update:
        csha = content_sha(path)
        if ent and ent.get("content_sha"):
            if verify and csha != ent["content_sha"]:
                raise LockError(f"{key}: cached rows ({csha[:12]}) don't match the lock "
                                f"({ent['content_sha'][:12]}). `api.py relock {key}` to accept, "
                                f"or delete data/{key}.jsonl and re-fetch.")
            return path
        _write_lock_entry(lock, key, csha, path, keep_rev=bool(ent))  # complete a prelock
        return path

    rev = None if update else (ent or {}).get("upstream_sha")
    n = _fb.fetch(key, revision=rev)
    if not n:
        return None
    csha = content_sha(path)
    if ent and not update and ent.get("content_sha") and csha != ent["content_sha"]:
        raise LockError(
            f"{key}: fetched rows ({csha[:12]}) differ from the locked snapshot "
            f"({ent['content_sha'][:12]}) — upstream drifted. `api.py relock {key}` to accept.")
    _write_lock_entry(lock, key, csha, path, keep_rev=bool(ent and not update))
    return path


def prelock(selection=None):
    """Pin upstream revisions for a selection WITHOUT downloading data (Hub refs only).
    content_sha is filled lazily on the first real fetch. Returns the count locked."""
    lock = _load_lock()
    sel = selection or list(_fb.REGISTRY)
    n = 0
    for k in sel:
        if k not in _fb.REGISTRY:
            continue
        dataset = _fb.dataset_of(k)
        sha = upstream_sha(dataset)
        ent = lock["benchmarks"].get(k, {})
        ent.update({"dataset": dataset, "upstream_sha": sha,
                    "content_sha": ent.get("content_sha"), "rows": ent.get("rows"),
                    "locked_at": datetime.datetime.now().isoformat(timespec="seconds")})
        lock["benchmarks"][k] = ent
        n += 1 if sha else 0
        print(f"  prelock {k:<15} {dataset} @ {(sha or '?')[:12]}")
    _save_lock(lock)
    return n


def lock_status():
    """Per-benchmark: locked?, present?, and whether upstream has drifted past the lock."""
    lock = _load_lock()["benchmarks"]
    rows = []
    for b in registry():
        k = b["key"]
        ent = lock.get(k)
        drift = None
        if ent:
            cur = upstream_sha(ent.get("dataset") or _fb.dataset_of(k))
            drift = bool(cur and ent.get("upstream_sha") and cur != ent["upstream_sha"])
        rows.append({"key": k, "name": b["name"], "tier": b["tier"], "domain": b["domain"],
                     "locked": bool(ent), "present": b["present"],
                     "upstream_sha": (ent or {}).get("upstream_sha"),
                     "rows": (ent or {}).get("rows"), "drift": drift})
    return {"api_version": API_VERSION, "benchmarks": rows}


class LockError(Exception):
    pass


# ---------- CLI ----------
def _select(arg):
    if not arg or arg == "all":
        return _fb.current_keys()
    if arg == "everything":
        return list(_fb.REGISTRY)
    if arg in _fb.REGISTRY:
        return [arg]
    ks = keys(domain=arg) or keys(tier=arg)
    if ks:
        return ks
    sys.exit(f"benchy.api: unknown selection '{arg}' (a key, domain, tier, 'all', or 'everything')")


def _cmd_status(_a):
    s = lock_status()
    print(f"benchy api v{s['api_version']} · lock: {LOCK_PATH}\n")
    print(f"  {'key':<15}{'tier':<9}{'rows':>6}  locked  drift")
    for r in s["benchmarks"]:
        lk = "✓" if r["locked"] else " "
        dr = "DRIFT" if r["drift"] else ("ok" if r["locked"] else "—")
        print(f"  {r['key']:<15}{r['tier']:<9}{(r['rows'] or '—'):>6}    {lk:^4}  {dr}")


def _cmd_lock(args):
    for k in _select(args[0] if args else None):
        try:
            p = fetch(k)
            print(f"  locked {k} -> {p}")
        except Exception as e:
            print(f"  ! {k}: {e}")


def _cmd_prelock(args):
    prelock(_select(args[0]) if args else list(_fb.REGISTRY))


def _cmd_relock(args):
    if not args:
        sys.exit("usage: api.py relock <key|all|everything>")
    for k in _select(args[0]):
        try:
            fetch(k, update=True)
            print(f"  relocked {k}")
        except Exception as e:
            print(f"  ! {k}: {e}")


def _cmd_verify(args):
    bad = 0
    for k in _select(args[0] if args else None):
        try:
            os.path.exists(_data_path(k)) and os.remove(_data_path(k))  # force a clean re-fetch
            fetch(k)
            print(f"  ok {k}")
        except Exception as e:
            bad += 1; print(f"  ! {k}: {e}")
    sys.exit(1 if bad else 0)


_CMDS = {"status": _cmd_status, "prelock": _cmd_prelock, "lock": _cmd_lock,
         "relock": _cmd_relock, "verify": _cmd_verify}


def main(argv):
    if not argv or argv[0] not in _CMDS:
        print(__doc__); sys.exit(0 if not argv else 2)
    _CMDS[argv[0]](argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
