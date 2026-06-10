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

CLI (<sel> is a key, domain, tier, 'all' or 'everything'; lock/relock/verify take several):
  python3 api.py status                 lock state + drift vs upstream for every set
  python3 api.py prelock [<sel>]        pin upstream revisions WITHOUT downloading data
  python3 api.py lock [<sel> ...]       fetch + lock (pin current upstream) — first time
  python3 api.py relock <sel> [...]     bump the lock to the latest upstream + re-hash
  python3 api.py verify [<sel> ...]     re-fetch pinned revision, check the content hash
"""
import datetime, json, os, sys, time, urllib.request

import benchy_common as _bc
import fetch_benchmarks as _fb

API_VERSION = 1
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK_PATH = os.path.join(HERE, "benchmarks.lock.json")


# ---------- lockfile ----------
def _load_lock():
    """PURE read: never renames/mutates the lockfile. Concurrent readers (the dashboard's
    /api/benchmarks, runners' check_dataset_lock) may catch a writer mid-flight; quarantining
    here would destroy a healthy lockfile. The destructive quarantine lives in
    _quarantine_corrupt_lock(), called only from the CLI write commands."""
    if os.path.exists(LOCK_PATH):
        try:
            return json.load(open(LOCK_PATH, encoding="utf-8"))
        except Exception as e:
            print(f"benchy.api: WARNING lockfile unreadable ({e}) — treating as empty "
                  f"(read-only; not touching {LOCK_PATH})", file=sys.stderr)
    return {"_api_version": API_VERSION, "benchmarks": {}}


def _quarantine_corrupt_lock():
    """CLI write paths only (lock/relock/verify): if the lockfile is genuinely corrupt,
    move it aside to .corrupt so the upcoming writes start from a clean slate. Re-reads and
    re-attempts json.load right here (a previous failed read may have been a writer
    mid-os.replace); only a load that fails NOW triggers the rename."""
    if not os.path.exists(LOCK_PATH):
        return
    try:
        with open(LOCK_PATH, encoding="utf-8") as f:
            json.load(f)
        return                          # healthy (or healed since the failed read)
    except Exception as e:
        corrupt = LOCK_PATH + ".corrupt"
        try:
            os.replace(LOCK_PATH, corrupt)  # quarantine for inspection (overwrites an older .corrupt)
        except OSError:
            return
        print(f"benchy.api: WARNING lockfile is corrupt ({e}) — moved to {corrupt}, "
              f"starting fresh", file=sys.stderr)


def _save_lock(lock):
    lock["_api_version"] = API_VERSION
    tmp = LOCK_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
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
    """Full SHA-256 of a data file — THE lockfile pin. One implementation, shared via
    benchy_common.sha256_file; runs.jsonl's data_sha is its 12-hex prefix."""
    return _bc.sha256_file(path)


def _data_path(key):
    return os.path.join(_fb.DATA, key + ".jsonl")


def data_path(key):
    """Public: where benchmark `key`'s rows live on disk (may not exist until fetch())."""
    return _data_path(key)


def _rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
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


# cross-process write exclusion around the lockfile read-modify-write (NOT held during
# downloads). Module-level so tests can shrink the windows.
_SENTINEL_TIMEOUT_S = 10.0   # how long a writer retries before giving up
_SENTINEL_STALE_S = 300.0    # a sentinel older than this is from a dead writer — reclaim it


def _acquire_write_sentinel(timeout=None, stale_s=None):
    """Take LOCK_PATH+'.write.lock' via O_CREAT|O_EXCL (atomic on every filesystem we care
    about). Retries briefly; a sentinel older than `stale_s` is treated as left behind by a
    crashed writer and removed. Caller MUST remove the returned path in a finally."""
    timeout = _SENTINEL_TIMEOUT_S if timeout is None else timeout
    stale_s = _SENTINEL_STALE_S if stale_s is None else stale_s
    sentinel = LOCK_PATH + ".write.lock"
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()} {datetime.datetime.now().isoformat()}\n".encode())
            finally:
                os.close(fd)
            return sentinel
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(sentinel) > stale_s:
                    os.remove(sentinel)     # stale: its writer died without cleaning up
                    continue
            except OSError:
                continue                    # vanished between EXCL-fail and stat — retry now
            if time.time() >= deadline:
                raise LockError(f"could not acquire {sentinel} within {timeout:.0f}s — another "
                                f"lock/relock/verify is writing; retry, or delete the file if "
                                f"no benchy process is running")
            time.sleep(0.05)


def _write_lock_entry(key, csha, path, keep_rev):
    """Persist one benchmark's lock entry with a FRESH single-key read-modify-write.
    The caller's lock snapshot may be minutes old (loaded before a download), so it is not
    accepted here — re-reading under the write sentinel means concurrent `api.py lock`
    writers can no longer drop each other's freshly-pinned keys."""
    dataset = _fb.dataset_of(key)
    prev = _load_lock()["benchmarks"].get(key, {})
    # resolve the revision BEFORE taking the sentinel: upstream_sha is a network call and
    # the sentinel must only bracket the local read-modify-write
    rev = prev.get("upstream_sha") if (keep_rev and prev.get("upstream_sha")) else upstream_sha(dataset)
    sentinel = _acquire_write_sentinel()
    try:
        lock = _load_lock()                 # fresh: pick up keys other writers just saved
        lock["benchmarks"][key] = {
            "dataset": dataset, "upstream_sha": rev, "content_sha": csha, "rows": _rows(path),
            "locked_at": datetime.datetime.now().isoformat(timespec="seconds")}
        _save_lock(lock)
    finally:
        try:
            os.remove(sentinel)
        except OSError:
            pass


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
        _write_lock_entry(key, csha, path, keep_rev=bool(ent))  # complete a prelock
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
    _write_lock_entry(key, csha, path, keep_rev=bool(ent and not update))
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
        ent = lock["benchmarks"].get(k, {})
        sha = upstream_sha(dataset) or ent.get("upstream_sha")  # keep the pin if upstream is unreachable
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


def _select_args(args):
    """Union of one-or-more selections, first-seen order (lock/relock/verify take several)."""
    if not args:
        return _select(None)
    out = []
    for a in args:
        out += [k for k in _select(a) if k not in out]
    return out


def _cmd_lock(args):
    _quarantine_corrupt_lock()      # write entry point: safe to move a corrupt lockfile aside
    bad = 0
    for k in _select_args(args):
        try:
            p = fetch(k)
            if not p:
                raise LockError("fetch returned no data (upstream unreachable?)")
            print(f"  locked {k} -> {p}")
        except Exception as e:
            bad += 1; print(f"  ! {k}: {e}")
    sys.exit(1 if bad else 0)   # non-zero so callers (the dashboard) can surface failures


def _cmd_prelock(args):
    prelock(_select(args[0]) if args else list(_fb.REGISTRY))


def _cmd_relock(args):
    if not args:
        sys.exit("usage: api.py relock <sel> [<sel> ...]")
    _quarantine_corrupt_lock()      # write entry point: safe to move a corrupt lockfile aside
    for k in _select_args(args):
        try:
            fetch(k, update=True)
            print(f"  relocked {k}")
        except Exception as e:
            print(f"  ! {k}: {e}")


def _cmd_verify(args):
    _quarantine_corrupt_lock()      # write entry point: safe to move a corrupt lockfile aside
    bad = 0
    for k in _select_args(args):
        path, bak = _data_path(k), _data_path(k) + ".bak"
        try:
            os.path.exists(path) and os.replace(path, bak)  # keep the original until re-fetch succeeds
            if fetch(k):
                os.path.exists(bak) and os.remove(bak)
                print(f"  ok {k}")
            else:
                os.path.exists(bak) and os.replace(bak, path)  # nothing verified — restore
                bad += 1; print(f"  ! {k}: re-fetch failed (upstream unreachable?) — original restored")
        except Exception as e:
            os.path.exists(bak) and os.replace(bak, path)
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
