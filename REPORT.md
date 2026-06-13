# A/B Coding Benchmark — coder-imatrix quants vs plain-2bit baseline

**Started:** 2026-06-10 · **Owner:** Andrea · **Status doc auto-updated by the tracking agent**

> **Question:** is a 2-bit GGUF of DeepSeek-V4-Flash quantized **with a coding imatrix**
> solidly better at coding workloads than the same 2-bit budget quantized with a
> **generic (chat) imatrix**? And does that gain come at the cost of a "knowledge tax"?

---

## TL;DR — FINAL VERDICT (2026-06-12 morning, all four models complete)

**The capture-driven refine WINS. `coder-q4boost-v2` — boost set corrected from live
imatrix captures (layer 41→39 swap) — is statistically significantly better than the
plain-2bit baseline on pooled paired code: 20 wins / 8 losses over 240 paired tasks,
exact McNemar p = 0.036 (< the pre-registered 0.05), with no detected knowledge tax
(11/9, p=0.82).** It also posts the best primary-benchmark score (LCB v6: 56.0%, +10.0
over baseline, +4.0 over v1) and the best/tied score on every other code leg.

| model | pooled code W/L | p | LCB primary | thinking @24k | verdict |
|---|---|---|---|---|---|
| coder-iq2 | 17/10 | 0.25 | 52.0 (+6.0) | 80.0 | directional |
| coder-q4boost v1 | 15/9 | 0.31 | 52.0 (+6.0) | 80.0 | directional |
| **coder-q4boost-v2** | **20/8** | **0.036 ✓** | **56.0 (+10.0)** | 73.3 | **SIGNIFICANT WIN** |

Honesty notes attached to the headline: (a) v2 was the **pre-registered confirmatory
test** (its prediction was written before any v2 task ran), which is why we report its
uncorrected p — but 3 models were compared against one baseline, disclosed; (b) LCB alone
remains directional for v2 (+10.0, p=0.125) — the pooled code metric is what crossed;
(c) the trade the refine made: −2pt on knowledge legs vs v1 (within noise) and **−1
thinking task** (80.0 → 73.3 at n=15) — see the layer-41 investigation below; (d) v2's
first thinking run was VOID (the updated harness hard-caps think at 4096 — artifact #6,
caught by failure classification) and was re-run at the true 24k budget on the same
harness as every other thinking number.

**The loop the article promised is closed and measured: benchmark → live capture →
refine (5-min reuse rebuild) → re-benchmark → statistically significant win.**

**→ Confirmatory status is now gated on the pre-registered held-out validation below** (two reviewer objections accepted: distribution overfitting via capture-calibration; endpoint switching — the pooled p=0.036 is supporting, not confirmatory).

---

## Pre-registration: held-out BigCodeBench validation — plain2bit vs coder-q4boost-v2

**Status: PRE-REGISTERED. Run not started.** Stamped 2026-06-12; the git commit carrying this section into REPORT.md — pushed to an off-machine remote before any held-out request (§8) — is the authoritative timestamp. No request against `bigcodebench_heldout.jsonl` has been served by any model, ever (verification and leak-channel audit below). **Decision (Andrea): the article does not publish before this run completes and is interpreted under the rules in §7.**

### 1. Why this document exists

Two reviewer objections against the campaign result (pooled paired code 20W/8L over 240 tasks, exact McNemar p=0.0357) were accepted as valid:

1. **Distribution overfitting (fatal as a confirmatory claim).** coder-q4boost-v2's boost set was derived from live imatrix captures of the very benchmark traffic it was then evaluated on. Every row in the campaign was, directly or via capture, part of the calibration loop.
2. **Endpoint switching.** The pre-registered primary endpoint was LCB v6 primary (n=50), which alone is not significant. The pooled p=0.036 was a supporting analysis and cannot be sold as confirmatory.

Both objections are answered by one thing only: a paired test on rows that (a) no model has ever executed and (b) never appeared in any capture traffic, with endpoint, analysis, and interpretation fixed before the first token is generated. That is this document.

### 2. Primary endpoint (unique — there is no other)

**Exact two-sided McNemar test, paired by row, `plain2bit` (baseline) vs `coder-q4boost-v2`, on all rows of `data/bigcodebench_heldout.jsonl` scored on both legs after the §5 error-drop rule (target 396), mode `nothink`, alpha = 0.05.**

- Test statistic: `analyze.py::binom_two_sided` verbatim — discordant pairs only, X ~ Bin(b+c, ½), p = min(1, 2·P(X ≤ min(b,c))). Ties (both-pass / both-fail) excluded from the test, reported in the table. b = baseline-only-pass, c = v2-only-pass. Zero discordants → p = 1.0.
- Dataset: `/Users/chinaski/Beep/benchy-eval-stable/data/bigcodebench_heldout.jsonl`, 396 rows, **full sha256 `df7449e2a3f391e710c2bb8db8101a3df1b20ae76f24daed0be8eb5a24b3fe86`** (sha256_12 `df7449e2a3f3`; re-verified at stamping time: 396 rows, hash matches).
- **Derivation rule (deterministic at row-set level):** `random.Random(1234).shuffle` over the 536 lines of `bigcodebench.jsonl` (full sha256 `791e3ce0e9e6ceb7e79a1ded305a5dc85588c73a90392085aea20b76d3977f22`; BigCodeBench v0.1.4 instruct, filtered to stdlib+numpy+pandas); the held-out set is the exact task_id complement of shuffle `[0:140]` (main runs = `[0:40]`, ext slice = `[40:140]`), **written back in source-file order**. It is set-identical to shuffle rows `[140:]`, but the committed file is not in shuffle order and 8 rows are re-serialized (JSON-equal, byte-different), so taking shuffle `[140:]` verbatim reproduces the rows, NOT the file hash. The registered artifact is the file itself, by the full sha256 above. The `rule` field of `data/bigcodebench_heldout.provenance.json` is updated to this wording in the stamping commit. Verification (run from `benchy-eval-stable`; all assertions pass at stamping time):

```
python3 - <<'EOF'
import json, random
src = open('data/bigcodebench.jsonl').readlines()
rows = list(src); random.Random(1234).shuffle(rows)
used = {json.loads(r)['task_id'] for r in rows[:140]}
held = [json.loads(l) for l in open('data/bigcodebench_heldout.jsonl')]
assert {h['task_id'] for h in held} == {json.loads(l)['task_id'] for l in src} - used   # exact complement
assert [h['task_id'] for h in held] == [json.loads(l)['task_id'] for l in src
                                        if json.loads(l)['task_id'] not in used]        # source order
EOF
```

- Never-seen status, verified twice and independently:
  1. By construction: the campaign only ever loaded shuffle `[0:140]`.
  2. By content: sha256 of every question in all 16 `results/details/bigcodebench*` files matched against source prompts — exactly 140 distinct rows used, 0 unmatched, complement == the 396. `benchy-dash/results` is a symlink to `benchy-eval-stable/results` (one physical store), so this audit covers both harnesses and every results tree; no `runs.jsonl` record in either checkout carries `data_sha df7449e2a3f3`. Since these rows were never served, they appear in no live-capture imatrix traffic either; the capture-calibration loop cannot touch them.
- **Leak-channel audit (exhaustive):**
  1. *Results stores on this machine:* `benchy-eval-stable/results` (single physical store; `benchy-dash/results` symlinks to it — the 16-file audit above covers both harnesses); `~/Beep/benchy/results` (pre-campaign checkout) contains no BigCodeBench datasets, runs, or details — verified; the backup tarball `benchy-eval-backup-20260611T1457.tar.gz` is a snapshot of the same store.
  2. *Static-corpus channel:* the imatrix calibration corpora (`coder-q4boost_corpus.txt` = humaneval+mbpp+mmlu_cs via the forge_corpus bridge; `general-baseline_corpus.txt`) contain zero BigCodeBench prompts from ANY split — verified by content probe against all 536 source rows, 0 hits.
  3. *Serving channels:* benchy runners (audited via details; every scripted invocation in all chain/run scripts and orchestrator.py is N=40 on `bigcodebench.jsonl` or N=0 on the 100-row ext file; details stream per-row, so even an aborted run leaves an audit trail); dashboard `/api/chat` (manual; operator attestation: never used with BigCodeBench content); healthchecks and warmups (fixed non-benchmark prompts).
  4. *Attestation (Andrea):* no BigCodeBench row outside shuffle `[0:140]` was ever sent to any model by any channel.

The benchmark name in `runs.jsonl` will be `bigcodebench_heldout`; only the tags `plain2bit` and `coder-q4boost-v2` enter the primary endpoint.

### 3. Prediction and power (written before data, to hold us honest)

**Predicted direction: c > b (v2 beats baseline among discordants).** Seen-BCB paired stats (pooled n=140, content-hash paired, last-wins records): baseline 64/140 (45.7%), v2 70/140 (50.0%), b=7, c=13, discordant rate 20/140 = 14.3%, v2 win share 13/20 = 0.65, **p = 0.263 — the seen pooled-BCB result is itself not significant.**

If the seen effect transfers exactly to n=396: expected discordants ≈ 57, point prediction b≈20, c≈37, **p ≈ 0.033** — just clearing alpha. Minimum detectable effect at ~57 discordants: c ≥ 37, i.e. win share ≥ 0.649, |c−b| ≥ 17.

**Honesty note:** at the seen win share (θ=0.65) and 14.3% discordant rate, this run has **~57% power** (θ=0.55 → 0.09, θ=0.60 → 0.27, θ=0.70 → 0.84). Type-I at θ=0.50 → 0.033 *conditional on 57 discordants* (McNemar is conditional inference; the unconditional probability of a directional false positive is ≈0.02). These power figures assume all 396 pairs score; the §5 error-drop rule tolerates row loss up to the 30% void threshold, and power falls with it (θ=0.65: ~0.53 at 10% dropped, ~0.47 at 20%, ~0.41 at 30%) — high transport-error legs are expensive even when not void. Even a fully real effect fails this test almost half the time. We accept that. This is a genuine confirmatory test, not a formality, and a p ≥ 0.05 outcome is informative under §7.

### 4. Run plan (FROZEN — no changes between stamping and completion of both legs)

**Generation backend (cloud CUDA, identical for both legs):**
- ds4 fork, **main @324cc5a**, single binary for both legs (no boostfix/plain split — the binary asymmetry of the local campaign does not recur here).
- CUDA full residency: **no** `--ssd-streaming`, no expert-cache flags, no `DS4_METAL_MODEL_VIEW_MAX_GIB`.
- **imatrix collector OFF on both legs** (no `--imatrix-out/--imatrix-every/--imatrix-min-requests`). The collector is verified non-output-neutral; uniform-OFF is registered here.
- Launch per leg: `./ds4-server -m <gguf> --port 8011 -c 16384`; poll `/v1/models` to HTTP 200; one warmup chat with the fixed string `"Reply with the single word: ready."` (max_tokens 8), identical on both legs, response discarded and logged; graceful SIGINT stop. The exact launch command and the server startup log of each leg (evidence of: loaded model path, no ssd-streaming, no collector flags, `-c 16384`) are retained as run artifacts. Hardware: RunPod H200 141GB single (or AWS 4× g6e.xlarge if quota arrives first — whichever runs the canary runs both legs; no backend switch mid-pair).
- **GGUF integrity (frozen, no rebuilds):**
  - `plain2bit` = `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`, 86,720,111,488 bytes, **sha256 `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`** (computed locally at stamping time).
  - `coder-q4boost-v2` = `DeepSeek-V4-Flash-coder-q4boost-v2.gguf`, 97,591,747,488 bytes, **sha256 `9245ddb0499ed77068506617f1fc7272039797582bbe99730e6409f45615aaab`** (build manifest `output.sha256`; recipe `recipes/coder-q4boost-v2.json`, boost set [30,31,34,37,39,42]).
  - Both files are hash-verified on the cloud host (`sha256sum`) after transfer and before each leg's server start; the verification output is appended to REPORT.md with the leg record. A mismatch voids the leg before it begins (§5 void list).
- Note for the record: this backend differs numerically from the local Metal+ssd-streaming campaign. Both legs share it, so the paired comparison is internal to the cloud environment; cross-environment absolute scores are not endpoints.

**Harness (campaign-faithful, identical for both legs, run from this Mac against the tunneled server):**

```
cd /Users/chinaski/Beep/benchy-dash        # benchy 0.2.0 @4678de3 + working-tree state hashed below
export BENCHY_SERVER=<tunneled cloud URL>
export BENCHY_MODEL=<gguf basename of this leg>   # stamps artifact identity into the run record
BENCHY_ALLOW_CODE_EXEC=1 BENCHY_CODE_TIMEOUT=30 MPLBACKEND=Agg \
  /Users/chinaski/Beep/benchy-eval-stable/.venv/bin/python eval_code.py \
  data/bigcodebench_heldout.jsonl 0 nothink <plain2bit | coder-q4boost-v2>
```

- **Working-tree code identity:** `eval_code.py` in benchy-dash is dirty vs @4678de3 (the campaign v2 legs ran this exact dirty state). The code that executes is pinned by content hash, not commit: **`eval_code.py` sha256 `1f92166ba2b0bd56c4d3fc47afa6bf1beaffb80cc65846e8dfdd38434b04251e`**, **`benchy_common.py` sha256 `950fb3fec682d0cbd7f68d6727a3466839f6bdeab011c0bef35b554af8a82eec`**. The working tree at run time must hash to exactly these values; a mismatch voids the leg.
- **Runtime dataset gate active:** `BENCHY_SKIP_LOCK_CHECK` is NOT set. A `benchmarks.lock.json` entry in benchy-dash pinning `bigcodebench_heldout` to `content_sha df7449e2a3f391e710c2bb8db8101a3df1b20ae76f24daed0be8eb5a24b3fe86` is added in the stamping commit, so dataset drift aborts the run pre-hoc; `data_sha` in the run record must additionally stamp `df7449e2a3f3` — a mismatch voids the leg.
- **`BENCHY_LIVE_STREAM` unset on both legs** (plain blocking `bc.chat`, no streaming client path); the dashboard's live generation box stays off for the duration of both legs.
- **N=0 = all 396 rows** (verified in `load()`: `rows if n == 0 else rows[:n]`), after `random.Random(1234).shuffle` — pairing by construction, no tag/model dependence; rows joined by index `i`. `data/bigcodebench_heldout.jsonl` symlinked into `benchy-dash/data/` before the run.
- Temperature 0.0 (hardcoded), body `seed: 1234`, nothink `max_tokens=1536` (the shared-constraint budget; truncation hypothesis already tested and rejected), sequential single requests, HTTP read timeout 600s, per-task exec timeout 30s.
- **No** `BENCHY_CODE_PRELUDE` — the typing prelude is LCB-only; BCB runs bare, as in the campaign.
- Code-exec venv: `benchy-eval-stable/.venv` with numpy 2.0.2, pandas 2.3.3, faker, matplotlib, pyfakefs (verified importable). v2's campaign BCB runs passed natively on this venv; **no post-hoc rescore step exists in this protocol.** `MPLBACKEND=Agg` is set uniformly on both legs (headless-matplotlib guard; uniform across legs, so pairing is unaffected).

**Frozen items:** dataset file + full sha256 + lock entry, N=0, seed, temperature, max_tokens, mode, exec timeout, context 16384, ds4 commit and binary, GGUF files + sha256, collector OFF, `BENCHY_LIVE_STREAM` unset, warmup string, venv package set, harness commit + working-tree file hashes, env vars above, tag names, analysis invocation (§7). Nothing is touched mid-leg or between legs.

### 5. Sanity gates (not endpoints)

- **err_rate guard (registered here for the held-out analysis):** a leg with err_rate > 0.30 (strict) is void — a server-broken run never counts as a result. Implementation locus (part of the §7 committed analyzer diff): benchy-dash writes transport-error rows as `error: true` with no `ok` key (unmodified `analyze.py::load_details` would KeyError, and the `ERR:` answer heuristic cannot see them). The modified `load_details` skips `error: true` rows; the existing common-index intersection then excludes that row index from pairing on both legs. Per-leg err_rate = (count of `error: true` rows on that leg) / 396; each error row counts toward the err_rate of the leg on which the error occurred (only that leg).
- **Completeness gate:** each leg's details file must contain exactly 396 rows (ok + error combined); fewer voids the leg as a harness failure.
- **The harness's own `invalid` flag (errors > 5%) is informational only;** a leg is void solely under the registered gates here.
- **Void list (exhaustive, algorithmic):** err_rate > 0.30; server crash/restart mid-leg; harness exception; `data_sha` mismatch; GGUF sha256 mismatch on the cloud host; harness file-hash mismatch (§4); completeness-gate failure.
- **Re-run policy — technical failure only.** A leg may be voided solely for an item on the void list. A voided leg voids the **pair**: after the fix is documented in REPORT.md (mechanism, not vibes — house rule), BOTH legs re-run in full on the unchanged frozen config. Results are never grounds for a re-run. Superseding records follow the existing last-clean-wins convention, with the voided run struck through and labeled VOID in REPORT.md.
- **MBPP+ canary (serving sanity, runs BEFORE the endpoint):** n=30, nothink, both models, on the cloud backend, same frozen launch line, harness, and env as the legs. The canary runs under the **distinct benchmark name `mbppplus_canary`** (`data/mbppplus.jsonl` symlinked as `data/mbppplus_canary.jsonl`), which is NOT in SUITE — so canary records can never collide via last-wins with the campaign `mbppplus` records or enter any analysis. Purpose: catch gross serving breakage of the kind already seen once (HTTP-500 storms, garbage outputs). Algorithmic gate: canary err_rate > 0.30 or accuracy < 50% (local reference ≈ 80–83%) on either model → serving is broken; fix and re-run the canary before any held-out request. Every canary attempt is recorded in `runs.jsonl` (automatic) and enumerated in REPORT.md with the specific fix applied between attempts. **Canary scores do not gate, adjust, or enter the endpoint in any way.** Moderate score shifts vs local are expected (different numerical backend) and block nothing.
- **No peeking.** Details files stream during a leg; interim tallies are not read and never trigger a stop. Single shot: one paired run, one analysis, no sequential testing.

### 6. Secondary / exploratory (gates nothing)

- `coder-iq2` and `coder-q4boost` (v1) legs on the same 396 rows under the identical protocol, if/when run — **only after the §7 verdict and §8 analysis output are committed** (any analyzer invocation prints the endpoint row; running them earlier re-opens a peeking channel). Labeled exploratory; no multiplicity claims; they do not modify the §7 verdict.
- Any subgroup looks (per-library, prompt length, layer-39 activation, etc.): exploratory, reported as such if reported at all.
- Cross-environment comparisons (cloud vs local absolute accuracy): descriptive only.

### 7. Interpretation rules (all three outcomes pre-committed)

**Analysis invocation (exact, registered):**

```
/Users/chinaski/Beep/benchy-eval-stable/.venv/bin/python \
  /Users/chinaski/Beep/benchy-eval-stable/analyze.py /Users/chinaski/Beep/benchy-dash/results
```

run once, after both legs complete. (`benchy-dash/results` is a symlink to `/Users/chinaski/Beep/benchy-eval-stable/results` — one physical store, the same one the benchy-dash RunWriter writes to; no copying or syncing of records between checkouts.)

**THE endpoint is the single row (benchmark=`bigcodebench_heldout`, model=`coder-q4boost-v2`, mode=`nothink`) of the per-benchmark paired McNemar table: its b, c, and exact p.** Every other line of the output — campaign benchmark rows, the canary, both pooled sections, and the script's printed "Verdict (decision rule)" section (which keys on the old `lcb_v6_func` primary and is inapplicable here) — is non-confirmatory, struck from the appended output, and not quoted in the article.

The modified analyzer is committed in the same commit as this stamp, and was dry-run against the existing campaign records before stamping: campaign p-values and verdict lines byte-identical to the unmodified analyzer, the only output change being the registered both-fail column (the held-out endpoint row appears once its records exist). The complete diff, with no other changes permitted:
- (a) add `bigcodebench_heldout` to `SUITE` only — explicitly **NOT** to `CODE_BENCH` or `KNOWLEDGE_BENCH`, so no pooled statistic ever mixes held-out with seen rows. Any aggregate that includes held-out rows alongside seen rows is not computed, not reported, and not citable.
- (b) the §5 error-row rule: `load_details` skips `error: true` rows (index then dropped from pairing on both legs via the existing intersection); per-leg err_rate = error-row count / 396, replacing the `ERR:` heuristic, which cannot see benchy-dash error rows.
- (c) a both-fail column, computed mechanically as paired_n − both_pass − b − c (reporting, not analysis).

`binom_two_sided` is used verbatim. One analysis, run once, after both legs complete.

| Outcome | Verdict | Article consequence |
|---|---|---|
| **p < 0.05 and c > b** | **Confirmatory pass.** | Article publishes with this as THE confirmatory result; the pooled seen p=0.036 is permanently demoted to supporting/exploratory. |
| **p < 0.05 and b > c** | **Refuted.** | Article must report the failure. The v2 efficacy claim is dead in its current form. |
| **p ≥ 0.05** | **Not confirmed.** | Article either does not publish the efficacy claim, or reports it explicitly as unconfirmed (with the §3 power note). No respinning as a "trend"; no post-hoc subgroup rescue. |

There is no fourth outcome and no re-analysis under alternative conventions.

### 8. Provenance and tamper-evidence

- **Stamping commit (single commit, before any held-out request is served):** this REPORT.md section; the modified `analyze.py` (§7 diff); via `git add -f` (the `/data/` path is gitignored) `data/bigcodebench_heldout.jsonl` and `data/bigcodebench_heldout.provenance.json` with its `rule` field updated to the §2 wording. The benchy-dash `benchmarks.lock.json` entry (§4) is committed in benchy-dash at the same time.
- **Off-machine anchor:** the stamping commit is **pushed to an off-machine remote (GitHub) before the canary**; the remote ref and commit hash are recorded in REPORT.md. A local-only commit is amendable and does not count as registration. (The GitHub push and the S3 bundle are the registration anchors; no reviewer-email anchor is used.)
- Run records append-only to `results/runs.jsonl` with full `run_meta` (model, **BENCHY_MODEL = gguf basename** per leg, server, benchy_sha, **data_sha = df7449e2a3f3**, host, py); per-question details files retained under `results/details/` and referenced by the `details` field (tag is not in the filename — the runs.jsonl record is the link). Per leg, the server launch command and startup log (§4) and the cloud-host GGUF hash verification are retained and appended to REPORT.md with the leg record.
- The analysis output — the §7 endpoint row (both-pass / both-fail / b / c / accuracies / exact p; both-fail derived per §7c) — is appended to REPORT.md directly below this section, with the §7 verdict line, in a separate commit after the run. Nothing else from the analyzer output is appended.

**Decision (Andrea): canary first, then plain2bit leg, then coder-q4boost-v2 leg, then one analysis. Whatever the number says, it goes in the article under the §7 rules.**

---

#### Registration record (post-stamp, append-only)

- Stamping commit `ae1965b8ec477d5871539df6008882f5b01fe90c` pushed to
  **https://github.com/andreaborio/benchy** branch `heldout-prereg` (2026-06-12 ~14:25 local).
- Additional off-machine anchors: git bundle + `REGISTRATION.txt` at
  `s3://beep-forgequant-bench/prereg/` (uploaded 14:14 local, before the push).
- §2.4 operator attestation confirmed by Andrea, 2026-06-12 (session, "puoi fare tutto tu"
  in response to the explicit attestation request).
- Reviewer-email anchor: dropped (not used). The off-machine GitHub push (above) plus the S3 bundle are the registration anchors; an email adds nothing.

---

#### Amendment 1 — execution backend (pre-data, 2026-06-12 ~20:05)

Stamped BEFORE any request against `bigcodebench_heldout` or `mbppplus_canary`: as of this
commit, `results/runs.jsonl` contains no record for either benchmark name, and no server has
been started since the pre-registration stamp.

**Reason.** AWS GPU quota was only partially granted (8/32 vCPU on-demand, 8/32 spot, both
cases still open at the time of this amendment). 2× g6e.xlarge = 89.4 GiB usable VRAM < v2
weights alone (90.9 GiB). Rather than mix on-demand and spot pools (interruption = pair void)
or wait indefinitely, the endpoint runs on the SAME backend as the entire campaign: the local
Mac, Metal + SSD streaming.

**What changes (only §4 "Generation backend"; endpoint §2, prediction §3, harness identity,
sanity gates §5, secondary §6, interpretation §7 and analysis invocation are UNTOUCHED):**
- Backend: local Mac (M-series, 64GB RAM), ds4 binary built 2026-06-12 13:50 from fork main
  @43a0bf5 — which is the registered 324cc5a plus one README-only docs commit
  (`git diff 324cc5a..43a0bf5 --stat` = README.md only, verified). SINGLE binary for both
  legs: `~/Beep/ds4/ds4-server`, cwd `~/Beep/ds4` (build log `/tmp/ds4_build_main_20260612.log`).
- Launch per leg: `./ds4-server -m <gguf> --port 8011 -c 16384 --ssd-streaming
  --ssd-streaming-cache-experts 40GB` — the campaign's proven serving config. Collector OFF on
  both legs (no `--imatrix-*` flags), as registered. NOTE: the campaign runs had the collector
  ON uniformly; these legs have it OFF uniformly. Pairing is internal to the legs, so the
  uniform-state requirement holds; absolute scores vs campaign BCB rows are not comparable
  anyway (different rows) and are not an endpoint.
- No tunnel: harness on the same host, `BENCHY_SERVER=http://127.0.0.1:8011`.
- GGUF integrity re-verified at amendment time on the actual serving files:
  baseline `efc7ed607ff2…` (computed 2026-06-12 14:0x), v2 `9245ddb0499e…` (computed
  2026-06-12 20:0x) — both equal the registered hashes.
- Execution driver: `~/Beep/forgequant/run_heldout_local.sh` — canary plain2bit → canary v2
  (algorithmic gates: err > 30% or accuracy < 50% aborts) → leg 1 plain2bit (396) → leg 2 v2
  (396), one serving cycle each, warmup string as registered, full logs under
  `~/Beep/heldout_run_logs/`. Leg outputs are not inspected before the single analysis.
- Hardware lock unchanged: whichever backend runs the canary runs both legs. The canary runs
  on THIS backend; the AWS path is closed for this endpoint (any future cloud run of these
  rows would be a separate, labeled, non-confirmatory exercise).

---

#### HELD-OUT RESULT (registered analysis, run once — 2026-06-13 02:06)

**Endpoint row** (the only citable line of the analyzer output, per §7):

| benchmark | model | both ok | both fail | base only (b) | model only (c) | delta | p (exact) | sig |
|---|---|---|---|---|---|---|---|---|
| bigcodebench_heldout | coder-q4boost-v2 | 176 | 188 | 16 | 16 | +0.0pt | 1.000 | no |

Both legs 192/396 = **48.5%**. Discordant pairs 32/396 (8.1%); split exactly 16/16.
Independently recomputed from the per-question files (pairing by question sha256, 396/396
matched): identical numbers.

**VERDICT (§7, pre-committed): p ≥ 0.05 → NOT CONFIRMED.** The capture-refine efficacy claim
did not survive its held-out confirmatory test. The §3 prediction (c > b, win share ~0.65,
expected p ≈ 0.033) did not transfer: observed win share 0.50, point estimate of zero effect.
Per §7: the article either does not publish the efficacy claim, or reports it explicitly as
unconfirmed with the §3 power note. No respinning as a trend; no post-hoc subgroup rescue.
The pooled seen p=0.0357 is permanently demoted to exploratory; the distribution-overfitting
objection (§1.1) stands as the simplest explanation of the seen-vs-held-out gap.

**Leg records (gates: all PASS).**
- Canary plain2bit: 24/30 = 80.0%, 0 errors (matches campaign reference 80.0).
- Canary coder-q4boost-v2: 25/30 = 83.3%, 0 errors (matches campaign reference 83.3).
- Leg 1 plain2bit: 396/396 rows, 0 error rows, 21:04–23:02 local (~17.7 s/task).
- Leg 2 coder-q4boost-v2: 396/396 rows, 0 error rows, 23:02–02:05 (~27.5 s/task).
- Single binary both legs (`~/Beep/ds4/ds4-server` @43a0bf5), collector OFF, launch line per
  Amendment 1; GGUF hashes verified same-day (baseline 14:0x, v2 20:0x) == registered values.
- Driver + full logs: `~/Beep/forgequant/run_heldout_local.sh`,
  `~/Beep/heldout_run_logs/` (status.txt, 4 server logs, warmup.log, per-phase stdout);
  full analyzer output retained at `~/Beep/heldout_run_logs/analysis_20260613T0206.md`
  (everything but the endpoint row struck per §7).
- No held-out output was inspected before this analysis (driver never parses leg output;
  progress monitored by row count and transport errors only).

---

## Previous verdict (v1-only, 2026-06-11 20:00 — superseded by the above)

**Both coding-imatrix quants are ahead or at par on the pre-registered code legs —
coder-iq2 is never behind anywhere; coder-q4boost loses one 20-item extension slice by a
single question — with no detected knowledge tax. The advantage did NOT reach statistical
significance at the N we ran (pre-registered McNemar threshold p<0.05).**

Final pooled paired evidence (corrected/re-scored; **240 paired nothink code tasks** and
180 knowledge tasks per model, +15 thinking tasks not in the pool):

| | code: pairs won/lost | p | knowledge: won/lost | p | LCB primary delta |
|---|---|---|---|---|---|
| coder-iq2 | 17 / 10 | 0.25 | 12 / 9 | 0.66 | +6.0pt (p=0.51) |
| coder-q4boost | 15 / 9 | 0.31 | 13 / 7 | 0.26 | +6.0pt (p=0.38) |

Honest reading: a ~62% discordant-win share in a consistent direction (both modes,
nothink and thinking) is a real directional result — but at 24-27 discordant pairs the
exact test cannot rule out luck. **Power, recomputed:** two-sided significance at this
win share first occurs at ~58-67 discordant pairs; at the observed ~10-11% discordance
rate that means **roughly 350-430 more paired code tasks per model (~12-24 GPU-hours)**.
The q4boost interim p=0.065 did not survive its extension slice (BCB-ext came in at par)
— a textbook illustration of why interim peeks aren't verdicts.

Robustness checks (verdict unchanged under all): excluding LCB pairs contaminated by the
nothink 1536-token generation cap (~4-8% of LCB tasks per model show truncation; only ONE
discordant pair is affected — an iq2 win — excluding it: iq2 16W/10L p=0.33, q4boost
unchanged, its truncations all fell on both-fail tasks). Excluding the canary MBPP+ from the pool
(it was pre-registered as a gate, not a metric): iq2 p=0.23, q4boost p=0.40. Excluding the
one rescore flip not explained by the module fix (BigCodeBench/88, a *baseline* win, so it
biases against our claim): iq2 p=0.17, q4boost p=0.21. ~14 uncorrected pairwise tests were
run across 2 correlated models (same baseline run, same imatrix); no correction applied —
moot here since nothing crossed the threshold anyway.

**What this buys the article:** "ahead or tied on essentially everything, behind almost
nowhere, no detected knowledge tax, thinking-mode advantage preserved" — with the honest
statistical caveat, and **coder-q4boost-v2** (capture-driven boost refine, suite in
progress) as the live test of whether putting bits exactly where measured traffic goes
sharpens the result.

_This verdict and every number in it were independently reproduced from the raw
per-question files by a 3-auditor adversarial verification pass (stats recompute, pairing
integrity, claims-vs-evidence); the corrections it demanded are incorporated above._

| | LiveCodeBench v6 (primary) | other code | knowledge legs | thinking @24k |
|---|---|---|---|---|
| plain2bit (baseline) | 46.0% | BCB 42.5 | SuperGPQA-CS 35.0 / MMLU-Pro-CS 52.5 | 73.3% |
| **coder-iq2** | **52.0% (+6.0)** | **BCB 50.0 (+7.5)** | 35.0 (=) / **56.2 (+3.7)** | **80.0% (+6.7)** |
| coder-q4boost | **52.0% (+6.0)** | **BCB 50.0 (+7.5)** | **37.0 (+2.0)** / **57.5 (+5.0)** | **80.0% (+6.7)** |

Direction: coder-iq2 ahead or tied on every code leg in both modes, never behind;
coder-q4boost ahead on most legs, at par on BCB-ext, behind by one question on the
20-item LCB-ext slice. No detected knowledge tax. Final verdict in the TL;DR above.

**Decision rule (from the research phase):** the coding-imatrix recipe *wins* iff its
**paired** LiveCodeBench delta vs plain2bit is positive **and** statistically meaningful
(exact McNemar on discordant pairs), **AND** its SuperGPQA-CS / MMLU-Pro-CS deltas are not
significantly negative (no knowledge tax).

---

## What is being compared

Three GGUFs of **DeepSeek-V4-Flash** (284B MoE / 13B active, 1M ctx, the community's
default locally-runnable agentic-coding model in 2026). Identical in every way **except how
the 2-bit budget is allocated**:

| tag | file | imatrix used to quantize | size |
|---|---|---|---|
| **plain2bit** | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | generic **chat** imatrix (NOT coding) | ~87 GB |
| **coder-iq2** | `DeepSeek-V4-Flash-coder-iq2.gguf` | `coder.dat` (HumanEval+MBPP+MMLU-CS calibrated) | ~87 GB |
| **coder-q4boost** | `DeepSeek-V4-Flash-coder-q4boost.gguf` | `coder.dat` + Q4_K upcast on the 6 code-hottest layers | ~97 GB |

- **coder-iq2** is the cleanest comparison: *identical* bit budget to the baseline, only the
  imatrix differs → isolates the value of coding-aware bit steering.
- **coder-q4boost** spends ~10 GB more (Q4_K on 6 layers) → tests whether the layer boost
  buys additional headroom on top of the imatrix.

Order of execution: `plain2bit` → `coder-q4boost` → `coder-iq2` (so baseline + flagship are
comparable first if anything stops mid-run).

---

## Why these benchmarks (and why NOT the originals)

The original recipe calibrated/intended HumanEval, MBPP, MMLU-CS. **These are saturated** for
a model this strong: in the very first (aborted) run, plain2bit opened **6/6 on HumanEval** —
zero discriminating power. A 2026 web-research pass (4-agent workflow) replaced them with
**non-saturated, contamination-resistant** sets that fit our two harness formats (MCQ jsonl;
HumanEval-style code-exec jsonl with local `assert` tests):

| benchmark | role | rows built | format | why |
|---|---|---|---|---|
| **LiveCodeBench v6 (functional, 2025+)** | **PRIMARY** code | 70 (sampled to 50/run) | code-exec | contamination-resistant (contest dates 2025-01→2025-04, after Flash's training); frontier <50% on hard; the benchmark the coding-imatrix hypothesis is *about* |
| **SuperGPQA-CS** | primary knowledge | 763 (→100/run) | MCQ (10-opt) | least-saturated MCQ (~60-65% frontier); detects an "imatrix tax" on CS knowledge |
| **MMLU-Pro-CS** | secondary knowledge | 410 (→80/run) | MCQ (10-opt) | cross-check on knowledge leg |
| **MBPP+ (EvalPlus)** | **canary** only | 378 (→30/run) | code-exec | rigorized MBPP tests; 10-min gate to catch a broken quant before spending the LCB budget — never a headline metric |
| **BigCodeBench** (libs-filtered) | optional code | 536 (→40/run) | code-exec | practical, library-using code (filtered to stdlib+numpy+pandas so tests actually run) |
| **LCB v6 (thinking)** | optional | 15/run | code-exec | quality probe with reasoning enabled |

**Dropped as saturated/legacy:** HumanEval, MBPP, MMLU-CS (cais/mmlu CS cluster).

Datasets, dataset ids, splits and row SHAs are recorded in `data/*.provenance.json`.

---

## Methodology / validity

- **Paired by construction.** All eval scripts use `SEED=1234`: same rows, same order, same
  deterministic option-shuffle (seeded by question text) for every model. So per-question
  correctness is directly comparable → **exact McNemar** on discordant pairs (roughly doubles
  effective power vs comparing independent accuracies).
- **Greedy, deterministic** decoding (`temperature=0.0`). Confirmed reproducible: MBPP+ canary
  scored exactly **80.0%** on two independent launches.
- **Honest scoring.** Code benchmarks **execute** model output against the real tests in an
  isolated subprocess (`BENCHY_ALLOW_CODE_EXEC=1`, 30s timeout). `err_rows` (transport/server
  failures) is logged per eval; an eval with >10% err_rows is auto-retried once, else dropped.
- **Live imatrix capture** is ON for every model (`ds4-server --imatrix-out`, snapshot every
  32 requests) — the real expert-activation paths under coding traffic are saved to
  `~/Beep/ds4-models/bench-captures/<tag>__coding-bench__20260610.dat` for post-hoc refines.

### Design choice: nothink as the primary mode (for the article)

All headline legs run with reasoning **disabled** (`think: false`, greedy, temp 0). This
is deliberate, not a shortcut:

1. **Statistical power per GPU-hour.** Measured on this machine: nothink ≈ 4–10 s/question,
   thinking ≈ 35–200 s/question (LCB ≈ 3.5 min/task). At equal wall-clock, thinking yields
   ~1/10 of the samples. A paired A/B is decided by **discordant pairs** (questions one
   model gets right and the other wrong); with thinking-sized N you never accumulate
   enough of them to reach significance.
2. **Internal validity is unaffected.** Both quants run the identical mode, prompt order,
   and option shuffle — the object of measurement is the *delta* between imatrices, not
   the model's absolute score. The mode choice changes sensitivity, not fairness.
3. **Thinking masks exactly the damage we want to measure.** Reasoning self-corrects
   quantization errors and compresses differences toward the ceiling — observed in the
   medical baselines (MedQA 56% nothink → 92–97% thinking on the same quant) and hinted
   at tonight: plain2bit scored *lower* on LCB with thinking (33.3%, n=15) than without
   (46.0%), consistent with degraded reasoning tokens accumulating errors over long
   generations at 2-bit. Direct generation exposes the quant's raw quality, which is what
   an imatrix changes.
4. **Less execution noise.** Long generations add timeouts/truncations uncorrelated with
   model quality.

**Caveat kept honest:** real coding-agent usage is thinking-mode, so each model also runs
a small **paired thinking probe** (LCB ×15, time-gated) as a directional sanity check —
not a headline metric. A dedicated thinking slice (~1h/model per 15 tasks) is the natural
follow-up if the nothink verdict needs confirmation in agent-mode conditions.

#### Thinking probes: VOIDED by a harness artifact (token-budget cap)

The plain2bit thinking number (33.3%, n=15) initially read as "thinking hurts codegen at
2-bit". A pre-registered expectation was written for the coder-iq2 probe (point estimate
6/15 from the paired nothink data on the identical tasks). Then **a manual inspection of
the failed transcripts (Andrea, 00:10) found the real mechanism**: with `max_tokens=4096`,
the reasoning chain on these contest problems consumes the whole budget — generation stops
mid-reasoning, **no code is ever emitted**, and the harness executes leftover prose, which
fails as a SyntaxError.

Failure classification of the plain2bit thinking probe (n=15):

- **5 PASS** · **1 genuine FAIL** (valid code, tests failed) · **9 = token-cap artifact**
  (still reasoning at cutoff, or code truncated mid-emission)

**Both thinking probes are therefore VOID as quality measures.** The coder-iq2 probe was
aborted mid-run to save GPU time. What survives, with evidence grading:

- **Solid:** the MCQ result — thinking lifted MedQA from 56% to 92–97% on the same 2-bit
  quant class (n=85, budget was sufficient there since the answer is one letter).
- **Solid, but a cost observation, not a quality one:** at 2-bit (~15 t/s) thinking-mode
  codegen on contest problems needs >4096 generated tokens per task — i.e. 10+ minutes per
  question. Operationally expensive regardless of quality.
- **Withdrawn:** any claim that thinking *quality* degrades on codegen at 2-bit. Unmeasured.

**Decision (Andrea, 00:30): B** — the thinking-degradation question matters for the
article, so the probes are being re-run overnight with a real budget. Design of the
re-run (`orchestrator_think.py`, launched ~00:40):

- same **15 paired LCB tasks** as the voided probes (same seed → directly comparable);
- generation budget **24576 tokens** — Andrea proposed 32k; trimmed to 24k on the
  grounds that at 2-bit, degenerate repetition loops are common and each 32k loop burns
  ~35 min of GPU; **any chain still running at 24k is degenerate looping, which counts
  as a model failure, not a harness artifact** (and will be classified as such, not
  silently scored);
- ctx raised to 28672; eval HTTP timeout raised to 3000s — the original 600s socket
  timeout was the *next* latent artifact in line (a 24k chain at ~15 t/s ≈ 25 min);
  both knobs are new env vars in `eval_code.py` (`BENCHY_THINK_MAX_TOKENS`,
  `BENCHY_HTTP_TIMEOUT`), defaults unchanged;
- order: plain2bit first (replaces its void record), then coder-iq2; imatrix capture to
  separate `__coding-bench-think__` files so the nothink captures stay pristine;
- morning read-out: pass@1 per model **plus a truncation classification** (PASS / genuine
  FAIL / still-capped-at-24k) — only measured tasks count toward the thinking-vs-nothink
  comparison.

_Method note kept for the article: the pre-registration practice did its job — the probe
band was declared before the data landed — and the artifact was caught by reading failed
transcripts (Andrea spotted reasoning prose where code should be) rather than trusting
the aggregate score. Both habits are worth keeping._

#### Re-run RESULTS (overnight 00:22–06:16, budget 24576): thinking does NOT degrade — it transforms

Same 15 paired LCB tasks, proper budget. Both models completed cleanly (rc=0, ~11–12 min
per task — the operational-cost note stands):

| | nothink (same 15 tasks) | thinking @4096 (VOID) | **thinking @24576** | classification |
|---|---|---|---|---|
| plain2bit | 6/15 (40.0%) | (5/15 — artifact) | **11/15 (73.3%)** | 11 PASS · 3 genuine FAIL · 1 degenerate loop |
| coder-iq2 | 6/15 (40.0%) | (aborted) | **12/15 (80.0%)** | 12 PASS · 2 genuine FAIL · 1 degenerate loop |

**Answers to the question that motivated the re-run** (does thinking degrade under 2-bit
quantization?):

1. **No collapse — the opposite.** Given room to finish its chains, the 2-bit model goes
   from 40% to 73–80% on the same contest problems. Reasoning survives 2-bit quantization
   and remains the single biggest quality lever, exactly as in the medical MCQ data.
   The earlier "thinking is a cost at 2-bit" reading was 100% harness artifact.
2. **The coding imatrix keeps its edge in thinking mode** (12 vs 11 PASS — directionally
   consistent with the nothink +6pt; n=15, so directional only).
3. **The one true 2-bit thinking pathology observed: degenerate loops** — 1/15 tasks per
   model (~7%) was still "reasoning" at 24,576 tokens with no code emitted. This, not
   quality loss, is the failure mode to watch (and a candidate metric for comparing
   quants: loop rate under long reasoning).
4. **The real cost is operational**: ~11–12 min/task at ~2k-token/min on this hardware —
   thinking-mode coding at 2-bit is a quality play, not a latency play.

_All thinking-mode imatrix activity captured to separate `__coding-bench-think__` files
(usable to check whether reasoning traffic lights up different experts than direct
generation — relevant for the refine step)._

### Harness fix applied mid-run (documented for reproducibility)

The first LCB run scored plain2bit at **16%** — but failure-mode analysis showed **31/50
fails were `NameError: name 'List' is not defined`**, i.e. the benchmark's own LeetCode-style
starter code (`def f(self, nums: List[int])`) failed at class-definition time because our
runner didn't pre-import `typing`. The **official LiveCodeBench executor pre-imports that
environment.** Fix: `eval_code.py` now injects a `BENCHY_CODE_PRELUDE`
(`from typing import *` + common modules) before the candidate, set by the orchestrator only
for `lcb*` datasets. Re-validated on the real failed candidates: NameErrors became genuine
PASS/AssertionError outcomes. Corrected baseline LCB = **46.0%**. The 16% number is void.

---

## Infrastructure

- **Stable benchy clone:** `~/Beep/benchy-eval-stable` at pinned submodule commit `81b3efa`
  (isolated from the live submodule another agent was editing).
- **Orchestrator:** `orchestrator.py` — serves each model with
  `ds4-server --ssd-streaming --ssd-streaming-cache-experts 40GB` on **port 8011** (cwd = ds4
  checkout so Metal sources load), runs the suite, SIGINT-stops the server (final imatrix
  flush), watchdog restarts + 1 retry on failure. Runs under `caffeinate` (no sleep).
- **venv:** `.venv` (numpy 2.0.2 + pandas 2.3.3) — required to execute EvalPlus/BCB tests.
- **Logs:** `orchestrator.log`, `status.json` (live state), `results/runs.jsonl`,
  `results/details/*.jsonl` (per-question), `evals/*.log` (raw per-eval stdout).
- **Dashboard:** `http://localhost:8051` (live feed; the `:8050` one is a different checkout —
  ignore it for this suite).
- **Analysis:** `analyze.py` → paired McNemar vs plain2bit.

---

## Results log (live)

Accuracy per (model, benchmark, mode). Updated as each eval finishes.

### plain2bit (baseline) — core complete

| benchmark | mode | N | accuracy | err_rows | time |
|---|---|---|---|---|---|
| MBPP+ (canary) | nothink | 30 | **80.0%** | 0% | 309s |
| **LiveCodeBench v6** | nothink | 50 | **46.0%** | 0% | 2541s |
| SuperGPQA-CS | nothink | 100 | **35.0%** | 0% | 655s |
| MMLU-Pro-CS | nothink | 80 | **52.5%** | 0% | 541s |
| BigCodeBench | nothink | 40 | **42.5%** | 0% | 716s |
| LCB v6 | thinking @4096 | 15 | ~~33.3%~~ VOID (token-cap artifact) | 0% | 3300s |
| LCB v6 | thinking @24k (re-run) | 15 | **73.3%** | 0% | 686s/q |

### coder-iq2 — running (relaunched 22:14, runs FIRST)

_Cleanest comparison: identical 87 GB bit budget vs baseline, only the imatrix differs._

| benchmark | mode | N | accuracy | vs plain2bit | err_rows | time |
|---|---|---|---|---|---|---|
| MBPP+ (canary) | nothink | 30 | **80.0%** | = (80.0) | 0% | 319s |
| **LiveCodeBench v6** | nothink | 50 | **52.0%** | **+6.0pt** (46.0) | 0% | 2812s |
| SuperGPQA-CS | nothink | 100 | **35.0%** | = (35.0) | 0% | 703s |
| MMLU-Pro-CS | nothink | 80 | **56.2%** | **+3.7pt** (52.5) | 0% | 595s |
| BigCodeBench | nothink | 40 | **50.0%** | **+7.5pt** (42.5) | 0% | 768s |
| LCB v6 | thinking @24k | 15 | **80.0%** | **+6.7pt** (73.3) | 0% | 727s/q |

**coder-iq2's suite is complete** (the @4096 thinking probes were voided as harness
artifact and re-run overnight with budget 24576 — see methodology). Imatrix captures
flushed for both models, nothink and thinking traffic in separate `.dat` files.

Paired McNemar on LCB (N=50): coder-iq2 wins 6 questions plain2bit misses, loses 3
(20 both-correct) → **+6pt, directionally positive, p=0.508 — not yet significant**
(only 9 discordant pairs). Per the research plan, power will be extended overnight with
a second paired LCB slice (the remaining 20 of 70 built items) and a BigCodeBench
extension (536 items built) before the final verdict.

### coder-q4boost — suite running on the PATCHED ds4 (2026-06-11)

| benchmark | mode | N | accuracy | vs plain2bit | vs coder-iq2 |
|---|---|---|---|---|---|
| MBPP+ (canary) | nothink | 30 | **83.3%** | +3.3 (80.0) | +3.3 (80.0) |
| **LiveCodeBench v6** | nothink | 50 | **52.0%** | **+6.0** (46.0) | = (52.0) |
| SuperGPQA-CS | nothink | 100 | **37.0%** | **+2.0** (35.0) | +2.0 (35.0) |
| MMLU-Pro-CS | nothink | 80 | **57.5%** | **+5.0** (52.5) | +1.3 (56.2) |
| BigCodeBench | nothink | 40 | **50.0%** | **+7.5** (42.5) | = (50.0) |
| LCB v6 | thinking @24k | 15 | **80.0%** (12/15) | **+6.7** (73.3) | = (80.0) |

Both coding-imatrix models beat the baseline by the same +6.0pt on the primary
benchmark; the Q4 boost adds a knowledge-leg gain on top (+2.0/+5.0 vs baseline).
Zero transport errors on the patched binary throughout.

## Calibration miss found via live captures: layer 39 (the refine, pre-registered)

The live imatrix captures (real benchmark traffic) vs the calibration (`coder.dat`)
expose exactly one materially mis-allocated boost slot:

| layer | calibration energy (rank) | real-traffic energy (rank) | boosted? |
|---|---|---|---|
| 30 | 12.5% (2) | 15.9% (**1**) | yes |
| 37 | 15.1% (1) | 13.6% (2) | yes |
| 42 | 6.7% (5) | 11.3% (3) | yes |
| **39** | **4.3% (10)** | **9.6% (4)** | **NO — missed** |
| 31 | 6.6% (6) | 7.1% (5) | yes |
| 34 | 9.5% (3) | 6.7% (6) | yes |
| 41 | 7.7% (4) | 5.4% (7) | yes (weakest in set) |

The `auto:6` energy pick was faithful to the calibration — but the calibration corpus
was built on HumanEval/MBPP/MMLU-CS, the very benchmarks this report retired as
saturated. 2025 contest-style code (LCB) more than doubles layer 39's energy share
(4.3% → 9.6%, rank 10 → 4). The calibration-vs-reality gap closes the article's loop:
**saturated calibration data ⇒ bits not quite where the real work is.**

**Refine v2 (pre-registered before testing):** `coder-q4boost-v2` = identical recipe but
boost set `[30, 31, 34, 37, 39, 42]` (swap 41 → 39). Same imatrix (`coder.dat`) so
`--reuse` applies: only 2 layers requantize (~5 min build, the post's headline feature
in action). **Decision (Andrea):** finish the v1 test program first (q4boost thinking +
extension slices for significance), then build and benchmark v2 as the 4th paired column.
Prediction to hold us honest: v2 should match or beat v1 on LCB/BCB (39 is where real
code lives) with no knowledge-leg regression; the cleanest detectable signal would be on
the tasks that activate layer 39 hardest.

### coder-q4boost — original block (resolved above, kept for the record)

First attempt returned **HTTP 500 on every request** (eval scored 0%, 100% err_rows —
that garbage record was stripped from runs.jsonl). Root cause from `server.log`:

```
ds4: Metal model range 50.64..51.76 GiB is not covered by mapped model views
ds4: gpu layer 30 ffn batch encode failed
ds4: gpu layer-major prefill layer 30 encode failed
```

**Mechanism** (`ds4_metal.m` `ds4_gpu_wrap_model_range` / model-view mapping): under
`--ssd-streaming` ds4 maps the GGUF as a few overlapping Metal buffers ("views"); the
overlap is sized to the largest **single** tensor so every tensor lies wholly inside one
view. coder-q4boost upcasts the 6 code-hottest layers to **Q4_K** (layer 30 is one), whose
FFN expert **batch** span (~1.12 GiB at file offset 50–52 GiB) appears to exceed that
overlap and straddle a view boundary → no view contains it → encode fails → 500. plain2bit
& coder-iq2 (uniform IQ2, 87 GB) don't hit this. **This is a ds4-server limitation with the
mixed-precision boosted model under SSD streaming — not an harness bug, not a model-quality
result.** ("invalid syntax" seen in that eval's log is downstream: the harness tried to
`exec` the `ERR:HTTP 500` string as Python.)

**RESOLVED (2026-06-11 morning) — ds4 patched, validated, q4boost running.**

Root cause (refined by a multi-agent source-audit workflow): two compounding issues.
(1) *Prefill*: under streaming's "batch selected addr" mode, per-layer maps use the
decode-static span set, which omits routed expert tensors; the Metal fallback for a
non-IQ2 layer wraps the full fused tensors via `ds4_gpu_wrap_model_range` → range not
covered → encode fails. (2) *Decode (latent)*: the expert cache is a single-size-class
slab allocator sized from the FIRST routed layer; off-size (Q4_K-boosted) experts would
poison the byte budget and can deadlock slab reuse after an mlock cap.

Patch (worktree `~/Beep/ds4-boostfix`, ~114 lines, 5 files, uncommitted):
- **Piece A**: `weights_streaming_layer_experts_uniform()` detects boosted layers
  (per-expert bytes ≠ slab class); the decode-span builders now include their exps
  tensors → mapped views cover both prefill and decode reads.
- **Piece B**: the slab size class is pre-seeded at startup and
  `note_expert_size` *rejects* off-size layers (freeze+reject instead of
  last-writer-wins) → boosted layers take the existing per-expert
  `wrap_model_exact_range` path. CUDA/ROCm no-op stubs added.
- Startup now logs: `mixed-precision model: 6/43 routed layers off the slab size class
  will bypass the expert cache`.

Validation (all green, 08:20–08:31):
1. **Diagnosis confirm** — unpatched binary + `DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1`
   (legacy full-layer maps): q4boost answers correctly. Root cause proven end-to-end.
2. **Default untouched** — patched binary on uniform coder-iq2 reproduces recorded
   benchmark answers **byte-identically** (3/3, greedy determinism).
3. **Patched canary** — q4boost MBPP+ **83.3% (25/30), 0 transport errors, ~14 s/task**
   (above both other models' 80.0%). Full suite launched on the patched server
   (cache lowered 40→24 GB to fund the ~20 GiB of extra mapped views).

---

## Timeline / ETA

- 18:23 first launch → aborted (ds4-server cwd / Metal sources) → fixed
- 19:00 launched legacy suite → spotted saturation (HumanEval 6/6) → stopped
- 19:00–19:50 research workflow + built 2026 datasets + harness validation
- 19:52 launched 2026 suite → LCB harness bug (typing prelude) → fixed
- 20:00 relaunched corrected suite
- ~22:05 plain2bit done (est.) → coder-q4boost (~2h20) → coder-iq2 (~2h20)
- **All tests done: ~02:30–03:00. Analysis + verdict: ~03:30.**

#### Truncation branch: CLOSED (hypothesis tested and rejected, 2026-06-11 evening)

The "capped tasks just need more space" hypothesis was tested by re-generating truncated
LCB tasks at a 3072 budget with a token-exact replication guard (greedy prefix must match
the stored truncated text; three guard iterations were needed to compare normalized text
fairly, and request shape — including the updated harness's `seed` field — had to be
replicated run-by-run). Result: **0 of 6 verified re-generations flipped FAIL→PASS**; one
task saturated even 3072 (13.8k chars). Combined with the earlier robustness check (only
1 discordant pair in the whole verdict was truncation-contaminated), the branch was cut
(Andrea's call, cost/benefit): **the official scores stand on the uniform 1536 budget,
documented as a shared constraint.** Coda chiusa con un finding: 3 tasks failed prefix replication — single-task test (i=12)
confirmed the cause: **ds4's `--imatrix-out` collector is not output-neutral** (it switches
the prefill path; near-tie logits resolve differently). With the collector enabled, the
regenerated prefix is token-identical. Greedy IS deterministic per full configuration —
all our benchmark runs had capture ON uniformly, so pairing fairness was never affected.

#### Side finding: boosted models generate longer answers — because they solve longer problems

Observed live (Andrea) and quantified on identical LCB tasks: mean answer length rises
with boost level (plain2bit 1057 ch on its own passing solutions → v2 1722 ch). But on
the 11 tasks **all four models pass**, the style gap shrinks to +3-13% (856→884-971 ch):
the verbosity gradient is mostly **composition** — boosted models solve harder tasks
whose correct solutions are simply longer. The extra tokens are earned, not babble. This
also explains the truncation gradient (2/3/4 cap-hits from baseline to q4boost) and
motivates the queued cap-lifted re-measurement (planned correction: re-generate the ~11
truncated tasks at 3072 budget after the v2 suite, amend records, last-wins).

#### Hypothesis to analyze: "thinking in the comments" and the layer-41 de-boost (Andrea, live observation)

> **Promoted to its own investigation: see [REASONING_LAYER.md](REASONING_LAYER.md)** —
> dedicated chapter with the full evidence table, competing hypotheses (H0/H1/H2),
> tonight's free ablation test (v2 vs v1 differ only by the 41↔39 swap), and the
> experiment plan for a separate article (single-layer ablation builds via `--reuse`).

Watching v2's live generations, the model appears to **smuggle a full reasoning chain into
code comments** when thinking is disabled. Real examples from v2's LCB run (nothink):

```
# For each value x, we need max subarray sum after removing all x.
# That is equivalent to: we split the array at each occurrence of x,
# ... But also we can combine segments? No, because removal removes all x...
```
```
# Let's derive:
# We start at -1. First move to 0: visit 0 once.
# For i from 0 to n-2: ... Each extra visit to i requires a round trip...
```

The mechanistic hypothesis worth recording: **v2 removed the Q4 boost from layer 41 — the
single most thinking-distinctive layer in the capture contrast analysis (5.9% think-vs-
nothink divergence)** — so de-precisioning it might push reasoning behavior elsewhere.

Current evidence does NOT yet support a v2-specific effect: comment-line density on the
31 shared LCB tasks is flat across all four quants (plain2bit 42.8%, iq2 44.0%, q4boost
39.8%, v2 40.7%) — "reasoning as comments" is a DeepSeek-V4-Flash nothink behavior on
hard problems, shared by the baseline. **Open analyses:** (1) v2's thinking@24k leg
(tonight) vs q4boost's — if layer 41 matters for reasoning, the de-boosted v2 should
diverge there (accuracy and/or chain length/loop rate); (2) semantic classification of
comments (derivation vs documentation) rather than raw density; (3) paired per-task
comment-block length deltas. Status: hypothesis, to analyze.

## Open caveats

- **Harness split (disclosed for exactness):** the core suites ran on the stable benchy
  clone @ 81b3efa; the extension slices (lcb_v6_ext + bigcodebench_ext, 120 of the 240
  pooled pairs) ran on the updated benchy @ 4678de3 (v0.2.0) with
  `BENCHY_SKIP_LOCK_CHECK=1` (custom datasets carry their own provenance files) and the
  `BENCHY_CODE_PRELUDE` port. Pairing fairness is preserved — all three models ran each
  slice on the identical harness, same seed/shuffle semantics (verified by independent
  audit) — but the environment differs *across* slices.
- **BCB rescore detail:** BigCodeBench tests import faker/matplotlib/pyfakefs beyond the
  dataset's `libs` field; after installing them, stored deterministic generations were
  re-executed: plain2bit ext 37→47 (10 flips), iq2 ext 44→48 (4 flips), zero PASS→FAIL
  anywhere. One flip (BigCodeBench/88, plain2bit) is unexplained by the module fix
  (likely timing): it is a baseline win, and excluding it the verdict is unchanged.
- **Binary asymmetry:** q4boost (and v2) are served by the patched ds4-boostfix binary
  with a 24GB expert cache vs the original binary at 40GB for the uniform models;
  byte-identity of the patched binary on uniform models was verified (3/3 greedy replay),
  but q4boost deltas formally conflate quant recipe with server patch.

- **Mixed-precision serving config**: q4boost runs with `--ssd-streaming-cache-experts 24GB`
  (vs 40GB for uniform models) to fund ~20GiB of pageable mapped views for the 6 boosted
  layers → decode ~25-30% slower (~9.9 t/s thinking vs ~12-15). Results are unaffected
  (greedy determinism; storage path doesn't change weight values) and the 3000s HTTP
  timeout has ~17% headroom at this speed. Raising the cache to 40GB on a 64GB box would
  likely *hurt* (wired cache would evict the pageable views the boosted layers touch every
  token). Queued follow-up: a 15-min 24/30/36GB sweep on t/s, after v2. Decision (Andrea,
  2026-06-11): keep 24GB, don't touch mid-leg.

- This harness cannot measure the most-reported quantized-Flash field failure — **tool-calling
  amnesia near 50k tokens** — so a coding-imatrix "win" here does not certify agentic tool-use
  fidelity at long context. Noted for the refine discussion.
