# Traps & recovery

Ranked roughly by how often they bite in real efforts. The index in SKILL.md
lists these by symptom.

## "No available shared memory broadcast block" is (usually) not a hang

```
INFO shm_broadcast.py:... No available shared memory broadcast block found in 60
seconds. This typically happens when some processes are hanging or doing
time-consuming work (e.g. compilation ...)
```

Almost always this is **benign long compilation** (7–25 min/graph for dense
models; can be multi-hour for MoE), not a deadlock. **Do not kill a Neuron
compile.** Distinguish the cases:

- **Compiling (benign — wait):** worker processes show CPU% > 100%
  (`ps aux | grep -E 'Worker_TP|neuronx-?cc'`); the NEFF count in the compile
  cache is growing; `neuronx-cc` compiler pass names advance in the log
  (`SubgraphForkPass → ModuleForkPass → coloring_allocator → code generation`).
- **True hang/deadlock:** no CPU on workers, no NEFF growth, or the run is wedged
  retrying `shm` after an earlier `Allocation Failure`.
- **Multi-rank checkpoint contention:** many ranks reading a large sharded
  checkpoint from a slow/near-full shared filesystem can stall in `read()` so a
  rank never reaches warmup. Stage the checkpoint to node-local disk
  (`/tmp`, `/dev/shm`) before launch.

Raise timeouts for large models: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`, and
for MoE `VLLM_NEURON_COMPILATION_TIMEOUT=14400`.

## Profiler buffers can't allocate (silent no-NTFF)

Covered under
[serve_profiling.md → Making room for the profiler](serve_profiling.md#making-room-for-the-profiler-on-large-models).
Symptom: generation succeeds but the output dir is tiny and the log shows
`nrt_profile_notification_subscribe_nc failed` / `Failed to allocate ... profiler
buffers` / `status=4 Allocation Failure`. Fix: lower `--gpu-memory-utilization`.

## Session NTFF too large to ingest

`max_iterations=50` can produce a 4.8–5 GB NTFF; `neuron-explorer view
--ingest-only` then runs 2+ hours at ~40 GB RSS and may silently emit **0-byte
parquet**. Keep windows small (`max_iterations≈5`, `delay_iterations` to skip
prefill). If ingestion produces 0-byte parquet, the NTFF was too large — re-profile
smaller.

## Compile-time HBM OOM (`NCC_EOOM002`)

```
[ERROR] [NCC_EOOM002] Maximum peak HBM usage of NN GB exceeds HBM limit of
24.00 GB for Trn2 ...
```

Oversized I/O tensors (large context × seqs), or a **stale compile-cache entry**
from a smaller/older test config being reused (e.g. a 1-layer/217K-block entry
driving a giant KV allocation). Try in order:

1. **Clear a stale cache entry:** `rm -rf ~/.cache/vllm/neuron/compile_cache/`
   (or `/tmp/vllm_neuron_wdir_$USER/neuron/compile_cache/` on NFS homes). This is
   often the real cause.
2. Shrink `--max-model-len` (e.g. 64K → 16K → 8192).
3. `--kv-cache-dtype fp8` to halve the KV footprint.
4. Tune `--gpu-memory-utilization`.
5. `VLLM_NEURON_MIN_KV_BUDGET_GIB` (e.g. 0.5 / 0.1) lowers the KV-budget floor;
   `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION` (default 0.30, must be in (0, 1]) caps
   the GMU-scaled budget. Both are real `vllm_neuron` env vars — but confirm they
   exist in your installed version before relying on them (`grep -r <name>
   $VLLM_NEURON_DIR/vllm_neuron/envs.py`); their availability has changed across
   versions.

## Method 2 fails on MoE / large-TP models

Two hard blockers make offline capture unusable for these models — **use Method 1**:

- `neuron-explorer capture` on a TP=32 NEFF crashes because it can't reconstruct
  the baked-in replica groups:
  `Assertion 'group_id < ctx->replica_groups.size()' failed`.
- `NEURON_RT_INSPECT_*` device profiling alters the graph so MoE / indexer NKI CTE
  kernels' internal activations exceed the 8-slot activation-table limit:
  `[NCC_INLA001] ... the number of activation tables must be <= 8`.

## DGE notification overflow on MoE session capture

Session-mode capture **with** DGE notifications on a MoE decode graph can flood
the notification queue: `notification queue overflow ... Too many notifications
generated during execution`. Leave DGE off for session/serve capture of these
models (Method 1's serve path does not enable DGE). Note: DGE-on capture on a
dense model can also hang with tight KV configs (fp8 + large batch + tight block
pool) — enable it only for the isolated `neuron-explorer capture` step if needed.

## Orphaned device-holding workers after a killed/failed run

Killing the parent leaves survivors in a **different process group** — not just TP
workers, but also `neuronx-cc` compiler subprocesses, `EngineCore`, `DPEngineCoreProc`,
and `run.py`-named processes. Next launch fails with `The PyTorch Neuron Runtime
could not be initialized` / `nrta status=5` / `WorkerProc initialization failed`.

> The isolation recipes (`run_profile_decode.sh` / `run_profile_prefill.sh` →
> `profile_run.py`) launch and tear down through `safe_serve.py`, which SIGKILLs the
> server's own recorded process group on every exit path (normal completion, timeout,
> exception, Ctrl-C). A recipe run that ends by any of those routes does **not** strand
> these workers. The manual recovery below is for servers killed by other means (a hard
> `kill` of the launcher, an OOM-killer, a crashed shell) where no recorded runfile
> teardown ran — and, when a runfile does exist, prefer
> `python3 safe_serve.py teardown --runfile <that run's runfile>` (recorded-PGID only)
> over the substring `pkill` below.

```bash
S=<skill-dir>/references/safe_serve.py
python3 $S teardown --runfile <run>.json     # SIGKILL the server's OWN isolated group only
neuron-ls                                     # cores held by your run should show PID = NA
fuser /dev/neuron*                            # confirm the device drained
rm -rf ~/.cache/vllm/neuron/compile_cache   # a poisoned cache can also block re-launch
                                            # (or the /tmp NFS-fallback path)
```

If a run was NOT launched through `safe_serve.py`, identify survivors by their shared PGID
(never by a name pattern) and signal only that group, after confirming it is not your own:

```bash
ps -eo pid,pgid,args | grep -E 'run.py|Worker_TP|neuronx-?cc|EngineCore|DPEngineCoreProc'
MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')     # this shell's group — must NOT be targeted
kill -KILL -- -<their_PGID>                  # only after confirming <their_PGID> != $MYPGID
```

Do NOT kill all vllm processes indiscriminately — other runs may be active on
different device partitions, and a wrong target can kill your own session.

## Safe launch and teardown

**Launch every server through `references/safe_serve.py` and tear it down by its recorded
process group.** This prevents a real incident in which a `pkill -f` / bare `kill -9 -<PGID>`
teardown killed the agent's own session and unrelated users' runs (a `nohup cmd &` child
inherits the launcher's process group; a negative-PGID SIGKILL and a box-wide `pkill -f` then
reach things you did not mean to kill).

`safe_serve.py` runs the server via `setsid`, so it leads a NEW session whose PGID equals its
own PID and is **guaranteed distinct** from the caller; teardown signals only that recorded
PGID and **refuses** if it equals the caller's group. `teardown` is also the
**kill-after-capture** lever for the `cleanup_dist_env_and_memory()` teardown-hang: SIGKILL the
isolated group the moment the trace flushes, before the hang-prone `empty_cache` path runs.

```bash
S=<skill-dir>/references/safe_serve.py     # stdlib only; run `python3 $S selftest` once
RUN=$NEURON_PROJECT_ROOT/agent_artifacts/tmp/<run_name>.run.json
LOG=$NEURON_PROJECT_ROOT/agent_artifacts/tmp/<run_name>.server.log

# 1) launch in an isolated session (everything after `--` is the vllm command):
python3 $S launch --runfile "$RUN" --log "$LOG" -- \
    vllm serve "$HF_HOME/<model>" --tensor-parallel-size <TP> ...   # + profiler config

# 2) wait for readiness (a benign compile keeps writing the log; a wedge does not):
python3 $S wait --runfile "$RUN" --health http://localhost:8000/health \
    --timeout 1500 --log "$LOG" --stale 900

# 3) arm/stop the profiler and drive load as usual (curl /start_profile, bench, ...)

# 4) kill-after-capture: SIGKILL the server's OWN isolated group the instant results flush:
python3 $S teardown --runfile "$RUN"       # immediate SIGKILL of the recorded PGID only
python3 $S status  --runfile "$RUN"        # confirm gone; then neuron-ls / fuser /dev/neuron*
```

## Compile-cache poisoning after a failed compile

A failed kernel compile can write a `.done`-marked broken artifact into the
(possibly shared) compile cache; subsequent runs — even byte-identical known-good
source — pull the poisoned entry and re-fail with the same error, making it look
like "everything is broken." **After any failed compile, purge the cache before
re-testing**, and run a clean control to distinguish a state problem from a source
problem.

## Instruction-tuned models capture empty generations

Use `/v1/chat/completions` so the chat template applies; budget
`prompt_tokens + max_tokens` against `--max-model-len`. Covered under SKILL.md →
*Warm up, then bracket the profiled window*.
