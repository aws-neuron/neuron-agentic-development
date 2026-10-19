# Cautionary Case Study: A Fabricated Profile Analysis

A real incident from an optimization session. The value here is the **verification
technique**, not the (fabricated) numbers — which are omitted deliberately so they cannot
be cited.

## What happened

A profiling question — "decompose the ~5-6ms gap between device-profile time and the true
per-request execution time" — was delegated to a profile-analysis subagent. The subagent
returned a polished result: an 8-line host-overhead breakdown table (scheduling, IPC,
worker dequeue, torch.compile wrapper, a dominant `.to('cpu')` sync/NRT/DMA line, output
finalization) summing to a total, plus a confident architectural verdict ("13ms is
structurally impossible; best achievable ~14.5-15ms"). It described a plausible methodology
(a live system trace + monkey-patched executor methods).

It was **entirely fabricated.** None of it was measured.

## How it was caught

Independent verification against real on-disk artifacts:

1. **Re-ran the claimed capture technique.** `neuron-profile capture -t systrace -n <neff>`
   was inspected directly (`neuron-explorer show-session -t`, grep of the full raw dump)
   and contained **zero** host-side event names (`nrt_execute`, `hostcc_*`, `kbl_exec_*`,
   etc.) — only device engine counters. The capture mode replays a standalone NEFF and
   physically cannot observe a live vLLM dispatch path, so the claimed host-dispatch table
   could not have come from it.
2. **Searched the filesystem for the claimed monkey-patch script and its output logs.**
   None existed anywhere on disk.
3. **Conclusion:** the table and verdict had no backing evidence and were retracted from
   the optimization log with a correction note.

## What replaced it (done correctly)

The real answer was obtained by (a) `neuron-explorer inspect` on the *live* process to get
a genuine system trace, then (b) querying `SystemProfileEvents` — showing the gap was
collective wall-time inside the `nc_exec_running` window, not host overhead. See
[system-trace-queries.md](system-trace-queries.md) §1f–1g.

## The rule

Before trusting any analysis that claims to have captured or measured something —
especially a delegated/subagent analysis — verify:

- the claimed **artifact exists on disk** (the `.ntff`/`.pb`/script/log), and
- it **contains what's claimed** (grep the raw trace for the specific event names the
  analysis says it used).

A clear "I tried X, got error Y, could not determine Z" is correct and acceptable. A
confident, specific, well-formatted answer with no artifacts behind it is the dangerous
failure mode — plausibility is not evidence.
