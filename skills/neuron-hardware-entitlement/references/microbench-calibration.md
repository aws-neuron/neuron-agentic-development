# Calibrating hardware constants by microbenchmarking

## When to read this

A floor is only as trustworthy as the two hardware constants it divides by: the
**tensor-engine peak FLOP/s** (compute floor) and the **HBM bandwidth bytes/s** (memory
floor). If a floor looks wrong — an operator the model should bind on memory comes out
compute-bound, or a measured kernel beats its own "lower bound" — the suspect is often a
hardware constant, not the model shapes. The model shapes come from the oracle and are
ground-truthed by reconciliation; the constants come from a hardware spec sheet and may not
reflect what a real kernel achieves. Microbenchmarking calibrates them.

**The file you are calibrating** is the parallelism-planner skill's
`references/trn2_constants.json` -- the single canonical copy. This skill does not ship its own:
it imports the planner's analytic engine, and the same file feeds every floor either skill
reports, so there is one file to calibrate, not two that can drift. Calibrate by editing that
copy (or, for a one-off, copy it beside your inputs and point at the copy).
rather than editing the shipped one in place.

## What a microbenchmark reveals

Grid-sweep a minimal repro of a single operator class across a size that scales the dominant
cost, and read the resulting time-vs-size curve. The curve has two diagnostic features:

- **The slope** is the effective constant. For a weight-streaming (memory-bound) op, slope =
  1 ÷ effective HBM bandwidth; for a matmul (compute-bound) op, slope = 1 ÷ effective
  tensor-engine peak. If the measured slope disagrees with the spec-sheet constant, the floor
  is computed against the wrong number — replace the constant in the hardware-constants file
  with the measured one (or a documented fraction of peak).
- **The flat offset** at small sizes is **fixed per-invocation overhead** — kernel dispatch,
  sync, setup. The analytic floor models neither; it is a pure rate × size bound. The offset
  tells you the floor cannot to be hit below a problem size where the offset is negligible,
  which is exactly the small-batch decode regime where dispatch overhead dominates. Use it to
  set expectations: a decode op whose floor is a few microseconds will never reach that floor
  if dispatch alone costs more.

## How to calibrate

1. **Pick the operator class and the scaling axis.** For HBM bandwidth, sweep a
   weight-streaming op over weight size (or batch, holding shape). For tensor-engine peak,
   sweep a large matmul over the contraction dimension.
2. **Sweep a minimal repro**, not the full model — isolate the one operator so the curve
   reflects one cost, not a mix.
3. **Fit the linear region.** Discard the small-size points dominated by the flat offset;
   fit the slope over the size range where the op is firmly in its regime.
4. **Compare slope to the spec constant.** A wrong slope means the wrong constant — update
   the hardware-constants JSON (owned by `neuron-parallelism-planner`) and
   rerun. A right slope but a measured kernel far above the floor means the kernel is
   inefficient (the headroom the floor is meant to expose), not a miscalibrated constant.
5. **Record the offset** as the dispatch floor — the size below which the analytic floor is
   not a reachable target.

## What calibration cannot fix

The floor is a rate × size bound. Microbenchmarking corrects the *rate* (slope) and bounds
the *unmodeled overhead* (offset). It does not change which operator binds on what — that is
the model's arithmetic intensity vs the ridge point, and the ridge point itself shifts only
when the calibrated peak or bandwidth shifts. If an operator's binding flips after
recalibration, that is the correct, intended consequence of using the real constants instead
of the spec-sheet ones.
