# Worked example (2): prefill-only embedding model, per-component compute floor

A second, contrasting worked example to the decode-oriented `worked-example.md`. This one
is **prefill-only, no KV cache, compute-floor-dominated at low TP and collective-bound at
high TP** — computed analytically (no oracle run) to show the component TE/HBM floor and
how it reconciles against measured device time.

## Run-spec

`generic-encoder-only-model`: H=4096, I=14336, D=128, n_q=32, n_kv=8, 32 layers,
prefill S=256, bf16, with bidirectional `S × S` attention. One LNC2 logical core contains
two physical NeuronCore-v3 tensor engines: `2 × 79 = 158 TFLOP/s` bf16.

## Per-component compute floor at TP=1 (S=256, one logical core)

FLOPs per layer = 2·S·(matmul dims); summed over 32 layers, divided by 158 TFLOP/s:

| Certified TE component | Compute floor | % of measured 66.2ms |
|---|---|---|
| **MLP (gate + up + down)** | **18.3ms** | **28%** |
| Attention projections (QKV + O) | 4.4ms | 7% |
| Attention scores (QK^T + A·V) | 0.2ms | <1% |
| **Total certified TE floor** | **22.9ms** | **35%** |

These are semantic component records rather than serialized operator latencies. The engine
sums the analytical work terms corresponding to the operators inside each component, applies
`max(component_TE, component_HBM)`, and then serializes the component records. This example has
no KV stream and its component records are TE-bound at TP=1, so the TE values above are also
the component floor after including each component's unique external activation reads/writes.
Applying a roofline independently to gate, up, down, Q, K, V, and O and
then summing those operator times would impose an unsupported operator-serialization policy.

**Read:** MLP matmuls remain the largest certified engine term — 2.89 TFLOP of work at
158 TFLOP/s ≈ 18.3ms. The profile also observed roughly 16ms of VectorEngine-attributed time
and 4.6ms of DMA/scheduling time, but those are evidence about the implementation, not
certified mandatory floor terms, so they are outside the published model floor.

## Reconciliation against the deployed bound (the high-TP surprise)

At TP=16 the certified TE floor drops to roughly 1.4ms (per-rank MLP shrinks to
I/16=896), while **measured device time is ~18ms**. This large gap is not evidence that the
matmul kernels are inefficient:

- Live system trace: collectives = **43% of the device window (7.8ms of 18ms)**, running
  ~99.9% serialized with compute (sequence-parallel `all_gather → matmul → reduce_scatter`).
- TensorEngine runs at ~100% efficiency **when scheduled** but is idle 75.7% of the window,
  92% of that idle ending on a `LDWEIGHTS` (DMA), not a VE op.

So the component TE/HBM floor is scoped correctly for the mandatory engine work, but it does
not explain the deployed time by itself. This is the canonical case where an engine-only
component floor would misroute optimization effort toward kernels when the real bind is
communication. See the collective-wall-time caveat in
[operator-floors.md](operator-floors.md).

*(Prefill-only pooling/embedding, trn2. Analytic floor; measured device numbers from a live
system-trace profile of the TP=16 config.)*
