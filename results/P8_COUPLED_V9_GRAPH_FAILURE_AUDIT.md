# P8 v9 CUDA-graph failure audit and evidence-only retry

Status: **original v1 result remains failed; CPU-only evidence-preservation patch ready.**

## Preserved observation

The first immutable-v9 graph run exited 1 at M64 with `M64 five graph replays
are not bitwise deterministic`.

- Protocol: `3281b800664e7987732a681959c561b0c74905271e34a3da07e1ea004a0a2843`
- Image: `sha256:ad6b26bf6d1f265d99b09383485ddef82a4acfaea43e28af46341ebb41da24e3`
- Executed harness: `c8f8bd0518b85164220c3b67f26297fb57164b88e63a46ff0cd29455fe74528a`
- Result: `53178a902f05fac9931a652b9eb17b0d7c5e3a17b2ca9008836a824d77f87b14`
- Execution: `283445e933914325ff6bf11cc0d632005f975295cad49bd5db4afbaf027b8436`

The failed result contains no completed-case or per-replay payload, so the
exception text alone cannot establish which receipt field differed. It must not
be retroactively reclassified as a pass.

## Exact predicate audit

For every replay, the harness first requires all six observable tensor hashes
(input payload, input scale, middle payload, middle scale, route output, final
output) to equal the fresh eager hashes. Only after all five replays satisfy that
gate does it compare the entire observation dictionaries after removing only
the replay number.

That whole dictionary also contains `schedule.route_to_physical_sha256`.
`map_route_order_physical_rows` independently checks row counts, prefix geometry,
expert ownership, bounds, injectivity, and surjectivity before producing the
route-order mapping. The runtime obtains each expert-local physical row through
an atomic increment of `expert_write_rows`; therefore valid physical ordering
can vary while the logical route-order tensors remain byte-identical.

This is a grounded rival explanation, not yet the measured cause of the first
failure because the first runner discarded partial receipts.

## Evidence-only retry patch

The harness now atomically writes `partial.json`:

- after each case's fresh eager observation;
- after every graph replay and its eager hash equality gate;
- after each whole-receipt determinism decision.

The frozen protocol SHA and original whole-receipt failure predicate are
unchanged. A retry under this source is diagnostic evidence only. If it fails,
the preserved partial must be hashed and copied alongside the new failure
receipt before any acceptance change.

## Proposed amendment after evidence

Only if the partial receipt shows all tensor hashes equal eager across all five
replays, with every schedule passing the existing bijection checks and variation
confined to route-to-physical layout, create a disclosed v2 protocol:

1. Retain exact eager equality and five-replay equality for all six observable
   tensor hashes.
2. Retain the full per-replay routing bijection, expert ownership, bounds, row
   count, prefix, active-span, dispatch and finite/numerical gates.
3. Preserve every raw `route_to_physical_sha256` as a nuisance receipt, but do
   not require those hashes to equal across replays.
4. Report v1 as failed by its broader predeclared rule; do not overwrite or
   reinterpret it.

If any tensor hash varies, a bijection check fails, or another schedule field
changes, this proposed amendment is not justified and the runtime remains
graph-unqualified.

No GPU, service, image, model, packer, or verifier action was performed in this
audit.
