# V6 localization preflight

2026-09-05. Independent critic verdict at runtime `c7388f2`: PASS for bounded
localization only, not product qualification. Broad P8 CPU suite: 764 passed.
The corrected ordinary allocation matches the frozen AST; sentinel fills now
occur only in diagnostic mode. The capture verifies consumed expert/route/tile
metadata, exactly one write, finite selected data and untouched unselected data.
The 512-byte input trace and dual-reference per-expert FC1 report meet the
predeclared requirements. Results cannot report closure or qualification.

V5 was already building when the broader default-allocation test exposed its
refactor. Preserve that build, but do not use it on the GPU. V6 contains the
allocation-preserving correction. Its Docker COPY operations are grouped by
destination using only the manifest-pinned staged inventory. A CPU test resolves
the COPY mapping and requires every install destination to match the manifest;
the in-image installed-byte verifier remains mandatory. No computational kernel
change is introduced by grouping COPY operations.

Decision before execution: permit one additional small v6 build under the same
8 GB conservative existing charge and an additional 64 MiB build reserve.
Combined reserved charge is 8,138,412,032 bytes (v5+v6+4 MiB diagnostic), below
the user's 30,000,000,000-byte ceiling. No model/teacher allocation, no klcstore
write, no new worktree. Recheck storage after build, then lock and idle GPU0
before the single diagnostic. Production remains off.

Diagnostic protocol remains
`c8be5ce44879efc9f4d821541d62aad09fcbb5ed2358c11ee16d6a14ba63a45c`.
The separate N128 expert-ID dispatch repair is disclosed in its receipt and
is not confused with diagnostic instrumentation. Contrary v4 evidence remains.
