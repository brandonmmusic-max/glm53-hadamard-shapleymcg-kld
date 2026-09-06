# V6 raw capture integrity failure and evidence amendment

2026-09-05. The first v6 diagnostic failed before comparison:
`selected raw FC1 row 1 retained sentinel words`.
Output: `p8-coupled-fixture-v1/m1-raw-localization-v6` under the native6 campaign.
Image: `sha256:43b61de24fa8f9323d154705fe37dfe66f39e9954f659895d067b1d99e38e868`.
No valid raw-FC1 comparison, closure or KLD claim follows.

Decision before retry: preserve the raw intermediate scratch and 512-byte input
trace to a create-only NPZ before completeness checks. The previous harness
reported the failed check but did not retain the buffer needed to diagnose it.
Keep the same image, fixture, input and all gates. One additional same-image
diagnostic is permitted within an additional 4 MiB evidence reserve; aggregate
reserved charge becomes 8,142,606,336 bytes, below the 30 GB user ceiling.
Use a fresh output directory. No image/kernel/math change, no protected data,
no production restore. This is an evidence-capture amendment, not a gate retry
intended to qualify the candidate.
