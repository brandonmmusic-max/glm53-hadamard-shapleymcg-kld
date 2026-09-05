# Layer-3 native v2 chronology caveat

The historical plan's `written_at` value, `2026-09-04T20:28:00-04:00`, is
inconsistent with its local Git history and recorded execution. Do not use
that field to establish preregistration or silently reinterpret its timezone.
The cause of the incorrect timestamp has not been established.

Verified local records:

- Commit `dfdc8e16871f8c635ac44d45fef2aa768143795f` first contains the plan.
  Both author and committer timestamps are `2026-09-04T16:12:30-04:00`.
- The plan blob in that commit and the current historical file have the same
  SHA-256: `aa9e476f070836438a695416edfa096b29e6c1f900782d4239f4224e63e5cb34`.
- The session log records launch setup at `2026-09-04T16:12:46-04:00`,
  completed KLD at `16:21:04-04:00`, and service restoration at `16:21:06-04:00`.
  Its path is
  `/media/brandonmusic/klcstore/bmxfp4-glm53/kld-v3/sessions/p8-identity-mcg-l3-native-deterministic-cf32-v2/session.log`.

These local records are consistent with committing the decision rule before
the run. Git timestamps and local logs are not independent trusted timestamps;
they do not justify a stronger external timestamping claim. Preserve the
original plan unchanged, including its inconsistent field, alongside this note.
No KLD value, interval, decision rule, or qualification boundary is changed.
The full-model P8 experiment is separate and remains unfinished.
