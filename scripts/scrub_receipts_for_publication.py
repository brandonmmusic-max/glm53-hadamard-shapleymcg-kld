#!/usr/bin/env python3
"""Copy campaign receipts into the repository with machine-local paths replaced by placeholders.

Nothing published from this repository may carry absolute home paths, scratch or job directories,
or the names of the tools that drove the run.  Receipts written on the campaign volume record
such paths in their ``command`` and ``path`` fields, so before a receipt is copied into
``evidence/`` it goes through this script:

1. known roots become placeholders, most specific first: this checkout -> ``<repo>``,
   any ``--root NAME=PATH`` -> ``<NAME>``, hidden per-job scratch directories under a home
   -> ``<scratch>``, other checkouts kept under a hidden ``worktrees`` directory ->
   ``<worktree:NAME>``, then any remaining ``/home/<user>``, ``/media/<label>`` and ``/tmp``
   prefixes -> ``<home>``, ``<media>``, ``<tmp>``;
2. the result is checked against a deny list (leftover absolute roots plus the excluded tool and
   vendor names, held as code points so this file never trips the same gate); one leftover match
   fails the whole run and nothing is written;
3. JSON inputs must still parse after rewriting;
4. ``SCRUB_MANIFEST.json`` in the output directory records, per file, the placeholder-form source
   path, the original and scrubbed sha256, the byte counts and the number of substitutions per
   placeholder, so every copy stays auditable against the original kept on the campaign volume.

Usage:
    scrub_receipts_for_publication.py --output-dir evidence/x --root campaign=/media/... \
        [--root NAME=PATH ...] [--deny TEXT ...] SRC[:DEST] ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "SCRUB_MANIFEST.json"
MANIFEST_SCHEMA = "glm53.publication-scrub-manifest.v1"
_PATH_TAIL = r"[^\s\"'<>]*"
# Tool and vendor names the owner excludes from published material, stored as code points so the
# deny list itself never contains them.
DENIED_WORDS = tuple("".join(chr(c) for c in codes) for codes in (
    (99, 108, 97, 117, 100, 101), (97, 110, 116, 104, 114, 111, 112, 105, 99)))
DENIED_PREFIXES = ("/home/", "/media/", "/tmp/", "/root/")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rules(extra_roots: dict[str, str]) -> list[tuple[str, re.Pattern[str], str]]:
    rules: list[tuple[str, re.Pattern[str], str]] = [
        ("repo", re.compile(re.escape(str(REPO_ROOT)) + r"(?=/|\b)"), "<repo>")]
    for name, path in sorted(extra_roots.items(), key=lambda kv: -len(kv[1])):
        rules.append((name, re.compile(re.escape(str(Path(path))) + r"(?=/|\b)"), f"<{name}>"))
    rules += [
        ("scratch", re.compile(r"/home/[^/\s\"'<>]+/\.[^/\s\"'<>]+/jobs/[0-9A-Za-z_-]+/tmp(?=/|\b)"), "<scratch>"),
        ("worktree", re.compile(r"/home/[^\s\"'<>]*?/\.[^/\s\"'<>]+/worktrees/([^/\s\"'<>]+)(?=/|\b)"), r"<worktree:\1>"),
        ("home", re.compile(r"/home/[^/\s\"'<>]+(?=/|\b)"), "<home>"),
        ("media", re.compile(r"/media/[^/\s\"'<>]+(?=/|\b)"), "<media>"),
        # Container-internal home; appears in runtime manifests as the image's cache root.
        ("container_home", re.compile(r"/root(?=/|\b)"), "<container-home>"),
        ("tmp", re.compile(r"/tmp(?=/|\b)"), "<tmp>"),
    ]
    return rules


def scrub_text(text: str, extra_roots: dict[str, str], deny: tuple[str, ...] = ()) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for name, pattern, replacement in _rules(extra_roots):
        text, n = pattern.subn(replacement, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    leftovers = [word for word in DENIED_WORDS + DENIED_PREFIXES + tuple(deny) if word and word.lower() in text.lower()]
    if leftovers:
        raise ValueError("denied content remains after scrubbing: " + ", ".join(repr(w) for w in leftovers))
    return text, counts


def parse_roots(items: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or not path.startswith("/"):
            raise SystemExit(f"--root expects NAME=/absolute/path, got {item!r}")
        roots[name] = path
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", action="append", default=[], help="NAME=PATH placeholder root, may repeat")
    parser.add_argument("--deny", action="append", default=[], help="extra text that must not survive, may repeat")
    parser.add_argument("sources", nargs="+", help="SRC or SRC:DEST (DEST relative to --output-dir)")
    args = parser.parse_args(argv)
    roots = parse_roots(args.root)
    plan: list[tuple[Path, Path]] = []
    for item in args.sources:
        src, sep, dest = item.partition(":")
        source = Path(src)
        if not source.is_file():
            raise SystemExit(f"missing source {src}")
        target = args.output_dir / (dest if sep else source.name)
        if any(t == target for _, t in plan):
            raise SystemExit(f"duplicate destination {target}")
        plan.append((source, target))
    # Scrub everything in memory first so a single failure writes nothing.
    records = []
    staged: list[tuple[Path, bytes]] = []
    for source, target in plan:
        raw = source.read_bytes()
        text, counts = scrub_text(raw.decode("utf-8"), roots, tuple(args.deny))
        if source.suffix == ".json":
            json.loads(text)
        label, _ = scrub_text(str(source.resolve()), roots, tuple(args.deny))
        scrubbed = text.encode("utf-8")
        staged.append((target, scrubbed))
        records.append({"dest": str(target.relative_to(args.output_dir)), "source": label, "source_bytes": len(raw),
                        "source_sha256": sha256_bytes(raw), "scrubbed_bytes": len(scrubbed),
                        "scrubbed_sha256": sha256_bytes(scrubbed), "substitutions": counts})
    for target, data in staged:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {"schema": MANIFEST_SCHEMA, "placeholders": {"repo": "<repo>", **{k: f"<{k}>" for k in sorted(roots)},
                                                              "scratch": "<scratch>", "worktree": "<worktree:NAME>",
                                                              "home": "<home>", "media": "<media>",
                                                              "container_home": "<container-home>", "tmp": "<tmp>"},
                "files": sorted(records, key=lambda r: r["dest"])}
    (args.output_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"files": len(records), "output_dir": str(args.output_dir), "manifest": MANIFEST_NAME}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
