"""Require completed full CUDA-graph capture on all four serving ranks."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def verify(text: str) -> dict:
    if 'enforce_eager=False' not in text or 'Breakable CUDA graph enabled' not in text:
        raise ValueError('graph-enabled serving configuration missing')
    # Runtime variants differ only in whether this progress label says decode.
    # Profiling or a 0% capture line is not sufficient evidence.
    if not re.search(r'Capturing (?:decode )?CUDA graphs \(FULL\):[^\r\n]*100%', text):
        raise ValueError('completed FULL graph capture missing')
    ranks = {int(x) for x in re.findall(
        r'\(Worker_TP(\d+)[^)]*\)[^\r\n]*Graph capturing finished', text
    )}
    if ranks != {0, 1, 2, 3}:
        raise ValueError(f'graph completion rank inventory differs: {sorted(ranks)}')
    return {'status': 'pass', 'full_capture_complete': True, 'ranks': sorted(ranks)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.log.read_text(errors='replace')), sort_keys=True))


if __name__ == '__main__':
    main()
