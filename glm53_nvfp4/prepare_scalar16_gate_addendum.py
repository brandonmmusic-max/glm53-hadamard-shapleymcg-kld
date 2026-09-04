"""Record a pre-result engineering addendum without weakening scientific claims."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .shard_index import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-plan", type=Path, required=True)
    parser.add_argument("--prior-powered-analysis", type=Path, required=True)
    parser.add_argument("--prior-kld-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = {
        "schema": "glm53-p8-scalar16-engineering-addendum.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "timing": "written while the four-GPU screen was running and before any raw part or screen result was opened",
        "strict_rule_status": "unchanged; report it exactly as preregistered",
        "engineering_rule": "the expert screen cannot block one developmental n=32 conditional-fit end-to-end teacher-KLD control; choose the lowest aggregate held-out NMSE arm among hybrid, scalar16, and MCG, freeze it, fit any 288-expert policy using fit captures only, and label its KLD exploratory unless the strict screen gate also passes",
        "reason": "prior evidence shows material expert/window heterogeneity and the acceptance metric is end-to-end KLD; however prior routed-NMSE signals have not reliably transferred to KLD",
        "claim_boundary": "no candidate beats NVFP4 unless end-to-end paired KLD improves and its BCa 95 percent upper delta bound is below zero; Shapley allocation cannot retroactively convert a null base candidate into a win",
        "data_boundary": "use the already-open conditional-fit n=32 teacher panel for this developmental control; do not open selection, confirmation, final, or the 28 confirmation logits",
        "algorithm_exclusion": "no LDLQ or BlockLDLQ code path, objective, or result is used",
        "inputs": {
            "strict_plan": {"path": str(args.strict_plan), "sha256": sha256_file(args.strict_plan)},
            "prior_powered_analysis": {"path": str(args.prior_powered_analysis), "sha256": sha256_file(args.prior_powered_analysis)},
            "prior_kld_analysis": {"path": str(args.prior_kld_analysis), "sha256": sha256_file(args.prior_kld_analysis)},
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(sha256_file(args.output))


if __name__ == "__main__":
    main()
