#!/usr/bin/env python3
"""Resolve immutable prerequisites and seal Tail-V2 CF32/product-speed plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm53_nvfp4 import p8_decode_protocol as protocol
from glm53_nvfp4 import p8_tail_repair_v2 as tail_v2
from glm53_nvfp4 import tail_v2_product_validation as validation


REPO = Path(__file__).resolve().parents[1]
ROLES = Path("/media/brandonmusic/klcstore/bmxfp4-glm53/roles/roles-codec-conditional-fit32-v1.json")
TEACHER = Path("/media/brandonmusic/klcstore/bmxfp4-glm53/teacher")
P8_RECEIPT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/p8-tail-repair-image-v2a/receipt.json")
FAILED_EXL3_RECEIPT = Path("/media/brandonmusic/nvme1n1p3/glm53-trellismx-native6/exl3-tail-v2-product-image-v1/receipt.json")
SOURCES = {
    "glm53_nvfp4/tail_v2_product_validation.py",
    "scripts/prepare_tail_v2_product_validation.py",
    "scripts/tail_v2_stream_score_delete.py",
    "glm53_nvfp4/tail_v2_product_runner.py",
    "scripts/build_exl3_tail_v2_product_image.py",
    "scripts/verify_exl3_tail_v2_activation_boundary.py",
    "runtime_patch/p8_tail_repair_v2/Dockerfile.exl3-product",
    "runtime_patch/p8_tail_repair_v2/kpool-tail-after-valid-history-v2.patch",
    "tests/test_tail_v2_product_validation.py",
} | tail_v2.source_files()


def load_receipt(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text())
    if (value.get("schema") != schema or value.get("status") != "complete"
            or value.get("gpu_used") is not False
            or not str(value.get("image_id", "")).startswith("sha256:")
            or not value.get("patched_kpool_sha256")):
        raise ValueError(f"unqualified Tail-V2 image receipt: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exl3-image-receipt", type=Path, required=True)
    parser.add_argument("--exl3-activation-receipt", type=Path, required=True)
    parser.add_argument("--weight-audit", type=Path, required=True)
    args = parser.parse_args()
    if (args.plan != args.plan.resolve() or args.output != args.output.resolve()
            or args.exl3_image_receipt != args.exl3_image_receipt.resolve()
            or args.exl3_activation_receipt != args.exl3_activation_receipt.resolve()
            or args.weight_audit != args.weight_audit.resolve()
            or args.plan.exists() or args.plan.with_suffix(".sha256").exists() or args.output.exists()):
        raise ValueError("fresh canonical plan/seal/output and canonical receipt required")
    p8 = load_receipt(P8_RECEIPT, "glm53.p8-tail-repair-image.v2")
    exl3 = load_receipt(args.exl3_image_receipt, "glm53.exl3-tail-v2-product-image.v1")
    if p8["patched_kpool_sha256"] != exl3["patched_kpool_sha256"]:
        raise ValueError("P8 and EXL3 do not contain the identical Tail-V2 selector")
    activation = json.loads(args.exl3_activation_receipt.read_text())
    if (activation.get("status") != "pass" or activation.get("image_id") != exl3["image_id"]
            or activation.get("source_sha256") != "ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45"
            or activation.get("image_receipt_sha256") != validation.sha(args.exl3_image_receipt)
            or activation.get("gpu_used") is not False):
        raise ValueError("EXL3 BF16 kernel-boundary source proof differs")
    failed = json.loads(FAILED_EXL3_RECEIPT.read_text())
    if failed.get("status") != "failed" or failed.get("tag") != "klc/glm53-exl3-tail-v2-product:v1":
        raise ValueError("preserved v1 failed build chronology differs")
    from glm53_nvfp4 import p8_weight_identity as weights
    weight_audit = json.loads(args.weight_audit.read_text())
    weights.verify_stats(weight_audit)
    windows = protocol.load_role_inputs(ROLES, TEACHER, verify_teacher_bytes=True)
    plan = {
        "schema": validation.PLAN_SCHEMA,
        "status": "sealed-before-gpu-execution",
        "output": str(args.output),
        "roles": str(ROLES),
        "roles_sha256": validation.sha(ROLES),
        "teacher_root": str(TEACHER),
        "weight_audit": {"path": str(args.weight_audit), "sha256": validation.sha(args.weight_audit)},
        "windows": windows,
        "opened_roles": ["conditional-fit"],
        "protected_roles_opened": [],
        "images": {"p8": p8["image_id"], "exl3": exl3["image_id"]},
        "image_receipts": {
            "p8": {"path": str(P8_RECEIPT), "sha256": validation.sha(P8_RECEIPT)},
            "exl3": {"path": str(args.exl3_image_receipt), "sha256": validation.sha(args.exl3_image_receipt)},
        },
        "exl3_activation_receipt": {"path": str(args.exl3_activation_receipt),
                                    "sha256": validation.sha(args.exl3_activation_receipt)},
        "preserved_failed_exl3_build_receipt": {"path": str(FAILED_EXL3_RECEIPT),
                                                 "sha256": validation.sha(FAILED_EXL3_RECEIPT)},
        "patched_kpool_sha256": p8["patched_kpool_sha256"],
        "topology": validation.TOPOLOGY,
        "product": validation.PRODUCT,
        "order": validation.ORDER,
        "cold_runs_per_arm": validation.REPEATS,
        "raw_bytes_per_window": validation.RAW_BYTES,
        "max_new_nvme_bytes": validation.MAX_NEW_BYTES,
        "storage_policy": "one window raw at a time; durable score and retirement receipts; raw unlinked before next request",
        "determinism": "five independent cold server starts per arm; each window raw SHA-256 must match repeat 1 bit-for-bit",
        "kld_analysis": "paired CF32 by window using repeat 1 only; repeats 2-5 are determinism replications, not added n",
        "speed": {
            "cold_runs_per_arm": 5,
            "order": validation.ORDER,
            "primary_prefill": "32K Prometheus kv_computed server tokens/s",
            "primary_decode": "C1 32K OpenAI continuous-usage output tokens/s",
            "secondary": "64K prefill and C1 decode",
            "capture_hook_enabled": False,
            "cuda_graphs": True,
        },
        "decision_before_result": {
            "quality": "report paired CF32 P8-minus-EXL3 KLD BCa CI; no post-hoc threshold",
            "determinism": "all 64 arm-window cells must have one raw SHA across five cold runs",
            "speed": "report all runs and medians; P8 must exceed EXL3 in both primary medians to call the product faster",
        },
        "claim_boundary": "Same Tail-V2 B12X/NVFP4 attention, graphs, TP4, four GPUs and no MTP; P8 noEP/DCP1/E4M3 versus production EXL3 EP4/DCP4/BF16 is a product comparison, not codec-only causality.",
        "source_sha256": {name: validation.sha(REPO / name) for name in sorted(SOURCES)},
        "retry_policy": "no resume or silent reroll; preserve failed receipts and amend before another attempt",
    }
    args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    args.plan.with_suffix(".sha256").write_text(validation.sha(args.plan) + "  " + args.plan.name + "\n")
    validation.authenticate_plan(args.plan)
    print(json.dumps({"status": plan["status"], "plan_sha256": validation.sha(args.plan),
                      "windows": len(windows), "peak_raw_bytes": validation.RAW_BYTES}, sort_keys=True))


if __name__ == "__main__":
    main()
