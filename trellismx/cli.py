"""CPU-only TrellisMX command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .capabilities import capabilities
from .codec import P4Payload, read_p4, serialize_p4, storage_accounting
from .protocol import P8_RATES


def _shape(text: str) -> tuple[int, int]:
    try:
        rows_text, columns_text = text.lower().split("x", 1)
        rows, columns = int(rows_text), int(columns_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "shape must be ROWSxCOLUMNS, for example 32x128"
        ) from error
    if rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError("rows and columns must be positive")
    return rows, columns


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _command_capabilities(_args: argparse.Namespace) -> int:
    _emit(capabilities())
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    expected = (
        {"expected_shape": args.expected_shape} if args.expected_shape else {}
    )
    payload: P4Payload = read_p4(args.path, **expected)
    _emit(
        {
            "status": "passed",
            "path": str(args.path),
            "shape": [payload.rows, payload.width],
            "storage": storage_accounting(payload),
        }
    )
    return 0


def _command_fixture(args: argparse.Namespace) -> int:
    # Import the scalar oracle lazily; validation does not need it.
    from glm53_nvfp4.p4_fixture import (
        fixture_expectation,
        make_fixture,
        verify_fixture,
    )

    if not args.write:
        _emit(verify_fixture(args.directory))
        return 0
    args.directory.mkdir(parents=True, exist_ok=True)
    payload_path = args.directory / "fixture.p4.safetensors"
    expectation_path = args.directory / "fixture.expected.json"
    if payload_path.exists() or expectation_path.exists():
        raise FileExistsError("refusing to overwrite fixture files")
    payload = make_fixture()
    payload_path.write_bytes(serialize_p4(payload))
    expectation_path.write_text(
        json.dumps(fixture_expectation(payload), sort_keys=True, indent=2) + "\n"
    )
    _emit(verify_fixture(args.directory))
    return 0


def _rates_from_args(args: argparse.Namespace) -> dict[int, int]:
    from .adapters import GLM53FlashAdapter

    if args.rate_map is not None:
        import json

        raw = json.loads(args.rate_map)
        if not isinstance(raw, dict):
            raise ValueError("--rate-map must be a JSON object keyed by layer")
        result: dict[int, int] = {}
        for layer, rate in raw.items():
            try:
                layer_int, rate_int = int(layer), int(rate)
            except (TypeError, ValueError) as error:
                raise ValueError("rate map layers and rates must be integers") from error
            if layer_int in result:
                raise ValueError("duplicate layer in rate map")
            result[layer_int] = rate_int
        return result
    if args.uniform_rate not in P8_RATES:
        raise ValueError(f"uniform P8 rate must be one of {P8_RATES}")
    return {layer: args.uniform_rate for layer in GLM53FlashAdapter.layers}


def _command_workflow(args: argparse.Namespace) -> int:
    from .workflow import WorkflowConfig, workflow_report
    from .provenance import EncoderBackendConfig, ProvenanceOverlay

    config = WorkflowConfig.from_file(args.config)
    provenance = (
        ProvenanceOverlay.from_file(args.provenance)
        if args.provenance is not None
        else None
    )
    backend = (
        EncoderBackendConfig.from_file(args.encoder_backend)
        if args.encoder_backend is not None
        else None
    )
    if provenance is None:
        report = workflow_report(config)
    else:
        from .workflow import workflow_report_with_provenance

        report = workflow_report_with_provenance(config, provenance, backend)
    if provenance is None and backend is not None:
        report["encoder_backend"] = backend.summary()
    _emit(report)
    return 0


def _command_p8_fixture(args: argparse.Namespace) -> int:
    from .p8_fixture import verify_fixture, write_fixture

    path = args.directory / "p8-reference.safetensors"
    if args.write:
        write_fixture(path)
    _emit(verify_fixture(path))
    return 0


def _command_sidecar_fixture(args: argparse.Namespace) -> int:
    from .sidecar import validate_sidecar, write_synthetic_sidecar

    path = args.directory / "p8-tp4-sidecar.safetensors"
    if args.write:
        write_synthetic_sidecar(
            path,
            layer=args.layer,
            rank=args.rank,
            bits=args.bits,
        )
    report = validate_sidecar(
        path,
        expected_layer=args.layer,
        expected_rank=args.rank,
        expected_bits=args.bits,
        production_geometry=False,
        experts=1,
        hidden=32,
        local_intermediate=32,
    )
    _emit(report.as_dict())
    return 0


def _command_p8_tensor_fixture(args: argparse.Namespace) -> int:
    from .p8_format import read_p8_tensor_file, synthetic_p8_payload, write_p8_tensor_file

    path = args.directory / "p8-tensors.safetensors"
    if args.write:
        payloads = {
            f"k{bits}": synthetic_p8_payload(f"k{bits}", bits=bits)
            for bits in (3, 4, 5)
        }
        write_p8_tensor_file(path, payloads)
    report = read_p8_tensor_file(path)
    _emit(report.storage_report())
    return 0


def _command_inspect_checkpoint(args: argparse.Namespace) -> int:
    from .adapters import adapter_for

    selected = adapter_for(args.architecture)
    if not hasattr(selected, "inspect_checkpoint"):
        raise ValueError("architecture adapter does not implement checkpoint inspection")
    _emit(
        selected.inspect_checkpoint(
            args.checkpoint,
            index_path=args.index,
            verify_files=not args.no_file_check,
        )
    )
    return 0


def _command_runtime_info(_args: argparse.Namespace) -> int:
    from .runtime import runtime_overlay_report

    _emit(runtime_overlay_report())
    return 0


def _command_validate_sidecar(args: argparse.Namespace) -> int:
    from .sidecar import validate_sidecar

    report = validate_sidecar(
        args.path,
        expected_layer=args.expected_layer,
        expected_rank=args.expected_rank,
        expected_bits=args.expected_bits,
        expected_design_sha256=args.expected_design_sha256,
        expected_transform_sha256=args.expected_transform_sha256,
        expected_scale_source_sha256=args.expected_scale_source_sha256,
        expected_file_sha256=args.expected_file_sha256,
    )
    _emit(report.as_dict())
    return 0


def _command_validate_p8_tensors(args: argparse.Namespace) -> int:
    from .p8_format import read_p8_tensor_file

    report = read_p8_tensor_file(args.path, expected_sha256=args.expected_sha256)
    _emit(report.storage_report())
    return 0


def _command_plan(args: argparse.Namespace) -> int:
    from .adapters import adapter_for

    selected = adapter_for(args.architecture)
    value = selected.plan(_rates_from_args(args), tensor_parallel_size=args.tensor_parallel_size)
    _emit(value.summary())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trellismx",
        description="CPU-only TrellisMX codec and artifact tools; no GPU execution",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capabilities_parser = subparsers.add_parser(
        "capabilities", help="show supported and explicitly unsupported operations"
    )
    capabilities_parser.set_defaults(handler=_command_capabilities)

    validate_parser = subparsers.add_parser(
        "validate", help="validate a portable P4 matrix artifact"
    )
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--expected-shape", type=_shape)
    validate_parser.set_defaults(handler=_command_validate)

    fixture_parser = subparsers.add_parser(
        "fixture", help="write or verify the deterministic CPU-only P4 fixture"
    )
    fixture_parser.add_argument("directory", type=Path)
    fixture_parser.add_argument(
        "--write", action="store_true", help="write files; refuse overwrite"
    )
    fixture_parser.set_defaults(handler=_command_fixture)

    plan_parser = subparsers.add_parser(
        "plan", help="produce a CPU-only architecture conversion plan; never reads a checkpoint"
    )
    plan_parser.add_argument("--architecture", required=True)
    plan_parser.add_argument("--uniform-rate", type=int, default=4)
    plan_parser.add_argument(
        "--rate-map", help='JSON object such as {"3":5,"4":4}; overrides --uniform-rate'
    )
    plan_parser.add_argument("--tensor-parallel-size", type=int, default=4)
    plan_parser.set_defaults(handler=_command_plan)

    workflow_parser = subparsers.add_parser(
        "workflow",
        help="summarize a portable workflow config and its research-source stages; does not execute them",
    )
    workflow_parser.add_argument("--config", type=Path, required=True)
    workflow_parser.add_argument(
        "--provenance", type=Path, help="portable provenance overlay; artifacts are not read"
    )
    workflow_parser.add_argument(
        "--encoder-backend", type=Path, help="disabled or operator-pinned backend descriptor"
    )
    workflow_parser.set_defaults(handler=_command_workflow)

    p8_fixture_parser = subparsers.add_parser(
        "p8-fixture", help="write or verify a deterministic K3/K4/K5 CPU reference fixture"
    )
    p8_fixture_parser.add_argument("directory", type=Path)
    p8_fixture_parser.add_argument("--write", action="store_true", help="write once; refuse overwrite")
    p8_fixture_parser.set_defaults(handler=_command_p8_fixture)

    sidecar_fixture_parser = subparsers.add_parser(
        "sidecar-fixture", help="write or verify a tiny synthetic full-coupled TP4 sidecar"
    )
    sidecar_fixture_parser.add_argument("directory", type=Path)
    sidecar_fixture_parser.add_argument("--write", action="store_true", help="write once; refuse overwrite")
    sidecar_fixture_parser.add_argument("--layer", type=int, default=3, choices=range(3, 45))
    sidecar_fixture_parser.add_argument("--rank", type=int, default=0, choices=range(4))
    sidecar_fixture_parser.add_argument("--bits", type=int, default=4, choices=(3, 4, 5))
    sidecar_fixture_parser.set_defaults(handler=_command_sidecar_fixture)

    p8_tensor_fixture_parser = subparsers.add_parser(
        "p8-tensor-fixture",
        help="write or verify a portable multi-rate P8 tensor container",
    )
    p8_tensor_fixture_parser.add_argument("directory", type=Path)
    p8_tensor_fixture_parser.add_argument(
        "--write", action="store_true", help="write once; refuse overwrite"
    )
    p8_tensor_fixture_parser.set_defaults(handler=_command_p8_tensor_fixture)

    inspect_parser = subparsers.add_parser(
        "inspect-checkpoint",
        help="validate architecture config and tensor index without reading tensor payloads",
    )
    inspect_parser.add_argument("--architecture", required=True)
    inspect_parser.add_argument("--checkpoint", type=Path, required=True)
    inspect_parser.add_argument("--index", type=Path)
    inspect_parser.add_argument(
        "--no-file-check", action="store_true", help="validate JSON mapping only"
    )
    inspect_parser.set_defaults(handler=_command_inspect_checkpoint)

    runtime_parser = subparsers.add_parser(
        "runtime-info",
        help="hash-check the optional source-only SM120 overlay; never imports runtime dependencies",
    )
    runtime_parser.set_defaults(handler=_command_runtime_info)

    validate_sidecar_parser = subparsers.add_parser(
        "validate-sidecar", help="validate one production full-coupled P8 TP4 sidecar"
    )
    validate_sidecar_parser.add_argument("path", type=Path)
    validate_sidecar_parser.add_argument("--expected-layer", type=int)
    validate_sidecar_parser.add_argument("--expected-rank", type=int)
    validate_sidecar_parser.add_argument("--expected-bits", type=int, choices=(3, 4, 5))
    validate_sidecar_parser.add_argument("--expected-design-sha256")
    validate_sidecar_parser.add_argument("--expected-transform-sha256")
    validate_sidecar_parser.add_argument("--expected-scale-source-sha256")
    validate_sidecar_parser.add_argument("--expected-file-sha256")
    validate_sidecar_parser.set_defaults(handler=_command_validate_sidecar)

    validate_p8_parser = subparsers.add_parser(
        "validate-p8-tensors", help="validate a portable model-independent P8 tensor container"
    )
    validate_p8_parser.add_argument("path", type=Path)
    validate_p8_parser.add_argument("--expected-sha256")
    validate_p8_parser.set_defaults(handler=_command_validate_p8_tensors)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.exit(2, f"trellismx: error: {error}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
