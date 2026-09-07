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

    config = WorkflowConfig.from_file(args.config)
    _emit(workflow_report(config))
    return 0


def _command_p8_fixture(args: argparse.Namespace) -> int:
    from .p8_fixture import verify_fixture, write_fixture

    path = args.directory / "p8-reference.safetensors"
    if args.write:
        write_fixture(path)
    _emit(verify_fixture(path))
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
    workflow_parser.set_defaults(handler=_command_workflow)

    p8_fixture_parser = subparsers.add_parser(
        "p8-fixture", help="write or verify a deterministic K3/K4/K5 CPU reference fixture"
    )
    p8_fixture_parser.add_argument("directory", type=Path)
    p8_fixture_parser.add_argument("--write", action="store_true", help="write once; refuse overwrite")
    p8_fixture_parser.set_defaults(handler=_command_p8_fixture)
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
