"""Host dispatch tests, including execution of the exact local vLLM methods.

No CUDA device is queried or launched. Runtime arithmetic/device/graph parity
remain independent gates. Source-path override permits a pinned reproduction.
"""
import ast
import contextlib
import copy
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import weakref
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from runtime_patch.p4_glm_serving import P4ServingConfig, _P4ServingMethod, install
from runtime_patch.p4_native_kernel import (
    P4NativeTPMoE, P4WorkspaceOwner, SOURCE, _CAPTURE_PINS, _verify_file, max_route_tiles,
)


ROOT = Path(__file__).resolve().parents[1]
VLLM = Path(os.environ.get("GLM53_VLLM_SOURCE", "/home/brandonmusic/KLC_SANDBOXES/glm53-exl3-k4-sm120/vllm"))
LAUNCHER = Path(os.environ.get("GLM53_KLD_LAUNCHER", "/home/brandonmusic/KLC_SANDBOXES/bmxfp4-glm53/scripts/run_kld_v3.sh"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def environment(tmp_path):
    design = tmp_path / "design.json"
    design.write_text('{"fixture":"host-only"}\n')
    entries = []
    for rank in range(4):
        path = tmp_path / f"p4-layer-003-tp4-rank-{rank}.safetensors"
        path.write_bytes(f"synthetic-sidecar-rank-{rank}".encode())
        entries.append(dict(layer=3, rank=rank, path=path.name, sha256=digest(path.read_bytes()), bytes=path.stat().st_size))
    manifest = dict(schema="glm53-p4-mcg-tp4-manifest.v1", codec_schema="glm53-p4-mcg-tp-rank.v2",
                    world_size=4, layers=[3], source_design_sha256=digest(design.read_bytes()), entries=entries)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return {"GLM53_P4_NATIVE": "1", "GLM53_P4_NATIVE_SIDECAR_DIR": str(tmp_path),
            "GLM53_P4_NATIVE_LAYERS": "3", "GLM53_P4_NATIVE_DESIGN": str(design),
            "GLM53_P4_NATIVE_MANIFEST": str(path), "GLM53_P4_NATIVE_MANIFEST_SHA256": digest(path.read_bytes())}


def mutate_manifest(env, action):
    path = Path(env["GLM53_P4_NATIVE_MANIFEST"])
    value = json.loads(path.read_text())
    action(value)
    path.write_text(json.dumps(value))
    env["GLM53_P4_NATIVE_MANIFEST_SHA256"] = digest(path.read_bytes())


class Carrier:
    @property
    def is_monolithic(self):
        return True

    def apply(self, *args, **kwargs):
        raise AssertionError("selected P4 layer fell back to stock")

    def process_weights_after_loading(self, *args, **kwargs):
        raise AssertionError("selected P4 layer used stock repacking")


class ModelOptCarrier(Carrier):
    pass


class Backend:
    backend_name = "b12x-p4-trellis-mxf4nvf4"
    new_workspace_owner = staticmethod(P4NativeTPMoE.new_workspace_owner)

    def __init__(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs
        self.calls = []

    def apply(self, x, weights, ids):
        self.calls.append((x, weights, ids))
        return x + 7  # Explicit host seam, not P4 arithmetic evidence.


def runner(method_type=Carrier, rank=0):
    parallel = SimpleNamespace(tp_size=4, ep_size=1, use_ep=False, enable_eplb=False,
                               dp_size=1, pcp_size=1, sp_size=1)
    config = SimpleNamespace(moe_parallel_config=parallel, tp_rank=rank, in_dtype=torch.bfloat16,
                             has_bias=False, is_lora_enabled=False)
    layer = SimpleNamespace(moe_config=config, quant_method=method_type(), global_num_experts=288,
                            local_num_experts=288, hidden_size=4096, intermediate_size_per_partition=512,
                            top_k=8, activation="silu", swiglu_limit=10.0, apply_router_weight_on_input=False,
                            expert_map=None, w13_weight=torch.zeros(1), w2_weight=torch.zeros(1))
    return SimpleNamespace(routed_experts=layer, routed_input_transform=None, routed_output_transform=None,
                           router=object(), shared_experts=object(), enable_dbo=False)


def model_for(value):
    return SimpleNamespace(FusedMoEFactory=lambda *args, **kwargs: value)


@pytest.mark.parametrize("method_type", [Carrier, ModelOptCarrier])
@pytest.mark.parametrize("rank", range(4))
def test_selected_glm_factory_retains_router_shared_experts_and_uses_pinned_p4(environment, method_type, rank, capsys):
    value = runner(method_type, rank)
    router, shared = value.router, value.shared_experts
    model = model_for(value)
    install(environment, components=(model, (Carrier, ModelOptCarrier), Backend))
    actual = model.FusedMoEFactory(prefix="model.layers.3.mlp.experts")
    assert actual is value and actual.router is router and actual.shared_experts is shared
    layer = actual.routed_experts
    method = layer.quant_method
    assert isinstance(method, method_type) and isinstance(method, _P4ServingMethod)
    assert not method.is_monolithic and not method.mk_can_overlap_shared_experts
    assert not method.supports_internal_mk and method.topk_indices_dtype == torch.int32
    with pytest.raises(RuntimeError, match="before physical"):
        method.apply(layer, None, None, None)
    method.process_weights_after_loading(layer)
    assert layer.w13_weight.numel() == layer.w2_weight.numel() == 0
    assert layer._glm53_p4_runtime.kwargs["tp_rank"] == rank
    assert layer._glm53_p4_runtime.kwargs["expected_file_sha256"] == digest(layer._glm53_p4_runtime.path.read_bytes())
    x = torch.zeros((1, 4096), dtype=torch.bfloat16)
    weights = torch.full((1, 8), .125)
    ids = torch.arange(8).reshape(1, 8)
    output = method.apply(layer, x, weights, ids, shared, x)
    assert torch.equal(output, x + 7)
    assert layer._glm53_p4_runtime.calls == [(x, weights, ids)]
    with pytest.raises(RuntimeError, match="twice"):
        method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="modular"):
        method.apply_monolithic()
    log = capsys.readouterr().out
    assert "schema=glm53-p4-mcg-tp-rank.v2" in log and "mma=mxf4nvf4" in log
    assert "FC1=trellis_E2M1_K64 SwiGLU=clamp10 FC2=trellis_E2M1_K64" in log
    assert "physical_bpw=4.5" in log


@pytest.mark.parametrize("prefix", ["model.layers.4.mlp.experts", "model.shared_experts", "model.layers.30.mlp.experts"])
def test_nonselected_layers_keep_exact_stock_method(environment, prefix):
    value = runner()
    original_method = value.routed_experts.quant_method
    model = model_for(value)
    install(environment, components=(model, (Carrier, ModelOptCarrier), Backend))
    assert model.FusedMoEFactory(prefix=prefix) is value
    assert value.routed_experts.quant_method is original_method
    assert type(original_method) is Carrier and original_method.is_monolithic


@pytest.mark.parametrize("scope,name,value", [
    ("parallel", "tp_size", 2), ("parallel", "ep_size", 4), ("parallel", "use_ep", True),
    ("parallel", "enable_eplb", True), ("parallel", "sp_size", 4), ("parallel", "dp_size", 2),
    ("layer", "intermediate_size_per_partition", 1024), ("layer", "swiglu_limit", None),
    ("layer", "activation", "gelu"), ("layer", "apply_router_weight_on_input", True),
    ("layer", "expert_map", torch.zeros(1)), ("runner", "routed_input_transform", object()),
    ("config", "has_bias", True), ("runner", "enable_dbo", True),
])
def test_unsupported_glm_semantics_fail_closed(environment, scope, name, value):
    actual = runner()
    layer = actual.routed_experts
    target = {"runner": actual, "layer": layer, "config": layer.moe_config,
              "parallel": layer.moe_config.moe_parallel_config}[scope]
    setattr(target, name, value)
    model = model_for(actual)
    install(environment, components=(model, (Carrier, ModelOptCarrier), Backend))
    with pytest.raises(ValueError):
        model.FusedMoEFactory(prefix="model.layers.3.mlp.experts")


@pytest.mark.parametrize("mutation", ["pin", "design", "rank", "duplicate", "schema", "escape", "size", "layer"])
def test_manifest_identity_and_rank_completeness(environment, mutation):
    if mutation == "pin":
        environment["GLM53_P4_NATIVE_MANIFEST_SHA256"] = "0" * 64
    elif mutation == "design":
        Path(environment["GLM53_P4_NATIVE_DESIGN"]).write_text("different")
    else:
        actions = {
            "rank": lambda m: m["entries"].pop(),
            "duplicate": lambda m: m["entries"].append(m["entries"][0]),
            "schema": lambda m: m.update(codec_schema="glm53-p4-mcg-tp-rank.v1"),
            "escape": lambda m: m["entries"][0].update(path="../weights.safetensors"),
            "size": lambda m: m["entries"][0].update(bytes=1),
            "layer": lambda m: m.update(layers=[3, 3]),
        }
        mutate_manifest(environment, actions[mutation])
    with pytest.raises(ValueError):
        P4ServingConfig.from_env(environment)


def test_sidecar_hash_checked_before_cuda_and_zero_scale_abi(environment):
    config = P4ServingConfig.from_env(environment)
    path, entry = config.sidecar(3, 0)
    _verify_file(path, entry["sha256"], entry["bytes"])
    path.write_bytes(b"X" * entry["bytes"])
    with pytest.raises(ValueError, match="content identity"):
        _verify_file(path, entry["sha256"], entry["bytes"])


def test_sitecustomize_aborts_instead_of_silently_serving_stock():
    env = dict(os.environ, GLM53_P4_NATIVE="bad", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "runtime_patch"))
    process = subprocess.run([sys.executable, "-c", "print('REACHED_APPLICATION')"], env=env,
                             capture_output=True, text=True)
    assert process.returncode == 78 and "REACHED_APPLICATION" not in process.stdout
    assert "GLM53_P4_NATIVE must be explicitly enabled" in process.stderr


@pytest.mark.parametrize("failure", ["import", "config", "system_exit", "reporting"])
def test_exact_sitecustomize_install_failure_is_never_swallowed(tmp_path, failure):
    # Execute the exact P4 sitecustomize block in an actual Python process,
    # using a synthetic import only to provoke each failure mode. -S prevents
    # startup from running a different copy before the test executes this one.
    module = tmp_path / "p4_glm_serving.py"
    if failure == "import":
        module.write_text("raise ImportError('P4_IMPORT_FAILURE')\n")
    else:
        error = "SystemExit(0)" if failure == "system_exit" else "ValueError('P4_CONFIG_FAILURE')"
        module.write_text(f"def install():\n    raise {error}\n")
    if failure == "reporting":
        (tmp_path / "traceback.py").write_text("def print_exc():\n    raise RuntimeError('BROKEN_REPORTING')\n")
    tree = ast.parse((ROOT / "runtime_patch/sitecustomize.py").read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Name) and node.test.id == "P4_NATIVE")
    code = "import os\nP4_NATIVE = '1'\n" + ast.unparse(block) + "\nprint('REACHED_APPLICATION')"
    process = subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path,
                             env=dict(os.environ, PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES=""),
                             capture_output=True, text=True)
    assert process.returncode == 78 and "REACHED_APPLICATION" not in process.stdout


@pytest.mark.parametrize("layers", ["03", " 3", "3 ", "4,3", "3,03", "3, 4", "3,4,4"])
def test_readiness_rejects_noncanonical_layer_syntax(environment, layers):
    environment["GLM53_P4_NATIVE_LAYERS"] = layers
    with pytest.raises(ValueError, match="layers"):
        P4ServingConfig.from_env(environment)


def add_manifest_layer(environment, layer):
    def add(manifest):
        manifest["layers"].append(layer)
        for entry in list(manifest["entries"][:4]):
            path = Path(environment["GLM53_P4_NATIVE_SIDECAR_DIR"]) / f"p4-layer-{layer:03d}-tp4-rank-{entry['rank']}.safetensors"
            path.write_bytes((Path(environment["GLM53_P4_NATIVE_SIDECAR_DIR"]) / entry["path"]).read_bytes())
            manifest["entries"].append(dict(entry, layer=layer, path=path.name))
    mutate_manifest(environment, add)


@pytest.mark.skipif(not LAUNCHER.is_file(), reason="exact parent launcher absent; readiness contract NOT verified")
@pytest.mark.parametrize("layers", ["3", "3,4", "3,4,44"])
def test_readiness_matches_exact_parent_launcher_grep(environment, layers, tmp_path, capsys):
    for layer in map(int, layers.split(",")[1:]):
        add_manifest_layer(environment, layer)
    environment["GLM53_P4_NATIVE_LAYERS"] = layers
    install(environment, components=(model_for(runner()), (Carrier,), Backend))
    log = capsys.readouterr().out
    (tmp_path / "server-ready.log").write_text(log)
    source = LAUNCHER.read_text()
    gate = next(line for line in source.splitlines() if 'grep -q "GLM53_P4_NATIVE_PATCH_ACTIVE' in line)
    # Run the actual shell expression, not a separately maintained regex.
    env = dict(os.environ, P4_NATIVE="1", P4_NATIVE_LAYERS=layers, SESSION=str(tmp_path))
    process = subprocess.run(["bash", "-eu", "-c", gate], env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    (tmp_path / "server-ready.log").write_text(log.replace("ldlq=false", "ldlq=true"))
    assert subprocess.run(["bash", "-eu", "-c", gate], env=env).returncode != 0
    print("launcher_sha256", digest(LAUNCHER.read_bytes()))


def test_model_owner_is_shared_only_by_same_namespace_and_rank(environment):
    add_manifest_layer(environment, 4)
    environment["GLM53_P4_NATIVE_LAYERS"] = "3,4"
    model = SimpleNamespace(FusedMoEFactory=lambda *args, **kwargs: runner(rank=kwargs.get("test_rank", 0)))
    install(environment, components=(model, (Carrier,), Backend))
    def owner(prefix, rank=0):
        layer = model.FusedMoEFactory(prefix=prefix, test_rank=rank).routed_experts
        method = layer.quant_method
        method.process_weights_after_loading(layer)
        assert layer._glm53_p4_runtime.kwargs["workspace_owner"] is method._glm53_p4_workspace_owner
        return method._glm53_p4_workspace_owner
    first = owner("first.layers.3.mlp.experts")
    assert first is owner("first.layers.4.mlp.experts")
    assert first is not owner("second.layers.3.mlp.experts")
    assert first is not owner("first.layers.3.mlp.experts", 1)
    with pytest.raises(RuntimeError, match="duplicate model namespace"):
        owner("first.layers.3.mlp.experts")


@pytest.mark.parametrize("dbo,ubatching", [(True, True), (False, True)])
def test_real_parallel_dbo_or_microbatching_rejected_before_factory(environment, dbo, ubatching):
    def forbidden(*args, **kwargs):
        raise AssertionError("carrier allocated before unsafe parallelism rejection")
    model = SimpleNamespace(FusedMoEFactory=forbidden)
    install(environment, components=(model, (Carrier,), Backend),
            parallel_config_getter=lambda: SimpleNamespace(enable_dbo=dbo, use_ubatching=ubatching))
    with pytest.raises(ValueError, match="DBO and microbatching disabled"):
        model.FusedMoEFactory(prefix="model.layers.3.mlp.experts")


def test_cold_compile_and_prepare_are_forbidden_inside_capture(monkeypatch):
    runtime = P4NativeTPMoE.__new__(P4NativeTPMoE)
    runtime.lib, runtime.prepared, runtime.device = None, False, torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: contextlib.nullcontext())
    with pytest.raises(RuntimeError, match="before graph capture"):
        runtime.compile()
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        runtime.prepare()


def cpu_runtime(owner=None):
    runtime = P4NativeTPMoE.__new__(P4NativeTPMoE)
    runtime.device, runtime.experts, runtime.hidden, runtime.intermediate, runtime.topk = torch.device("cpu"), 3, 128, 64, 2
    runtime.workspace_owner = owner if owner is not None else P4WorkspaceOwner()
    return runtime


def test_graph_address_reuse_is_model_shape_stream_capture_scoped_and_activation_only():
    runtime = cpu_runtime()
    work = runtime.workspace(17, 101)
    assert work is runtime.workspace(17, 101)
    assert work is not runtime.workspace(17, 102) and work is not runtime.workspace(33, 101)
    assert work is cpu_runtime(runtime.workspace_owner).workspace(17, 101)
    assert work is not cpu_runtime().workspace(17, 101)
    captured = runtime.workspace(17, 101, 0)
    assert captured is not work and captured is runtime.workspace(17, 101, 0)
    assert captured is not runtime.workspace(17, 101, 1)
    assert not any("weight" in name or "trellis" in name for name in work)
    assert work["a1"].dtype == torch.int32 and work["sfa1"].dtype == torch.uint8
    assert work["a1"].numel() * 8 == 17 * 128
    assert work["sfa1"].numel() * 16 == 17 * 128
    assert len({value.data_ptr() for value in work.values()}) == len(work)


def test_capture_pins_external_weights_inputs_and_scratch_after_wrapper_is_gone():
    runtime = cpu_runtime()
    owner = runtime.workspace_owner
    runtime.tensors = {"physical_stream": torch.arange(16, dtype=torch.int16)}
    weight_ref = weakref.ref(runtime.tensors["physical_stream"])
    work = runtime.workspace(2, 101, 17)
    address, scratch_ref = work["output"].data_ptr(), weakref.ref(work["output"])
    activation, weights = torch.zeros(2), torch.zeros(3)
    activation_ref, weights_ref = weakref.ref(activation), weakref.ref(weights)
    runtime._pin_capture(17, activation, weights)
    runtime_ref = weakref.ref(runtime)
    del runtime, work, activation, weights
    gc.collect()
    assert runtime_ref() is not None and weight_ref() is not None
    assert scratch_ref().data_ptr() == address
    assert activation_ref() is not None and weights_ref() is not None
    assert runtime_ref() in _CAPTURE_PINS[(owner, 17)]["runtimes"].values()


def test_eager_owner_can_be_reclaimed_without_graph_pins():
    runtime = cpu_runtime()
    work = runtime.workspace(2, 101)
    owner_ref, scratch_ref = weakref.ref(runtime.workspace_owner), weakref.ref(work["output"])
    del runtime, work
    gc.collect()
    assert owner_ref() is None and scratch_ref() is None


def test_host_owner_guard_rejects_real_thread_overlap_and_recovers_from_error():
    owner = P4WorkspaceOwner()
    errors = []
    def conflicting_call():
        try:
            with owner.dispatch():
                raise AssertionError("concurrent owner entered critical region")
        except RuntimeError as error:
            errors.append(str(error))
    with owner.dispatch():
        thread = threading.Thread(target=conflicting_call)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive() and errors and "concurrent/reentrant" in errors[0]
        with pytest.raises(RuntimeError, match="concurrent/reentrant"):
            with owner.dispatch():
                pass
        with P4WorkspaceOwner().dispatch():
            pass  # independent model remains independent
    with pytest.raises(ValueError):
        with owner.dispatch():
            raise ValueError("intentional host failure")
    with owner.dispatch():
        pass


@pytest.mark.parametrize("active,sequence,expected", [(0, 0, None), (1, 0, 0), (1, 1234, 1234)])
def test_capture_id_host_abi_distinguishes_eager_and_valid_zero(active, sequence, expected):
    runtime = cpu_runtime()
    def query(stream, active_pointer, id_pointer):
        assert stream == 4321
        ctypes.cast(active_pointer, ctypes.POINTER(ctypes.c_int))[0] = active
        ctypes.cast(id_pointer, ctypes.POINTER(ctypes.c_ulonglong))[0] = sequence
        return 0
    runtime.lib = SimpleNamespace(p4_capture_state=query)
    assert runtime.capture_id(4321) == expected


@pytest.mark.parametrize("status,active", [(901, 0), (0, -1)])
def test_capture_query_error_fails_closed(status, active):
    runtime = cpu_runtime()
    def query(stream, active_pointer, id_pointer):
        ctypes.cast(active_pointer, ctypes.POINTER(ctypes.c_int))[0] = active
        return status
    runtime.lib = SimpleNamespace(p4_capture_state=query)
    with pytest.raises(RuntimeError):
        runtime.capture_id(4321)


def test_sparse_launch_bound_is_exact_for_exhaustive_count_partitions():
    # DP enumerates every count partition for up to 7 experts and 128 routes.
    maximum = [0] + [-999] * 128
    for experts in range(1, 8):
        maximum = [max(maximum[routes - count] + (count + 15) // 16
                       for count in range(routes + 1)) for routes in range(129)]
        assert maximum == [max_route_tiles(routes, experts) for routes in range(129)]
    assert max_route_tiles(8, 288) == 8
    assert max_route_tiles(2048 * 8, 288) == 1294


def test_exactly_two_A_quantizers_with_both_weight_decodes_in_native_prologue():
    source = SOURCE.read_text()
    tree = ast.parse((ROOT / "runtime_patch/p4_native_kernel.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    quantizers = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "p4_quantize"]
    projections = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "project"]
    assert len(quantizers) == len(projections) == 2
    assert ast.literal_eval(projections[0].args[2]) == "w13"
    assert ast.literal_eval(projections[1].args[2]) == "w2"
    assert ast.literal_eval(quantizers[0].args[5]) == 1  # BF16 input directly; no x.float allocation
    assert ast.literal_eval(quantizers[1].args[5]) == 0  # FP32 SwiGLU
    produce = source[source.index("void produce("):source.index("// Tiles are grouped")]
    assert "weight_word(" in produce and "__nv_cvt_float_to_fp4" not in produce
    assert "x_scale[x_row" in produce and "p4::scale_byte_offset" in produce
    assert "cudaDeviceSynchronize" not in source and "cudaStreamSynchronize" not in source
    runtime_source = (ROOT / "runtime_patch/p4_native_kernel.py").read_text()
    assert "current_stream(self.device)" in runtime_source and "v.record_stream(current_stream)" in runtime_source
    assert "torch.sort(ids, stable=True, out=" in runtime_source
    assert "result = output.to(torch.bfloat16)" in runtime_source  # caller owns returned storage


@pytest.mark.parametrize("capture_id", [None, 0, 42])
def test_real_hotpath_C_ABI_order_routes_and_stream_without_CUDA(monkeypatch, capture_id):
    from glm53_nvfp4.p4_reference import synthetic_payload
    runtime = cpu_runtime()
    runtime.limit, runtime.tensors = 10.0, synthetic_payload()[0]
    runtime.prepared = True
    calls = []
    stream = SimpleNamespace(cuda_stream=4321)

    class Library:
        def p4_capture_state(self, stream_id, active_pointer, id_pointer):
            assert stream_id == 4321
            ctypes.cast(active_pointer, ctypes.POINTER(ctypes.c_int))[0] = int(capture_id is not None)
            ctypes.cast(id_pointer, ctypes.POINTER(ctypes.c_ulonglong))[0] = capture_id or 0
            return 0

        def p4_quantize(self, inp, packed, scales, rows, k, bf16, stream_id):
            assert stream_id == 4321
            calls.append(("quantize", rows, k, bf16))
            return 0

        def p4_project(self, inp, sf, trellis, scales, globals_, order, offsets, tiles, out,
                       experts, routes, n, k, projections, divisor, stream_id):
            assert stream_id == 4321 and (experts, routes) == (3, 34)
            counts = list((ctypes.c_int64 * 4).from_address(offsets))
            tiled = list((ctypes.c_int64 * 4).from_address(tiles))
            assert counts == [0, 12, 23, 34] and tiled == [0, 1, 2, 3]
            name = "w13" if projections == 2 else "w2"
            assert trellis == runtime.tensors[name + "_trellis"].data_ptr()
            assert scales == runtime.tensors[name + "_scale_e4m3"].data_ptr()
            assert globals_ == runtime.tensors[name + "_global_scale"].data_ptr()
            calls.append((name, n, k, projections, divisor))
            return 0

        def p4_swiglu(self, gu, mid, routes, n, limit, stream_id):
            assert (routes, n, limit, stream_id) == (34, 64, 10.0, 4321)
            calls.append(("swiglu",))
            return 0

        def p4_sum(self, routed, weights, output, m, topk, hidden, stream_id):
            assert (m, topk, hidden, stream_id) == (17, 2, 128, 4321)
            values = (ctypes.c_float * (m * hidden)).from_address(output)
            for i in range(m * hidden):
                values[i] = 7.0
            calls.append(("sum",))
            return 0

    runtime.lib = Library()
    monkeypatch.setattr(torch.cuda, "device", lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, value: None)
    x = torch.zeros((17, 128), dtype=torch.bfloat16)
    ids = (torch.arange(34) % 3).reshape(17, 2)
    weights = torch.full((17, 2), .5)
    output = runtime(x, weights, ids)
    assert torch.equal(output, torch.full_like(x, 7))
    assert calls == [("quantize", 17, 128, 1), ("w13", 64, 128, 2, 2), ("swiglu",),
                     ("quantize", 34, 64, 0), ("w2", 128, 64, 1, 1), ("sum",)]
    assert output.data_ptr() != runtime.workspace(17, 4321, capture_id)["output"].data_ptr()
    with runtime.workspace_owner.dispatch():
        with pytest.raises(RuntimeError, match="concurrent/reentrant"):
            runtime(x, weights, ids)
    assert len(calls) == 6  # the rejected call did not enqueue any stage


def extract_method(relative, class_name, name):
    path = VLLM / "vllm" / relative
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace = {"torch": torch, "os": os}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.skipif(not (VLLM / "vllm").is_dir(), reason="exact vLLM source absent; source integration NOT verified")
@pytest.mark.parametrize("carrier_path,carrier_class", [
    ("model_executor/layers/fused_moe/unquantized_fused_moe_method.py", "UnquantizedFusedMoEMethod"),
    ("model_executor/layers/quantization/modelopt.py", "ModelOptNvFp4FusedMoE"),
])
def test_real_vllm_modular_method_and_real_carrier_signatures_reach_p4(environment, carrier_path, carrier_class):
    methods = {name: extract_method(carrier_path, carrier_class, name)
               for name in ("apply", "process_weights_after_loading")}
    real_carrier = type(carrier_class, (Carrier,), methods)
    actual = runner(real_carrier)
    layer = actual.routed_experts
    layer.forward_modular = MethodType(extract_method("model_executor/layers/fused_moe/routed_experts.py",
                                                      "RoutedExperts", "forward_modular"), layer)
    layer._ensure_moe_quant_config_init = MethodType(extract_method(
        "model_executor/layers/fused_moe/routed_experts.py", "RoutedExperts", "_ensure_moe_quant_config_init"), layer)
    model = model_for(actual)
    install(environment, components=(model, (real_carrier,), Backend))
    model.FusedMoEFactory(prefix="model.layers.3.mlp.experts")
    layer.quant_method.process_weights_after_loading(layer)
    layer._ensure_moe_quant_config_init()
    assert layer.quant_method.moe_quant_config is None
    x = torch.zeros((1, 4096), dtype=torch.bfloat16)
    weights, ids = torch.full((1, 8), .125), torch.arange(8).reshape(1, 8)
    output = layer.forward_modular(x, weights, ids, actual.shared_experts, x)
    assert torch.equal(output, x + 7)
    assert layer.quant_method.apply.__func__ is _P4ServingMethod.apply
    assert layer.quant_method.process_weights_after_loading.__func__ is _P4ServingMethod.process_weights_after_loading


@pytest.mark.skipif(not (VLLM / "vllm").is_dir(), reason="exact vLLM source absent; source integration NOT verified")
def test_exact_vllm_factory_loader_and_runner_ownership_contract(capsys):
    sources = {
        "model": "models/glm5next/nvidia/model.py",
        "factory": "model_executor/layers/fused_moe/layer.py",
        "runner": "model_executor/layers/fused_moe/runner/moe_runner.py",
        "loader": "model_executor/model_loader/utils.py",
        "config": "model_executor/layers/fused_moe/config.py",
        "parallel": "config/parallel.py",
    }
    text = {name: (VLLM / "vllm" / path).read_text() for name, path in sources.items()}
    assert "self.experts = FusedMoEFactory(" in text["model"]
    assert "swiglu_limit=swiglu_limit" in text["model"]
    assert "runner = runner_cls(" in text["factory"] and "routed_experts=routed_experts" in text["factory"]
    assert "quant_method.process_weights_after_loading(module)" in text["loader"]
    assert "topk_weights, topk_ids = self.router.select_experts(" in text["runner"]
    assert "fused_out = self.routed_experts.forward_modular(" in text["runner"]
    assert "self._quant_method.mk_can_overlap_shared_experts" in text["runner"]
    assert "tensor_model_parallel_all_reduce(" in text["runner"]
    assert "return self.moe_parallel_config.tp_rank" in text["config"]
    assert "enable_dbo=vllm_config.parallel_config.enable_dbo" in text["factory"]
    assert "self.enable_dbo = enable_dbo" in text["runner"]
    use_ubatching = extract_method("config/parallel.py", "ParallelConfig", "use_ubatching")
    assert not use_ubatching(SimpleNamespace(enable_dbo=False, ubatch_size=1))
    assert use_ubatching(SimpleNamespace(enable_dbo=False, ubatch_size=2))
    assert use_ubatching(SimpleNamespace(enable_dbo=True, ubatch_size=0))
    for path in sources.values():
        print("vllm_source_sha256", path, digest((VLLM / "vllm" / path).read_bytes()))


def test_capture_query_uses_cuda_sequence_and_invalidated_capture_fails_closed(capsys):
    source = SOURCE.read_text()
    query = source[source.index('extern "C" int p4_capture_state'):source.index('extern "C" int p4_prepare')]
    assert "cudaStreamGetCaptureInfo(stream, &capture, &id)" in query
    assert "cudaStreamCaptureStatusInvalidated" in query
    assert "cudaErrorStreamCaptureInvalidated" in query
    assert "cudaStreamCaptureStatusActive" in query and "cudaStreamCaptureStatusNone" in query
    header = Path("/usr/local/cuda-13.2/include/cuda_runtime_api.h")
    if header.is_file():
        actual = header.read_text()
        assert "unique over the lifetime of the process" in actual
        assert "cudaStreamGetCaptureInfo(cudaStream_t stream," in actual
        print("cuda_runtime_header_sha256", digest(header.read_bytes()))
