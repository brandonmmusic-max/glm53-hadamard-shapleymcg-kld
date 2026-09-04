import ast
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITECUSTOMIZE = ROOT / "runtime_patch" / "sitecustomize.py"
VLLM_ROUTED = Path(
    "/home/brandonmusic/KLC_SANDBOXES/glm53-r19-port/r19-vllm-port-work/"
    "vllm/model_executor/layers/fused_moe/routed_experts.py"
)


def _extract_function(name):
    tree = ast.parse(SITECUSTOMIZE.read_text())
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == name)
    node = copy.deepcopy(node)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"_P8N_ORIGINAL_MODELOPT_GET_QUANT_CONFIG": lambda self, layer: ("stock", layer)}
    exec(compile(ast.fix_missing_locations(module), str(SITECUSTOMIZE), "exec"), namespace)
    return namespace[name]


def test_native_modelopt_bypasses_released_carrier_quant_config():
    wrapper = _extract_function("_p8n_modelopt_get_fused_moe_quant_config")
    native = type("Method", (), {"_glm53_p8_native": True})()
    stock = type("Method", (), {"_glm53_p8_native": False})()
    assert wrapper(native, object()) is None
    marker = object()
    assert wrapper(stock, marker) == ("stock", marker)


def test_exact_routed_hook_and_runtime_assignment_are_present():
    routed = VLLM_ROUTED.read_text()
    source = SITECUSTOMIZE.read_text()
    assert "self.quant_method.get_fused_moe_quant_config(self)" in routed
    assert "_P8NativeModelOptNvFp4FusedMoE.get_fused_moe_quant_config = (" in source
    assert "_p8n_modelopt_get_fused_moe_quant_config" in source
