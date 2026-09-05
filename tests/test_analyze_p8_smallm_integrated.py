import pytest

from glm53_nvfp4.analyze_p8_smallm_integrated import analyze


def fixtures(decode=75.6):
    row = {"context_tokens": 32768, "concurrency": 1, "aggregate_tps": decode,
           "hardware_summary": {"temp_max_c": 71}}
    result = {"results": [row], "prefill": {"32768": {
        "client_tok_per_sec": 7166, "hardware_summary": {"temp_max_c": 62}}}}
    graph = {"full_capture_complete": True, "ranks": [0, 1, 2, 3], "status": "pass"}
    execution = {"exit_code": 0, "protected_roles_opened": [], "ldlq": False}
    lines = [f"GLM53_P8_NATIVE_FORWARD layer={layer} rank={rank} small_m_scheduler=true"
             for layer in range(3, 45) for rank in range(4)]
    return result, graph, execution, "\n".join(lines)


def test_integrated_pass_preserves_failed_product_gate():
    result = analyze(*fixtures())
    assert result["integration_status"] == "pass"
    assert result["product_gate"] == "fail"
    assert result["small_m_forward_pairs"] == 168


def test_rejects_incomplete_forward_inventory():
    values = list(fixtures())
    values[3] = values[3].splitlines()[0]
    with pytest.raises(ValueError, match="inventory"):
        analyze(*values)
