from glm53_nvfp4.verify_rotation_runtime import parse_layers, verify


def test_verify_rotation_runtime_exact_coverage():
    lines = []
    digest = "a" * 64
    for layer in (3, 4):
        for rank in (0, 1):
            lines.append(
                "GLM53_BLOCK_ROTATION_FORWARD "
                f"layer={layer} rank={rank} rotation_sha256={digest} "
                "hidden_width=4096 dtype=torch.bfloat16"
            )
    result = verify("\n".join(lines), {3, 4}, {0, 1})
    assert result["status"] == "pass"
    assert result["forward_pairs"] == 4


def test_parse_layers_accepts_ranges_and_items():
    assert parse_layers("3-5,9") == {3, 4, 5, 9}
