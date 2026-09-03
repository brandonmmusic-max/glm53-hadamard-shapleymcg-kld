from glm53_nvfp4.verify_mixed_rotation_runtime import verify_mixed


def test_verify_mixed_rotation_runtime_exact_coverage():
    digest = "d" * 64
    lines = []
    for rank in (0, 1):
        lines.extend(
            [
                "GLM53_BLOCK_ROTATION_FORWARD "
                f"layer=3 rank={rank} rotation_sha256={digest} "
                "hidden_width=4096 dtype=torch.bfloat16",
                "GLM53_BLOCK_MID_ROTATION_FORWARD "
                f"layer=3 rank={rank} rotation_sha256={digest} "
                "hidden_width=2048 dtype=torch.bfloat16",
                "GLM53_MXFP6_H16_ALL_PROJECTION_PATCH_ACTIVE "
                f"prefix=model.language_model.layers.4.mlp.experts rank={rank} "
                "rotation_sha256=c0cce70ab9288f764401571e81431a5af51cd595f2beb6f864ee517e8bdf89be "
                "input=python-block16 mid=fused-w6a8-block16",
                "GLM53_MXFP6_H16_INPUT_FORWARD "
                f"prefix=model.language_model.layers.4.mlp.experts rank={rank} "
                "rotation_sha256=c0cce70ab9288f764401571e81431a5af51cd595f2beb6f864ee517e8bdf89be "
                "hidden_width=4096 dtype=torch.bfloat16",
            ]
        )
    result = verify_mixed("\n".join(lines), {3, 4}, {4}, {0, 1})
    assert result["status"] == "pass"
    assert result["mxfp6_active_pairs"] == 2
    assert result["nvfp4_proof"]["mid_forward_pairs"] == 2
