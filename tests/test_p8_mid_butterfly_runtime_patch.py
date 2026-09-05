import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sitecustomize_contains_fail_closed_mid_butterfly_reference():
    source = (ROOT / "runtime_patch/sitecustomize.py").read_text()
    ast.parse(source)
    assert '"mid-butterfly"' in source
    assert "GLM53_P8_MID_BUTTERFLY_ANGLE_PI" in source
    assert "_p8_shared_butterfly16_staged(middle)" in source
    assert "work[:, left] = cosine * a - sine * b" in source
    assert "work[:, right] = sine * a + cosine * b" in source
    assert "shared_butterfly_p00625" in source


def test_kld_launcher_forwards_only_the_frozen_angle():
    source = (ROOT / "scripts/run_kld_v3.sh").read_text()
    assert "P8_MID_BUTTERFLY_ANGLE_PI=${GLM53_P8_MID_BUTTERFLY_ANGLE_PI:-}" in source
    assert '[ "$P8_MID_BUTTERFLY_ANGLE_PI" = 0.0625 ]' in source
    assert '-e GLM53_P8_MID_BUTTERFLY_ANGLE_PI="$P8_MID_BUTTERFLY_ANGLE_PI"' in source
