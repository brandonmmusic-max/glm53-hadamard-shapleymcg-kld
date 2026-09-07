import trellismx
from trellismx.codec import P4_CONTRACT, P8_CONTRACT


def test_public_name_and_capabilities_are_explicit() -> None:
    assert trellismx.__version__ == "0.1.0"
    observed = trellismx.capabilities()
    assert observed["name"] == "TrellisMX"
    assert observed["cpu"]["p4_matrix_codec"]["status"] == "supported"
    assert observed["cpu"]["p8_reference_codec"]["status"] == "supported"
    assert observed["cpu"]["p8_reference_codec"]["rates"] == [3, 4, 5]
    assert observed["cpu"]["p8_checkpoint_conversion"]["status"] == "not-exposed-by-this-cli"
    assert observed["cpu"]["architecture_adapters"]["status"] == "glm53-flash-research-only"
    assert observed["cpu"]["architecture_adapters"]["unsupported_policy"] == "reject before reading or encoding model weights"
    assert observed["cpu"]["p8_viterbi_encoder_backend"]["status"] == "external-plugin-required"
    assert observed["cpu"]["p8_viterbi_encoder_backend"]["default_enabled"] is False
    assert observed["device"]["status"] == "not-run-by-this-package"


def test_contracts_keep_p4_and_p8_separate() -> None:
    assert P4_CONTRACT["schema"] == "glm53-p4-mcg-matrix.v1"
    assert P4_CONTRACT["alphabet"] == "e2m1"
    assert P8_CONTRACT["compute_operand"] == "e4m3"
    assert P8_CONTRACT["mma"] == "mxf8f6f4"
    assert P8_CONTRACT["stored_rates"] == (3, 4, 5)
    assert P8_CONTRACT["general_converter"] is False
