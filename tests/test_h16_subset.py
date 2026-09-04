import pytest

from glm53_nvfp4.h16_subset import parse_layers


def test_parse_layers_accepts_ranges_and_singletons() -> None:
    assert parse_layers("3-6,9,11-12") == [3, 4, 5, 6, 9, 11, 12]


@pytest.mark.parametrize("spec", ["", "2", "45", "9-3"])
def test_parse_layers_rejects_invalid_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_layers(spec)
