import shlex

import pytest

from glm53_nvfp4 import p8_exl3_fp8_decode_control as control


def test_fixed_panel_storage_and_product_baselines():
    assert control.RAW_BYTES == 10_145_259_520
    assert control.HEADROOM == 20 * 2**30
    assert len(control.WINDOW_IDS) == 4
    assert control.PRODUCT == {"stock": "p8", "exl3": "exl3"}
    assert set(control.BASELINES) == {"stock", "exl3"}
    for arm in control.BASELINES:
        assert set(control.BASELINES[arm]) == set(control.WINDOW_IDS)
        assert sum(control.BASELINES[arm].values()) / 4 == pytest.approx(control.BASELINE_MEANS[arm])


def test_selected_panel_is_domain_balanced():
    rows = control.selected_windows()
    assert [row["id"] for row in rows] == list(control.WINDOW_IDS)
    assert len({row["domain"] for row in rows}) == 4


def test_exact_product_receipts_and_recipes_are_available():
    receipts = control.image_receipts()
    recipes = control.recipes()
    assert set(receipts) == set(recipes) == {"stock", "exl3"}
    assert receipts["stock"]["status"] == receipts["exl3"]["status"] == "complete"
    assert control.sha(control.P8_RECIPE) == control.P8_RECIPE_SHA
    assert control.sha(control.EXL3_RECIPE) == control.EXL3_RECIPE_SHA


def test_replace_option_changes_exactly_one_cli_value():
    command = "python -m vllm.entrypoints.openai.api_server --kv-cache-dtype nvfp4_ds_mla --port 8000"
    changed = control.replace_option(command, "--kv-cache-dtype", "fp8_ds_mla")
    tokens = shlex.split(changed)
    assert tokens[tokens.index("--kv-cache-dtype") + 1] == "fp8_ds_mla"
    assert tokens[tokens.index("--port") + 1] == "8000"


def test_runtime_contract_is_product_specific_and_mtp_off():
    assert control.model_name("stock").endswith("-p8")
    assert control.model_name("n64").endswith("-p8")
    assert control.model_name("exl3").endswith("-exl3")
    assert control.slot_name({"stage": "control", "arm": "n64"}) == "stock"
    assert control.slot_name({"stage": "control", "arm": "exl3"}) == "exl3"
    assert control.ORDER == ({"stage": "control", "arm": "stock"}, {"stage": "control", "arm": "exl3"})
