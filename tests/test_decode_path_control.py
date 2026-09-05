from glm53_nvfp4 import decode_path_control as control


def test_fixed_storage_and_panel_geometry():
    assert control.RAW_BYTES == 10_145_259_520
    assert len(control.WINDOW_IDS) == 4
    assert len(set(control.WINDOW_IDS)) == 4
    assert sum(control.P8_WINDOW_KLD.values()) / 4 == control.P8_MEAN


def test_control_image_receipts_are_bound_to_exact_parents():
    rows = control.load_image_receipts()
    assert set(rows) == {"stock", "exl3"}
    for arm, row in rows.items():
        assert row["parent_image_id"] == control.PARENTS[arm]
        assert row["status"] == "complete"


def test_selected_panel_is_domain_balanced():
    rows = control.selected_windows()
    assert [row["id"] for row in rows] == list(control.WINDOW_IDS)
    assert len({row["domain"] for row in rows}) == 4
