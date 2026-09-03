from glm53_nvfp4.prefetch_partial_capture import merge_windows


def test_merge_windows_preserves_only_selected_consecutive_ranges():
    assert merge_windows([7, 3, 4, 4, 11, 12, 13, 20]) == [
        (3, 4),
        (7, 7),
        (11, 13),
        (20, 20),
    ]
