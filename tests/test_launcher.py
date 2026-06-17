from storepose.launcher import _col_window


def test_no_scroll_when_all_columns_fit():
    # window collapses to 0 whenever the viewport can show every column
    assert _col_window(col=5, col_off=3, n_cols=7, max_visible=7) == 0
    assert _col_window(col=5, col_off=3, n_cols=7, max_visible=10) == 0


def test_window_slides_right_to_keep_selection_visible():
    # col past the right edge pulls the window so col sits at the last slot
    assert _col_window(col=4, col_off=0, n_cols=7, max_visible=3) == 2
    assert _col_window(col=6, col_off=0, n_cols=7, max_visible=3) == 4


def test_window_slides_left_to_keep_selection_visible():
    assert _col_window(col=1, col_off=4, n_cols=7, max_visible=3) == 1


def test_window_holds_when_selection_already_visible():
    assert _col_window(col=3, col_off=2, n_cols=7, max_visible=3) == 2


def test_window_clamped_to_last_full_page():
    # can't scroll past the point where the final column is flush right
    assert _col_window(col=6, col_off=99, n_cols=7, max_visible=4) == 3


def test_degenerate_widths_return_zero():
    assert _col_window(col=3, col_off=2, n_cols=7, max_visible=0) == 0
