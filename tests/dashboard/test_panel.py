import numpy as np

from storepose.dashboard.panel import (
    PanelData,
    composite,
    panel_data,
    render_panel,
)
from storepose.dashboard.state import DashboardState


def _empty_data(show_alt=False):
    return PanelData(
        busy=None,
        summary={"in_line": 0, "at_pos": 0, "avg_line_s": 0.0, "avg_pos_s": 0.0,
                 "avg_total_s": 0.0, "served_count": 0},
        checkouts={"mashgin_avg": 0.0, "mashgin_n": 0, "other_avg": 0.0,
                   "other_n": 0, "delta": 0.0},
        occ=[],
        show_alt=show_alt,
    )


def test_render_panel_shape_and_dtype():
    img = render_panel(240, 720, _empty_data())
    assert img.shape == (720, 240, 3)
    assert img.dtype == np.uint8


def test_render_panel_handles_populated_data():
    data = PanelData(
        busy=(12.0, "High", 4.2),
        summary={"in_line": 3, "at_pos": 1, "avg_line_s": 22.5, "avg_pos_s": 9.1,
                 "avg_total_s": 31.6, "served_count": 7},
        checkouts={"mashgin_avg": 9.1, "mashgin_n": 7, "other_avg": 14.0,
                   "other_n": 2, "delta": 4.9},
        occ=[(float(t), t % 4, 1) for t in range(60)],
        show_alt=True,
    )
    img = render_panel(260, 700, data)
    assert img.shape == (700, 260, 3)


def test_composite_video_is_three_quarters_with_even_width():
    video = np.zeros((480, 600, 3), np.uint8)
    out = composite(video, _empty_data())
    assert out.shape[0] == 480                     # height unchanged
    assert out.shape[1] % 2 == 0                   # avc1 needs even width
    # video occupies (close to) the left three quarters
    assert abs(video.shape[1] / out.shape[1] - 0.75) < 0.02
    # the video region is left untouched
    assert np.array_equal(out[:, :600], video)


def test_panel_data_pulls_from_dashboard_state():
    st = DashboardState()
    st.observe(0.0, 2, 1)
    st.add_visit(1.0, 20.0, 8.0, "served")
    st.set_busy(1.0, "Medium", 1.8)
    data = panel_data(st, show_alt=True)
    assert data.busy == (1.0, "Medium", 1.8)
    assert data.summary["in_line"] == 2
    assert data.summary["served_count"] == 1
    assert data.show_alt is True
