import pytest

from storepose.config import AppConfig, from_args


def test_defaults():
    cfg = from_args([])
    assert cfg.source == 0
    assert cfg.mode == "balanced"
    assert cfg.device == "mps"
    assert cfg.show_fps is True


def test_parses_flags():
    cfg = from_args(
        ["--source", "2", "--mode", "lightweight", "--det-conf", "0.6",
         "--kpt-thr", "0.4", "--device", "mps", "--no-fps"]
    )
    assert cfg.source == 2
    assert cfg.mode == "lightweight"
    assert cfg.det_conf == 0.6
    assert cfg.kpt_thr == 0.4
    assert cfg.device == "mps"
    assert cfg.show_fps is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "ultra"},
        {"device": "cuda"},
        {"det_conf": 1.5},
        {"kpt_thr": -0.1},
        {"busy_window": 0},
        {"busy_metric": "vibes"},
        {"busy_medium_max": 1.0, "busy_low_max": 2.0},
        {"busy_hysteresis": -0.5},
    ],
)
def test_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        AppConfig(**kwargs)


def test_busy_log_implies_busy():
    cfg = from_args(["--busy-log", "out.csv", "--busy-metric", "mean_wait",
                     "--busy-window", "300"])
    assert cfg.busy is True
    assert cfg.busy_log == "out.csv"
    assert cfg.busy_metric == "mean_wait"
    assert cfg.busy_window == 300.0


def test_invalid_choice_exits():
    with pytest.raises(SystemExit):
        from_args(["--mode", "nope"])


def test_tracking_defaults():
    c = from_args([])
    assert c.track is True
    assert c.hold_seconds == 1.5
    assert c.min_hits == 3
    assert c.iou_thr == 0.3
    assert c.smooth is True
    assert c.smooth_cutoff == 1.0
    assert c.smooth_beta == 0.007


def test_tracking_flags():
    c = from_args([
        "--no-track", "--no-smooth", "--hold-seconds", "2.5",
        "--min-hits", "5", "--iou-thr", "0.4",
        "--smooth-cutoff", "0.5", "--smooth-beta", "0.01",
    ])
    assert c.track is False
    assert c.smooth is False
    assert c.hold_seconds == 2.5
    assert c.min_hits == 5
    assert c.iou_thr == 0.4
    assert c.smooth_cutoff == 0.5
    assert c.smooth_beta == 0.01


@pytest.mark.parametrize("kwargs", [
    {"min_hits": 0},
    {"iou_thr": 1.5},
    {"hold_seconds": -1.0},
    {"smooth_cutoff": 0.0},
    {"smooth_beta": -0.1},
])
def test_tracking_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        AppConfig(**kwargs)


def test_max_overlap_default_and_flag():
    assert from_args([]).max_overlap == 0.5
    assert from_args(["--max-overlap", "0.7"]).max_overlap == 0.7


def test_max_overlap_validation():
    with pytest.raises(ValueError):
        AppConfig(max_overlap=1.5)


def test_queue_defaults():
    c = from_args([])
    assert c.zone is None
    assert c.define_zone is False
    assert c.wait_enter_frames == 5
    assert c.wait_exit_seconds == 2.0
    assert c.wait_log is None


def test_queue_flags():
    c = from_args([
        "--zone", "zones/sco.json", "--define-zone",
        "--wait-enter-frames", "8",
        "--wait-exit-seconds", "3.0", "--wait-log", "waits.csv",
    ])
    assert c.zone == "zones/sco.json"
    assert c.define_zone is True
    assert c.wait_enter_frames == 8
    assert c.wait_exit_seconds == 3.0
    assert c.wait_log == "waits.csv"


@pytest.mark.parametrize("kwargs", [
    {"wait_enter_frames": 0},
    {"wait_exit_seconds": -0.5},
])
def test_queue_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        AppConfig(**kwargs)


def test_zone_foot_band_default_flag_and_validation():
    assert from_args([]).zone_foot_band == 0.3
    assert from_args(["--zone-foot-band", "0.25"]).zone_foot_band == 0.25
    with pytest.raises(ValueError):
        AppConfig(zone_foot_band=0.0)


def test_reid_defaults_on():
    cfg = from_args([])
    assert cfg.reid is True
    assert cfg.reid_seconds == 5.0
    assert cfg.reid_thr == 0.6


def test_no_reid_flag_disables():
    assert from_args(["--no-reid"]).reid is False


def test_reid_flags_parse():
    cfg = from_args(["--reid-seconds", "8", "--reid-thr", "0.4"])
    assert cfg.reid_seconds == 8.0 and cfg.reid_thr == 0.4


def test_reid_seconds_must_be_nonnegative():
    import pytest
    with pytest.raises(ValueError):
        AppConfig(reid_seconds=-1.0)


def test_reid_thr_must_be_in_range():
    import pytest
    with pytest.raises(ValueError):
        AppConfig(reid_thr=2.0)


def test_det_conf_and_overlap_defaults():
    cfg = from_args([])
    assert cfg.det_conf == 0.7
    assert cfg.det_overlap == 0.8


def test_det_overlap_flag_parses():
    assert from_args(["--det-overlap", "0.9"]).det_overlap == 0.9


def test_det_overlap_must_be_in_range():
    with pytest.raises(ValueError):
        AppConfig(det_overlap=1.5)


def test_pos_zone_flags():
    cfg = from_args(["--pos-zone", "zones/p.json"])
    assert cfg.pos_zone == "zones/p.json"
    assert cfg.define_pos_zone is False
    assert from_args(["--define-pos-zone"]).define_pos_zone is True


def test_pos_zone_defaults_none():
    cfg = from_args([])
    assert cfg.pos_zone is None
    assert cfg.define_pos_zone is False
