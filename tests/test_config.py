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
    ],
)
def test_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        AppConfig(**kwargs)


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
