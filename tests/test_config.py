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
