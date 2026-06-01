import pytest

from storepose.fps import FpsMeter


def test_first_tick_is_zero():
    meter = FpsMeter(clock=iter([0.0]).__next__)
    assert meter.tick() == 0.0


def test_constant_interval_gives_expected_fps():
    # 0.0, 0.1, 0.2 -> two intervals of 0.1s -> 10 fps
    times = iter([0.0, 0.1, 0.2])
    meter = FpsMeter(clock=times.__next__)
    meter.tick()
    assert meter.tick() == pytest.approx(10.0)
    assert meter.tick() == pytest.approx(10.0)


def test_rolls_over_window():
    times = iter([0.0, 1.0, 1.1])  # window=2 keeps only last two stamps
    meter = FpsMeter(window=2, clock=times.__next__)
    meter.tick()
    meter.tick()
    # now stamps are [1.0]; next tick -> [1.0, 1.1] -> 10 fps
    assert meter.tick() == pytest.approx(10.0)


def test_rejects_tiny_window():
    with pytest.raises(ValueError):
        FpsMeter(window=1)
