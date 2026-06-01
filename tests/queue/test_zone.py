from storepose.queue.zone import Zone


def _square():
    return Zone([(0, 0), (100, 0), (100, 100), (0, 100)])


def test_contains_inside_and_outside():
    z = _square()
    assert z.contains((50, 50)) is True
    assert z.contains((150, 50)) is False
    assert z.contains((-5, 50)) is False


def test_contains_on_boundary():
    assert _square().contains((0, 50)) is True


def test_degenerate_zone_contains_nothing():
    assert Zone([(0, 0), (10, 10)]).contains((5, 5)) is False


def test_json_round_trip(tmp_path):
    z = _square()
    path = str(tmp_path / "zone.json")
    z.save(path)
    loaded = Zone.load(path)
    assert loaded.points == z.points
    assert loaded.contains((50, 50)) is True


def test_coverage_fully_inside_and_outside():
    z = _square()
    assert z.coverage([10, 10, 90, 90]) == 1.0
    assert z.coverage([200, 200, 300, 300]) == 0.0


def test_coverage_partial_is_about_half():
    z = _square()
    # box spans x 50..150 (right half outside the 0..100 square), y inside
    cov = z.coverage([50, 10, 150, 90])
    assert 0.4 < cov < 0.6
