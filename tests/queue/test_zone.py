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


def test_multi_contour_contains_union():
    z = Zone.from_polygons([
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])
    assert z.contains((50, 50)) is True
    assert z.contains((250, 250)) is True
    assert z.contains((150, 150)) is False


def test_multi_contour_coverage_union():
    z = Zone.from_polygons([
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])
    assert z.coverage([10, 10, 90, 90]) == 1.0
    assert z.coverage([210, 210, 290, 290]) == 1.0
    assert z.coverage([120, 120, 180, 180]) == 0.0


def test_multi_contour_json_round_trip(tmp_path):
    z = Zone.from_polygons([
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(200, 200), (300, 200), (300, 300), (200, 300)],
    ])
    path = str(tmp_path / "z.json")
    z.save(path)
    loaded = Zone.load(path)
    assert len(loaded.polygons) == 2
    assert loaded.contains((250, 250)) is True


def test_legacy_points_json_still_loads(tmp_path):
    import json
    path = str(tmp_path / "legacy.json")
    with open(path, "w") as f:
        json.dump({"points": [[0, 0], [100, 0], [100, 100], [0, 100]]}, f)
    z = Zone.load(path)
    assert len(z.polygons) == 1
    assert z.contains((50, 50)) is True


def test_single_polygon_back_compat():
    z = Zone([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert len(z.polygons) == 1
    assert z.points == [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert z.contains((50, 50)) is True
