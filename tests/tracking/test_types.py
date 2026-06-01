import numpy as np

from storepose.tracking.types import TrackedPerson


def test_tracked_person_fields():
    p = TrackedPerson(
        id=3,
        box=np.array([0, 0, 10, 20], float),
        keypoints=np.zeros((17, 2), float),
        scores=np.ones(17, float),
        coasting=False,
        color=(0, 255, 0),
    )
    assert p.id == 3
    assert p.coasting is False
    assert p.color == (0, 255, 0)
    assert p.keypoints.shape == (17, 2)


def test_tracked_person_coasting_has_no_pose():
    p = TrackedPerson(
        id=1, box=np.array([0, 0, 5, 5], float),
        keypoints=None, scores=None, coasting=True, color=(255, 0, 0),
    )
    assert p.keypoints is None
    assert p.scores is None
    assert p.coasting is True
