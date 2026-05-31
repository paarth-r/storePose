import numpy as np

from storepose.config import AppConfig
from storepose.pipeline import FrameResult, PosePipeline
from storepose.pose import NUM_KEYPOINTS, PoseEstimator


class FakeDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def detect(self, frame):
        return self._boxes


class FakePoseModel:
    """Stand-in for rtmlib RTMPose: returns one skeleton per box."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame, bboxes):
        self.calls += 1
        n = len(bboxes)
        return (
            np.zeros((n, NUM_KEYPOINTS, 2), np.float32),
            np.ones((n, NUM_KEYPOINTS), np.float32),
        )


def _pose_with(fake):
    cfg = AppConfig()
    return PoseEstimator(spec=None, config=cfg, model=fake)


def test_pose_short_circuits_on_empty_boxes():
    fake = FakePoseModel()
    pose = _pose_with(fake)
    kpts, scores = pose.estimate(np.zeros((4, 4, 3)), np.empty((0, 4)))
    assert kpts.shape == (0, NUM_KEYPOINTS, 2)
    assert scores.shape == (0, NUM_KEYPOINTS)
    assert fake.calls == 0  # model never invoked when no people


def test_pose_runs_model_when_boxes_present():
    fake = FakePoseModel()
    pose = _pose_with(fake)
    boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20]], np.float32)
    kpts, scores = pose.estimate(np.zeros((30, 30, 3)), boxes)
    assert kpts.shape == (2, NUM_KEYPOINTS, 2)
    assert fake.calls == 1


def test_pipeline_composes_and_counts():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    pipeline = PosePipeline(
        AppConfig(),
        detector=FakeDetector(boxes),
        pose=_pose_with(FakePoseModel()),
    )
    result = pipeline.process(np.zeros((40, 40, 3), np.uint8))
    assert isinstance(result, FrameResult)
    assert result.count == 2
    assert result.keypoints.shape == (2, NUM_KEYPOINTS, 2)


def test_pipeline_handles_no_people():
    pipeline = PosePipeline(
        AppConfig(),
        detector=FakeDetector(np.empty((0, 4), np.float32)),
        pose=_pose_with(FakePoseModel()),
    )
    result = pipeline.process(np.zeros((40, 40, 3), np.uint8))
    assert result.count == 0
