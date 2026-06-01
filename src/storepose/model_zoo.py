"""Model URL/size resolution for each mode, sourced from rtmlib.

We read rtmlib's own ``Body.MODE`` table so model URLs stay in sync with the
installed rtmlib version rather than being hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rtmlib import Body


@dataclass(frozen=True)
class ModelSpec:
    """ONNX model location and input size for a pipeline stage."""

    url: str
    input_size: tuple[int, int]


@dataclass(frozen=True)
class ModeSpec:
    """Detector + pose model specs for a given rtmlib mode."""

    detector: ModelSpec
    pose: ModelSpec


def resolve(mode: str) -> ModeSpec:
    """Return the detector + pose :class:`ModelSpec` pair for ``mode``."""
    try:
        entry = Body.MODE[mode]
    except KeyError as exc:  # pragma: no cover - guarded earlier by AppConfig
        raise ValueError(f"unknown mode: {mode!r}") from exc
    return ModeSpec(
        detector=ModelSpec(entry["det"], tuple(entry["det_input_size"])),
        pose=ModelSpec(entry["pose"], tuple(entry["pose_input_size"])),
    )
