"""Resolve and cache OSNet ReID ONNX weights (the ReID analogue of model_zoo).

URLs and checksums are pinned (see the plan's "obtain weights" step). Weights
download once into ~/.cache/storepose/reid/ and are checksum-verified on use.
"""
from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "storepose" / "reid"


@dataclass(frozen=True)
class ReidSpec:
    """A downloadable OSNet ONNX model."""
    url: str
    sha256: str
    filename: str


# Pinned 512-d MSMT17 embedding models. The exports fix the ONNX batch dim
# (OsnetAppearance chunk-and-pads to it). osnet-x1 is not yet pinned.
SPECS: dict[str, ReidSpec] = {
    "osnet-x1": ReidSpec(url="", sha256="", filename="osnet_x1_0.onnx"),
    "osnet-x025": ReidSpec(
        url="https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx",
        sha256="e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277",
        filename="osnet_x0_25_msmt17.onnx",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(backend: str) -> Path:
    """Return a local path to ``backend``'s ONNX weights, downloading if needed."""
    try:
        spec = SPECS[backend]
    except KeyError as exc:
        raise ValueError(f"no ReID weights for backend {backend!r}") from exc
    if not spec.url:
        raise RuntimeError(f"ReID weights URL for {backend!r} is not pinned yet")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CACHE_DIR / spec.filename
    if not dest.is_file():
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(spec.url, tmp)
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)  # don't leave a truncated .part behind
            raise
    actual = _sha256(dest)
    if spec.sha256 and actual != spec.sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"ReID weights {spec.filename} checksum mismatch "
            f"(expected {spec.sha256}, got {actual})"
        )
    return dest
