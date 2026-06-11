"""Metrics for the busy classifier: treat Low/Medium/High as an ordinal scale.

Plain accuracy is misleading for an ordinal 3-class problem (confusing Low with
High is far worse than confusing Low with Medium), so we also report:

- within-1 accuracy: fraction within one level of truth (the "no gross errors"
  metric — for 3 classes this is everything except Low<->High confusions);
- mean absolute error on the 0/1/2 scale (how far off, on average);
- per-class precision/recall;
- the full confusion matrix.

Only windows present in *both* prediction and ground truth are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..busy.types import BusyLevel

_LEVELS = (BusyLevel.LOW, BusyLevel.MEDIUM, BusyLevel.HIGH)


@dataclass
class EvalReport:
    n: int
    accuracy: float
    within_one: float
    mae: float
    confusion: list[list[int]]  # rows = true, cols = pred, order LOW/MED/HIGH
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        names = [lv.label for lv in _LEVELS]
        lines = [
            f"windows scored : {self.n}",
            f"accuracy       : {self.accuracy:.3f}",
            f"within-1 acc   : {self.within_one:.3f}",
            f"ordinal MAE    : {self.mae:.3f}",
            "",
            "confusion (rows=true, cols=pred):",
            "            " + "".join(f"{n:>8}" for n in names),
        ]
        for i, name in enumerate(names):
            row = "".join(f"{c:>8}" for c in self.confusion[i])
            lines.append(f"{name:>10}  {row}")
        lines.append("")
        lines.append("per-class      precision   recall")
        for name in names:
            lines.append(
                f"{name:>10}  {self.precision[name]:>10.3f} {self.recall[name]:>8.3f}"
            )
        return "\n".join(lines)


def evaluate(
    truth: dict[int, BusyLevel], pred: dict[int, BusyLevel]
) -> EvalReport:
    """Compare predicted vs. ground-truth window labels."""
    keys = sorted(set(truth) & set(pred))
    n = len(keys)
    confusion = [[0, 0, 0] for _ in _LEVELS]
    correct = 0
    within = 0
    abs_err = 0
    for k in keys:
        t, p = truth[k], pred[k]
        confusion[int(t)][int(p)] += 1
        d = abs(int(t) - int(p))
        abs_err += d
        if d == 0:
            correct += 1
        if d <= 1:
            within += 1

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    for i, lv in enumerate(_LEVELS):
        tp = confusion[i][i]
        col = sum(confusion[r][i] for r in range(len(_LEVELS)))  # predicted as i
        row = sum(confusion[i])  # truly i
        precision[lv.label] = tp / col if col else 0.0
        recall[lv.label] = tp / row if row else 0.0

    return EvalReport(
        n=n,
        accuracy=correct / n if n else 0.0,
        within_one=within / n if n else 0.0,
        mae=abs_err / n if n else 0.0,
        confusion=confusion,
        precision=precision,
        recall=recall,
    )
