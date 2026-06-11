from storepose.busy.types import BusyLevel
from storepose.eval.metrics import evaluate

LOW, MED, HIGH = BusyLevel.LOW, BusyLevel.MEDIUM, BusyLevel.HIGH


def test_perfect_match():
    truth = {0: LOW, 1: MED, 2: HIGH}
    rep = evaluate(truth, dict(truth))
    assert rep.n == 3
    assert rep.accuracy == 1.0
    assert rep.within_one == 1.0
    assert rep.mae == 0.0


def test_only_shared_windows_scored():
    truth = {0: LOW, 1: MED, 2: HIGH}
    pred = {1: MED, 2: HIGH}  # window 0 missing from pred
    rep = evaluate(truth, pred)
    assert rep.n == 2
    assert rep.accuracy == 1.0


def test_ordinal_mae_and_within_one():
    # one Low<->High gross error, one adjacent error, one correct
    truth = {0: LOW, 1: MED, 2: HIGH}
    pred = {0: HIGH, 1: HIGH, 2: HIGH}
    rep = evaluate(truth, pred)
    assert rep.accuracy == 1 / 3            # only window 2 correct
    assert rep.within_one == 2 / 3          # window 1 within 1, window 0 not
    assert rep.mae == (2 + 1 + 0) / 3


def test_confusion_and_per_class():
    truth = {0: LOW, 1: LOW, 2: HIGH}
    pred = {0: LOW, 1: MED, 2: HIGH}
    rep = evaluate(truth, pred)
    # true LOW row: one predicted LOW, one predicted MED
    assert rep.confusion[int(LOW)][int(LOW)] == 1
    assert rep.confusion[int(LOW)][int(MED)] == 1
    assert rep.recall["Low"] == 0.5         # 1 of 2 true-Low recovered
    assert rep.precision["High"] == 1.0
