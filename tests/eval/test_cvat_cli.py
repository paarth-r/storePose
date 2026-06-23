from __future__ import annotations

import csv

import busy_report
from tests.eval.test_cvat_import import SAMPLE_XML


def _write(path, text):
    path.write_text(text)
    return str(path)


def test_import_cvat_writes_gt_csv(tmp_path, capsys):
    xml = _write(tmp_path / "export.xml", SAMPLE_XML)
    out = str(tmp_path / "gt.csv")
    rc = busy_report.main(["import-cvat", xml, "--fps", "10", "--step", "1", "-o", out])
    assert rc == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows[0].keys() >= {"t_s", "occupancy"}


def test_eval_occupancy_runs(tmp_path, capsys):
    # GT: occupancy 1 at t=1.0; waits.csv: one wait covering [1,2)
    gt = tmp_path / "gt.csv"
    gt.write_text("t_s,occupancy\n1.000,1\n")
    waits = tmp_path / "waits.csv"
    waits.write_text("id,entered_s,exited_s,wait_seconds\n0,0.5,2.0,1.5\n")
    rc = busy_report.main(["eval-occupancy", str(gt), str(waits), "--step", "1"])
    assert rc == 0
    assert "occupancy MAE" in capsys.readouterr().out
