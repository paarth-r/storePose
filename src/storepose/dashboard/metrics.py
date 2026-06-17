"""Pure series builders turning DashboardState snapshots into the JSON payload."""
from __future__ import annotations


def moving_average(times: list[float], values: list[float], window: float) -> list[float]:
    """Trailing-``window`` mean of ``values`` aligned to ``times`` (same length)."""
    out: list[float] = []
    j = 0
    for i in range(len(times)):
        while times[j] < times[i] - window:
            j += 1
        seg = values[j:i + 1]
        out.append(sum(seg) / len(seg) if seg else 0.0)
    return out


def occupancy_series(occ: list, ma_window: float = 30.0) -> dict:
    t = [s[0] for s in occ]
    waiting = [s[1] for s in occ]
    serving = [s[2] for s in occ]
    return {
        "t": t, "waiting": waiting, "serving": serving,
        "waiting_ma": moving_average(t, [float(w) for w in waiting], ma_window),
        "serving_ma": moving_average(t, [float(s) for s in serving], ma_window),
    }


def wait_serve_series(visits: list, window: float = 120.0) -> dict:
    served = [v for v in visits if v.outcome == "served"]
    t = [v.t for v in served]
    return {
        "t": t,
        "wait_ma": moving_average(t, [v.wait_seconds for v in served], window),
        "serve_ma": moving_average(t, [v.serving_seconds for v in served], window),
    }


def throughput_series(visits: list, bucket: float = 60.0) -> dict:
    served = sorted(v.t for v in visits if v.outcome == "served")
    if not served:
        return {"t": [], "served_per_min": []}
    start, end = served[0], served[-1]
    counts: dict[int, int] = {}
    for tt in served:
        b = int((tt - start) // bucket)
        counts[b] = counts.get(b, 0) + 1
    n = int((end - start) // bucket) + 1
    return {
        "t": [start + b * bucket for b in range(n)],
        "served_per_min": [counts.get(b, 0) * (60.0 / bucket) for b in range(n)],
    }


def summary_stats(occ: list, visits: list) -> dict:
    """Headline numbers: current occupancy + average line/POS/total times."""
    in_line = occ[-1][1] if occ else 0
    at_pos = occ[-1][2] if occ else 0
    served = [v for v in visits if v.outcome == "served"]
    n = len(served)
    if n:
        avg_line = sum(v.wait_seconds for v in served) / n
        avg_pos = sum(v.serving_seconds for v in served) / n
        avg_total = sum(v.wait_seconds + v.serving_seconds for v in served) / n
    else:
        avg_line = avg_pos = avg_total = 0.0
    return {
        "in_line": in_line, "at_pos": at_pos,
        "avg_line_s": avg_line, "avg_pos_s": avg_pos, "avg_total_s": avg_total,
        "served_count": n,
    }


_BUSY_IDX = {"Low": 0, "Medium": 1, "High": 2}


def busy_series(current, history: list) -> dict:
    """Current busy label/value plus its history as a 0/1/2 step series."""
    if current:
        cur = {"level": current[1], "value": current[2]}
    else:
        cur = {"level": None, "value": 0.0}
    return {
        "current": cur,
        "t": [b[0] for b in history],
        "level_idx": [_BUSY_IDX.get(b[1], 0) for b in history],
        "value": [b[2] for b in history],
    }


def checkout_stats(visits: list, num_mashgins: int = 1) -> dict:
    """Avg serve time per person at the Mashgin vs non-Mashgin checkout + delta.

    ``num_mashgins`` is how many Mashgin self-checkout kiosks run in parallel:
    they process people simultaneously, so the *effective* per-customer time of
    the Mashgin system is the measured per-person serve time divided by the kiosk
    count. ``mashgin_avg_eff`` is that effective figure and the ``delta`` (seconds
    saved per person vs the staffed lane) is computed from it. ``mashgin_avg``
    stays the raw per-person measurement.
    """
    n = max(1, num_mashgins)
    mash = [v for v in visits if v.outcome == "served"]
    other = [v for v in visits if v.outcome == "served_other"]

    def avg(vs):
        return sum(v.serving_seconds for v in vs) / len(vs) if vs else 0.0

    m_avg, o_avg = avg(mash), avg(other)
    m_eff = m_avg / n
    return {
        "mashgin_avg": m_avg, "mashgin_avg_eff": m_eff, "num_mashgins": n,
        "mashgin_n": len(mash),
        "other_avg": o_avg, "other_n": len(other),
        "delta": o_avg - m_eff,  # seconds the Mashgin system saves per person
    }


def checkout_series(visits: list, window: float = 120.0) -> dict:
    """Trailing-window mean serve time per checkout, as two time series."""
    mash = [v for v in visits if v.outcome == "served"]
    other = [v for v in visits if v.outcome == "served_other"]
    mt, ot = [v.t for v in mash], [v.t for v in other]
    return {
        "t_mashgin": mt,
        "mashgin_ma": moving_average(mt, [v.serving_seconds for v in mash], window),
        "t_other": ot,
        "other_ma": moving_average(ot, [v.serving_seconds for v in other], window),
    }


def build_payload(snapshot: tuple[list, list], busy: tuple = (None, []),
                  debug: tuple = (None, []), num_mashgins: int = 1) -> dict:
    occ, visits = snapshot
    busy_current, busy_history = busy
    debug_frame, debug_rows = debug
    now = occ[-1][0] if occ else 0.0
    checkouts = checkout_stats(visits, num_mashgins)
    checkouts["series"] = checkout_series(visits)
    return {
        "now": now,
        "summary": summary_stats(occ, visits),
        "busy": busy_series(busy_current, busy_history),
        "checkouts": checkouts,
        "occupancy": occupancy_series(occ),
        "wait_serve": wait_serve_series(visits),
        "throughput": throughput_series(visits),
        "debug": {"frame": debug_frame, "rows": debug_rows},
    }
