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


def build_payload(snapshot: tuple[list, list]) -> dict:
    occ, visits = snapshot
    now = occ[-1][0] if occ else 0.0
    return {
        "now": now,
        "occupancy": occupancy_series(occ),
        "wait_serve": wait_serve_series(visits),
        "throughput": throughput_series(visits),
    }
