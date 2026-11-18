"""GTA VI launch-day Twitch forecast (make.fun gta6-twitch-ccu-forecast-01).

Reads one case per run (schema "makefun/twitch-ccu-case/v1") and writes, for
each of hours 2..24 of the release day, five quantiles (5/25/50/75/95%) of
that hour's Twitch concurrent-viewer count, scored by pinball loss on the
log scale.

Method (fitted at run time from the case's own `comparables`, so the same
program adapts to the practice cases and to the final):

1.  Pick the launches that look like GTA VI: a console-led game that
    unlocked at midnight local time, rolling around the world.  A launch
    qualifies when its release day has an hour-1 print and at least 20 of
    its 24 hourly prints, its day peak is at most 4x its hour-1 print
    (the first unlock wave was already in by hour 1 -- a single global
    unlock hours later, e.g. Silksong or Black Myth: Wukong, has a peak
    hundreds of times its hour-1 print and a totally different shape),
    and it had no early-access surge in the 3 days before release
    (max hourly viewers 72..25 hours before day start below half the
    release-day peak -- this drops Hogwarts Legacy, Starfield, Diablo IV,
    Assassin's Creed Shadows, DOOM: The Dark Ages, Death Stranding 2 and
    the like, whose release-day curves are already-decayed early-access
    curves).  If fewer than 8 launches qualify, the peak-ratio rule is
    relaxed; if still fewer than 8, fall back to the house persistence
    baseline.

2.  For each hour h = 2..24, regress the log ratio log(v_h / v_1) across
    the qualifying launches on log(v_1) (centred).  This is the key
    regularity in the data: launches that already drew a huge hour-1
    crowd peak at or near hour 1 and stay flat, while smaller hour-1
    launches grow 2-3x into the US evening (leave-one-out correlation
    between log hour-1 print and the evening log ratio is about -0.85).
    The regression's intercept at the target's log hour-1 print is the
    median log ratio for that hour.

3.  A second, weaker signal: the hour just before midnight (23:00-00:00
    UTC, which already holds the New Zealand / early-Europe wave) relative
    to hour 1.  Where at least 8 qualifying launches have it, the
    residuals of step 2 are regressed on it too and half of that
    adjustment is added (leave-one-out correlation with the evening ratio
    is about +0.5).  The residual spread of the regression, floored at
    0.15 and capped at 0.90 log units, is the forecast's log spread.

4.  Quantiles are the median times exp(z_tau * sigma), z the standard
    normal quantiles, rounded half-up to 2 decimals.

Leave-one-out pinball loss on the 16 qualifying launches (each held out
in turn, model refit on the rest): 0.180, against 0.276 for the house
persistence baseline on the same launches and 0.225 for the
scale-free empirical-quantile curve.  The program is deterministic
(no randomness, no wall-clock or dict-order dependence), standard
library only, and runs in well under a second per case.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# A launch qualifies as GTA-like when its release-day peak is at most this
# many times its hour-1 print, and its pre-release surge (max hourly
# viewers 72..25 h before day start) is below this fraction of its peak.
MAX_PEAK_OVER_HOUR1 = 4.0
MAX_PRERELEASE_OVER_PEAK = 0.5
MIN_TRAIN = 8
MIN_DAY_HOURS = 20

SIGMA_FLOOR = 0.15
SIGMA_CAP = 0.90
BASELINE_SIGMA = 0.15


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_offset(from_text, day_start):
    """Whole hours from day_start to `from_text` (negative before it)."""
    delta = parse_time(from_text) - day_start
    return int(delta.total_seconds() // 3600)


def release_day_curve(hourly, day_start):
    """{hour 1..24: viewers} for the UTC release day, from hourly points."""
    curve = {}
    for point in hourly or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = hour_offset(point["from"], day_start) + 1
        if 1 <= hour <= 24:
            curve[hour] = viewers
    return curve


def prerelease_max(hourly, day_start):
    """Max hourly viewers 72..25 h before day start (excludes the last
    24 h, which legitimately hold the first rolling-unlock wave)."""
    best = 0
    for point in hourly or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        off = hour_offset(point["from"], day_start)
        if -72 <= off <= -25 and viewers > best:
            best = viewers
    return best


def hour_before_midnight(hourly, day_start):
    """Viewers in the hour 23:00-00:00 UTC before day start, or None."""
    for point in hourly or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        if hour_offset(point["from"], day_start) == -1:
            return viewers
    return None


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx
             if vx > 0 else 0.0)
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd = statistics.pstdev(resid) if n > 2 else 0.3
    return intercept, slope, sd, resid


def round2(x):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def clamp(value, low, high):
    return max(low, min(high, value))


def qualifying_launches(case):
    """(strict, relaxed) training sets from the case's comparables.

    Each entry is (log hour-1 print, log hour-before-midnight / hour-1 or
    None, {hour: log viewers / hour-1}).  Strict applies both the
    peak-ratio and the early-access rules; relaxed drops the peak rule.
    """
    strict = []
    relaxed = []
    for comp in case.get("comparables") or []:
        hourly = comp.get("hourly") or []
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        curve = release_day_curve(hourly, day_start)
        if 1 not in curve or len(curve) < MIN_DAY_HOURS:
            continue
        peak = max(curve.values())
        hour1 = curve[1]
        before = hour_before_midnight(hourly, day_start)
        entry = (math.log(hour1),
                 math.log(before / hour1) if before else None,
                 {h: math.log(v / hour1) for h, v in curve.items()})
        relaxed.append(entry)
        if (peak / hour1 <= MAX_PEAK_OVER_HOUR1
                and prerelease_max(hourly, day_start)
                < MAX_PRERELEASE_OVER_PEAK * peak):
            strict.append(entry)
    return strict, relaxed


def regression_forecast(train, p1, target_before_ratio):
    """Per-hour median log ratio + spread from the training launches."""
    log_p1 = math.log(p1)
    log_before = (math.log(target_before_ratio)
                  if target_before_ratio else None)
    hours = {}
    for hour in range(2, 25):
        rows = [(lp, lb, ratios[hour]) for lp, lb, ratios in train
                if hour in ratios]
        if len(rows) < 3:
            continue
        xs = [r[0] for r in rows]
        ys = [r[2] for r in rows]
        mean_x = sum(xs) / len(xs)
        intercept, slope, sd, resid = ols([x - mean_x for x in xs], ys)
        adjust = clamp(slope * (log_p1 - mean_x), -1.2, 1.2)
        if log_before is not None:
            paired = [(r[1], e) for r, e in zip(rows, resid)
                      if r[1] is not None]
            if len(paired) >= MIN_TRAIN:
                mean_b = sum(b for b, _ in paired) / len(paired)
                var_b = sum((b - mean_b) ** 2 for b, _ in paired)
                slope2 = (sum((b - mean_b) * e for (b, e) in paired) / var_b
                          if var_b > 0 else 0.0)
                adjust += clamp(0.5 * slope2 * (log_before - mean_b),
                                -0.6, 0.6)
                sd = statistics.pstdev(
                    [e - slope2 * (b - mean_b) for (b, e) in paired])
        sigma = clamp(sd, SIGMA_FLOOR, SIGMA_CAP)
        centre = intercept + adjust
        hours[hour] = [p1 * math.exp(centre + z * sigma) for z in Z]
    return hours


def baseline_forecast(p1):
    """House persistence baseline: every hour stays at hour 1's print,
    spread widening like a random walk in log space."""
    return {h: [p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(h - 1))
                for z in Z] for h in range(2, 25)}


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, "
                         f"not {LEVELS}")
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if known is not None and known.get("viewers"):
        p1 = max(1, known["viewers"])
    else:
        hourly = case.get("history_hourly") or []
        p1 = max(1, hourly[-1]["viewers"]) if hourly else 1
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        day_start = None
    before = None
    if day_start is not None:
        before = hour_before_midnight(case.get("history_hourly") or [],
                                      day_start)
    target_ratio = (before / p1) if before else None

    strict, relaxed = qualifying_launches(case)
    train = strict if len(strict) >= MIN_TRAIN else relaxed
    if len(train) >= MIN_TRAIN:
        quantiles = regression_forecast(train, p1, target_ratio)
        missing = [h for h in range(2, 25) if h not in quantiles]
        if missing:
            base = baseline_forecast(p1)
            for h in missing:
                quantiles[h] = base[h]
    else:
        quantiles = baseline_forecast(p1)

    rows = []
    for row in case["forecast"]:
        hour = row["hour"]
        qs = []
        prev = 0.0
        for q in quantiles[hour]:
            value = round2(clamp(q, 1, 1_000_000_000))
            value = max(value, prev)
            qs.append(value)
            prev = value
        rows.append({"hour": hour, "q": qs})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
