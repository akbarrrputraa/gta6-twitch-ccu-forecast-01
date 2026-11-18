import json, math, glob, statistics
from datetime import datetime, timedelta

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

def parse(t): return datetime.fromisoformat(t.replace('Z','+00:00'))

def day_curve(comp):
    """release-day hourly viewers dict hour->viewers (hour 1..24), from comp hourly points."""
    ds = parse(comp['day_start'])
    out = {}
    for p in comp['hourly']:
        t = parse(p['from'])
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and p['viewers'] and p['viewers'] > 0:
            out[h] = p['viewers']
    return out

def pinball(y, q, tau):
    d = math.log(max(y,1)) - math.log(max(q,1e-9))
    return 2*max(tau*d, (tau-1)*d)

def score_curve(actual, forecast):
    # forecast: hour -> [q05..q95]; actual: hour->viewers
    tot, n = 0.0, 0
    for h, y in actual.items():
        if h < 2 or h not in forecast: continue
        for tau, q in zip(LEVELS, forecast[h]):
            tot += pinball(y, q, tau); n += 1
    return tot/n if n else float('nan'), n

# load comparables from p00 (38 launches, excludes 4 practice launches)
c = json.load(open('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in'))
comps = []
for comp in c['comparables']:
    curve = day_curve(comp)
    if 1 in curve and len(curve) >= 20:
        comps.append((comp['name'], comp, curve))
print(f"{len(comps)} comparables with full-ish release day")
for name, comp, curve in comps:
    print(f"  {name:55s} h1={curve[1]:9.0f} peak={max(curve.values()):9.0f} peak_h={max(curve,key=curve.get)} tier={str(comp.get('tier'))[:40] if comp.get('tier') else comp.get('unlock','')[:40]}")

# log-ratio stats across all
import collections
ratios = collections.defaultdict(list)
for name, comp, curve in comps:
    for h in range(2,25):
        if h in curve:
            ratios[h].append(math.log(curve[h]/curve[1]))
print("\nhour: n  median_logratio  mean  std  | median ratio")
for h in range(2,25):
    r = ratios[h]
    print(f"h{h:2d}: n={len(r):2d} med={statistics.median(r):+.3f} mean={statistics.mean(r):+.3f} sd={statistics.pstdev(r):.3f} | ratio_med={math.exp(statistics.median(r)):.2f}")
# correlation of log(p1) with ratios at some hours
import math as m
for h in [5,8,12,18,24]:
    xs=[]; ys=[]
    for name, comp, curve in comps:
        if h in curve:
            xs.append(m.log(curve[1])); ys.append(m.log(curve[h]/curve[1]))
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    print(f"corr log(p1) vs r_h{h}: {cov/m.sqrt(vx*vy):+.2f} (n={len(xs)})")
