import json, math, statistics
from datetime import datetime

Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

def parse(t): return datetime.fromisoformat(t.replace('Z','+00:00'))
def pinball(y,q,tau):
    d = math.log(max(y,1)) - math.log(max(q,1e-9))
    return 2*max(tau*d,(tau-1)*d)
def score(actual, fc):
    tot,n=0.0,0
    for h,y in actual.items():
        if h<2 or h not in fc: continue
        for tau,q in zip(LEVELS,fc[h]): tot+=pinball(y,q,tau); n+=1
    return tot/n if n else float('nan')

def day_curve(comp):
    ds=parse(comp['day_start']); out={}
    for p in comp['hourly']:
        h=int((parse(p['from'])-ds).total_seconds()//3600)+1
        if 1<=h<=24 and p['viewers'] and p['viewers']>0: out[h]=p['viewers']
    return out

c=json.load(open('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in'))
allc=[]
for comp in c['comparables']:
    curve=day_curve(comp)
    if 1 in curve and len(curve)>=20: allc.append((comp['name'],curve))
# rolling-midnight-like: peak/h1 <= 4 (h1 already substantial => unlock wave mostly in before/at h1)
ROLL=[(n,cu) for n,cu in allc if max(cu.values())/cu[1] <= 4.0]
print(f"rolling subset: {len(ROLL)} of {len(allc)}")
print("  ", [n for n,_ in ROLL])

def median_curve(train):
    med={}; sd={}
    for h in range(2,25):
        rs=[math.log(cu[h]/cu[1]) for _,cu in train if h in cu]
        med[h]=statistics.median(rs)
        sd[h]=statistics.pstdev(rs) if len(rs)>1 else 0.3
    return med, sd

def smooth(d, w=1):
    # 3-point moving average on hours 2..24
    out={}
    for h in range(2,25):
        vals=[d[k] for k in (h-1,h,h+1) if k in d]
        out[h]=sum(vals)/len(vals)
    return out

def run_eval(subset, label, models):
    print(f"\n=== {label} (n={len(subset)}) ===")
    res={m:[] for m in models}
    for i,(name,curve) in enumerate(subset):
        train=[x for j,x in enumerate(subset) if j!=i]
        p1=curve[1]
        for m in models:
            fc={}
            if m=='baseline':
                for h in range(2,25):
                    fc[h]=[p1*math.exp(z*0.15*math.sqrt(h-1)) for z in Z]
            elif m.startswith('med'):
                med,sd=median_curve(train)
                sm = smooth(med) if 'sm' in m else med
                mult = float(m.split('_')[-1]) if m.split('_')[-1].replace('.','').isdigit() else 1.0
                for h in range(2,25):
                    s=max(sd[h]*mult,0.12)
                    fc[h]=[p1*math.exp(sm[h]+z*s) for z in Z]
            elif m.startswith('empq'):
                for h in range(2,25):
                    rs=sorted(math.log(cu[h]/cu[1]) for _,cu in train if h in cu)
                    qs=[]
                    for tau in LEVELS:
                        idx=tau*(len(rs)-1); lo=int(idx); frac=idx-lo
                        v=rs[lo]*(1-frac)+rs[min(lo+1,len(rs)-1)]*frac
                        qs.append(p1*math.exp(v))
                    fc[h]=qs
            res[m].append(score(curve,fc))
    for m in models:
        v=res[m]
        print(f"  {m:18s} mean pinball {statistics.mean(v):.4f}  median {statistics.median(v):.4f}  worst {max(v):.4f}")

models=['baseline','med','med_sm','med_sm_0.8','med_sm_1.2','empq']
run_eval(ROLL,'ROLLING subset LOO',models)
run_eval(allc,'ALL comparables LOO',models)
