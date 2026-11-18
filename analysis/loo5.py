import json, math, statistics
from datetime import datetime, timedelta
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
def pre_hours(comp):
    ds=parse(comp['day_start']); out={}
    for p in comp['hourly']:
        h=int((parse(p['from'])-ds).total_seconds()//3600)
        if -72<=h<0 and p['viewers'] and p['viewers']>0: out[h]=p['viewers']
    return out
def ols(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    vx=sum((x-mx)**2 for x in xs)
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/vx if vx>0 else 0.0
    a=my-b*mx
    resid=[y-(a+b*x) for x,y in zip(xs,ys)]
    return a,b,(statistics.pstdev(resid) if n>2 else 0.3)
def reg_fc(train, p1, pre, sd_floor=0.15, sd_cap=0.9):
    fc={}; lp1=math.log(p1)
    lh0=math.log(pre[-1]/p1) if -1 in pre else None
    for h in range(2,25):
        rows=[(math.log(cu[1]), math.log(pr[-1]/cu[1]) if -1 in pr else None, math.log(cu[h]/cu[1]))
              for _,cu,pr,*_ in train if h in cu]
        xs=[r[0] for r in rows]; ys=[r[2] for r in rows]
        mx=sum(xs)/len(xs)
        a,b,sd=ols([x-mx for x in xs], ys)
        adj=max(-1.2,min(1.2,b*(lp1-mx)))
        if lh0 is not None:
            rows2=[r for r in rows if r[1] is not None]
            if len(rows2)>=8:
                res=[y-(a+b*(x-mx)) for x,_,y in rows2]
                mh=sum(r[1] for r in rows2)/len(rows2)
                vx=sum((r[1]-mh)**2 for r in rows2)
                b2=sum((r[1]-mh)*e for r,e in zip(rows2,res))/vx if vx>0 else 0
                adj+=max(-0.6,min(0.6,0.5*b2*(lh0-mh)))
                sd=statistics.pstdev([e-b2*(r[1]-mh) for r,e in zip(rows2,res)]) if len(rows2)>2 else sd
        s=min(sd_cap,max(sd_floor,sd))
        fc[h]=[p1*math.exp(a+adj+z*s) for z in Z]
    return fc

c=json.load(open('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in'))
data=[]
for comp in c['comparables']:
    curve=day_curve(comp); pre=pre_hours(comp)
    if 1 in curve and len(curve)>=20:
        peak=max(curve.values())
        premax=max([v for k,v in pre.items() if k<=-25], default=0)
        data.append((comp['name'],curve,pre,peak/curve[1],premax/peak if peak else 9))
print("name: peak/h1, pre3d_max/peak")
for n,cu,pr,ratio,cont in data:
    print(f"  {n:45s} peak/h1={ratio:7.2f} pre/peak={cont:5.2f}")
AUTO=[d for d in data if d[3]<=4.0 and d[4]<0.5]
print("\nAUTO-CLEAN n=",len(AUTO),[d[0] for d in AUTO])
ROLL=[d for d in data if d[3]<=4.0]
def loo(subset,label):
    vals=[]
    for i,(name,curve,pre,*_ ) in enumerate(subset):
        train=[x for j,x in enumerate(subset) if j!=i]
        vals.append(score(curve, reg_fc(train,curve[1],pre)))
    print(f"{label}: mean {statistics.mean(vals):.4f} median {statistics.median(vals):.4f} worst {max(vals):.4f}")
loo(AUTO,'AUTO-CLEAN reg')
loo(ROLL,'ROLL reg (ref)')
# AUTO-CLEAN fit, evaluated on ALL rolling targets (train excludes target if present)
vals=[]
for name,curve,pre,*_ in ROLL:
    train=[x for x in AUTO if x[0]!=name]
    vals.append(score(curve, reg_fc(train,curve[1],pre)))
print(f"AUTO-CLEAN fit -> ROLL targets: mean {statistics.mean(vals):.4f} median {statistics.median(vals):.4f}")
# what if pre/peak threshold 0.35 / 0.7?
for thr in [0.35,0.7]:
    S=[d for d in data if d[3]<=4.0 and d[4]<thr]
    print(f"thr {thr}: n={len(S)}")
    loo(S,f'AUTO thr={thr}')
