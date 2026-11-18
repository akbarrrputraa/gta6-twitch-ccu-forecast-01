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
def pre_hours(comp):
    ds=parse(comp['day_start']); out={}
    for p in comp['hourly']:
        h=int((parse(p['from'])-ds).total_seconds()//3600)
        if -48<=h<0 and p['viewers'] and p['viewers']>0: out[h]=p['viewers']
    return out

c=json.load(open('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in'))
CLEAN_NAMES = {'ELDEN RING','Ghost of Yōtei','Monster Hunter Wilds','Call of Duty: Modern Warfare 4',
 'The Legend of Zelda: Tears of the Kingdom','Borderlands 4','FINAL FANTASY VII REBIRTH',
 'The Last of Us Part II','Ghost of Tsushima','Stellar Blade','Cyberpunk 2077','Elden Ring Nightreign',
 'Call of Duty: Black Ops 7'}
data=[]
for comp in c['comparables']:
    curve=day_curve(comp)
    if 1 in curve and len(curve)>=20:
        data.append((comp['name'],curve,pre_hours(comp)))
ROLL=[d for d in data if max(d[1].values())/d[1][1] <= 4.0]
CLEAN=[d for d in data if d[0] in CLEAN_NAMES]

def ols(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    vx=sum((x-mx)**2 for x in xs)
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/vx if vx>0 else 0.0
    a=my-b*mx
    resid=[y-(a+b*x) for x,y in zip(xs,ys)]
    sd=statistics.pstdev(resid) if n>2 else 0.3
    return a,b,sd

def reg_fc(train, p1, pre, use_h0=False, shrink=1.0, sd_floor=0.15, sd_cap=0.9):
    """per-hour OLS of log-ratio on centered log(p1) (+ centered log(h0/h1) if use_h0)."""
    fc={}
    lp1=math.log(p1)
    lh0=math.log(pre[-1]/p1) if (use_h0 and -1 in pre) else None
    for h in range(2,25):
        rows=[(math.log(cu[1]), math.log(pre2[-1]/cu[1]) if -1 in pre2 else None, math.log(cu[h]/cu[1]))
              for _,cu,pre2 in train if h in cu]
        xs=[r[0] for r in rows]; ys=[r[2] for r in rows]
        mx=sum(xs)/len(xs)
        a,b,sd=ols([x-mx for x in xs], ys)
        adj=b*shrink*(lp1-mx)
        if use_h0 and lh0 is not None:
            rows2=[r for r in rows if r[1] is not None]
            if len(rows2)>=8:
                # residual regression on lh0
                res=[y-(a+b*(x-mx)) for x,_,y in rows2]
                mh=sum(r[1] for r in rows2)/len(rows2)
                vx=sum((r[1]-mh)**2 for r in rows2)
                b2=sum((r[1]-mh)*e for r,e in zip(rows2,res))/vx if vx>0 else 0
                adj+=0.5*b2*(lh0-mh)  # half weight: weak signal, small n
                sd=statistics.pstdev([e-b2*(r[1]-mh) for r,e in zip(rows2,res)]) if len(rows2)>2 else sd
        adj=max(-1.2,min(1.2,adj))
        s=min(sd_cap,max(sd_floor,sd))
        fc[h]=[p1*math.exp(a+adj+z*s) for z in Z]
    return fc

def empq_fc(train, p1):
    fc={}
    for h in range(2,25):
        rs=sorted(math.log(cu[h]/cu[1]) for _,cu,_ in train if h in cu)
        qs=[]
        for tau in LEVELS:
            idx=tau*(len(rs)-1); lo=int(idx); frac=idx-lo
            v=rs[lo]*(1-frac)+rs[min(lo+1,len(rs)-1)]*frac
            qs.append(p1*math.exp(v))
        fc[h]=qs
    return fc

def loo(subset,label,fn):
    vals=[]
    for i,(name,curve,pre) in enumerate(subset):
        train=[x for j,x in enumerate(subset) if j!=i]
        vals.append(score(curve, fn(train,curve[1],pre)))
    print(f"{label}: mean {statistics.mean(vals):.4f} median {statistics.median(vals):.4f} worst {max(vals):.4f}")
    return statistics.mean(vals)

loo(ROLL,'ROLL reg logp1', lambda tr,p1,pre: reg_fc(tr,p1,pre))
loo(ROLL,'ROLL reg logp1+h0', lambda tr,p1,pre: reg_fc(tr,p1,pre,use_h0=True))
loo(ROLL,'ROLL reg logp1 shrink0.7', lambda tr,p1,pre: reg_fc(tr,p1,pre,shrink=0.7))
loo(CLEAN,'CLEAN reg logp1', lambda tr,p1,pre: reg_fc(tr,p1,pre))
loo(CLEAN,'CLEAN reg logp1+h0', lambda tr,p1,pre: reg_fc(tr,p1,pre,use_h0=True))
loo(ROLL,'ROLL empq (ref)', lambda tr,p1,pre: empq_fc(tr,p1))

# sanity: predictions at various p1 using full ROLL fit
print("\nMedian ratio by hour at different hour-1 prints (ROLL reg logp1):")
for p1 in [50000, 200000, 1000000, 2000000]:
    fc=reg_fc(ROLL,p1,{-1: int(p1*1.1)})
    print(f" p1={p1:>8,}: " + " ".join(f"h{h}:{fc[h][2]/p1:.2f}" for h in [2,5,8,12,16,19,20,24]))
print("\nMedian ratio by hour (ROLL empq, scale-free):")
fc=empq_fc(ROLL,100000)
print("  " + " ".join(f"h{h}:{fc[h][2]/100000:.2f}" for h in [2,5,8,12,16,19,20,24]))
