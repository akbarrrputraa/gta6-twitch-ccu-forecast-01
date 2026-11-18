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
    """hourly viewers for hours before day start: dict offset(-1,-2,...)->viewers"""
    ds=parse(comp['day_start']); out={}
    for p in comp['hourly']:
        h=int((parse(p['from'])-ds).total_seconds()//3600)  # 0-based offset; -1 = 23:00 prev day
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
print("CLEAN n=",len(CLEAN),[n for n,_,_ in CLEAN])

# predictive signals within CLEAN: corr of log(h0/h1), log(p1) with evening ratios
def corr(xs,ys):
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    return cov/math.sqrt(vx*vy) if vx>0 and vy>0 else 0
for h in [12,18,20,24]:
    xs=[];ys=[];xs2=[];ys2=[]
    for n,cu,pre in CLEAN:
        if h in cu:
            ys.append(math.log(cu[h]/cu[1])); xs.append(math.log(cu[1]))
            if -1 in pre: xs2.append(math.log(pre[-1]/cu[1])); ys2.append(math.log(cu[h]/cu[1]))
    print(f"h{h}: corr log(p1)={corr(xs,ys):+.2f} (n={len(xs)}); corr log(h0/h1)={corr(xs2,ys2):+.2f} (n={len(xs2)})")
# also peak ratio vs h0/h1
xs=[];ys=[]
for n,cu,pre in CLEAN:
    if -1 in pre:
        xs.append(math.log(pre[-1]/cu[1])); ys.append(math.log(max(cu.values())/cu[1]))
print(f"corr log(h0/h1) vs log(peak/h1): {corr(xs,ys):+.2f} (n={len(xs)})")
for n,cu,pre in CLEAN:
    print(f"  {n:45s} h0/h1={pre.get(-1,0)/cu[1] if -1 in pre else float('nan'):6.2f} h1={cu[1]:8.0f} peak/h1={max(cu.values())/cu[1]:.2f}")

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
def smooth_fc(fc):
    out={}
    for h in range(2,25):
        out[h]=[sum(fc[k][j] for k in (h-1,h,h+1) if k in fc)/len([k for k in (h-1,h,h+1) if k in fc]) for j in range(5)]
        # keep monotone within row
        row=out[h]
        for j in range(1,5): row[j]=max(row[j],row[j-1])
        out[h]=row
    return out

def loo_eval(subset,label,fn):
    vals=[]
    for i,(name,curve,pre) in enumerate(subset):
        train=[x for j,x in enumerate(subset) if j!=i]
        fc=fn(train,curve[1])
        vals.append(score(curve,fc))
    print(f"{label}: mean {statistics.mean(vals):.4f} median {statistics.median(vals):.4f} worst {max(vals):.4f}")
    return statistics.mean(vals)

loo_eval(CLEAN,'CLEAN empq', empq_fc)
loo_eval(CLEAN,'CLEAN empq smoothed', lambda tr,p1: smooth_fc(empq_fc(tr,p1)))
loo_eval(ROLL,'ROLL empq', empq_fc)
# blend: quantiles averaged in log space between CLEAN and ROLL fits
def blend_fc(train_clean, train_roll, p1, w=0.7):
    fc1=empq_fc(train_clean,p1); fc2=empq_fc(train_roll,p1)
    return {h:[math.exp(w*math.log(fc1[h][j])+(1-w)*math.log(fc2[h][j])) for j in range(5)] for h in range(2,25)}
vals=[]
for i,(name,curve,pre) in enumerate(CLEAN):
    tc=[x for j,x in enumerate(CLEAN) if j!=i]
    tr=[x for x in ROLL if x[0]!=name]
    vals.append(score(curve, blend_fc(tc,tr,curve[1],0.7)))
print(f"CLEAN blend 0.7clean/0.3roll: mean {statistics.mean(vals):.4f} median {statistics.median(vals):.4f} worst {max(vals):.4f}")
# baseline on clean
vals=[]
for name,curve,pre in CLEAN:
    fc={h:[curve[1]*math.exp(z*0.15*math.sqrt(h-1)) for z in Z] for h in range(2,25)}
    vals.append(score(curve,fc))
print(f"CLEAN baseline: mean {statistics.mean(vals):.4f} median {statistics.median(vals):.4f} worst {max(vals):.4f}")
