import json, math
from datetime import datetime
def parse(t): return datetime.fromisoformat(t.replace('Z','+00:00'))
c=json.load(open('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p00.in'))
for comp in c['comparables']:
    ds=parse(comp['day_start'])
    curve={}
    for p in comp['hourly']:
        h=int((parse(p['from'])-ds).total_seconds()//3600)+1
        if 1<=h<=24 and p['viewers'] and p['viewers']>0: curve[h]=p['viewers']
    if 1 not in curve: continue
    peak=max(curve.values()); ph=max(curve,key=curve.get)
    print(f"{comp['name']:45s} h1={curve[1]:8.0f} peak/h1={peak/curve[1]:6.2f} peak_h={ph:2d} plats={str(comp.get('platforms'))[:28]:28s} | {str(comp.get('unlock'))[:150]}")
