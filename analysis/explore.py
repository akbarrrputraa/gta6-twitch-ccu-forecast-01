import json, math, glob
from datetime import datetime, timezone

def parse(t): return datetime.fromisoformat(t.replace('Z','+00:00'))

for f in sorted(glob.glob('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01-kit/gta6-twitch-ccu-forecast-01-kit/input/p0*.in')):
    c=json.load(open(f))
    print("="*100)
    print(c['id'], c['game']['name'], 'day_start', c['day_start'], 'known', c['known'])
    print('notes:', c.get('notes'))
    print('n comparables:', len(c['comparables']), 'history pts:', len(c['history']), 'history_hourly:', len(c['history_hourly']))
    for comp in c['comparables'][:3]:
        print(' comp:', comp['name'], 'day_start', comp['day_start'], 'unlock', comp.get('unlock'), 'platforms', comp.get('platforms'), 'n hourly', len(comp['hourly']))
