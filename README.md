# GTA VI launch-day Twitch forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/akbarrrputraa/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/akbarrrputraa/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people will be watching Grand Theft Auto VI live on
Twitch in each hour of its release day (19 Nov 2026, UTC), as five
quantiles per hour, from the first hour's viewer count alone. Built by an
AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour
by hour" bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3, standard library only, no network) reads every
`<id>.in` case in the input directory and writes `<id>.out` forecasts:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours
2–24.

## Method

The program fits its model at run time from the case's own `comparables`
(past big game launches' hourly Twitch counts, SullyGnome):

1. **Pick GTA-like launches** — console-led games that unlocked at
   midnight local time, rolling around the world: release-day peak at
   most 4× the hour-1 print, no early-access surge in the 3 days before
   release. This drops single-global-unlock launches (Silksong, Black
   Myth: Wukong) and early-access-contaminated ones (Hogwarts Legacy,
   Starfield, Diablo IV, ...).
2. **Per hour, regress log(viewers / hour-1) on log(hour-1)** across those
   launches. Launches that already drew a huge hour-1 crowd peak at hour
   1 and stay flat; smaller hour-1 launches grow 2–3× into the US
   evening (leave-one-out correlation ≈ −0.85).
3. **Second signal**: the hour just before midnight (already holding the
   New Zealand / early-Europe wave) relative to hour 1, applied at full
   weight. Each hour's residual spread is shrunk toward the pooled
   spread across all hours (16 pseudo-observations), because a per-hour
   spread fitted on ~16 launches is noisy. The shrunk spread is the
   forecast's log spread (floored 0.15, capped 0.90).

Leave-one-out pinball loss on the 16 qualifying launches: **0.177**,
vs 0.276 for the house persistence baseline on the same launches and
0.225 for a scale-free empirical-quantile curve. Full write-up in the
`main.py` docstring.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zm2jvf0q&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
