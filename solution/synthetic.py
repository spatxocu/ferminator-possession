"""A made-up match with the same array shapes as the real tracking, for testing code paths.

Nothing here comes from the competition data. It exists so the pipeline can be exercised (does it
run, does every label come out, is it deterministic) without opening the real files. The football
is crude on purpose: possession passes between teammates, is sometimes lost to the other team,
and play stops now and then with the ball out of the pitch and everyone walking.
"""

import numpy as np

# Params
N_FRAMES = 199319
N_SLOTS = 46
PITCH = (105.3, 68.0)


def make_match(seed=0, n_frames=N_FRAMES):
    """(players, ball, team) shaped like load.load(): (n,46,3), (n,3), (46,)."""
    rng = np.random.default_rng(seed)
    team = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    active = np.r_[0:11, 23:34]                       # 11 home and 11 away slots on the pitch

    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    home = rng.uniform([10, 5], [95, 63], size=(22, 2))
    owner = int(active[0])
    target = home[0].copy()
    pace = 2.5
    dead_until = -1
    for f in range(n_frames):
        if f % 250 == 0:
            pace = rng.choice([0.6, 2.5, 3.5], p=[0.3, 0.4, 0.3])
        home += rng.normal(0, pace / 25, size=home.shape)
        home = np.clip(home, [1, 1], [PITCH[0] - 1, PITCH[1] - 1])
        players[f, active, :2] = home
        players[f, active, 2] = pace + rng.normal(0, 0.2, size=22)

        k = int(np.flatnonzero(active == owner)[0])
        if f < dead_until:
            ball[f, :2] = (-3.0, home[k, 1])         # out of play beyond the goal line
        else:
            target = home[k] + 0.5
            if rng.random() < 1 / 60:                # a pass: usually a teammate, sometimes lost
                same = team[owner] if rng.random() < 0.8 else 1 - team[owner]
                owner = int(rng.choice(active[team[active] == same]))
            if rng.random() < 1 / 1500:              # a stoppage of 5-20 s
                dead_until = f + int(rng.integers(125, 500))
            ball[f, :2] = target
        if rng.random() < 0.4:                       # the ball is often invisible
            ball[f, :2] = np.nan
    return players, ball, team
