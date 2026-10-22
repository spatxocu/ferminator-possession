"""Frame-by-frame ball possession from the automatic tracking.

Three stages, each a plain function over numpy arrays:

  1. clean_ball      drop impossible ball jumps, bridge short gaps by linear interpolation
  2. touches         per frame, which team (if any) is touching the ball
  3. label_match     a state machine: dead-ball stretches from exits and stoppages, and during
                     live play the ball belongs to the last team that touched it

The possession rule follows CHALLENGE.md: a loose or contested ball stays with the last team that
had it, and possession only changes when the other team actually reaches the ball.
"""

import numpy as np

# Params
FPS = 25
PITCH = (105.3, 68.0)
HALVES = ((11185, 90651), (110808, 190197))  # graded video frames, inclusive

DEFAULTS = dict(
    max_ball_speed=45.0,   # m/s; faster apparent jumps are tracker errors, not kicks
    max_gap=50,            # frames; bridge ball gaps up to 2 s by interpolation
    touch_radius=1.0,      # m; nearest player this close (and closer than any opponent) touches
    contest_margin=0.0,    # m; the toucher must be this much nearer than the nearest opponent
    switch_lag=10,         # frames; label the new team this long after its first touch
    max_break=3,           # frames; touch frames this close together are one touch episode
    control_frames=10,     # touching frames that make a lone episode controlled (0.4 s)
    out_margin=0.5,        # m beyond the line before the ball counts as out
    min_dead=50,           # frames; a stoppage lasts at least 2 s before a restart is looked for
    kick_speed=6.0,        # m/s; ball speed over the next 0.5 s that counts as a restart kick
    pace_window=50,        # frames; half-width of the team-pace average (+/- 2 s)
    slow_speed=1.1,        # m/s; mean player speed below this is a walk, not play. 0 disables
    min_slow=75,           # frames; a walk this long (3 s) is a stoppage
    slow_backoff=150,      # frames; look this far back for the pace peak that marks the whistle
    restart_window=100,    # frames; restart kick accepted from this long before the walk ends
    restart_grace=25,      # frames; with no kick seen, play restarts this long after the walk
)


def clean_ball(ball, p):
    """Ball x, y with tracker jumps removed and short gaps interpolated. Also: was it observed."""
    xy = ball[:, :2].astype(np.float64).copy()
    seen = ~np.isnan(xy[:, 0])

    # Remove points that imply an impossible speed from the previous observed point.
    idx = np.flatnonzero(seen)
    step = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1) / np.maximum(np.diff(idx), 1) * FPS
    bad = idx[1:][step > p["max_ball_speed"]]
    xy[bad] = np.nan
    seen[bad] = False

    # Linear interpolation across gaps no longer than max_gap.
    idx = np.flatnonzero(seen)
    filled = xy.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= p["max_gap"]:
            w = np.linspace(0, 1, b - a + 1)[1:-1, None]
            filled[a + 1:b] = xy[a] * (1 - w) + xy[b] * w
    return filled, seen


def team_distances(players, team, xy):
    """Distance from the ball to the nearest home and the nearest away player, per frame."""
    d = np.linalg.norm(players[:, :, :2] - xy[:, None, :], axis=2)
    home = np.nanmin(np.where(team[None, :] == 0, d, np.inf), axis=1)
    away = np.nanmin(np.where(team[None, :] == 1, d, np.inf), axis=1)
    home[~np.isfinite(home)] = np.nan
    away[~np.isfinite(away)] = np.nan
    return home, away


def touches(d_home, d_away, p):
    """-1 nobody, 0 home, 1 away: the nearest player is within touch_radius of the ball and
    clearly nearer than any opponent. A 50/50 within contest_margin is nobody's touch yet."""
    out = np.full(len(d_home), -1, dtype=np.int8)
    h = np.nan_to_num(d_home, nan=np.inf)
    a = np.nan_to_num(d_away, nan=np.inf)
    m = p["contest_margin"]
    out[(h < p["touch_radius"]) & (h + m <= a)] = 0
    out[(a < p["touch_radius"]) & (a + m <= h)] = 1
    return out


def ball_speed_ahead(xy, horizon):
    """Mean ball speed over the next `horizon` frames, m/s. NaN where unknown."""
    ahead = np.full(len(xy), np.nan)
    ahead[:-horizon] = np.linalg.norm(xy[horizon:] - xy[:-horizon], axis=1) / horizon * FPS
    return ahead


def out_of_play(xy, p):
    m = p["out_margin"]
    x, y = xy[:, 0], xy[:, 1]
    return (x < -m) | (x > PITCH[0] + m) | (y < -m) | (y > PITCH[1] + m)


def rolling_mean(x, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN."""
    total = np.cumsum(np.r_[0, np.nan_to_num(x)])
    count = np.cumsum(np.r_[0, ~np.isnan(x)])
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half_width), np.minimum(len(x), i + half_width + 1)
    return (total[hi] - total[lo]) / np.maximum(count[hi] - count[lo], 1)


def slow_stoppages(players, p):
    """(start, end) frames of stretches where the whole team slows to a walk: a stoppage the ball
    never left the pitch for -- a foul, an offside, an injury, a goal celebration.

    The referee whistles before the players slow down, and players decelerate from the moment of
    the whistle, so the stoppage is dated from the peak of the team's pace in the slow_backoff
    frames before the walk begins."""
    speeds = players[:, :, 2]
    tracked = np.sum(~np.isnan(speeds), axis=1)
    mean_speed = np.where(tracked > 0, np.nansum(speeds, axis=1) / np.maximum(tracked, 1), np.nan)
    pace = rolling_mean(mean_speed, p["pace_window"])
    slow = np.r_[False, pace < p["slow_speed"], False]
    starts = np.flatnonzero(~slow[:-1] & slow[1:])
    ends = np.flatnonzero(slow[:-1] & ~slow[1:]) - 1
    out = []
    for s, e in zip(starts, ends):
        if e - s + 1 < p["min_slow"]:
            continue
        lo = max(0, s - p["slow_backoff"])
        out.append((lo + int(np.argmax(pace[lo:s + 1])), e))
    return out


def dead_half(first, last, touch, out, speed_ahead, stoppages, p):
    """Boolean dead-ball mask for one half.

    Play stops when the ball leaves the pitch or a slow stoppage begins, and restarts at the
    first kick after that: a player on the ball, in play, about to travel at kicking speed. In a
    slow stoppage that kick is only accepted near the end of the slow stretch, because players
    rolling the ball to the taker look like a kick but are not the restart. If no kick is seen,
    play is taken to have restarted once the players are moving again."""
    n = last - first + 1
    dead = np.zeros(n, dtype=bool)
    stop_end = np.full(n, -1)
    for s, e in stoppages:
        if first <= s <= last:
            stop_end[s - first] = max(stop_end[s - first], e - first)
    is_dead, since, not_before, give_up = False, 0, 0, n
    for i in range(n):
        f = first + i
        if stop_end[i] >= 0:
            if not is_dead:
                is_dead, since, give_up = True, i, n
            not_before = max(not_before, stop_end[i] - p["restart_window"])
            give_up = stop_end[i] + p["restart_grace"] if give_up == n else max(
                give_up, stop_end[i] + p["restart_grace"])
        if not is_dead:
            if out[f]:
                is_dead, since, not_before, give_up = True, i, 0, n
        elif out[f]:
            give_up = n  # back off the pitch: wait for a real restart
        elif (i - since >= p["min_dead"] and i >= not_before and touch[f] >= 0
              and speed_ahead[f] >= p["kick_speed"]) or i > give_up:
            is_dead, not_before, give_up = False, 0, n
        dead[i] = is_dead
    return dead


def touch_episodes(touch, max_break):
    """Runs of one team touching the ball, merging breaks of up to max_break frames.

    Returned as (team, first frame, last frame, number of touching frames)."""
    idx = np.flatnonzero(touch >= 0)
    episodes = []
    for f in idx:
        t = int(touch[f])
        if episodes and episodes[-1][0] == t and f - episodes[-1][2] <= max_break + 1:
            team, start, _, count = episodes[-1]
            episodes[-1] = (team, start, f, count + 1)
        else:
            episodes.append((t, f, f, 1))
    return episodes


def possession_run(first, last, episodes, p):
    """Team per frame over one stretch of live play, first..last inclusive. -1 where unknown.

    The stretch opens with whoever touches the ball first (the restart). After that the other
    team takes over only with controlled possession: it keeps the ball for control_frames, or the
    next touch after its own is by a teammate (a completed pass or a re-touch). A lone touch
    followed by the original team getting the ball back is a deflection and changes nothing.
    """
    n = last - first + 1
    team_at = np.full(n, -1, dtype=np.int8)
    # Episodes that reach into the run: the restart taker often has the ball before the restart.
    eps = [e for e in episodes if e[2] >= first and e[1] <= last]
    if not eps:
        return team_at

    switches = [(first, eps[0][0])]
    team = eps[0][0]
    for k, (t, start, end, count) in enumerate(eps[1:], 1):
        if t == team:
            continue
        controlled = count >= p["control_frames"]
        followed = k + 1 < len(eps) and eps[k + 1][0] == t
        if controlled or followed:
            team = t
            switches.append((start + p["switch_lag"], t))

    for frame, t in switches:
        team_at[max(frame, first) - first:] = t
    return team_at


def label_half(first, last, touch, out, speed_ahead, stoppages, episodes, p):
    """Labels for one half, frames first..last inclusive, as an array of 'H'/'A'/'D'/'X'."""
    dead = dead_half(first, last, touch, out, speed_ahead, stoppages, p)
    labels = np.full(len(dead), "D", dtype="<U1")

    alive = np.flatnonzero(~dead)
    if len(alive):
        breaks = np.flatnonzero(np.diff(alive) > 1)
        for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks, len(alive) - 1]):
            lo, hi = alive[a], alive[b]
            team_at = possession_run(first + lo, first + hi, episodes, p)
            labels[lo:hi + 1] = np.where(team_at == 0, "H", np.where(team_at == 1, "A", "X"))
    return fill_unknown(labels)


def fill_unknown(labels):
    """Live frames nobody was seen touching take the last team named in the half (the next one at
    the head of a half). The ball stays with the last team that had it."""
    team = np.where((labels == "H") | (labels == "A"), labels, "")
    idx = np.flatnonzero(team != "")
    if not len(idx):
        return labels
    unknown = np.flatnonzero(labels == "X")
    prev = np.searchsorted(idx, unknown, side="right") - 1
    labels[unknown] = team[idx[np.maximum(prev, 0)]]
    return labels


def predict(players, ball, team, params=None, n_frames=None):
    p = dict(DEFAULTS, **(params or {}))
    n_frames = n_frames or len(ball)
    xy, _ = clean_ball(ball, p)
    d_home, d_away = team_distances(players, team, xy)
    touch = touches(d_home, d_away, p)
    out = out_of_play(xy, p)
    speed = ball_speed_ahead(xy, FPS // 2)

    episodes = touch_episodes(touch, p["max_break"])
    stoppages = slow_stoppages(players, p) if p["slow_speed"] > 0 else []

    labels = np.full(n_frames, "X", dtype="<U1")
    for first, last in HALVES:
        labels[first:last + 1] = label_half(first, last, touch, out, speed, stoppages,
                                            episodes, p)
    return labels
