"""Who has the ball, judged from the two teams' shapes alone.

A team in possession spreads out and pushes up; the team without it gets compact and drops back
or presses. Those shapes are visible on every frame, including the 41% where the ball is not.

The model is trained on the method's own confident output -- the middle of long possessions it
found from touches -- so no hand labels are involved. The public ground truth window is left out
of training and is used only to report how often the model agrees with it.

Features are taken from the home team's point of view with x flipped so that home always attacks
towards +x, which makes the two halves one problem.
"""

import numpy as np

# Params
FPS = 25
PITCH = (105.3, 68.0)
HALVES = ((11185, 90651), (110808, 190197))
PUBLIC_WINDOW = (11185, 18684)   # left out of training
MIN_SPELL = 125                  # frames; only possessions at least 5 s long are training data
TRIM = 25                        # frames cut from each end of a training possession
STEP = 5                         # frames; one training sample every 0.2 s
L2 = 1.0


def home_attacks_right(players, team, first, last):
    """True when the home team defends x = 0 in this half: its players sit on average further
    left than the away team's."""
    x = players[first:last + 1, :, 0]
    home = np.nanmean(np.where(team[None, :] == 0, x, np.nan))
    away = np.nanmean(np.where(team[None, :] == 1, x, np.nan))
    return home < away


def _team_stats(x, y, speed, member):
    """Centroid, spread, players past halfway and mean speed for one team, per frame."""
    m = member & ~np.isnan(x)
    n = np.maximum(m.sum(axis=1), 1)
    xs, ys = np.where(m, x, 0.0), np.where(m, y, 0.0)
    cx, cy = xs.sum(axis=1) / n, ys.sum(axis=1) / n
    sx = np.sqrt(np.maximum((xs ** 2).sum(axis=1) / n - cx ** 2, 0))
    sy = np.sqrt(np.maximum((ys ** 2).sum(axis=1) / n - cy ** 2, 0))
    past = (m & (x > PITCH[0] / 2)).sum(axis=1)
    spd = np.where(m, np.nan_to_num(speed), 0.0).sum(axis=1) / n
    return cx, cy, sx, sy, past, spd


def _velocity(c):
    """Centred velocity over one second, m/s."""
    v = np.zeros_like(c)
    h = FPS // 2
    v[h:-h] = (c[2 * h:] - c[:-2 * h]) / (2 * h) * FPS
    return v


def features(players, team_f, xy, seen):
    """(n_frames, 14) features, home-perspective with home attacking +x. Zero outside halves."""
    n = len(players)
    out = np.zeros((n, 14))
    team_f = team_f if team_f.ndim == 2 else np.repeat(team_f[None, :], n, axis=0)
    base_team = team_f[0]
    for first, last in HALVES:
        span = slice(first, last + 1)
        x = players[span, :, 0].astype(np.float64)
        y = players[span, :, 1].astype(np.float64)
        bx = xy[span, 0].copy()
        if not home_attacks_right(players, base_team, first, last):
            x = PITCH[0] - x
            bx = PITCH[0] - bx
        speed = players[span, :, 2]
        hcx, _, hsx, hsy, hpast, hspd = _team_stats(x, y, speed, team_f[span] == 0)
        acx, _, asx, asy, apast, aspd = _team_stats(x, y, speed, team_f[span] == 1)
        # Away players past halfway in home's frame are in their own half; count them as such.
        aown = (team_f[span] == 1).sum(axis=1) - apast
        ball_seen = seen[span].astype(np.float64)
        ball_rel = np.where(np.isnan(bx), 0.0, (bx - PITCH[0] / 2) / PITCH[0])
        out[span] = np.column_stack([
            hcx, acx, _velocity(hcx), _velocity(acx),
            hsx, hsy, asx, asy, hpast, aown, hspd, aspd,
            ball_rel, ball_seen])
    return out


def _fit_logistic(X, y, l2=L2, iterations=25):
    """L2-regularised logistic regression by Newton's method. Returns weights incl. bias."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    reg = np.eye(len(w)) * l2
    reg[0, 0] = 0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        grad = Xb.T @ (p - y) + reg @ w
        hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def training_frames(labels):
    """Frames in the middle of long H/A spells, outside the public window, every STEP frames."""
    frames = []
    for first, last in HALVES:
        seg = labels[first:last + 1]
        changes = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        for a, b in zip(np.r_[0, changes], np.r_[changes, len(seg)]):
            if seg[a] in ("H", "A") and b - a >= MIN_SPELL:
                frames.append(np.arange(first + a + TRIM, first + b - TRIM, STEP))
    frames = np.concatenate(frames) if frames else np.array([], dtype=int)
    lo, hi = PUBLIC_WINDOW
    return frames[(frames < lo) | (frames > hi)]


def home_probability(players, team_f, xy, seen, labels, smooth=12):
    """P(home has the ball) per frame from team shape, trained on `labels`. NaN outside halves."""
    X = features(players, team_f, xy, seen)
    train = training_frames(labels)
    prob = np.full(len(X), np.nan)
    if len(train) < 100:
        return prob
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-9
    w = _fit_logistic((X[train] - mu) / sd, (labels[train] == "H").astype(np.float64))
    for first, last in HALVES:
        Z = (X[first:last + 1] - mu) / sd
        p = 1 / (1 + np.exp(-np.clip(w[0] + Z @ w[1:], -30, 30)))
        if smooth:
            c = np.cumsum(np.r_[0, p])
            i = np.arange(len(p))
            lo, hi = np.maximum(0, i - smooth), np.minimum(len(p), i + smooth + 1)
            p = (c[hi] - c[lo]) / (hi - lo)
        prob[first:last + 1] = p
    return prob


def veto_short(labels, prob, max_len, min_p):
    """Absorb a possession shorter than max_len frames, sandwiched between two spells of the other
    team, when the shape model gives its team a mean probability below min_p over its frames.
    Real quick turnovers survive whenever the shapes agree with them."""
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.r_[0, changes], np.r_[changes, len(labels)]
    for k in range(1, len(starts) - 1):
        a, b = starts[k], ends[k]
        mid, before, after = labels[a], labels[starts[k - 1]], labels[starts[k + 1]]
        if not (mid in ("H", "A") and before == after and before in ("H", "A") and before != mid
                and b - a < max_len):
            continue
        p_home = np.nanmean(prob[a:b])
        p_team = p_home if mid == "H" else 1 - p_home
        if np.isfinite(p_team) and p_team < min_p:
            labels[a:b] = before
    return labels
