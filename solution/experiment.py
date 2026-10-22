"""Compare variants of the method. Prints scores and match-level totals only -- never data rows.

    python solution/experiment.py                 # every variant in VARIANTS, on the real data
    python solution/experiment.py --synthetic     # same, on a made-up match (code-path check)

Each line: public-window spellF1 and macroF1 (the 5-minute public ground truth), then full-match
totals that need no ground truth: number of possessions, their mean length, number of dead-ball
stretches and the dead-ball share. A variant that looks better on the 5 minutes but makes the
full-match totals unrealistic is not better.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession  # noqa: E402
import devscore  # noqa: E402

# Params
GRADED = np.r_[11185:90652, 110808:190198]

# name -> parameter overrides of possession.DEFAULTS. "current" is the submitted method.
VARIANTS = {
    "current": {},
    # Re-time switches inside ball gaps with the shape model: gap length, cost, weight.
    "gap2s_c5_w1": {"gap_viterbi": 50},
    "gap1s_c5_w1": {"gap_viterbi": 25},
    "gap4s_c5_w1": {"gap_viterbi": 100},
    "gap2s_c10_w1": {"gap_viterbi": 50, "gap_cost": 10.0},
    "gap2s_c2_w1": {"gap_viterbi": 50, "gap_cost": 2.0},
    "gap2s_c5_w0.5": {"gap_viterbi": 50, "gap_weight": 0.5},
    "gap2s_c5_double": {"gap_viterbi": 50, "gap_double": 1},
}
# Tried and dropped (see ferminator.md): merge_short, placed_restart, whistle_sync, walk_guard.
# Kept: keeper_fix, shape veto (4 s, 0.35).


def shape_report(players, ball, team):
    """How good is the team-shape model? Agreement with the public ground truth (never used in
    training) and with our own labels over the full match. Aggregates only."""
    extras = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = possession.predict(players, ball, team, extras=extras)
    prob = extras["shape_prob"]
    gt = devscore.gt_labels()
    live = np.isin(gt, ["H", "A"]) & ~np.isnan(prob)
    gt_acc = np.mean((prob[live] > 0.5) == (gt[live] == "H")) if live.any() else np.nan
    ours = np.isin(labels, ["H", "A"]) & ~np.isnan(prob)
    our_acc = np.mean((prob[ours] > 0.5) == (labels[ours] == "H"))
    confident = live & (np.abs(prob - 0.5) > 0.25)
    conf_acc = (np.mean((prob[confident] > 0.5) == (gt[confident] == "H"))
                if confident.any() else np.nan)
    print("shape model: public-truth accuracy {:.3f} | when confident (p<0.25 or >0.75) {:.3f} "
          "on {:.0%} of frames | agrees with our labels over the match {:.3f}".format(
              gt_acc, conf_acc, confident.sum() / max(live.sum(), 1), our_acc))


def totals(labels):
    graded = labels[GRADED]
    live = np.isin(graded, ["H", "A"])
    changes = np.flatnonzero(graded[1:] != graded[:-1]) + 1
    runs = np.split(graded, changes)
    possessions = [len(r) for r in runs if r[0] in ("H", "A")]
    dead = sum(1 for r in runs if r[0] == "D")
    return (len(possessions), np.mean(possessions) / possession.FPS if possessions else 0.0,
            dead, 1.0 - live.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic", action="store_true", help="use a made-up match")
    parser.add_argument("--only", nargs="+", help="run only these variants")
    args = parser.parse_args()

    if args.synthetic:
        from synthetic import make_match
        players, ball, team = make_match()
    else:
        from load import load
        players, ball, team = load()

    shape_report(players, ball, team)
    print("{:<26} {:>8} {:>8} {:>6} {:>6} {:>5} {:>6}".format(
        "variant", "spellF1", "macroF1", "poss", "len_s", "dead", "Dshare"))
    for name, params in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = possession.predict(players, ball, team, params)
        result = devscore.score(labels)
        n_poss, mean_len, n_dead, dead_share = totals(labels)
        print("{:<26} {:>8.4f} {:>8.4f} {:>6} {:>6.1f} {:>5} {:>6.3f}".format(
            name, result["spell_f1_mean"], result["macro_f1"], n_poss, mean_len, n_dead,
            dead_share))


if __name__ == '__main__':
    main()
