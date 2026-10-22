"""A full-match sanity check that needs no ground truth: agreement with the Smart Tagging.

The tagging is automatic and its intervals are phases, not possessions, so this is never a score
to optimise. It is a second opinion over all 106 minutes, where the public ground truth covers 5,
and a variant that agrees much less with it than another has probably broken something.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))
import load_smart_tagging as lst  # noqa: E402

# Params
# Phases in which the tagged team is the one with the ball.
ATTACKING = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
             "ATTACKING TRANSITION"}
TEAMS = {"Barcelona": "H", "Atletic Club": "A"}
_events = None


def events():
    global _events
    if _events is None:
        _events = lst.load()
    return _events


def agreement(labels):
    """(team agreement in attacking phases, share of those frames called dead, set-piece restarts
    whose team matches the first live label after the tag starts)."""
    agree = total = dead = 0
    for e in events():
        team = TEAMS.get(e["tags"].get("Team"))
        if e["code"] not in ATTACKING or team is None:
            continue
        seg = labels[e["start_frame"]:e["end_frame"]]
        live = seg[(seg == "H") | (seg == "A")]
        agree += int(np.sum(live == team))
        total += len(live)
        dead += int(np.sum(seg == "D"))

    hits = n = 0
    for e in events():
        team = TEAMS.get(e["tags"].get("Team"))
        if e["code"] != "SET PIECES" or team is None:
            continue
        seg = labels[e["start_frame"]:e["end_frame"] + 250]
        live = seg[(seg == "H") | (seg == "A")]
        if len(live):
            n += 1
            hits += live[0] == team
    return agree / max(total, 1), dead / max(total + dead, 1), hits / max(n, 1)
