"""Read the tactical ATD into dense numpy arrays, indexed by video frame.

The raw file is one line per frame, "<frame_id>:<46 x 'x,y,speed'>:<ball x,y,speed>". Swapping both
separators for commas turns it into a plain 142-column CSV that pandas parses in seconds, instead
of the minutes a per-line Python loop takes.

The parsed arrays are cached next to the data (never in the repository) as one .npz file.

    players  (n_frames, 46, 3)  x, y in metres, speed in m/s; NaN where the slot is not tracked
    ball     (n_frames, 3)      same, for the ball
    team     (46,)              0 home, 1 away, per track slot
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402

# Params
N_SLOTS = 46
CACHE_NAME = "atd_tactical_cache.npz"


def team_of_each_track(meta_path):
    """0 for home slots, 1 for away. The first <Team> in the EPTS metadata is the home side."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_frames):
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32).to_numpy()
    if table.shape[1] != 1 + 3 * N_SLOTS + 3:
        raise ValueError("unexpected ATD width {}".format(table.shape[1]))

    frames = table[:, 0].astype(np.int64) - 1  # frame_id counts from 1, video frames from 0
    players = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players[frames] = table[:, 1:1 + 3 * N_SLOTS].reshape(-1, N_SLOTS, 3)
    ball[frames] = table[:, 1 + 3 * N_SLOTS:]
    return players, ball


def load(game=GAME, data_dir=None):
    """(players, ball, team) in metres. Parses once, then reads the cache."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(os.path.dirname(raw_path), CACHE_NAME)
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(raw_path):
        stored = np.load(cache)
        return stored["players"], stored["ball"], stored["team"]

    players, ball = parse_raw(raw_path, source_frames("tactical", game))
    length, width = GAMES[game]["pitch"]
    for array in (players, ball):
        array[..., 0] *= length
        array[..., 1] *= width
    team = team_of_each_track(meta_path)
    np.savez(cache, players=players, ball=ball, team=team)
    return players, ball, team


if __name__ == '__main__':
    players, ball, team = load()
    print("players", players.shape, "ball", ball.shape, "home slots", int(np.sum(team == 0)))
    print("ball visible on {:.1%} of frames".format(np.mean(~np.isnan(ball[:, 0]))))
