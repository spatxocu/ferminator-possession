"""Regenerate the submission from the provided data. Deterministic: same data, same file.

    python solution/run.py                      # writes evaluation/results/ferminator.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from load import load  # noqa: E402
import possession  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
N_FRAMES = 199319


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="ferminator", help="file name, without .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    players, ball, team = load()
    labels = possession.predict(players, ball, team, n_frames=N_FRAMES)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)

    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("->", os.path.relpath(path, ROOT))
