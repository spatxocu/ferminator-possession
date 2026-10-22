"""Rebuild the submission twice, confirm both runs are byte-identical, validate it and score it.

    python solution/rebuild.py

Prints label counts, the reproducibility verdict and the public-window scores only.
"""

import filecmp
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "ferminator"


def run(*args):
    return subprocess.run([sys.executable] + list(args), cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


if __name__ == '__main__':
    check_dir = tempfile.mkdtemp()
    run("solution/run.py", "--out", check_dir)
    print(run("solution/run.py").strip())
    target = os.path.join(ROOT, "evaluation", "results", NAME + ".csv")
    same = filecmp.cmp(target, os.path.join(check_dir, NAME + ".csv"), shallow=False)
    print("REPRODUCIBLE: both runs identical" if same else "NOT REPRODUCIBLE: runs differ")
    print(run("evaluation/validate_submission.py", target).strip().splitlines()[-2])
    run("evaluation/run_evaluation.py")
    import pandas as pd
    board = pd.read_csv(os.path.join(ROOT, "evaluation", "scoreboard.csv"))
    print(board[["rank", "submission", "spell_f1_mean", "macro_f1"]].to_string(index=False))
