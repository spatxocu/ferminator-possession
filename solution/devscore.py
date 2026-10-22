"""Score label arrays against the public 5 minutes without writing files. Development aid only."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
_gt = None


def gt_labels():
    global _gt
    if _gt is None:
        _gt = pm.read_label_csv(GT_PATH)["label"].astype(str).str.strip().str.upper().values
    return _gt


def score(labels, bootstrap_n=0):
    result = pm.evaluate(pm.label_codes(gt_labels()), pm.label_codes(np.asarray(labels)),
                         bootstrap_n=bootstrap_n)
    return result


def line(name, labels):
    r = score(labels)
    return "{:<34} spellF1 {:.4f}  macroF1 {:.4f}  poss {:>3} (gt {})  aliveDead {:.3f}  team {:.3f}".format(
        name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
        r["alive_dead_f1"], r["team_acc_gt_alive"])
