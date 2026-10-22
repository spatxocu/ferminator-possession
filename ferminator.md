# Ball possession from automatic tracking — technical document

**Team ferminator** · More than a Hack 2027, Phase 1 · FC Barcelona vs Athletic Club

## 1. The approach in one paragraph

We treat possession as a **state that changes on events**, not as a per-frame classification. A
dead-ball detector splits each half into stretches of live play. Inside each stretch, possession
belongs to the last team that **controlled** the ball, and it changes only when the other team
takes control: a player keeps the ball for a moment, or a teammate is the next to touch it. Loose
balls, deflections and 50/50 challenges change nothing. That is the definition in CHALLENGE.md
written as code, and it is what keeps the number of possessions realistic, which the ranking
metric (spellF1) rewards above everything else.

On top of that, a small **team-shape model** judges who has the ball from how the two teams are
positioned and moving. That information is there on every frame, including the 41% where the
ball is not. The model is trained on our own confident output, never on hand labels or the
public ground truth, and it acts only as a veto on short, doubtful possessions.

The whole method is plain numpy and runs in under a minute. Apart from the 14-weight logistic
regression, it consists of interpretable thresholds, so there is little to overfit to the 5
public minutes.

## 2. Pipeline

### 2.1 Ball cleaning (`clean_ball`)
- Ball observations implying a speed above **45 m/s** from the previous observation are tracker
  errors and are removed. No kick reaches that speed.
- Gaps of up to **2 s** are bridged by linear interpolation. Longer gaps are left empty, and
  possession simply holds through them.

### 2.2 Goalkeeper repair (`keeper_teams`)
Track slots carry one team for the whole match, but the tracker sometimes hands a slot to a
different person. One case can be fixed from geometry alone:
- Split every slot into segments wherever it is lost or jumps more than 3 m in one frame.
- A segment lasting at least 2 s in which the slot is the deepest player near one goal (within
  20 m of the line, central 40 m) for over 80% of its frames is that goal's keeper.
- Its team is corrected to the team defending that goal. Each half's goal sides come from the
  teams' mean positions.

This removes 6 possessions over the full match that came from a keeper tracked with the
opponent's team.

### 2.3 Touches (`touches`, `touch_episodes`)
- A team **touches** the ball on a frame when its nearest player is within **1.0 m** of the ball
  and nearer than every opponent.
- Touch frames of one team separated by at most 3 frames form a **touch episode**: one player or
  team on the ball.

### 2.4 Dead ball (`dead_half`, `slow_stoppages`)
Play stops on either of two triggers:
1. **The ball leaves the pitch:** more than 0.5 m beyond a touchline or goal line.
2. **The whole team slows to a walk:** the ±2 s mean speed of all tracked players drops below
   **1.1 m/s** for at least 3 s. This catches fouls, offsides, injuries and goal celebrations,
   where the ball never leaves the pitch (or the tracker loses it). In the public window, live
   play almost never goes below 1.3 m/s. Because players slow down *after* the whistle, the
   stoppage is dated from the peak of the team's pace in the 6 s before the slowdown.

Play **restarts** at the first kick after the stoppage: the ball is on the pitch, a player is on
it, at least 2 s have passed, and the ball is about to travel at kicking speed (≥ 6 m/s over the
next 0.5 s). In slowdown stoppages the kick is only accepted near the end of the slow stretch,
because players rolling the ball to the taker look like a kick. If no kick is seen, play
restarts 1 s after the players are moving again.

### 2.5 Possession (`possession_run`)
For each stretch of live play:
- It opens with the team of the first touch: the restart taker, who often has the ball before
  the restart.
- An episode by the *other* team switches possession only if it is **controlled**: at least
  10 touching frames (0.4 s), **or** the next episode is by the same team (a completed pass, or
  winning the ball and touching it again).
- Otherwise it is a deflection or a lost duel and is ignored.
- The switch is labelled 10 frames (0.4 s) after the new team's first touch. In the public
  ground truth the label changes as the ball leaves the new possessor, slightly after contact.
- Live frames with no touch at all yet take the last team named in the half.

### 2.6 Team-shape veto (`shape.py`)
A team with the ball spreads out and pushes up, while the team without it gets compact and drops
back or presses. We turn that into a model and use it to veto doubtful possessions.

**Features:** 14 per frame, from the home team's point of view with x flipped so that home
always attacks towards +x:
- each team's centroid x and its velocity over 1 s
- each team's spread (std of x and y)
- home players past halfway, and away players in their own half
- each team's mean speed
- the ball's x position (0 when unseen), and whether it is seen

**Model:** L2-regularised logistic regression, fitted by Newton's method in numpy.

**Training data:** our own labels, never the truth. We take the middle of every possession
lasting at least 5 s, trimming 1 s at each end and sampling every 0.2 s. The public 5-minute
window is excluded from training.

**Quality:** on that held-out window the model agrees with the ground truth on **88.8%** of
live frames, and on 92.2% of the 86% of frames where it is confident (p < 0.25 or > 0.75).

**Use:** a possession is vetoed and absorbed by its neighbours when all of these hold:
- it is shorter than **4 s**
- it sits between two spells of the other team, with no stoppage in between
- the model's mean probability for its team over its frames is below **0.35**

Real quick turnovers survive whenever the shapes agree with them.

## 3. Possession rules we implemented

| situation | our label |
|---|---|
| loose ball, ball in the air, ball not visible | stays with the last team in control |
| defender's single touch, after which the attacking team regains the ball | no change (deflection) |
| defender wins the ball and keeps it ≥ 0.4 s, or passes to a teammate | change of possession |
| 50/50, both teams within 1 m with no clear nearest | nobody's touch: no change |
| ball over a line | `D` from the moment it crosses, until the restart kick |
| foul / offside / injury (ball stays on the pitch) | `D` from the whistle (pace peak) to the restart |
| restart | the restart taker's team |
| outside the two halves | `X` (never scored) |

## 4. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| tactical ATD (players + ball) | **yes**, everything is built on it | positions for touches, player speeds for stoppages |
| tactical ATD metadata | **yes** | team of each track slot |
| Smart Tagging | **only as a sanity check** (`solution/tagcheck.py`) | Agreement with tagged attacking phases over the full match. It is automatic, its intervals are phases rather than possessions, and throw-ins carry no team, so we did not let it drive the labels. |
| homography | no | Tracking positions are already in pitch coordinates. It would only matter for reading the video. |
| broadcast video / ATD | no | Not synchronised to the tactical timeline, with cuts and replays. Aligning it is a project of its own. |
| tactical video | no | The tracking already contains what we need. A ball detector on the video is the main extension we would build next (§6). |

## 5. Results and how we validated

On the public 5 minutes (`python evaluation/run_evaluation.py`):

| | spellF1 | macroF1 | possessions (truth: 17) |
|---|---|---|---|
| **this method** | **0.600** | **0.879** | 18 |
| this method without the team-shape veto | 0.567 | 0.866 | 19 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 (published floor) | 0.326 | 0.621 | 26 |

The public window holds only 17 possessions, and the scorer warns that spellF1 gaps below about
0.33 are inside its bootstrap noise there. So we did **not** pick parameters by maximising the
public score. Instead:
- **Stable region, not the peak.** We swept the touch radius (0.8–1.5 m), control time (5–20
  frames) and stoppage speed (0.9–1.2 m/s), and chose values in the middle of the flat region
  rather than the best single cell. For example, 1.1 m/s rather than 1.2 m/s for stoppages, even
  though 1.2 scores higher on the public window.
- **Full-match plausibility.** Our labels give 249 possessions averaging 15.2 s, against 14.2 s
  in the public sample. They also give 105 stoppages and a dead-ball share of 40%, in line with
  typical ball-in-play figures for professional football.
- **Independent second opinion.** Our team labels agree with Smart Tagging's attacking-phase
  teams across the whole match more often than the official baselines do.

### Ideas we tried and dropped
Each was built behind a switch and compared with `python solution/experiment.py`, which prints
the public-window scores next to full-match totals:

| idea | result | decision |
|---|---|---|
| merge possessions under 1–2 s sandwiched between spells of the other team | public spellF1 0.529 / 0.529 / 0.588 for 1.0 / 1.5 / 2.0 s: inconsistent, and the truth contains real 0.8 s possessions | dropped |
| restart from a placed, stationary ball | almost never fires: restarts are already found by the kick rule | dropped |
| whistle time from players braking together | never fires: the provided player speeds are too smooth for a sharp braking moment | dropped |
| reject slow phases in which passes keep flowing | no slow phase qualified: the stoppage detector shows no such false positives | dropped |
| goalkeeper repair | public unchanged, 6 false possessions removed over the full match | **kept** |
| team-shape veto, 2 / 4 / 6 s | public spellF1 0.588 / 0.600 / 0.643 and macroF1 0.873 / 0.879 / 0.894: every setting better than without it, and the 4 s score is the same for any threshold from 0.25 to 0.45 | **kept at 4 s / 0.35**, the middle of the stable region. The 6 s version scores higher but merges more (16.1 s mean possession), and over-merging is costly on the full match |

## 6. Limitations and what we would do next

- **Track team errors.** Track slots sometimes carry the wrong team after an identity switch. In
  the public window, an Athletic-labelled track at Barcelona's goalkeeper's position produced a
  false possession. A fix: re-assign slot teams by formation, since a player usually sits among
  their own team's shape.
- **Ball visible on only ~59% of graded frames.** Possession holds through gaps, so a change of
  possession that happens entirely while the ball is invisible is detected late or missed. The
  team-shape model currently only *removes* doubtful possessions. Letting it also *place*
  switches inside long ball gaps (for example with a two-state Viterbi over touches plus shape
  evidence) is the natural next step. A ball detector on the tactical video (the homography
  maps it straight to the pitch) would fill many gaps directly.
- **Stoppage timing.** The whistle is inferred from the players' pace. Short stoppages with a
  quick restart (a fast free kick) can be missed, and slow build-up phases at the back can be
  mistaken for stoppages.
- **Aerial balls.** The ball track is a 2-D projection. A high ball "passes" near players it
  never touches. Ball-height cues (apparent speed and trajectory) would help.

## 7. Reproduction

```bash
pip install -r solution/requirements.txt        # numpy==2.4.6, pandas==3.0.3 (Python 3.14)
# unpack the match data into ./data, or point MTAH_DATA_DIR at it
python solution/run.py --name ferminator        # -> evaluation/results/ferminator.csv
python evaluation/validate_submission.py evaluation/results/ferminator.csv
python evaluation/run_evaluation.py             # score on the public 5 minutes
```

The output is deterministic: running it twice gives a byte-identical file. The first run parses
the 196 MB tracking file and caches it as `.npz` next to the data (never inside the repository).
Later runs take a few seconds.

Code: `solution/load.py` (data), `solution/possession.py` (the method, all parameters in
`DEFAULTS`), `solution/run.py` (entry point), `solution/experiment.py` (variant comparison),
`solution/synthetic.py` (a made-up match for testing code without the data),
`solution/devscore.py` and `solution/tagcheck.py` (development checks only).

## 8. Tools used

The code was written with the help of Claude, Anthropic's AI assistant. Early in development, small
extracts of the match data were shown to the assistant during analysis. Once we found that the
terms forbid this, we reported it to the organisers. From then on, the assistant only saw code,
the public documentation, synthetic data and aggregate scores printed by our scripts.
