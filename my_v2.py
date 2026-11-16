"""My second possession detector (v2) -- adds ball and player speed to dead-ball detection.

Put this file in the repo root (next to games.py) and run:

    python my_v2.py

It writes evaluation/results/my_v2.csv. Then check it:

    python evaluation/validate_submission.py evaluation/results/my_v2.csv
    python evaluation/run_evaluation.py        (public 5 min: pipeline check ONLY, never for tuning)

V2 keeps the V1 possession state machine. Its only intended change is dead-ball detection:
it adds ball speed and whole-team player movement as extra evidence.
"""
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require

# ---------------------------------------------------------------- knobs (football reasoning)
# Do NOT tune these against the public 5 minutes (RULES.md / EVALUATION.md say why).
PITCH_L, PITCH_W = GAMES[GAME]["pitch"]     # metres: 105.3 x 68.0
N_FRAMES = GAMES[GAME]["n_video_frames"]    # 199319
FPS = 25

MAX_BALL_GAP = 25        # fill ball gaps up to 1 s by a straight line; longer gaps stay missing
CONTROL_RADIUS_M = 3.0   # a player this close to the ball can be "in control"
MARGIN_M = 1.0           # ...and must be this much closer than the other team's nearest player
SWITCH_FRAMES = 50       # the other team must look in control for ~2 s before possession flips
STILL_FRAMES = 100       # ball moved less than STILL_METRES over 4 s => stoppage
STILL_METRES = 1.0

# V2: extra speed-based dead-ball clues. These are initial football-reasoning
# thresholds, not values tuned against the public 5-minute ground truth.
BALL_SLOW_MPS = 0.5             # ball speed at or below this is considered slow
PLAYER_SLOW_MPS = 1.0           # a tracked player below this speed is moving slowly
PLAYER_MEAN_SLOW_MPS = 1.5      # overall mean player speed must also be low
SLOW_PLAYER_FRACTION = 0.65     # at least 65% of visible players must be slow
SPEED_WINDOW_FRAMES = 25        # smooth speed evidence over 1 second
MISSING_BALL_WINDOW_FRAMES = 75 # stricter player-only evidence over 3 seconds
MIN_TRACKED_PLAYERS = 10        # ignore player-speed evidence with too few tracks

MIN_DEAD_FRAMES = 50     # ignore dead-ball blips shorter than 2 s (they would split possessions)
CLOSE_GAP_FRAMES = 75    # a "live" blip shorter than 3 s between two stoppages is really the same stoppage

OUT_CSV = os.path.join("evaluation", "results", "my_v2.csv")


# ---------------------------------------------------------------- 1. read the files
def _tag(el):
    return el.tag.split("}")[-1]        # drop any XML namespace


def load_team_slots(xml_path):
    """Which of the 46 track slots belong to Barcelona (home) and Athletic (away)."""
    root = ET.parse(xml_path).getroot()
    home_id = None
    for el in root.iter():
        if _tag(el) == "Team" and "barcelona" in " ".join(el.itertext()).lower():
            home_id = el.get("id")
    if home_id is None:
        raise ValueError("could not find the Barcelona team in the metadata XML")
    home, away = [], []
    for el in root.iter():
        if _tag(el) == "Player":
            slot = int(el.get("id")) - 1                  # slot index 0..45
            (home if el.get("teamId") == home_id else away).append(slot)
    assert sorted(home + away) == list(range(46)), "expected slots 1..46 exactly once"
    return np.array(sorted(home)), np.array(sorted(away))


def load_tracking(txt_path):
    """Raw tracking text -> arrays indexed by VIDEO frame.

    Each line is   frame_id : 46 x "x,y,speed" separated by ";" : ball "x,y,speed"
    and video_frame = frame_id - 1 (DATA.md). Missing values are NaN.
    """
    players = np.full((N_FRAMES, 46, 2), np.nan, dtype=np.float32)
    player_speed = np.full((N_FRAMES, 46), np.nan, dtype=np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(N_FRAMES, np.nan, dtype=np.float32)
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_id, player_part, ball_part = line.split(":")[:3]
            vf = int(frame_id) - 1
            if not 0 <= vf < N_FRAMES:
                continue
            p = np.array(player_part.replace(";", ",").split(","), dtype=np.float32)
            assert p.size == 46 * 3, "line for frame %s has %d numbers, expected 138" % (frame_id, p.size)
            player_rows = p.reshape(46, 3)
            players[vf] = player_rows[:, :2]
            player_speed[vf] = player_rows[:, 2]

            # The ball record contains x, y, speed. Preserve the speed as well
            # as its position; allow the speed to remain NaN if it is absent.
            ball_values = np.array(
                ball_part.split(";")[0].split(","), dtype=np.float32
            )
            if ball_values.size >= 2:
                ball[vf] = ball_values[:2]
            if ball_values.size >= 3:
                ball_speed[vf] = ball_values[2]
    return players, player_speed, ball, ball_speed


# ---------------------------------------------------------------- 2. clean / measure
def fill_short_gaps(xy, max_gap):
    """Straight-line fill of ball gaps up to max_gap frames. Longer gaps are left as NaN."""
    out = xy.copy()
    missing = np.isnan(xy[:, 0])
    t, n = 0, len(xy)
    while t < n:
        if not missing[t]:
            t += 1
            continue
        start = t
        while t < n and missing[t]:
            t += 1
        end = t                                            # first frame with the ball again
        if start > 0 and end < n and end - start <= max_gap:
            idx = np.arange(start, end)
            for k in (0, 1):
                out[start:end, k] = np.interp(idx, [start - 1, end], [xy[start - 1, k], xy[end, k]])
    return out


def nearest_distance(players_m, ball_m, slots):
    """Distance in metres from the ball to the closest player among `slots` (NaN if unknown)."""
    d = np.hypot(players_m[:, slots, 0] - ball_m[:, None, 0],
                 players_m[:, slots, 1] - ball_m[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.min(axis=1)
    nearest[np.isinf(nearest)] = np.nan
    return nearest


def remove_short_runs(mask, min_len):
    """Turn True-runs shorter than min_len into False."""
    mask = mask.copy()
    t, n = 0, len(mask)
    while t < n:
        if not mask[t]:
            t += 1
            continue
        start = t
        while t < n and mask[t]:
            t += 1
        if t - start < min_len:
            mask[start:t] = False
    return mask


def rolling_nanmean(values, window):
    """Rolling mean that ignores NaN values; also returns valid-value counts."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    kernel = np.ones(window, dtype=np.float32)
    sums = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    counts = np.convolve(valid.astype(np.float32), kernel, mode="same")
    means = np.full(values.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts


def player_quiet_signal(player_speed):
    """Find frames where the tracked players, considered together, are moving slowly."""
    valid = np.isfinite(player_speed) & (player_speed >= 0)
    counts = valid.sum(axis=1)
    safe_speeds = np.where(valid, player_speed, 0.0)
    mean_speed = np.full(len(player_speed), np.nan, dtype=np.float32)
    np.divide(safe_speeds.sum(axis=1), counts, out=mean_speed, where=counts > 0)

    slow_counts = ((player_speed <= PLAYER_SLOW_MPS) & valid).sum(axis=1)
    slow_fraction = np.full(len(player_speed), np.nan, dtype=np.float32)
    np.divide(slow_counts, counts, out=slow_fraction, where=counts > 0)

    quiet = (counts >= MIN_TRACKED_PLAYERS)
    quiet &= (mean_speed <= PLAYER_MEAN_SLOW_MPS)
    quiet &= (slow_fraction >= SLOW_PLAYER_FRACTION)
    # NaN means there were too few reliable player speeds to make a decision.
    return np.where(counts >= MIN_TRACKED_PLAYERS, quiet.astype(np.float32), np.nan)


def find_dead_ball(ball_norm, ball_m, raw_ball, ball_speed, player_speed):
    """Combine V1 position rules with V2 ball-speed and player-movement clues."""
    with np.errstate(invalid="ignore"):
        outside = ((ball_norm[:, 0] < 0) | (ball_norm[:, 0] > 1) |
                   (ball_norm[:, 1] < 0) | (ball_norm[:, 1] > 1))
        moved = np.hypot(ball_m[STILL_FRAMES:, 0] - ball_m[:-STILL_FRAMES, 0],
                         ball_m[STILL_FRAMES:, 1] - ball_m[:-STILL_FRAMES, 1])
        still = np.zeros(len(ball_m), dtype=bool)
        still[STILL_FRAMES:] = moved < STILL_METRES        # window (t-50 .. t) was still
    # a still window marks all the frames inside it as dead
    c = np.concatenate([[0], np.cumsum(still)])
    t = np.arange(len(ball_m))
    hi = np.minimum(t + STILL_FRAMES + 1, len(ball_m))
    still_cover = (c[hi] - c[t]) > 0
    # V2 evidence A: the ball is genuinely slow over about one second.
    ball_speed_mean, ball_speed_count = rolling_nanmean(ball_speed, SPEED_WINDOW_FRAMES)
    enough_ball_speed = ball_speed_count >= max(5, SPEED_WINDOW_FRAMES // 4)
    ball_slow = enough_ball_speed & (ball_speed_mean <= BALL_SLOW_MPS)

    # V2 evidence B: many players are moving slowly at the same time.
    quiet_raw = player_quiet_signal(player_speed)
    quiet_1s, quiet_1s_count = rolling_nanmean(quiet_raw, SPEED_WINDOW_FRAMES)
    players_quiet_1s = (quiet_1s_count >= SPEED_WINDOW_FRAMES // 2) & (quiet_1s >= 0.70)

    quiet_3s, quiet_3s_count = rolling_nanmean(quiet_raw, MISSING_BALL_WINDOW_FRAMES)
    players_quiet_3s = (quiet_3s_count >= MISSING_BALL_WINDOW_FRAMES // 2) & (quiet_3s >= 0.75)

    ball_visible = np.isfinite(raw_ball[:, 0]) & np.isfinite(raw_ball[:, 1])
    ball_missing = ~ball_visible

    # When the ball is visible, require BOTH a slow ball and broadly slow players.
    # When the ball is missing, require a longer period of broadly slow players.
    speed_based_dead = (ball_slow & players_quiet_1s) | (ball_missing & players_quiet_3s)

    dead = outside | still_cover | speed_based_dead
    dead = ~remove_short_runs(~dead, CLOSE_GAP_FRAMES)     # bridge tiny live gaps inside a stoppage
    return remove_short_runs(dead, MIN_DEAD_FRAMES)         # drop tiny dead blips inside play


# ---------------------------------------------------------------- 3. the state machine
def label_half(d_home, d_away, dead, first, last):
    """Return {frame: 'H'|'A'|'D'|'X'} for one half (inclusive frame range)."""
    labels = ["X"] * (last - first + 1)
    state, challenger, streak_start = None, 0, 0
    for i, t in enumerate(range(first, last + 1)):
        if dead[t]:
            labels[i] = "D"
            state, challenger = None, 0      # after a stoppage there is a fresh start
            continue
        dh, da = d_home[t], d_away[t]
        if dh != dh or da != da:             # NaN: ball not seen -> change nothing
            evidence = "missing"
        elif dh < CONTROL_RADIUS_M and dh + MARGIN_M < da:
            evidence = "H"
        elif da < CONTROL_RADIUS_M and da + MARGIN_M < dh:
            evidence = "A"
        else:
            evidence = "loose"               # nobody clearly in control

        if evidence == "missing":
            pass
        elif state is None:
            if evidence in ("H", "A"):
                state = evidence
        elif evidence == state:
            challenger = 0
        elif evidence == "loose":
            challenger = max(0, challenger - 1)
        else:                                # the other team looks in control
            if challenger == 0:
                streak_start = i                 # remember when this streak began
            challenger += 1
            if challenger >= SWITCH_FRAMES:
                for j in range(streak_start, i): # backdate: possession changed when the streak began
                    labels[j] = evidence
                state, challenger = evidence, 0
        labels[i] = state if state else "X"

    # frames with no state yet (before the first clear evidence) take the label that follows
    nxt = None
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] in ("H", "A"):
            nxt = labels[i]
        elif labels[i] == "D":
            nxt = None
        elif nxt:
            labels[i] = nxt
    prev = None                              # anything left takes the label before it
    for i in range(len(labels)):
        if labels[i] == "X" and prev:
            labels[i] = prev
        elif labels[i] != "X":
            prev = labels[i]
    return labels


# ---------------------------------------------------------------- 4. report (sanity checks)
def spells(labels):
    """Lengths (frames) of possessions: runs of H or of A. D (and X) end a run."""
    out, cur, n = [], None, 0
    for lab in labels:
        if lab == cur and lab in ("H", "A"):
            n += 1
        else:
            if cur in ("H", "A"):
                out.append(n)
            cur, n = lab, 1
    if cur in ("H", "A"):
        out.append(n)
    return out


def main():
    t0 = time.time()
    meta, raw = require(["atd_meta", "atd_raw"])
    home, away = load_team_slots(meta)
    print("slots: home=%d away=%d" % (len(home), len(away)))

    print("reading tracking (this takes a minute)...")
    players, player_speed, ball, ball_speed = load_tracking(raw)

    blocks = half_video_blocks(GAME)
    graded = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    # sanity numbers the docs publish: ball visible ~59.3% of graded frames, ~20.5 players tracked
    print("ball visible on graded frames: %.1f%%  (docs: 59.3%%)"
          % (100 * np.mean(~np.isnan(ball[graded, 0]))))
    print("mean tracked slots on graded frames: %.1f  (docs: 20.5)"
          % np.mean((~np.isnan(players[graded, :, 0])).sum(axis=1)))
    valid_ball_speed = np.isfinite(ball_speed[graded])
    if np.any(valid_ball_speed):
        print("ball speed available on graded frames: %.1f%%"
              % (100 * np.mean(valid_ball_speed)))
    print("mean valid player speeds on graded frames: %.1f"
          % np.mean(np.isfinite(player_speed[graded]).sum(axis=1)))

    ball_norm = fill_short_gaps(ball, MAX_BALL_GAP)       # still 0-1 coordinates
    scale = np.array([PITCH_L, PITCH_W], dtype=np.float32)
    ball_m = ball_norm * scale                            # metres
    players_m = players * scale

    d_home = nearest_distance(players_m, ball_m, home).tolist()
    d_away = nearest_distance(players_m, ball_m, away).tolist()
    dead = find_dead_ball(ball_norm, ball_m, ball, ball_speed, player_speed)

    all_labels = ["X"] * N_FRAMES
    for half, first, last in blocks:
        part = label_half(d_home, d_away, dead, first, last)
        all_labels[first:last + 1] = part
        s = spells(part)
        print("half %d: %d possessions, mean %.1fs, dead-ball share %.1f%%" % (
            half, len(s), np.mean(s) / FPS if s else 0, 100 * part.count("D") / len(part)))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        f.write("frame,label\n")
        f.write("\n".join("%d,%s" % (i, lab) for i, lab in enumerate(all_labels)))
        f.write("\n")
    print("wrote %s (%d rows) in %.0fs" % (OUT_CSV, N_FRAMES, time.time() - t0))


if __name__ == "__main__":
    main()
