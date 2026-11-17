<div align="center">

# ⚽ PITCH COMMAND AI
### Frame-Level Ball Possession from Tactical Tracking Data

[![Competition](https://shields.io)]()
[![Python](https://shields.io)]()
[![Pipeline](https://shields.io)]()
[![Zero ML](https://shields.io)]()

**Team:** `KYLIAN YAMAL DS` | **Final Submission:** `evaluation/results/my_v4.csv`  
**Squad:** Gaddam Yashas • Kaivalya R Nayak • Kevin Alexandar

*No black-box neural networks. No external data. Just pure tactical geometry, physics, and football IQ.*

```text
╔═════════════════════════════════════╤═════════════════════════════════════╗
║ ┌───────────┐                       │                       ┌───────────┐ ║
║ │           │                 . - ~ ┴ ~ - .                 │           │ ║
║ │  ┌─────┐  │              .-'             '-.              │  ┌─────┐  │ ║
║ │  │  H  │  │             /         ●         \             │  │  A  │  │ ║
║ │  └─────┘  │            │      105.3m × 68m   │            │  └─────┘  │ ║
║ │           │             \                   /             │           │ ║
║ └───────────┘              '-.             .-'              └───────────┘ ║
║                               ' - ~ ┬ ~ - '                               ║
╚═════════════════════════════════════╧═════════════════════════════════════╝
```

</div>

---

## 🏟️ Match Overview

This repository contains our solution for **More than a Hack 2027 (Phase 1)**. Every single frame of the tactical recording (`25 fps`) is evaluated by our transparent, rule-based pipeline (`my_v4.py`) and classified into one of four match states:

| Code | Match State | Tactical Meaning |
| :--- | :--- | :--- |
| **`H`** | 🏠 **Home Possession** | Home squad holds clear, sustained control of the ball. |
| **`A`** | ✈️ **Away Possession** | Away squad holds clear, sustained control of the ball. |
| **`D`** | 🛑 **Dead Ball** | Play is stopped (out of bounds, set-piece setup, or walking pace). |
| **`X`** | ⏱️ **Out of Play Window** | Frames outside the graded first and second halves defined in `games.py`. |

> **Fair Play Declaration:** Public ground truth was used strictly to verify frame alignment, CSV schema formatting, and dead-ball definitions. **Zero parameters were tuned on ground-truth labels.**

---

## 📁 Repository Layout

Our solution builds directly on top of the official challenge starter repository:

```text
├── data/                      # Raw match tracking & XML files (excluded from git)
├── evaluation/
│   ├── validate_submission.py # Official submission format & label validator
│   └── results/               # Output directory for my_v4.csv
├── starter/                   # Official challenge baseline scripts
├── games.py                   # Pitch dimensions (105.3m × 68.0m) & graded half boundaries
├── my_v1.py                   # Iteration 1: Initial proximity possession baseline
├── my_v2.py                   # Iteration 2: Basic dead-ball filtering
├── my_v3.py                   # Iteration 3: Smart Tagging set-piece windows
├── my_v4.py                   # 🏆 FINAL PIPELINE: 5-clue dead-ball detector + hysteresis state machine
├── requirements.txt           # Base challenge environment dependencies
├── requirements-starter.txt   # Starter baseline dependencies
├── requirements-mine.txt      # Exact dependency versions for my_v4.py (numpy, pandas, kloppy)
└── CHALLENGE.md / DATA.md / EVALUATION.md / FAQ.md / OVERVIEW.md / RULES.md / SUBMISSION.md
```

---

## 📋 The Tactical Scouting Report (Data Pipeline)

We strip away noisy broadcast feeds and build our match engine purely on pitch-coordinate geometry imported from `games.py`:

* **✅ What Enters the Pitch:**
  * **ATD Tracking:** Player & ball coordinates and player speeds (`video_frame = frame_id - 1`).
  * **Squad Sheet:** Team assignments across all `46` track slots from the metadata XML.
  * **Smart Tagging:** Set-piece start timestamps (`round(seconds × 25)`) and event types only.
  * **Pitch Dimensions:** Normalized `0–1` coordinates scaled to `105.3m × 68.0m` with graded half boundaries from `games.py`.
* **❌ Left on the Bench:**
  * Unaligned broadcast video, homography transforms, raw video pixels, unreliable tracker ball-speed channels, and Smart Tagging `Team` / `phase` tags.
* **🩹 Handling Missing Ball Data (`41%` of Graded Frames):**
  * Short blind spots (`≤ 1.0 s`) are bridged using linear interpolation.
  * Longer tracking dropouts remain missing—our possession state machine simply **holds the current team's possession** until fresh evidence arrives.

---

## 🚩 Component 1: The Assistant Referee (Dead-Ball Detector)

A frame is flagged as **Dead Ball (`D`)** if *any* of these five whistles blow:

1. **Out of Bounds:** Ball coordinates cross outside the `105.3m × 68.0m` touchlines or goal lines.
2. **Frozen Ball:** The ball travels less than `1.0 m` across a `4.0 s` window.
3. **Walking Pace:** Mean team speed (averaged over `4.0 s` with `≥ 8` valid players between `0–12 m/s`) drops to `≤ 1.3 m/s` while the ball moves `≤ 6.0 m` in the surrounding `2.0 s`.
4. **Set-Piece Buffer Zones:** Dynamic pre/post windows around Smart Tagging starts (derived from median stoppage profiles), snapping back early if the ball exited play for `≥ 0.2 s` in the prior `25 s`:
   * **Throw-in:** `-2s / +6s` | **Corner:** `-8s / +3s` | **Goal Kick:** `-5s / +6s` | **Free Kick:** `-6s / +8s` | **Kick-off:** `-10s / +3s`
5. **Pre-Kickoff Hold:** From the opening graded frame of each half until `3.0 s` after that half's `Kick Off (Start)` tag.

> **VAR Cleanup Pass:** Live gaps shorter than `3.0 s` trapped between dead frames are merged into dead play, and dead-ball blips shorter than `2.0 s` are wiped out to prevent tracking glitches from splitting real possessions.

---

## ⚡ Component 2: Possession State Machine

How do we decide who owns the ball? **Control requires proximity and commitment:**

```text
[Nearest Player <= 3.0m] AND [>= 1.0m Closer Than Opponent]
                         │
                         ▼ (Sustained for 50 frames / 2.0s)
        🔄 POSSESSION FLIPS (Backdated to start of streak!)
```

* **Contested / Loose Balls:** Frames where neither team meets the proximity rule decrement the takeover counter; loose balls stay with the last team in control.
* **Post-Stoppage Restart:** After any dead-ball stretch (`D`), the state resets. The first team to establish clear proximity control claims possession, and intermediate restart frames inherit that team's label.

---

## 📊 Telemetry & Sanity Checks

Running `my_v4.py` across the full match data reproduces expected tracking telemetry and produces realistic match flow:

| Telemetry Check | Pipeline Output |
| :--- | :--- |
| **Ball Visibility Reproduced** | `59.3%` of graded frames |
| **Active Tracked Slots** | `20.5` average slots |
| **Dead-Ball Share (`D`)** | `~42%` of graded frames |
| **Total Possessions Detected** | `~200` spells (`~17 s` mean duration) |

---

## 🚀 Kick-Off: Reproducing `my_v4.csv`

Follow these steps to run our final deterministic pipeline and validate the submission file:

### 1. Set Up the Locker Room
Ensure you have **Python 3.9+** installed, then install the exact pinned dependencies (`numpy`, `pandas`, `kloppy`):
```bash
pip install -r requirements-mine.txt
```

### 2. Verify Match Data Placement
Place the downloaded competition match files inside the `data/` directory (already ignored by `.gitignore`):
```text
data/
```

### 3. Run the Final Match Engine (`v4`)
Execute the deterministic script to generate the frame-level predictions:
```bash
python my_v4.py
```
*Output written to:* `evaluation/results/my_v4.csv`

### 4. Official VAR Validation
Run the official validation script in `evaluation/` to verify schema compliance and valid labels (`H`, `A`, `D`, `X`):
```bash
python evaluation/validate_submission.py evaluation/results/my_v4.csv
```

---

## 🔮 Second-Half Tactical Adjustments (Limitations & Roadmap)

* **Goal-Kick Speed Overlap:** The `1.3 m/s` walking-pace threshold sits close to slow build-up play and can occasionally over-flag slow live sequences as dead.
* **Window-Based Restarts:** Fixed Smart Tagging buffers can delay restart detection by a few seconds; future versions will trigger restarts on **first purposeful ball acceleration**.
* **Proximity-Only Control:** Adding explicit **touch/pass impulse detection** and utilizing the Smart Tagging `Team` field on restarts will sharpen turnover precision.
* **Multi-Match Calibration:** Current thresholds are grounded in football domain logic; training parameters across multiple labeled matches will improve cross-match generalization.

---

<div align="center">

**Built with ⚽ and pure geometry by KYLIAN YAMAL DS**  
*Refer to `CHALLENGE.md`, `RULES.md`, and `SUBMISSION.md` for official competition guidelines.*

</div>
