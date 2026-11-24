# motion-and-diffusion

VD Computation final project — visual experiments on top of motion diffusion
(output target: 1080×1350, 30 fps video).

Model: [A-MDM](https://github.com/Yi-Shi94/AMDM) (auto-regressive motion diffusion, LaFAN1 15-step checkpoint),
vendored as a git submodule in `third_party/AMDM`.

## Setup

```bash
git clone --recursive https://github.com/seungho2k/motion-and-diffusion.git
cd motion-and-diffusion
./scripts/setup.sh          # uv env (Python 3.10) + A-MDM checkpoints
```

LaFAN1 BVHs (needed by experiment 3 for dance start poses; CC BY-NC-ND 4.0):

```bash
mkdir -p data/lafan1 && curl -L -o /tmp/lafan1.zip \
  https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip \
  && unzip -o /tmp/lafan1.zip -d data/lafan1
```

## Experiment 6 — endless real-time A-MDM with a repeating τ cycle

```bash
uv run python exp6_live.py       # open http://localhost:8080
```

A child process (`src/live.py`) keeps generating new frames forever (no GIL contention with the viewer;
~40 frames/s on an M5 CPU, ~36 with the t=0 overlay). The diffusion level follows a cycle (ramp / pulses /
none) that repeats every `cycle` seconds while the motion itself is always new. Display at 30 or 15 fps; at
15 fps either keep real speed (generate 2, show 1) or play in slow motion. `restart` re-seeds from the start
clip; `Export` renders the last N seconds.

`motions: 3` generates three motions at once and shows them side by side along the camera's right vector,
each pinned at its tracked joint (default Hips); the floor grid is turned off automatically. They start
from the same pose with different seeds (or from frames offset along the clip). One motion runs on a CPU
worker (~36 frames/s), three run batched on an MPS worker (~34 frames/s; MPS cost barely grows with batch
size, whereas three parallel CPU processes only reached ~20 frames/s).

`3 holding hands`: the middle motion is pinned at its hips and holds the neighbours' hands (middle left hand ↔
left partner's right hand, middle right hand ↔ right partner's left hand; `swap hands` flips them).
`2 holding hands`: A's right hand holds B's left hand and that junction is pinned on screen. `hand link`:
`attach` moves the partner bodies so the held hands touch; `tether` (3 only) keeps the bodies in their slots
and draws a line between the hands. Holding hands is only a placement: each body is tracked at its held hand
and the hands are brought to the same screen point; no guidance changes the motion (an earlier hand-reach
guidance stretched the arms, since position channels carry no bone-length constraint).

Pinned points (tracked joint, middle hips, held-hand junction) are placed at the camera's look-at point, so
they stay at the screen centre even after zooming toward the cursor or panning.

Skeletons end at the `LeftHand` / `RightHand` joints. When holding hands, the partner is placed so its Hand
joint coincides with the other Hand joint, and that point is pinned at the screen centre (2 persons).

`τ ladder`: one clean motion (fed back fully denoised) shown side by side at evenly spaced τ from
`ladder max τ` down to 0 (`ladder steps`, default 6: 15, 12, 9, 6, 3, 0), each pinned at its hips, with τ
labels. All levels come from the same reverse chain and noise of each frame, so the ladder costs one denoise
per frame.

## Experiment 5 — text timeline with DiP (+ pulses / ramp / dip)

```bash
uv run python exp5_dip.py        # open http://localhost:8080
```

Uses **DiP**, the autoregressive text-to-motion diffusion planner from [CLoSD](https://github.com/GuyTevet/CLoSD)
(submodule `third_party/CLoSD`; only the planner is used, no simulator). DistilBERT text encoder, 10 diffusion
steps, 20-frame prefix → 40-frame plan, HumanML3D features (22 SMPL joints, 20 fps; resampled to 30 fps).
Assets are fetched one by one from the `guytevet/CLoSD` Hugging Face repo on first run (~250 MB, not the 19 GB
snapshot); `src/shims/clip` stands in for openai-clip, which this BERT checkpoint never calls.

- **Text timeline** — one line per segment, `seconds: prompt`; total length; text guidance (CFG)
- **Schedule** — same pulses / ramp / dip as experiment 2 (τ range 0–10), pulse frames in 30 fps video frames
- **Generation** — `commit frames`: frames kept from each 40-frame plan before replanning (receding horizon,
  so a schedule can change every few frames); feed back `xt` / `x0` / `final`; root from display / final
- Display, camera, overlay t=0, export, saved panel state as in the other viewers

## Experiment 3 — two dancers and a gravity between their joints

```bash
uv run python exp3_gravity.py    # open http://localhost:8080
```

Two independent A-MDM rollouts (LaFAN1 prior, started from dance clips) are coupled only by a softened
inverse-square pull applied to the model's clean-frame prediction during denoising (`src/gravity.py`).
Pulses of pull drag the bodies together; the last `guide stop step` diffusion steps run prior-only, so each
pose stays plausible while the trajectory becomes impossible. Defaults: 1200 frames, pulse width 4 frames.

- **Dancers** — start clip / frame for A and B, separation, facing
- **Gravity** — which joints pull (hands → partner, arms → partner, hands → head/chest, all ↔ all),
  pulse strength (≈ m/frame at 1 m) / count / width, constant pull, softening, repulsion radius,
  `momentum` (an impulse keeps dragging), `body pull` (share applied as a rigid whole-body drag outside the
  prior — the prior alone only lets limbs reach an arm's length), `lift off floor`, guide stop step
- **Display / Camera / Playback / Export** — as in experiment 2; track the midpoint of the two Hips (default) or any joint of A / B

## BVH playback (E2-Create Motion Bank samples, LaFAN1)

```bash
uv run python bvh_viewer.py      # open http://localhost:8081
```

Plays any `.bvh` found under `.cache/e2_motionbank/` and `data/lafan1/` (or `--dir`), with speed, scrubbing,
loop, root/hand trails, tracking of any joint and the same camera / display controls as the experiments. E2 samples
were pulled from the Zenodo archive with ranged requests (the 52 GB zip is almost all video).

## Experiment 2 — noise ↔ human (dip or pulses)

```bash
uv run python exp2_viewer.py     # open http://localhost:8080
```

Each frame is denoised only down to a level τ(f) (15 = pure noise, 0 = clean) and the
result is fed back as the condition for the next frame. The viewer exposes:

- **Schedule τ(f)** — `pulses` (default): clean generation with short pulses back toward noise at given
  frames (`150, 420, 700`) or evenly spaced; each rises to `peak τ` over `attack` frames and smoothly falls
  back over `decay` frames. `ramp`: τ starts at `ramp max τ`, falls to base by `ramp in end` (fraction of
  the clip; 0 = no opening), stays clean, and from `ramp out start` grows back to the max at the last frame
  (the opening mirrors the ending) along a curve (linear / ease in / ease out / ease in-out / exponential,
  `steepness`). `dip`: noise with one recognition window (high / low / center / width / sharpness)
- **Start clip** — any LaFAN1 clip in `data/lafan1` (default `dance1_subject1`) and frame
- **Generation** — seed, start pose, feed-back state (`xt` / `x0` / mix / `final`), noise scale & temporal coherence, root source, clip
- **Display** — `joints from`: `position` (joint-position channels; spheres scatter freely) or
  `rotation` (FK from 6D local rotations; bone lengths kept, skeleton twists); xt / x0 / ghost.
  Joints and bones are flat, unlit white (#ffffff) screen-facing circles on a black background (no shadows);
  `overlay t=0` also runs each frame's denoising to the end with the same noise and draws that skeleton in
  white while the current-t skeleton turns yellow (generation ~3x slower while on);
  joints / bones can be hidden; `reduced skeleton` (default on) drops Spine, Spine1, Spine2 and both
  shoulders: one torso bone Hips–Neck and the arms attached at the Neck; joint size, bone width, grid and background colour are adjustable
- **Camera** — tracking (default joint: RightHand) subtracts the tracked joint from every point before it
  is sent, so that joint sits exactly at the view centre in the same update as the new pose; the camera
  itself stays put. `follow height` can freeze vertical tracking. Orbit / zoom / pan in the viewport work
  normally and are mirrored into the azimuth / elevation / distance sliders; `reset view` restores defaults.
  `orthographic` removes perspective (telephoto approximation: 0.5° fov from far away with the same
  framing, since viser has no orthographic camera). Export renders from the viewer's current camera
- **Playback** — `space` toggles play / pause
- **Export MP4** — renders 1080×1350 @ 30 fps into `outputs/` with a JSON of all parameters

Panel state of both viewers is saved on every change to `.cache/gui_state/exp{2,3}.json` and restored at
start-up (`reset panel to defaults` forgets it).

## Layout

```
exp2_viewer.py        viser app for experiment 2
src/amdm_runtime.py   model loading, partial denoising per frame, decoding (position / rotation)
src/schedule.py       τ(f) schedules
exp3_gravity.py       viser app for experiment 3
src/gravity.py        two-dancer generation with joint-to-joint gravity guidance
src/viewer_common.py  shared viser pieces (camera rig, space key, dot drawing, curve plot)
bvh_viewer.py         viser BVH playback (port 8081)
src/bvh_simple.py     minimal BVH reader + forward kinematics
exp6_live.py          viser app for experiment 6 (real-time)
src/live.py           endless A-MDM generation in a child process
exp5_dip.py           viser app for experiment 5 (DiP)
src/dip_runtime.py    DiP loader / planner without the CLoSD stack
src/dip_gen.py        text-timeline generation with receding-horizon tau schedules
src/shims/clip        placeholder for openai-clip
third_party/CLoSD     CLoSD (submodule)
src/camera.py         joint-tracking camera
scripts/setup.sh      environment + checkpoint setup
third_party/AMDM      A-MDM (submodule)
```
