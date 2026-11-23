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

## Experiment 3 — two dancers and a gravity between their joints

```bash
uv run python exp3_gravity.py    # open http://localhost:8080
```

Two independent A-MDM rollouts (LaFAN1 prior, started from dance clips) are coupled only by a softened
inverse-square pull applied to the model's clean-frame prediction during denoising (`src/gravity.py`).
Pulses of pull drag the bodies together; the last `guide stop step` diffusion steps run prior-only, so each
pose stays plausible while the trajectory becomes impossible. Fingers (absent in LaFAN1) are procedural
damped spring chains with a soft ballet-hand rest shape (`src/fingers.py`), lagging behind the wrist and
pulled toward the partner too.

- **Dancers** — start clip / frame for A and B, separation, facing
- **Gravity** — which joints pull (hands → partner, arms → partner, hands → head/chest, all ↔ all),
  pulse strength / count / width, constant pull, softening, repulsion radius, guide stop step, root pull
- **Fingers** — stiffness, damping, gravity on fingers, curl
- **Display / Camera / Playback / Export** — as in experiment 2; track the pair centre or any joint of A / B

## Experiment 2 — noise → human → noise

```bash
uv run python exp2_viewer.py     # open http://localhost:8080
```

Each frame is denoised only down to a level τ(f) (15 = pure noise, 0 = clean) and the
result is fed back as the condition for the next frame. The viewer exposes:

- **Schedule τ(f)** — dip curve (high / low / center / width / sharpness)
- **Generation** — seed, start pose, feed-back state (`xt` / `x0` / mix / `final`), noise scale & temporal coherence, root source, clip
- **Display** — `joints from`: `position` (joint-position channels; spheres scatter freely) or
  `rotation` (FK from 6D local rotations; bone lengths kept, skeleton twists); xt / x0 / ghost.
  Joints and bones are flat, unlit white (#ffffff) screen-facing circles on a black background (no shadows);
  `overlay t=0` also runs each frame's denoising to the end with the same noise and draws that skeleton in
  white while the current-t skeleton turns yellow (generation ~3x slower while on);
  joints / bones can be hidden; `reduced skeleton` (default on) drops Spine, Spine1 and both
  shoulders: one torso bone Hips–Spine2 and the arms attached at the Neck; joint size, bone width, grid and background colour are adjustable
- **Camera** — tracking (default joint: RightHand) subtracts the tracked joint from every point before it
  is sent, so that joint sits exactly at the view centre in the same update as the new pose; the camera
  itself stays put. `follow height` can freeze vertical tracking. Orbit / zoom / pan in the viewport work
  normally and are mirrored into the azimuth / elevation / distance sliders; `reset view` restores defaults.
  `orthographic` removes perspective (telephoto approximation: 0.5° fov from far away with the same
  framing, since viser has no orthographic camera). Export renders from the viewer's current camera
- **Playback** — `space` toggles play / pause
- **Export MP4** — renders 1080×1350 @ 30 fps into `outputs/` with a JSON of all parameters

## Layout

```
exp2_viewer.py        viser app for experiment 2
src/amdm_runtime.py   model loading, partial denoising per frame, decoding (position / rotation)
src/schedule.py       τ(f) schedules
exp3_gravity.py       viser app for experiment 3
src/gravity.py        two-dancer generation with joint-to-joint gravity guidance
src/fingers.py        procedural spring fingers
src/viewer_common.py  shared viser pieces (camera rig, space key, dot drawing, curve plot)
src/camera.py         joint-tracking camera
scripts/setup.sh      environment + checkpoint setup
third_party/AMDM      A-MDM (submodule)
```
