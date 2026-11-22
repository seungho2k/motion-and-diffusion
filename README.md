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

## Experiment 2 — noise → human → noise

```bash
uv run python exp2_viewer.py     # open http://localhost:8080
```

Each frame is denoised only down to a level τ(f) (15 = pure noise, 0 = clean) and the
result is fed back as the condition for the next frame. The viewer exposes:

- **Schedule τ(f)** — dip curve (high / low / center / width / sharpness)
- **Generation** — seed, start pose, feed-back state (`xt` / `x0` / mix), noise scale & temporal coherence, root source, clip
- **Display** — `joints from`: `position` (joint-position channels; spheres scatter freely) or
  `rotation` (FK from 6D local rotations; bone lengths kept, skeleton twists); xt / x0 / ghost.
  Joints and bones are flat, unlit white (#ffffff) screen-facing circles on a black background (no shadows);
  joints / bones can be hidden; joint size, bone width, grid and background colour are adjustable
- **Camera** — tracking keeps the chosen joint at the view centre by moving the world (not the camera), so
  each frame is a single atomic scene update. `smoothing (s)` applies a critically damped follow
  (0 = rigid lock), `follow height` can freeze vertical tracking. Orbit / zoom / pan in the viewport work
  normally and are mirrored into the azimuth / elevation / distance sliders; `reset view` restores defaults.
  Export renders from the viewer's current camera
- **Playback** — `space` toggles play / pause
- **Export MP4** — renders 1080×1350 @ 30 fps into `outputs/` with a JSON of all parameters

## Layout

```
exp2_viewer.py        viser app for experiment 2
src/amdm_runtime.py   model loading, partial denoising per frame, decoding (position / rotation)
src/schedule.py       τ(f) schedules
src/camera.py         joint-tracking camera
scripts/setup.sh      environment + checkpoint setup
third_party/AMDM      A-MDM (submodule)
```
