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
- **Generation** — seed, start pose, feed-back state (`xt` / `x0` / mix / `final`), noise scale & temporal coherence, root source, clip
- **Display** — `joints from`: `position` (joint-position channels; spheres scatter freely) or
  `rotation` (FK from 6D local rotations; bone lengths kept, skeleton twists); xt / x0 / ghost.
  Joints and bones are flat, unlit white (#ffffff) screen-facing circles on a black background (no shadows);
  `overlay t=0` also runs each frame's denoising to the end with the same noise and draws that skeleton in
  white while the current-t skeleton turns yellow (generation ~3x slower while on);
  joints / bones can be hidden; `reduced skeleton` (default on) drops Spine, Spine1 and both
  shoulders: one torso bone Hips–Spine2 and the arms attached at the Neck; joint size, bone width, grid and background colour are adjustable
- **Mesh** — `render`: skeleton / mesh / mesh + skeleton. The body is the Quaternius *Universal Base
  Characters* (CC0, male / female, `assets/characters/`), scaled to the LaFAN1 hip height and driven from
  the 22 joint positions (`src/character.py`: framed pelvis / chest, aim-swing for limbs, rigid
  fingers). `mesh stretch`: `proportional` (segments scale with joint distance / rest length — clean
  poses keep the mesh's proportions, noise stretches it), `exact` (bones on the joints), `rigid`.
  Materials: `flat` (single unshaded colour: the body is skinned on the CPU and its surface drawn as
  dense unlit dots, because viser tone-maps every lit material), toon3, standard. With the t=0 overlay the current-t body is yellow
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
src/camera.py         joint-tracking camera
src/character.py      glTF skinned character loader + LaFAN1 -> UE-rig retargeting
assets/characters     Quaternius Universal Base Characters (CC0) geometry
scripts/setup.sh      environment + checkpoint setup
third_party/AMDM      A-MDM (submodule)
```
