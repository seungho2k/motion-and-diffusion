"""Experiment 3: two dancers pulled together by a gravity between their joints.

    uv run python exp3_gravity.py           # then open http://localhost:8080

Each dancer is an independent A-MDM rollout (LaFAN1 prior); the only coupling is a
softened inverse-square pull applied to the model's clean-frame prediction during
denoising (see src/gravity.py).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
import time
from dataclasses import asdict

import imageio.v2 as imageio
import numpy as np
import viser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from amdm_runtime import AMDMRuntime  # noqa: E402
from gravity import PAIR_PRESETS, GravityParams, PairGenerator  # noqa: E402
from viewer_common import CameraRig, CurvePlot, PersistentGui, add_dots, add_space_toggle, line_dots, setup_scene  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
VIDEO_W, VIDEO_H, FPS = 1080, 1350, 30
HIDDEN_WHEN_REDUCED = ("Spine", "Spine1", "Spine2", "LeftShoulder", "RightShoulder")


def main():
    rt = AMDMRuntime("lafan1")
    pg = PairGenerator(rt)
    names = rt.joint_names
    J = rt.J
    clips = pg.dance_clips()
    if not clips:
        sys.exit("LaFAN1 BVHs not found in data/lafan1 (see README).")
    dance = [c for c in clips if c.startswith("dance")]

    server = viser.ViserServer(port=8080)
    grid = setup_scene(server)
    state = {"res": None, "frame": 0, "playing": True, "busy": False}
    lock = threading.Lock()
    d = GravityParams()

    # ------------------------------------------------------------------ GUI
    with server.gui.add_folder("Dancers"):
        g_frames = server.gui.add_slider("frames", 60, 1800, 30, d.num_frames)
        g_seed = server.gui.add_number("seed", d.seed, step=1)
        g_clip_a = server.gui.add_dropdown("A start clip", tuple(clips), initial_value=d.clip_a)
        g_frame_a = server.gui.add_slider("A start frame", 0, 6000, 10, d.frame_a)
        g_clip_b = server.gui.add_dropdown("B start clip", tuple(clips), initial_value=d.clip_b)
        g_frame_b = server.gui.add_slider("B start frame", 0, 6000, 10, d.frame_b)
        g_sep = server.gui.add_slider("separation (m)", 0.3, 5.0, 0.05, d.separation)
        g_face = server.gui.add_checkbox("face each other", d.face_each_other)
    with server.gui.add_folder("Gravity"):
        g_preset = server.gui.add_dropdown("pull", tuple(PAIR_PRESETS), initial_value=d.preset)
        g_strength = server.gui.add_slider("pulse strength", 0.0, 3.0, 0.01, d.strength,
                                           hint="~metres per frame a joint is dragged at 1 m distance.")
        g_base = server.gui.add_slider("constant pull", 0.0, 1.0, 0.01, d.base)
        g_pulses = server.gui.add_slider("pulses", 0, 12, 1, d.pulses)
        g_pwidth = server.gui.add_slider("pulse width (fr)", 2.0, 200.0, 1.0, d.pulse_width)
        g_soft = server.gui.add_slider("softening (m)", 0.02, 1.0, 0.01, d.softening)
        g_repulse = server.gui.add_slider("repulse radius (m)", 0.0, 0.6, 0.01, d.repulse_radius)
        g_stop = server.gui.add_slider(
            "guide stop step", 0, rt.T - 1, 1, 2,
            hint="Pull is applied for diffusion steps t >= this; the remaining steps are prior-only "
                 "(higher = more natural poses, weaker pull).")
        g_momentum = server.gui.add_slider(
            "momentum", 0.0, 0.98, 0.01, d.momentum,
            hint="Pull accumulates into velocity: a pulse keeps dragging afterwards (decay per frame).")
        g_bodypull = server.gui.add_slider(
            "body pull", 0.0, 1.0, 0.05, d.body_pull,
            hint="Share of the pull that drags the whole body rigidly (outside the prior). "
                 "The rest reaches with the limbs through diffusion guidance.")
        g_lift = server.gui.add_checkbox("lift off floor", d.lift)
        g_gen = server.gui.add_button("Generate", color="blue")
        g_auto = server.gui.add_checkbox("auto regenerate", False)
    with server.gui.add_folder("Display"):
        g_joints = server.gui.add_checkbox("joints", True)
        g_reduced = server.gui.add_checkbox("reduced skeleton", True)
        g_bones = server.gui.add_checkbox("bones", True)
        g_jsize = server.gui.add_slider("joint size", 0.005, 0.2, 0.005, 0.04)
        g_bwidth = server.gui.add_slider("bone width", 0.002, 0.05, 0.001, 0.008)
        g_col_a = server.gui.add_rgb("color A", (255, 255, 255))
        g_col_b = server.gui.add_rgb("color B", (255, 255, 255))
        g_grid = server.gui.add_checkbox("grid", True)
        g_bg = server.gui.add_rgb("background", (0, 0, 0))
    with server.gui.add_folder("Camera"):
        track_opts = ("hips midpoint", "off") + tuple(f"A: {n}" for n in names) + tuple(f"B: {n}" for n in names)
        g_track = server.gui.add_dropdown("track", track_opts, initial_value="hips midpoint",
                                          hint="hips midpoint: centre between the two dancers' Hips joints.")
        g_follow_y = server.gui.add_checkbox("follow height", False)
        cam = CameraRig(server, dist=5.0)
    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("playing", True, hint="space: play / pause")
        g_frame = server.gui.add_slider("frame", 0, d.num_frames - 1, 1, 0)
        g_info = server.gui.add_markdown("")
        g_export = server.gui.add_button("Export MP4 (1080x1350)", color="green")
        plot = CurvePlot(server, color=(255, 214, 0))
        g_resetgui = server.gui.add_button("reset panel to defaults",
                                           hint="Forget the saved panel state (restart to apply).")
    add_space_toggle(server)

    # panel state is saved on every change and restored at start-up
    persisted = {k: v for k, v in locals().items()
                 if k.startswith("g_") and k not in ("g_frame", "g_play")
                 and type(v).__name__ in ("GuiSliderHandle", "GuiCheckboxHandle", "GuiDropdownHandle",
                                          "GuiNumberHandle", "GuiRgbHandle")}
    persisted.update(cam_az=cam.az, cam_el=cam.el, cam_dist=cam.dist, cam_fov=cam.fov, cam_ortho=cam.ortho)
    pstate = PersistentGui(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache/gui_state/exp3.json"),
                           persisted)
    pstate.restore()
    pstate.start_saving()
    g_resetgui.on_click(lambda _: (pstate.reset(), setattr(g_info, "content", "saved panel state cleared; restart to apply")))

    # -------------------------------------------------------------- skeletons
    hide = [names.index(n) for n in HIDDEN_WHEN_REDUCED]
    keep = np.array([i for i in range(J) if i not in hide])
    bridge = [(names.index(a), names.index(b)) for a, b in
              (("Hips", "Neck"), ("Neck", "LeftArm"), ("Neck", "RightArm"))]
    body_links_full = np.asarray(rt.links)
    body_links_red = np.concatenate([
        np.array([l for l in body_links_full if l[0] not in hide and l[1] not in hide]), np.array(bridge)])

    class Dancer:
        def __init__(self, name):
            self.joints = add_dots(server, f"/{name}/joints", (255, 255, 255), g_jsize.value)
            self.bones = add_dots(server, f"/{name}/bones", (255, 255, 255), g_bwidth.value)

        def style(self, color):
            for h in (self.joints, self.bones):
                h.colors = np.asarray(color, np.uint8)[None]
            self.joints.point_size = g_jsize.value
            self.bones.point_size = g_bwidth.value

        def show(self, body: np.ndarray):
            self.joints.visible = g_joints.value
            if self.joints.visible:
                self.joints.points = (body[keep] if g_reduced.value else body).astype(np.float32)
            self.bones.visible = g_bones.value
            if self.bones.visible:
                self.bones.points = line_dots(body, body_links_red if g_reduced.value else body_links_full,
                                              g_bwidth.value)

    dancers = [Dancer("A"), Dancer("B")]

    def apply_style(_=None):
        dancers[0].style(g_col_a.value)
        dancers[1].style(g_col_b.value)
        grid.visible = g_grid.value
        server.scene.set_background_image(np.full((4, 4, 3), g_bg.value, np.uint8))
        show_frame(state["frame"])

    # -------------------------------------------------------------- frames
    def anchor(res, i) -> np.ndarray:
        opt = g_track.value
        j = res.joints
        if opt == "off":
            return np.zeros(3)
        if opt == "hips midpoint":
            a = 0.5 * (j[0, i, 0] + j[1, i, 0])
            y0 = 0.5 * (j[0, 0, 0, 1] + j[1, 0, 0, 1])
        else:
            who, jn = opt.split(": ")
            k = names.index(jn)
            b = 0 if who == "A" else 1
            a, y0 = j[b, i, k], j[b, 0, k, 1]
        a = a.astype(np.float64).copy()
        if not g_follow_y.value:
            a[1] = y0
        return a

    def show_frame(i: int):
        res = state["res"]
        if res is None:
            return
        n = res.joints.shape[1]
        i = int(np.clip(i, 0, n - 1))
        a = anchor(res, i)
        with server.atomic():
            grid.position = -a
            for b in range(2):
                dancers[b].show(res.joints[b, i] - a)
        g_info.content = f"frame **{i}** / {n - 1} &nbsp; pull = **{res.gravity[i]:.2f}**"

    # -------------------------------------------------------------- generate
    def params() -> GravityParams:
        return GravityParams(
            num_frames=int(g_frames.value), seed=int(g_seed.value),
            clip_a=g_clip_a.value, clip_b=g_clip_b.value,
            frame_a=int(g_frame_a.value), frame_b=int(g_frame_b.value),
            separation=g_sep.value, face_each_other=g_face.value, preset=g_preset.value,
            strength=g_strength.value, base=g_base.value, pulses=int(g_pulses.value),
            pulse_width=g_pwidth.value, softening=g_soft.value, repulse_radius=g_repulse.value,
            guide_stop=int(g_stop.value), momentum=g_momentum.value, body_pull=g_bodypull.value,
            lift=g_lift.value)

    def update_plot(res):
        plot.set_values(res.gravity, vmax=max(1e-3, float(res.gravity.max())) * 1.1)

    def regenerate(_=None):
        if state["busy"]:
            return
        state["busy"] = True
        g_gen.disabled = True

        def work():
            try:
                t0 = time.time()
                p = params()
                res = pg.generate(p, progress=lambda i, n: setattr(
                    g_info, "content", f"generating {i}/{n} ..."))
                with lock:
                    state["res"] = res
                    g_frame.max = res.joints.shape[1] - 1
                update_plot(res)
                print(f"generated {p.num_frames} frames x2 in {time.time() - t0:.1f}s", flush=True)
                show_frame(state["frame"])
            except Exception as e:  # surface errors in the panel
                g_info.content = f"generation failed: {e}"
                print("generation failed:", repr(e), flush=True)
            finally:
                state["busy"] = False
                g_gen.disabled = False
        threading.Thread(target=work, daemon=True).start()

    g_gen.on_click(regenerate)
    for h in (g_frames, g_seed, g_clip_a, g_frame_a, g_clip_b, g_frame_b, g_sep, g_face, g_preset,
              g_strength, g_base, g_pulses, g_pwidth, g_soft, g_repulse, g_stop, g_momentum, g_bodypull,
              g_lift):
        h.on_update(lambda _: g_auto.value and regenerate())

    for h in (g_joints, g_reduced, g_bones, g_track, g_follow_y):
        h.on_update(lambda _: show_frame(state["frame"]))
    for h in (g_jsize, g_bwidth, g_col_a, g_col_b, g_grid, g_bg):
        h.on_update(apply_style)

    def on_frame(_):
        if not state["playing"]:
            state["frame"] = int(g_frame.value)
            show_frame(state["frame"])
            plot.set_cursor(state["frame"])
    g_frame.on_update(on_frame)
    g_play.on_update(lambda _: state.update(playing=g_play.value))

    # -------------------------------------------------------------- export
    def export(event: viser.GuiEvent):
        client, res = event.client, state["res"]
        if client is None or res is None:
            return
        g_play.value = False
        state["playing"] = False
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"exp3_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        meta = {"gravity": asdict(params()), "camera": cam.meta(client),
                "track": g_track.value, "follow_height": g_follow_y.value,
                "display": {"joints": g_joints.value, "reduced": g_reduced.value, "bones": g_bones.value,
                            "joint_size": g_jsize.value, "bone_width": g_bwidth.value, "color_a": list(g_col_a.value),
                            "color_b": list(g_col_b.value), "grid": g_grid.value, "background": list(g_bg.value)}}
        with open(path.replace(".mp4", ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        pos, wxyz, fov = np.array(client.camera.position), np.array(client.camera.wxyz), float(client.camera.fov)
        g_export.disabled = True
        writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        try:
            n = res.joints.shape[1]
            for i in range(n):
                show_frame(i)
                img = client.get_render(VIDEO_H, VIDEO_W, position=pos, wxyz=wxyz, fov=fov,
                                        transport_format="png")
                if img.shape[-1] == 4:
                    al = img[..., 3:4].astype(np.float32) / 255.0
                    img = img[..., :3] * al + np.asarray(g_bg.value, np.float32) * (1 - al)
                writer.append_data(img[..., :3].astype(np.uint8))
                if i % 30 == 0:
                    g_info.content = f"exporting {i}/{n} ..."
        except Exception as e:
            g_info.content = f"export failed: {e}"
            print("export failed:", repr(e), flush=True)
            return
        finally:
            writer.close()
            g_export.disabled = False
        g_info.content = f"saved `{path}`"
        print("saved", path, flush=True)
    g_export.on_click(export)

    # -------------------------------------------------------------- loop
    apply_style()
    regenerate()
    print("open http://localhost:8080", flush=True)
    while True:
        t0 = time.time()
        res = state["res"]
        if state["playing"] and res is not None:
            state["frame"] = (state["frame"] + 1) % res.joints.shape[1]
            show_frame(state["frame"])
            g_frame.value = state["frame"]
            if state["frame"] % 6 == 0:
                plot.set_cursor(state["frame"])
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))


if __name__ == "__main__":
    main()
