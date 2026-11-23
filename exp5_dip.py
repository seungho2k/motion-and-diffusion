"""Experiment 5: text-timeline motion with DiP + diffusion-level schedule (pulses / ramp / dip).

    uv run python exp5_dip.py               # then open http://localhost:8080

Same idea as experiment 2, but the prior is DiP (CLoSD's autoregressive text-to-motion
diffusion, HumanML3D, 20 fps), so the motion follows a timeline of text prompts.
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
import torch
import viser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dip_gen import DipGenParams, generate  # noqa: E402
from dip_runtime import JOINT_NAMES, LINKS, DiPRuntime  # noqa: E402
from schedule import DipSchedule, PulseSchedule, RampSchedule  # noqa: E402
from viewer_common import (CameraRig, CurvePlot, PersistentGui, add_dots, add_space_toggle,  # noqa: E402
                           line_dots, setup_scene)

OUT_DIR = os.path.join(ROOT, "outputs")
VIDEO_W, VIDEO_H, FPS = 1080, 1350, 30
HIDDEN_WHEN_REDUCED = ("Spine1", "Spine2", "LeftCollar", "RightCollar")
DEFAULT_TIMELINE = """0: a person walks forward slowly and gracefully
6: a person spins around slowly with both arms raised
14: a person dances like a ballerina, stretching one leg back
22: a person bows slowly"""


def main():
    torch.set_num_threads(4)
    rt = DiPRuntime()
    T, names, J = rt.T, JOINT_NAMES, len(JOINT_NAMES)
    server = viser.ViserServer(port=8080)
    grid = setup_scene(server)
    state = {"res": None, "frame": 0, "busy": False}

    # ------------------------------------------------------------------ GUI
    with server.gui.add_folder("Text timeline"):
        g_text = server.gui.add_text("prompts", DEFAULT_TIMELINE, multiline=True,
                                     hint="One line per segment: 'seconds: prompt'.")
        g_secs = server.gui.add_slider("seconds", 2.0, 90.0, 1.0, 30.0)
        g_guid = server.gui.add_slider("text guidance", 1.0, 12.0, 0.5, 7.5)
    with server.gui.add_folder("Schedule  τ(f)"):
        g_stype = server.gui.add_dropdown("type", ("pulses", "ramp", "dip", "none"), initial_value="pulses")
        g_pframes = server.gui.add_text("pulse frames", "", hint="Video frames (30 fps), e.g. 150, 420. Empty = evenly spaced.")
        g_pcount = server.gui.add_slider("pulse count", 0, 20, 1, 3)
        g_ppeak = server.gui.add_slider("pulse peak τ", 0.5, float(T), 0.5, 8.0)
        g_pattack = server.gui.add_slider("pulse attack (fr)", 0, 20, 1, 2)
        g_pdecay = server.gui.add_slider("pulse decay (fr)", 1, 90, 1, 6)
        g_rfade = server.gui.add_slider("ramp in end", 0.0, 0.99, 0.01, 0.2)
        g_rstart = server.gui.add_slider("ramp out start", 0.0, 0.99, 0.01, 0.7)
        g_rpeak = server.gui.add_slider("ramp max τ", 0.5, float(T), 0.5, float(T))
        g_rcurve = server.gui.add_dropdown("ramp curve", ("linear", "ease in", "ease out", "ease in-out", "exponential"),
                                           initial_value="ease in")
        g_rpower = server.gui.add_slider("ramp steepness", 0.2, 10.0, 0.1, 2.0)
        g_high = server.gui.add_slider("dip high τ", 0.0, float(T), 0.1, float(T))
        g_low = server.gui.add_slider("dip low τ", 0.0, float(T), 0.1, 0.0)
        g_center = server.gui.add_slider("dip center", 0.0, 1.0, 0.01, 0.5)
        g_width = server.gui.add_slider("dip half width (fr)", 2.0, 2700.0, 1.0, 300.0)
        g_sharp = server.gui.add_slider("dip sharpness", 0.5, 12.0, 0.1, 2.0)
        g_base = server.gui.add_slider("base τ", 0.0, float(T), 0.1, 0.0)
    with server.gui.add_folder("Generation"):
        g_seed = server.gui.add_number("seed", 0, step=1)
        g_commit = server.gui.add_slider("commit frames", 1, 40, 1, 4,
                                         hint="Frames (20 fps) kept from each 40-frame plan before replanning. "
                                              "Smaller = finer schedule resolution, slower.")
        g_feed = server.gui.add_dropdown("feed back", ("xt", "x0", "final"), initial_value="xt")
        g_root = server.gui.add_dropdown("root from", ("display", "final"), initial_value="display")
        g_nscale = server.gui.add_slider("noise scale", 0.0, 2.0, 0.05, 1.0)
        g_gen = server.gui.add_button("Generate", color="blue")
        g_auto = server.gui.add_checkbox("auto regenerate", False)
    with server.gui.add_folder("Display"):
        g_final = server.gui.add_checkbox("overlay t=0", False)
        g_joints = server.gui.add_checkbox("joints", True)
        g_reduced = server.gui.add_checkbox("reduced skeleton", True)
        g_bones = server.gui.add_checkbox("bones", True)
        g_jsize = server.gui.add_slider("joint size", 0.005, 0.2, 0.005, 0.04)
        g_bwidth = server.gui.add_slider("bone width", 0.002, 0.05, 0.001, 0.008)
        g_grid = server.gui.add_checkbox("grid", True)
        g_bg = server.gui.add_rgb("background", (0, 0, 0))
    with server.gui.add_folder("Camera"):
        g_track = server.gui.add_dropdown("track", ("off",) + tuple(names), initial_value="Pelvis")
        g_follow_y = server.gui.add_checkbox("follow height", False)
        cam = CameraRig(server, dist=3.5)
    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("playing", True, hint="space: play / pause")
        g_frame = server.gui.add_slider("frame", 0, 899, 1, 0)
        g_info = server.gui.add_markdown("")
        plot = CurvePlot(server, color=(255, 214, 0))
        g_export = server.gui.add_button("Export MP4 (1080x1350)", color="green")
        g_resetgui = server.gui.add_button("reset panel to defaults")
    add_space_toggle(server)

    persisted = {k: v for k, v in locals().items()
                 if k.startswith("g_") and k not in ("g_frame", "g_play")
                 and type(v).__name__ in ("GuiSliderHandle", "GuiCheckboxHandle", "GuiDropdownHandle",
                                          "GuiNumberHandle", "GuiRgbHandle", "GuiTextHandle")}
    persisted.update(cam_az=cam.az, cam_el=cam.el, cam_dist=cam.dist, cam_fov=cam.fov, cam_ortho=cam.ortho)
    pstate = PersistentGui(os.path.join(ROOT, ".cache/gui_state/exp5.json"), persisted)
    pstate.restore()
    pstate.start_saving()
    g_resetgui.on_click(lambda _: (pstate.reset(), setattr(g_info, "content", "panel state cleared; restart to apply")))

    groups = {"pulses": (g_pframes, g_pcount, g_ppeak, g_pattack, g_pdecay),
              "ramp": (g_rfade, g_rstart, g_rpeak, g_rcurve, g_rpower),
              "dip": (g_high, g_low, g_center, g_width, g_sharp)}

    def sched_visibility(_=None):
        for k, hs in groups.items():
            for h in hs:
                h.visible = g_stype.value == k
        g_base.visible = g_stype.value in ("pulses", "ramp", "none")
    g_stype.on_update(sched_visibility)
    sched_visibility()

    # -------------------------------------------------------------- drawing
    hide = [names.index(n) for n in HIDDEN_WHEN_REDUCED]
    keep = np.array([i for i in range(J) if i not in hide])
    links_red = np.concatenate([np.array([l for l in LINKS if l[0] not in hide and l[1] not in hide]),
                                np.array([(names.index(a), names.index(b)) for a, b in
                                          (("Pelvis", "Spine3"), ("Neck", "LeftShoulder"),
                                           ("Neck", "RightShoulder"))])])

    class Skel:
        def __init__(self, name, color):
            self.j = add_dots(server, f"/{name}/joints", color, g_jsize.value)
            self.b = add_dots(server, f"/{name}/bones", color, g_bwidth.value)

        def style(self):
            self.j.point_size = g_jsize.value
            self.b.point_size = g_bwidth.value

        def show(self, pts):
            on = pts is not None
            self.j.visible = on and g_joints.value
            self.b.visible = on and g_bones.value
            if self.j.visible:
                self.j.points = (pts[keep] if g_reduced.value else pts).astype(np.float32)
            if self.b.visible:
                self.b.points = line_dots(pts, links_red if g_reduced.value else LINKS, g_bwidth.value)

    sk_final = Skel("final", (255, 255, 255))
    sk_main = Skel("main", (255, 255, 255))
    sk_main_y = Skel("main_y", (255, 214, 0))

    def apply_style(_=None):
        for s in (sk_final, sk_main, sk_main_y):
            s.style()
        grid.visible = g_grid.value
        server.scene.set_background_image(np.full((4, 4, 3), g_bg.value, np.uint8))
        show_frame(state["frame"])

    def anchor(res, i):
        if g_track.value == "off":
            return np.zeros(3)
        k = names.index(g_track.value)
        a = res.joints[i, k].astype(np.float64).copy()
        if not g_follow_y.value:
            a[1] = float(np.median(res.joints[:, k, 1]))
        return a

    def show_frame(i: int):
        res = state["res"]
        if res is None:
            return
        n = len(res.levels)
        i = int(np.clip(i, 0, n - 1))
        a = anchor(res, i)
        overlay = g_final.value
        with server.atomic():
            grid.position = -a
            sk_main.show(None if overlay else res.joints[i] - a)
            sk_main_y.show(res.joints[i] - a if overlay else None)
            sk_final.show(res.joints_final[i] - a if overlay else None)
        g_info.content = (f"frame **{i}** / {n - 1} &nbsp; τ = **{res.levels[i]:.1f}** / {T}<br>"
                          f"*{res.prompts[i]}*")

    # -------------------------------------------------------------- generate
    def schedule():
        n = int(g_secs.value * FPS)
        if g_stype.value == "pulses":
            s = PulseSchedule(frames=g_pframes.value, count=int(g_pcount.value), peak=g_ppeak.value,
                              attack=int(g_pattack.value), decay=int(g_pdecay.value), base=g_base.value)
        elif g_stype.value == "ramp":
            s = RampSchedule(fade_in=g_rfade.value, start=g_rstart.value, peak=g_rpeak.value, base=g_base.value,
                             curve=g_rcurve.value, power=g_rpower.value)
        elif g_stype.value == "dip":
            s = DipSchedule(high=g_high.value, low=g_low.value, center=g_center.value,
                            half_width=g_width.value, sharpness=g_sharp.value)
        else:
            return None, np.full(n, g_base.value)
        return s, np.clip(s.levels(n), 0, T)

    def params() -> DipGenParams:
        return DipGenParams(timeline=g_text.value, seconds=g_secs.value, seed=int(g_seed.value),
                            commit=int(g_commit.value), guidance=g_guid.value, feed=g_feed.value,
                            root_source=g_root.value, noise_scale=g_nscale.value)

    def regenerate(_=None):
        if state["busy"]:
            return
        state["busy"] = True
        g_gen.disabled = True

        def work():
            try:
                t0 = time.time()
                _, levels = schedule()
                res = generate(rt, params(), levels,
                               progress=lambda i, n: setattr(g_info, "content", f"generating {i}/{n} ..."))
                state["res"] = res
                g_frame.max = len(res.levels) - 1
                plot.set_values(res.levels, vmax=float(T))
                print(f"generated {len(res.levels)} frames in {time.time() - t0:.1f}s", flush=True)
                show_frame(state["frame"])
            except Exception as e:
                g_info.content = f"generation failed: {e}"
                print("generation failed:", repr(e), flush=True)
            finally:
                state["busy"] = False
                g_gen.disabled = False
        threading.Thread(target=work, daemon=True).start()

    g_gen.on_click(regenerate)
    for h in (g_text, g_secs, g_guid, g_stype, g_pframes, g_pcount, g_ppeak, g_pattack, g_pdecay, g_rfade,
              g_rstart, g_rpeak, g_rcurve, g_rpower, g_high, g_low, g_center, g_width, g_sharp, g_base,
              g_seed, g_commit, g_feed, g_root, g_nscale):
        h.on_update(lambda _: g_auto.value and regenerate())
    for h in (g_final, g_joints, g_reduced, g_bones, g_track, g_follow_y):
        h.on_update(lambda _: show_frame(state["frame"]))
    for h in (g_jsize, g_bwidth, g_grid, g_bg):
        h.on_update(apply_style)

    def on_frame(_):
        if not g_play.value:
            state["frame"] = int(g_frame.value)
            show_frame(state["frame"])
            plot.set_cursor(state["frame"])
    g_frame.on_update(on_frame)

    # -------------------------------------------------------------- export
    def export(event: viser.GuiEvent):
        client, res = event.client, state["res"]
        if client is None or res is None:
            return
        g_play.value = False
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"exp5_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        sched, _ = schedule()
        meta = {"gen": asdict(params()), "schedule_type": g_stype.value,
                "schedule": asdict(sched) if sched is not None else {"base": g_base.value},
                "camera": cam.meta(client), "track": g_track.value, "overlay_t0": g_final.value}
        with open(path.replace(".mp4", ".json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        pos, wxyz, fov = np.array(client.camera.position), np.array(client.camera.wxyz), float(client.camera.fov)
        g_export.disabled = True
        writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        try:
            n = len(res.levels)
            for i in range(n):
                show_frame(i)
                img = client.get_render(VIDEO_H, VIDEO_W, position=pos, wxyz=wxyz, fov=fov, transport_format="png")
                if img.shape[-1] == 4:
                    al = img[..., 3:4].astype(np.float32) / 255.0
                    img = img[..., :3] * al + np.asarray(g_bg.value, np.float32) * (1 - al)
                writer.append_data(img[..., :3].astype(np.uint8))
                if i % 30 == 0:
                    g_info.content = f"exporting {i}/{n} ..."
            g_info.content = f"saved `{path}`"
            print("saved", path, flush=True)
        except Exception as e:
            g_info.content = f"export failed: {e}"
        finally:
            writer.close()
            g_export.disabled = False
    g_export.on_click(export)

    # -------------------------------------------------------------- loop
    apply_style()
    regenerate()
    print("open http://localhost:8080", flush=True)
    while True:
        t0 = time.time()
        res = state["res"]
        if g_play.value and res is not None and not g_export.disabled:
            state["frame"] = (state["frame"] + 1) % len(res.levels)
            show_frame(state["frame"])
            g_frame.value = state["frame"]
            if state["frame"] % 6 == 0:
                plot.set_cursor(state["frame"])
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))


if __name__ == "__main__":
    main()
