"""Play back BVH mocap (E2-Create Motion Bank samples, LaFAN1, ...) in viser.

    uv run python bvh_viewer.py                 # http://localhost:8081
    uv run python bvh_viewer.py --port 8082 --dir path/to/bvhs
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import threading
import time

import numpy as np
import viser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from bvh_simple import load_bvh  # noqa: E402
from viewer_common import (CameraRig, PersistentGui, add_dots, add_space_toggle,  # noqa: E402
                           line_dots, setup_scene)

DEFAULT_DIRS = [os.path.join(ROOT, ".cache/e2_motionbank"), os.path.join(ROOT, "data/lafan1")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--dir", action="append", help="folder(s) searched recursively for .bvh")
    args = ap.parse_args()

    dirs = args.dir or DEFAULT_DIRS
    files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "**/*.bvh"), recursive=True))
    if not files:
        sys.exit(f"no .bvh files under {dirs}")
    # short readable labels, e.g. "E2/Music_Improvisation/Amber_Pansters/mocap_skeleton_proc"
    def label(f):
        rel = os.path.relpath(f, ROOT)
        rel = rel.replace(".cache/e2_motionbank/E2-Create_MotionBank_Dataset/", "E2/")
        rel = rel.replace("data/lafan1/", "LaFAN1/").replace("/mocap/", "/")
        return rel[:-4]
    labels = {label(f): f for f in files}
    e2_first = sorted(labels, key=lambda k: (not k.startswith("E2/"), k))

    # joint names across all files (header only), for the track dropdown
    def header_joints(f):
        out = []
        with open(f) as fh:
            for line in fh:
                t = line.split()
                if t and t[0] in ("ROOT", "JOINT"):
                    out.append(t[1])
                elif t and t[0] == "MOTION":
                    break
        return out
    all_joints = []
    for f in files:
        for n in header_joints(f):
            if n not in all_joints:
                all_joints.append(n)

    server = viser.ViserServer(port=args.port)
    grid = setup_scene(server)
    state = {"clip": None, "t": 0.0, "playing": True, "loading": False}

    with server.gui.add_folder("Clip"):
        g_file = server.gui.add_dropdown("file", tuple(e2_first), initial_value=e2_first[0])
        g_info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("playing", True, hint="space: play / pause")
        g_speed = server.gui.add_slider("speed", 0.1, 3.0, 0.05, 1.0)
        g_time = server.gui.add_slider("time (s)", 0.0, 1.0, 0.01, 0.0)
        g_loop = server.gui.add_checkbox("loop", True)
    with server.gui.add_folder("Display"):
        g_joints = server.gui.add_checkbox("joints", True)
        g_ends = server.gui.add_checkbox("end sites", False)
        g_bones = server.gui.add_checkbox("bones", True)
        g_jsize = server.gui.add_slider("joint size", 0.005, 0.2, 0.005, 0.04)
        g_bwidth = server.gui.add_slider("bone width", 0.002, 0.05, 0.001, 0.008)
        g_trail = server.gui.add_slider("trail (s)", 0.0, 5.0, 0.1, 0.0,
                                        hint="Show the root and hand paths over the last N seconds.")
        g_grid = server.gui.add_checkbox("grid", True)
        g_bg = server.gui.add_rgb("background", (0, 0, 0))
    with server.gui.add_folder("Camera"):
        g_track = server.gui.add_dropdown(
            "track", ("off",) + tuple(all_joints), initial_value="Hips" if "Hips" in all_joints else "off",
            hint="Joint kept at the view centre (falls back to the root if this clip lacks it).")
        g_follow_y = server.gui.add_checkbox("follow height", False)
        cam = CameraRig(server, dist=4.0)
    add_space_toggle(server)

    persisted = {k: v for k, v in locals().items()
                 if k.startswith("g_") and k not in ("g_time", "g_play")
                 and type(v).__name__ in ("GuiSliderHandle", "GuiCheckboxHandle", "GuiDropdownHandle",
                                          "GuiNumberHandle", "GuiRgbHandle")}
    persisted.update(cam_az=cam.az, cam_el=cam.el, cam_dist=cam.dist, cam_fov=cam.fov, cam_ortho=cam.ortho)
    pstate = PersistentGui(os.path.join(ROOT, ".cache/gui_state/bvh_viewer.json"), persisted)
    pstate.restore()
    pstate.start_saving()

    joints_h = add_dots(server, "/body/joints", (255, 255, 255), g_jsize.value)
    bones_h = add_dots(server, "/body/bones", (255, 255, 255), g_bwidth.value)
    trail_h = add_dots(server, "/body/trail", (255, 214, 0), 0.01)

    def load(_=None):
        if state["loading"]:
            return
        state["loading"] = True
        g_info.content = "loading ..."

        def work():
            try:
                t0 = time.time()
                b = load_bvh(labels[g_file.value])
                scale = 0.01 if np.abs(b.positions[..., 1]).max() > 20 else 1.0   # cm -> m
                b.positions = (b.positions * scale).astype(np.float32)
                state["clip"] = b
                state["y_med"] = np.median(b.positions[:, :, 1], axis=0)   # per-joint typical height
                state["t"] = 0.0
                dur = len(b.positions) / b.fps
                g_time.max = max(dur, 0.01)
                hips_y = b.positions[:, 0, 1]
                g_info.content = (f"**{len(b.names) - int(b.is_end.sum())} joints**, {len(b.positions)} frames @ "
                                  f"{b.fps:.0f} fps = {dur / 60:.1f} min<br>"
                                  f"hips height p1/p50/p99: {np.percentile(hips_y, 1):.2f} / "
                                  f"{np.percentile(hips_y, 50):.2f} / {np.percentile(hips_y, 99):.2f} m "
                                  f"(loaded in {time.time() - t0:.1f}s)")
                print(f"loaded {g_file.value}: {len(b.positions)} frames", flush=True)
            except Exception as e:
                g_info.content = f"load failed: {e}"
                print("load failed:", repr(e), flush=True)
            finally:
                state["loading"] = False
        threading.Thread(target=work, daemon=True).start()

    def apply_style(_=None):
        joints_h.point_size = g_jsize.value
        bones_h.point_size = g_bwidth.value
        grid.visible = g_grid.value
        server.scene.set_background_image(np.full((4, 4, 3), g_bg.value, np.uint8))

    def show(t: float):
        b = state["clip"]
        if b is None:
            return
        n = len(b.positions)
        i = int(np.clip(round(t * b.fps), 0, n - 1))
        P = b.positions[i]
        a = np.zeros(3, np.float32)
        if g_track.value != "off":
            j = b.names.index(g_track.value) if g_track.value in b.names else 0
            a = P[j].copy()
            if not g_follow_y.value:
                a[1] = state["y_med"][j]   # look at this joint's typical height in the clip
        Q = P - a
        with server.atomic():
            grid.position = -a
            keep = np.ones(len(Q), bool) if g_ends.value else ~b.is_end
            joints_h.visible = g_joints.value
            if g_joints.value:
                joints_h.points = Q[keep]
            links = b.links if g_ends.value else np.array([l for l in b.links if not b.is_end[l[1]]])
            bones_h.visible = g_bones.value
            if g_bones.value:
                bones_h.points = line_dots(Q, links, g_bwidth.value)
            trail_h.visible = g_trail.value > 0
            if g_trail.value > 0:
                k0 = max(0, i - int(g_trail.value * b.fps))
                ids = [0] + [j for j, nm in enumerate(b.names) if nm.endswith("Hand")]
                trail_h.points = (b.positions[k0:i + 1, ids] - a).reshape(-1, 3)

    g_file.on_update(load)
    for h in (g_jsize, g_bwidth, g_grid, g_bg):
        h.on_update(apply_style)

    def on_time(e):
        if e.client is not None:           # user scrubbed
            state["t"] = g_time.value
            show(state["t"])
    g_time.on_update(on_time)
    g_play.on_update(lambda _: state.update(playing=g_play.value))
    for h in (g_joints, g_ends, g_bones, g_trail, g_track, g_follow_y):
        h.on_update(lambda _: show(state["t"]))

    apply_style()
    load()
    print(f"open http://localhost:{args.port}", flush=True)
    last = time.time()
    tick = 0
    while True:
        now = time.time()
        dt, last = now - last, now
        b = state["clip"]
        if b is not None and state["playing"]:
            dur = len(b.positions) / b.fps
            state["t"] += dt * g_speed.value
            if state["t"] > dur:
                state["t"] = state["t"] % dur if g_loop.value else dur
            show(state["t"])
            tick += 1
            if tick % 6 == 0:
                g_time.value = round(state["t"], 2)
        time.sleep(max(0.0, 1 / 60 - (time.time() - now)))


if __name__ == "__main__":
    main()
