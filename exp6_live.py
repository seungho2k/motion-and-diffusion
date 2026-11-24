"""Experiment 6: endless real-time A-MDM generation with a repeating tau cycle.

    uv run python exp6_live.py              # then open http://localhost:8080

A child process generates new frames forever; the diffusion level follows a cyclic
schedule (ramp / pulses / none) that repeats every `cycle` seconds, while the motion
itself keeps being newly generated. Display at 30 or 15 fps; at 15 fps either keep
real speed (generate 2, show 1) or play in slow motion (generate 1, show 1).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from collections import deque

import imageio.v2 as imageio
import numpy as np
import viser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from amdm_runtime import DECODE_MODES, AMDMRuntime  # noqa: E402
from live import LiveGenerator  # noqa: E402
from schedule import PulseSchedule, RampSchedule  # noqa: E402
from viewer_common import (CameraRig, CurvePlot, PersistentGui, add_dots, add_space_toggle,  # noqa: E402
                           line_dots, setup_scene)

OUT_DIR = os.path.join(ROOT, "outputs")
VIDEO_W, VIDEO_H = 1080, 1350
GEN_FPS = 30                       # A-MDM / LaFAN1 frame rate
HIDDEN_WHEN_REDUCED = ("Spine", "Spine1", "Spine2", "LeftShoulder", "RightShoulder")


def main():
    rt = AMDMRuntime("lafan1")
    T, J, names = rt.T, rt.J, rt.joint_names
    lafan_dir = os.path.join(ROOT, "data/lafan1")
    clips = sorted(f[:-4] for f in os.listdir(lafan_dir) if f.endswith(".bvh")) if os.path.isdir(lafan_dir) else []

    live = LiveGenerator()
    live.start()

    server = viser.ViserServer(port=8080)
    grid = setup_scene(server)

    # ------------------------------------------------------------------ GUI
    with server.gui.add_folder("Cycle  τ(f)"):
        g_stype = server.gui.add_dropdown("type", ("ramp", "pulses", "none"), initial_value="ramp")
        g_cycle = server.gui.add_slider("cycle (s)", 2.0, 120.0, 0.5, 20.0,
                                        hint="The schedule repeats with this period; the motion never repeats.")
        g_rfade = server.gui.add_slider("ramp in end", 0.0, 0.99, 0.01, 0.2)
        g_rstart = server.gui.add_slider("ramp out start", 0.0, 0.99, 0.01, 0.7)
        g_rpeak = server.gui.add_slider("ramp max τ", 0.5, float(T), 0.5, float(T))
        g_rcurve = server.gui.add_dropdown("ramp curve", ("linear", "ease in", "ease out", "ease in-out", "exponential"),
                                           initial_value="ease in")
        g_rpower = server.gui.add_slider("ramp steepness", 0.2, 10.0, 0.1, 2.0)
        g_pframes = server.gui.add_text("pulse frames", "", hint="Frames within one cycle (30 fps). Empty = evenly spaced.")
        g_pcount = server.gui.add_slider("pulse count", 0, 20, 1, 2)
        g_ppeak = server.gui.add_slider("pulse peak τ", 0.5, float(T), 0.5, 12.0)
        g_pattack = server.gui.add_slider("pulse attack (fr)", 0, 20, 1, 2)
        g_pdecay = server.gui.add_slider("pulse decay (fr)", 1, 90, 1, 6)
        g_base = server.gui.add_slider("base τ", 0.0, float(T), 0.1, 0.0)
    with server.gui.add_folder("Motions"):
        g_lanes = server.gui.add_dropdown("motions", ("1", "3", "3 holding hands", "2 holding hands", "τ ladder"),
                                          initial_value="1",
                                          hint="3: three motions generated at once, side by side, each pinned at its hips "
                                               "(grid is turned off automatically).")
        g_spacing = server.gui.add_slider("spacing (m)", 0.3, 4.0, 0.05, 0.8)
        g_lsteps = server.gui.add_slider("ladder steps", 2, 8, 1, 6,
                                         hint="τ ladder: one clean motion shown side by side at evenly spaced τ, "
                                              "from `ladder max τ` (left) down to 0 (right).")
        g_lmax = server.gui.add_slider("ladder max τ", 0.5, 15.0, 0.5, 15.0)
        g_llabels = server.gui.add_checkbox("ladder labels", True)
        g_hlink = server.gui.add_dropdown(
            "hand link", ("attach", "tether"), initial_value="attach",
            hint="attach: move the partner bodies so the held hands touch (they get dragged by the hand). "
                 "tether (3 only): bodies stay in their slots, a line joins the held hands.")
        g_hpull = server.gui.add_slider(
            "hand reach", 0.0, 1.0, 0.05, 0.5,
            hint="Diffusion guidance pulling the held arms toward each other (bodies at `spacing`): share of "
                 "the gap closed per diffusion step. 0 = display-only linking.")
        g_hstop = server.gui.add_slider(
            "prior cleanup steps", 0, 5, 1, 0,
            hint="Last diffusion steps run without the pull: more = more natural poses, hands further apart.")
        g_hswap = server.gui.add_checkbox("swap hands", False,
                                          hint="Use the other hand on each side of every held pair.")
        g_same = server.gui.add_checkbox("same start pose", True,
                                         hint="All motions start from the same pose with different seeds "
                                              "(they diverge); off: start frames offset along the clip.")
        g_offset = server.gui.add_slider("start frame offset", 0, 2000, 10, 300)
    with server.gui.add_folder("Generation"):
        g_clip = server.gui.add_dropdown("start clip", ("sample",) + tuple(clips),
                                         initial_value="dance1_subject1" if "dance1_subject1" in clips else "sample")
        g_init = server.gui.add_slider("init frame", 0, 7000, 1, 300)
        g_seed = server.gui.add_number("seed", 0, step=1)
        g_restart = server.gui.add_button("restart", color="blue")
        g_feed = server.gui.add_dropdown("feed back", ("xt", "x0", "mix", "final"), initial_value="xt")
        g_root = server.gui.add_dropdown("root from", ("display", "x0", "final", "none"), initial_value="display")
        g_nscale = server.gui.add_slider("noise scale", 0.0, 2.0, 0.05, 1.0)
        g_ncoh = server.gui.add_slider("noise coherence", 0.0, 0.99, 0.01, 0.0)
        g_clipv = server.gui.add_slider("clip", 1.0, 10.0, 0.5, 4.0)
        g_decode = server.gui.add_dropdown("joints from", DECODE_MODES, initial_value="position")
    with server.gui.add_folder("Display"):
        g_final = server.gui.add_checkbox("overlay t=0", False,
                                          hint="Draw the same frame fully denoised in white, current t in yellow "
                                               "(costs a full denoise every frame).")
        g_joints = server.gui.add_checkbox("joints", True)
        g_reduced = server.gui.add_checkbox("reduced skeleton", True)
        g_bones = server.gui.add_checkbox("bones", True)
        g_jsize = server.gui.add_slider("joint size", 0.005, 0.2, 0.005, 0.04)
        g_bwidth = server.gui.add_slider("bone width", 0.002, 0.05, 0.001, 0.008)
        g_grid = server.gui.add_checkbox("grid", True)
        g_bg = server.gui.add_rgb("background", (0, 0, 0))
    with server.gui.add_folder("Camera"):
        g_track = server.gui.add_dropdown("track", ("off",) + tuple(names), initial_value="Hips")
        g_follow_y = server.gui.add_checkbox("follow height", False)
        cam = CameraRig(server, dist=3.5)
    with server.gui.add_folder("Playback"):
        g_play = server.gui.add_checkbox("playing", True, hint="space: play / pause")
        g_fps = server.gui.add_dropdown("display fps", ("30", "15"), initial_value="30")
        g_realspeed = server.gui.add_checkbox("15 fps keeps real speed", True,
                                              hint="On: generate 2 frames per shown frame. Off: slow motion (half speed).")
        g_info = server.gui.add_markdown("")
        plot = CurvePlot(server, color=(255, 214, 0))
        g_rec = server.gui.add_slider("export last (s)", 1.0, 120.0, 1.0, 20.0)
        g_export = server.gui.add_button("Export MP4 (1080x1350)", color="green")
        g_resetgui = server.gui.add_button("reset panel to defaults")
    add_space_toggle(server)

    persisted = {k: v for k, v in locals().items()
                 if k.startswith("g_") and k != "g_play"
                 and type(v).__name__ in ("GuiSliderHandle", "GuiCheckboxHandle", "GuiDropdownHandle",
                                          "GuiNumberHandle", "GuiRgbHandle", "GuiTextHandle")}
    persisted.update(cam_az=cam.az, cam_el=cam.el, cam_dist=cam.dist, cam_fov=cam.fov, cam_ortho=cam.ortho)
    pstate = PersistentGui(os.path.join(ROOT, ".cache/gui_state/exp6.json"), persisted)
    pstate.restore()
    pstate.start_saving()
    g_resetgui.on_click(lambda _: (pstate.reset(), setattr(g_info, "content", "panel state cleared; restart to apply")))

    groups = {"ramp": (g_rfade, g_rstart, g_rpeak, g_rcurve, g_rpower),
              "pulses": (g_pframes, g_pcount, g_ppeak, g_pattack, g_pdecay)}

    def sched_visibility(_=None):
        for k, hs in groups.items():
            for h in hs:
                h.visible = g_stype.value == k
    g_stype.on_update(sched_visibility)
    sched_visibility()

    # -------------------------------------------------------------- drawing
    hide = [names.index(n) for n in HIDDEN_WHEN_REDUCED]
    keep = np.array([i for i in range(J) if i not in hide])
    links_full = np.asarray(rt.links)
    links_red = np.concatenate([np.array([l for l in links_full if l[0] not in hide and l[1] not in hide]),
                                np.array([(names.index(a), names.index(b)) for a, b in
                                          (("Hips", "Neck"), ("Neck", "LeftArm"), ("Neck", "RightArm"))])])

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
                self.b.points = line_dots(pts, links_red if g_reduced.value else links_full, g_bwidth.value)

    MAX_LANES = 8
    lanes_sk = [(Skel(f"l{k}/final", (255, 255, 255)), Skel(f"l{k}/main", (255, 255, 255)),
                 Skel(f"l{k}/main_y", (255, 214, 0))) for k in range(MAX_LANES)]
    track_ref = {"y": None}

    def ladder() -> bool:
        return g_lanes.value == "τ ladder"

    def ladder_levels() -> list:
        return [float(v) for v in np.linspace(g_lmax.value, 0.0, int(g_lsteps.value))]

    def n_lanes() -> int:
        """Motions generated (the ladder generates one and shows it several times)."""
        return 1 if ladder() else int(g_lanes.value.split()[0])

    def screen_centre() -> np.ndarray:
        """Point the viewer looks at (screen centre); pinned joints are placed here."""
        for c in server.get_clients().values():
            return np.asarray(c.camera.look_at, np.float64)
        return np.zeros(3)

    def holding() -> bool:
        return "holding" in g_lanes.value

    LH, RH, HIPS = names.index("LeftHand"), names.index("RightHand"), names.index("Hips")

    def hold_shift(Pa, ha, Pb, hb):
        """Translation of body b so its Hand joint coincides with body a's Hand joint."""
        return Pa[ha] - Pb[hb]

    ladder_labels = [server.scene.add_label(f"/ladder_label{k}", "", visible=False, anchor="bottom-center", font_screen_scale=1.2) for k in range(8)]
    link_dots = add_dots(server, "/hand_links", (255, 255, 255), g_bwidth.value)

    def apply_style(_=None):
        for trio in lanes_sk:
            for sk in trio:
                sk.style()
        grid.visible = g_grid.value and n_lanes() == 1 and not ladder()   # several bodies: no shared floor
        server.scene.set_background_image(np.full((4, 4, 3), g_bg.value, np.uint8))

    def lane_slots(B: int) -> np.ndarray:
        """Screen-horizontal slots: positions along the camera's right vector, centred."""
        az = np.deg2rad(cam.az.value)
        right = np.array([np.cos(az), 0.0, -np.sin(az)])     # perpendicular to the view direction
        ks = np.arange(B) - (B - 1) / 2
        return ks[:, None] * g_spacing.value * right[None]

    def anchor(joints):
        if g_track.value == "off":
            return np.zeros(3)
        a = joints[names.index(g_track.value)].astype(np.float64).copy()
        if not g_follow_y.value:
            if track_ref["y"] is None:
                track_ref["y"] = a[1]
            a[1] = track_ref["y"]
        return a

    def draw(f):
        B = f.joints.shape[0]
        overlay = g_final.value
        links = []
        a_hand, b_hand = (LH, RH) if g_hswap.value else (RH, LH)     # (left partner's hand, right partner's hand)
        hand_links = []                                               # [(lane, joint, lane, joint)]
        if B == 1:
            shifts = [-anchor(f.joints[0])]
        elif holding() and B == 2:
            # B's Hand joint on A's Hand joint; that point is pinned at the screen centre
            sb = hold_shift(f.joints[0], a_hand, f.joints[1], b_hand)
            clasp = 0.5 * (f.joints[0, a_hand] + f.joints[1, b_hand] + sb)
            shifts = [-clasp, sb - clasp]
        elif holding() and B == 3:
            # middle pinned at its hips; neighbours hold the middle's hands
            mid = -f.joints[1, HIPS]
            m_left, m_right = (RH, LH) if g_hswap.value else (LH, RH)
            if g_hlink.value == "tether":
                slots = lane_slots(3)
                shifts = [slots[0] - f.joints[0, HIPS], mid, slots[2] - f.joints[2, HIPS]]
                hand_links = [(1, m_left, 0, a_hand), (1, m_right, 2, b_hand)]
            else:
                # partners' Hand joints on the middle's Hand joints
                shifts = [mid - hold_shift(f.joints[0], a_hand, f.joints[1], m_left), mid,
                          mid + hold_shift(f.joints[1], m_right, f.joints[2], b_hand)]
        else:   # each motion pinned at its tracked joint (default Hips), side by side on screen
            k = names.index(g_track.value) if g_track.value != "off" else 0
            slots = lane_slots(B)
            shifts = [slots[b] - f.joints[b, k] for b in range(B)]
        # the pinned point sits at the screen centre (the camera's look-at), not the world origin
        c = screen_centre()
        shifts = [sh + c for sh in shifts]
        links = [(f.joints[la, ja] + shifts[la], f.joints[lb, jb] + shifts[lb]) for la, ja, lb, jb in hand_links]
        show_labels = ladder() and g_llabels.value
        lv = ladder_levels()
        with server.atomic():
            for k, lab in enumerate(ladder_labels):
                lab.visible = show_labels and k < B
                if lab.visible:
                    lab.position = tuple(lane_slots(B)[k] + c + np.array([0.0, 1.05, 0.0]))
                    txt = f"τ = {lv[k]:.1f}" if k < len(lv) else ""
                    if lab.text != txt:
                        lab.text = txt
            grid.position = shifts[0]
            link_dots.visible = bool(links) and g_bones.value
            if link_dots.visible:
                pts = np.array([p for pair in links for p in pair], np.float32)
                link_dots.points = line_dots(pts, np.array([[2 * k, 2 * k + 1] for k in range(len(links))]),
                                             g_bwidth.value)
            for b, (sk_final, sk_main, sk_main_y) in enumerate(lanes_sk):
                if b < B:
                    sh = shifts[b]
                    sk_main.show(None if overlay else f.joints[b] + sh)
                    sk_main_y.show(f.joints[b] + sh if overlay else None)
                    sk_final.show(f.joints_final[b] + sh if overlay else None)
                else:
                    for sk in (sk_final, sk_main, sk_main_y):
                        sk.show(None)

    # -------------------------------------------------------------- live control
    def cycle_levels() -> np.ndarray:
        n = max(2, int(round(g_cycle.value * GEN_FPS)))
        if g_stype.value == "ramp":
            lv = RampSchedule(fade_in=g_rfade.value, start=g_rstart.value, peak=g_rpeak.value,
                              base=g_base.value, curve=g_rcurve.value, power=g_rpower.value).levels(n)
        elif g_stype.value == "pulses":
            lv = PulseSchedule(frames=g_pframes.value, count=int(g_pcount.value), peak=g_ppeak.value,
                               attack=int(g_pattack.value), decay=int(g_pdecay.value), base=g_base.value).levels(n)
        else:
            lv = np.full(n, g_base.value)
        return np.clip(lv, 0, T)

    def skip() -> int:
        return 2 if (g_fps.value == "15" and g_realspeed.value) else 1

    def hand_config():
        B = n_lanes()
        if not holding():
            return dict(hand_pairs=[], slots=None, hand_pull=0.0)
        a_hand, b_hand = (LH, RH) if g_hswap.value else (RH, LH)
        m_left, m_right = (RH, LH) if g_hswap.value else (LH, RH)
        pairs = [(0, a_hand, 1, b_hand)] if B == 2 else [(1, m_left, 0, a_hand), (1, m_right, 2, b_hand)]
        return dict(hand_pairs=pairs, slots=lane_slots(B), hand_pull=g_hpull.value, guide_stop=int(g_hstop.value))

    def sync(_=None):
        lv = cycle_levels()
        state["levels"] = lv
        plot.set_values(lv, vmax=float(T))
        live.configure(**hand_config(), levels=lv, feed=g_feed.value, root=g_root.value, decode=g_decode.value,
                       noise_scale=g_nscale.value, noise_coherence=g_ncoh.value, clip=g_clipv.value,
                       want_final=g_final.value, skip=skip())

    def restart(_=None):
        frames = rt.clip_frames("" if g_clip.value == "sample" else g_clip.value)
        track_ref["y"] = None
        history.clear()
        B = n_lanes()
        if g_same.value:
            starts = [frames[int(g_init.value) % len(frames)]] * B
        else:
            starts = [frames[(int(g_init.value) + b * int(g_offset.value)) % len(frames)] for b in range(B)]
        live.configure(**hand_config(), ladder=ladder_levels() if ladder() else None)
        live.reset(np.stack(starts), [int(g_seed.value) + b for b in range(B)])
        apply_style()

    state = {"levels": np.zeros(1)}
    history: deque = deque(maxlen=int(120 * GEN_FPS))
    for h in (g_hpull, g_hstop, g_hswap, g_spacing, cam.az):
        h.on_update(lambda _: live.configure(**hand_config()))
    for h in (g_stype, g_cycle, g_rfade, g_rstart, g_rpeak, g_rcurve, g_rpower, g_pframes, g_pcount, g_ppeak,
              g_pattack, g_pdecay, g_base, g_feed, g_root, g_nscale, g_ncoh, g_clipv, g_decode, g_final,
              g_fps, g_realspeed):
        h.on_update(sync)
    link_dots_ref = link_dots
    g_bwidth.on_update(lambda _: setattr(link_dots_ref, "point_size", g_bwidth.value))
    for h in (g_clip, g_init, g_seed, g_lanes, g_same, g_offset, g_lsteps, g_lmax):
        h.on_update(restart)
    g_restart.on_click(restart)
    for h in (g_jsize, g_bwidth, g_grid, g_bg):
        h.on_update(apply_style)
    g_track.on_update(lambda _: track_ref.update(y=None))

    # -------------------------------------------------------------- export (recent history)
    def export(event: viser.GuiEvent):
        client = event.client
        fps = int(g_fps.value)
        frames = list(history)[-int(g_rec.value * fps):]
        if client is None or not frames:
            return
        was = g_play.value
        g_play.value = False
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"exp6_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        meta = {k: (list(v.value) if isinstance(v.value, tuple) else v.value) for k, v in persisted.items()}
        meta["levels"] = [round(float(f.level), 3) for f in frames]
        meta["camera"] = cam.meta(client)
        with open(path.replace(".mp4", ".json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        pos, wxyz, fov = np.array(client.camera.position), np.array(client.camera.wxyz), float(client.camera.fov)
        g_export.disabled = True
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        try:
            for k, f in enumerate(frames):
                draw(f)
                img = client.get_render(VIDEO_H, VIDEO_W, position=pos, wxyz=wxyz, fov=fov, transport_format="png")
                if img.shape[-1] == 4:
                    al = img[..., 3:4].astype(np.float32) / 255.0
                    img = img[..., :3] * al + np.asarray(g_bg.value, np.float32) * (1 - al)
                writer.append_data(img[..., :3].astype(np.uint8))
                if k % 30 == 0:
                    g_info.content = f"exporting {k}/{len(frames)} ..."
            g_info.content = f"saved `{path}`"
            print("saved", path, flush=True)
        except Exception as e:
            g_info.content = f"export failed: {e}"
        finally:
            writer.close()
            g_export.disabled = False
            g_play.value = was
    g_export.on_click(export)

    # -------------------------------------------------------------- loop
    apply_style()
    sync()
    restart()
    print("open http://localhost:8080", flush=True)
    shown, t_win, late = 0, time.time(), 0
    next_t = time.time()
    while True:
        period = 1.0 / int(g_fps.value)
        if g_play.value and not g_export.disabled:
            f = live.pop(timeout=period)
            if f is None:
                late += 1
            else:
                draw(f)
                history.append(f)
                shown += 1
                lv = state["levels"]
                if shown % 5 == 0:
                    plot.set_cursor(f.index % len(lv))
                now = time.time()
                if now - t_win >= 2.0:
                    fps_now = shown / (now - t_win)
                    g_info.content = (f"frame **{f.index}** &nbsp; τ = **{f.level:.1f}** &nbsp; "
                                      f"cycle {f.index // len(lv) + 1} &nbsp; display **{fps_now:.1f}** fps"
                                      f"{' (generation behind)' if late else ''}")
                    shown, t_win, late = 0, now, 0
        else:
            time.sleep(0.05)
            next_t = time.time()
            continue
        # fixed-rate pacing: aim at absolute ticks so timing errors do not accumulate
        next_t += period
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.time()


if __name__ == "__main__":
    main()
