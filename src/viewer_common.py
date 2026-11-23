"""Shared viser pieces: dark scene, space-key playback, orbit camera rig, dot drawing."""
from __future__ import annotations

import time

import numpy as np
import viser

from camera import TrackCam, orbit_from

ORTHO_FOV_DEG = 0.5


def setup_scene(server: viser.ViserServer):
    server.gui.configure_theme(dark_mode=True)
    server.scene.set_up_direction("+y")
    return server.scene.add_grid(
        "/grid", width=40, height=40, plane="xz", cell_size=0.5, section_size=2.0,
        cell_color=(40, 40, 40), section_color=(70, 70, 70), shadow_opacity=0.0,
        infinite_grid=True, fade_distance=200.0, fade_from="origin")


def add_space_toggle(server: viser.ViserServer, label: str = "playing"):
    """Space clicks the checkbox labelled `label` (viser has no key events; add_html
    injects raw HTML, so an inline onerror handler registers the listener)."""
    server.gui.add_html(f"""<img src="data:," style="display:none" onerror="
      if (!window.__vdcSpace) {{ window.__vdcSpace = true;
        window.addEventListener('keydown', function (e) {{
          if (e.code !== 'Space' || e.repeat) return;
          var a = document.activeElement;
          if (a && (a.tagName === 'TEXTAREA' || (a.tagName === 'INPUT' && a.type !== 'checkbox'))) return;
          var el = Array.from(document.querySelectorAll('label, p, span, div'))
            .find(function (n) {{ return n.childElementCount === 0 && n.textContent.trim() === '{label}'; }});
          if (el) {{ e.preventDefault(); e.stopPropagation(); el.click(); }}
        }}, true);
      }}">""")


class CameraRig:
    """Orbit sliders mirrored with the viewer camera; the scene is shifted, not the camera."""

    def __init__(self, server: viser.ViserServer, az=30.0, el=10.0, dist=4.0, fov=40.0):
        self.server = server
        self.defaults = (az, el, dist, fov)
        self.az = server.gui.add_slider("azimuth", -180.0, 180.0, 1.0, az)
        self.el = server.gui.add_slider("elevation", -89.0, 89.0, 1.0, el)
        self.dist = server.gui.add_slider("distance", 0.1, 30.0, 0.05, dist)
        self.fov = server.gui.add_slider("fov", 10.0, 100.0, 1.0, fov)
        self.ortho = server.gui.add_checkbox(
            "orthographic", False,
            hint="Telephoto approximation (0.5 deg fov from far away, same framing).")
        self.reset = server.gui.add_button("reset view")
        self.ignore_until: dict[int, float] = {}

        self.ortho.on_update(lambda _: self.push())
        for h in (self.az, self.el, self.dist, self.fov):
            h.on_update(lambda e: e.client is not None and self.push())
        self.reset.on_click(lambda _: self._reset())
        server.on_client_connect(self._on_client)

    def _scale(self) -> float:
        if not self.ortho.value:
            return 1.0
        return np.tan(np.deg2rad(self.fov.value) / 2) / np.tan(np.deg2rad(ORTHO_FOV_DEG) / 2)

    def push(self, force: bool = False):
        real = max(self.dist.value, 0.05) * self._scale()   # typing "0.5" passes through "0"
        offset, _ = TrackCam(azimuth=self.az.value, elevation=self.el.value,
                             distance=real).pose(np.zeros((1, 3)))
        for _, c in self.server.get_clients().items():
            look = np.zeros(3) if force else np.asarray(c.camera.look_at)
            c.camera.fov = np.deg2rad(ORTHO_FOV_DEG if self.ortho.value else self.fov.value)
            c.camera.near = 1.0 if self.ortho.value else 0.01   # near > ~1 hides the bg image
            c.camera.far = real * 4 + 1000.0
            try:
                c.camera.position = look + offset
                c.camera.look_at = look
            except ValueError:   # transient degenerate pose (e.g. client mid-update)
                pass

    def _reset(self):
        self.az.value, self.el.value, self.dist.value, self.fov.value = self.defaults
        self.push(force=True)

    def _on_client(self, client: viser.ClientHandle):
        cid = client.client_id

        @client.camera.on_update
        def _(camera: viser.CameraHandle):
            if time.time() < self.ignore_until.get(cid, 0.0):
                return
            orb = orbit_from(np.asarray(camera.position), np.asarray(camera.look_at))
            if orb is None:
                return
            az, el, dist = orb
            dist /= self._scale()
            if np.allclose((az, el, dist), (self.az.value, self.el.value, self.dist.value), atol=1e-2):
                return
            self.az.value = float(np.clip(az, self.az.min, self.az.max))
            self.el.value = float(np.clip(el, self.el.min, self.el.max))
            self.dist.value = float(np.clip(dist, self.dist.min, self.dist.max))

        self.ignore_until[cid] = time.time() + 1.0
        self.push(force=True)

    def meta(self, client: viser.ClientHandle | None = None) -> dict:
        m = {"azimuth": self.az.value, "elevation": self.el.value, "distance": self.dist.value,
             "fov": self.fov.value, "orthographic": self.ortho.value}
        if client is not None:
            m.update(position=np.asarray(client.camera.position).tolist(),
                     wxyz=np.asarray(client.camera.wxyz).tolist(), fov_rad=float(client.camera.fov))
        return m


def add_dots(server, name, color, size, n=1):
    return server.scene.add_point_cloud(name, np.zeros((n, 3), np.float32), color, point_size=size,
                                        point_shape="circle", point_shading="flat", precision="float32")


def line_dots(pts: np.ndarray, links: np.ndarray, width: float, max_pts: int = 40000) -> np.ndarray:
    """Densely sampled dots along each link (flat, seamless lines)."""
    a, b = pts[links[:, 0]], pts[links[:, 1]]
    lengths = np.linalg.norm(b - a, axis=1)
    step = max(width * 0.35, 1e-4)
    n = np.clip(np.ceil(lengths / step).astype(int) + 1, 2, None)
    if n.sum() > max_pts:
        n = np.maximum(2, (n * max_pts / n.sum()).astype(int))
    out = [a[k] + np.linspace(0, 1, n[k])[:, None] * (b[k] - a[k]) for k in range(len(n))]
    return np.concatenate(out).astype(np.float32)


class CurvePlot:
    """Small curve drawn into an image (viser's uplot widget rendered empty in our apps)."""

    def __init__(self, server: viser.ViserServer, color=(255, 214, 0), w: int = 600, h: int = 170,
                 label: str | None = None):
        self.w, self.h, self.color = w, h, np.array(color, np.uint8)
        self.values = None
        self.vmax = 1.0
        self.cursor = None
        self.handle = server.gui.add_image(self._render(), label=label, format="png")

    def set_values(self, values: np.ndarray, vmax: float | None = None):
        self.values = np.asarray(values, np.float64)
        self.vmax = float(vmax if vmax is not None else max(1e-6, self.values.max()))
        self.handle.image = self._render()

    def set_cursor(self, i: int):
        if self.values is None or i == self.cursor:
            return
        self.cursor = i
        self.handle.image = self._render()

    def _render(self) -> np.ndarray:
        w, h, pad = self.w, self.h, 8
        img = np.full((h, w, 3), 24, np.uint8)
        img[h - pad, pad:w - pad] = 70                                  # baseline
        if self.values is None or len(self.values) < 2:
            return img
        n = len(self.values)
        xs = pad + np.arange(n) * (w - 2 * pad - 1) / (n - 1)
        ys = (h - pad) - np.clip(self.values / self.vmax, 0, 1) * (h - 2 * pad)
        for k in range(n - 1):                                          # thick polyline
            m = int(max(abs(xs[k + 1] - xs[k]), abs(ys[k + 1] - ys[k]))) + 1
            px = np.linspace(xs[k], xs[k + 1], m).round().astype(int)
            py = np.linspace(ys[k], ys[k + 1], m).round().astype(int)
            for dy in (-1, 0, 1):
                img[np.clip(py + dy, 0, h - 1), np.clip(px, 0, w - 1)] = self.color
        if self.cursor is not None:
            cx = int(round(pad + self.cursor * (w - 2 * pad - 1) / (n - 1)))
            img[pad:h - pad, max(cx - 1, 0):cx + 1] = 200
        return img


class PersistentGui:
    """Saves GUI values to a JSON file on every change and restores them at start-up.

    Call `restore()` right after creating the handles and before registering callbacks
    that would react to the restored values.
    """

    def __init__(self, path: str, handles: dict):
        import json as _json
        import os as _os
        self._json, self._os = _json, _os
        self.path = path
        self.handles = handles

    def restore(self):
        if not self._os.path.exists(self.path):
            return
        try:
            saved = self._json.load(open(self.path))
        except Exception:
            return
        for k, v in saved.items():
            h = self.handles.get(k)
            if h is None:
                continue
            try:
                opts = getattr(h, "options", None)
                if opts is not None and v not in opts:
                    continue
                h.value = tuple(v) if isinstance(v, list) else v
            except Exception:
                pass

    def start_saving(self):
        for h in self.handles.values():
            h.on_update(lambda _: self.save())

    def save(self):
        data = {}
        for k, h in self.handles.items():
            v = h.value
            data[k] = list(v) if isinstance(v, tuple) else v
        self._os.makedirs(self._os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            self._json.dump(data, f, indent=1)
        self._os.replace(tmp, self.path)

    def reset(self):
        if self._os.path.exists(self.path):
            self._os.remove(self.path)
