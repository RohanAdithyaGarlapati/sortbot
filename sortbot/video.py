"""Two ways to watch the simulation: record an MP4, or open a live 3D window.

Both are used the same way. Pass the object as ``on_step`` and it is called after every physics
step; it decides on its own when to grab a frame.

    out = open_output(model, data, "videos/demo.mp4", view=False)
    ...
    mujoco.mj_step(model, data); out()
    ...
    out.close()

Recording works everywhere, including servers with no screen. The live window needs a display;
on macOS, start scripts that use it with ``mjpython`` instead of ``python``.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np


def make_camera(lookat=(0.42, 0.0, 0.30), distance=1.6, azimuth=190.0, elevation=-28.0) -> mujoco.MjvCamera:
    """A free camera that orbits a point. Azimuth and elevation are in degrees."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def add_markers(scene: mujoco.MjvScene, markers) -> None:
    """Draw small spheres (pos, rgba, radius) into a scene, for trails and targets."""
    for pos, rgba, radius in markers:
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0.0, 0.0]), np.asarray(pos, dtype=float),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scene.ngeom += 1


def _wrap(draw, text: str, font, max_width: float) -> list[str]:
    """Split text into lines that fit within max_width pixels."""
    lines, current = [], ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current]


def _load_font(size: int, mono: bool = False):
    from PIL import ImageFont

    names = (["DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf", "Courier New.ttf"] if mono
             else ["DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc", "arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow 10.1+
    except TypeError:
        return ImageFont.load_default()


class Recorder:
    """Renders frames from a camera and writes them to an MP4 file.

    Text overlays (all optional, change them any time):
        ``lines``   a list of strings drawn top-left, handy for live numbers
        ``command`` a typed command, drawn in a terminal-style box at the top
        ``status``  a short caption drawn at the bottom
    """

    def __init__(self, model, data, path, width=1280, height=720, fps=30, camera=None, font_scale=1.0,
                 speed=1.0):
        import imageio.v2 as imageio

        self.model, self.data = model, data
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.fps = width, height, fps
        self.speed = speed   # playback speed: 1.25 makes the video 25% faster than real time
        self.renderer = mujoco.Renderer(model, height, width)
        self.camera = camera or make_camera()
        self.writer = imageio.get_writer(
            str(self.path), fps=fps, codec="libx264", quality=None, macro_block_size=2,
            pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast", "-crf", "20"])
        self.next_frame_time = 0.0
        self.frames = 0
        self.markers: list = []
        self.lines: list[str] = []
        self.command: str | None = None
        self.status: str | None = None
        s = font_scale * height / 720
        self._font = _load_font(int(26 * s))
        self._mono = _load_font(int(24 * s), mono=True)
        self._big_mono = _load_font(int(34 * s), mono=True)
        self._scale = s

    def __call__(self) -> None:
        if self.data.time + 1e-9 >= self.next_frame_time:
            self.capture()
            self.next_frame_time += self.speed / self.fps

    def capture(self) -> None:
        self.renderer.update_scene(self.data, camera=self.camera)
        add_markers(self.renderer.scene, self.markers)
        frame = self.renderer.render()
        if self.lines or self.command or self.status:
            frame = self._draw_overlays(frame)
        self.writer.append_data(frame)
        self.frames += 1

    def hold(self, seconds: float) -> None:
        """Repeat the current view for a while without advancing physics (for title cards)."""
        for _ in range(int(seconds * self.fps)):
            self.capture()

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.fromarray(frame).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        s, W, H = self._scale, self.width, self.height
        pad = int(18 * s)

        if self.lines:
            line_h = int(32 * s)
            box_w = max(draw.textlength(line, font=self._mono) for line in self.lines) + int(28 * s)
            draw.rounded_rectangle([pad, pad, pad + box_w, pad + line_h * len(self.lines) + int(16 * s)],
                                   radius=int(10 * s), fill=(12, 16, 24, 170))
            for i, line in enumerate(self.lines):
                draw.text((pad + int(14 * s), pad + int(8 * s) + i * line_h), line, font=self._mono,
                          fill=(255, 255, 255, 255))

        if self.command is not None:
            text_x = pad + int(64 * s)
            lines = _wrap(draw, self.command, self._big_mono, W - pad - int(24 * s) - text_x)
            line_h, inner = int(46 * s), int(18 * s)
            box_h = 2 * inner + line_h * len(lines)
            draw.rounded_rectangle([pad, pad, W - pad, pad + box_h], radius=int(14 * s), fill=(12, 16, 24, 215))
            first_y = pad + inner + (line_h - int(34 * s)) // 2
            draw.text((pad + int(24 * s), first_y), ">", font=self._big_mono, fill=(110, 230, 140, 255))
            for i, line in enumerate(lines):
                draw.text((text_x, first_y + i * line_h), line, font=self._big_mono, fill=(240, 240, 240, 255))

        if self.status:
            tw = draw.textlength(self.status, font=self._font)
            box_w, box_h = tw + int(40 * s), int(50 * s)
            x0, y0 = (W - box_w) / 2, H - pad - box_h
            draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=int(25 * s), fill=(12, 16, 24, 190))
            draw.text((x0 + int(20 * s), y0 + (box_h - int(28 * s)) / 2), self.status, font=self._font,
                      fill=(255, 255, 255, 255))

        return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
        print(f"Saved {self.frames / self.fps:.1f} s of video to {self.path}")


class LiveView:
    """Shows the simulation in an interactive window, slowed to real time.

    Drag with the mouse to orbit, scroll to zoom, double-click a body to select it.
    """

    def __init__(self, model, data, camera=None, fps=60):
        import mujoco.viewer

        self.model, self.data = model, data
        self.viewer = mujoco.viewer.launch_passive(model, data)
        cam = camera or make_camera()
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = cam.lookat
            self.viewer.cam.distance = cam.distance
            self.viewer.cam.azimuth = cam.azimuth
            self.viewer.cam.elevation = cam.elevation
        self.frame_dt = 1.0 / fps
        self.next_frame_time = 0.0
        self.wall_start = time.perf_counter()
        self.sim_start = data.time
        self.markers: list = []
        self.lines: list[str] = []
        self.command: str | None = None
        self.status: str | None = None
        self._last_status = None

    def __call__(self) -> None:
        if self.data.time + 1e-9 < self.next_frame_time:
            return
        self.next_frame_time += self.frame_dt
        if not self.viewer.is_running():
            raise SystemExit("Viewer closed.")
        if self.status and self.status != self._last_status:
            print(f"  [{self.status}]")
            self._last_status = self.status
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            add_markers(self.viewer.user_scn, self.markers)
        self.viewer.sync()
        # Sleep so one simulated second takes one real second.
        ahead = (self.data.time - self.sim_start) - (time.perf_counter() - self.wall_start)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -0.25:   # we were paused (for example waiting for you to type): restart the clock
            self.wall_start, self.sim_start = time.perf_counter(), self.data.time

    def hold(self, seconds: float) -> None:
        time.sleep(seconds)
        self.wall_start += seconds

    def close(self) -> None:
        self.viewer.close()


def open_output(model, data, video_path, view: bool = False, **kwargs):
    """Return a LiveView when ``view`` is True, otherwise a Recorder that writes ``video_path``."""
    if view:
        return LiveView(model, data, camera=kwargs.get("camera"))
    return Recorder(model, data, video_path, **kwargs)
