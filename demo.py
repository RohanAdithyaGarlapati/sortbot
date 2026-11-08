"""Make the LinkedIn demo video: a vertical clip with typed commands and live captions.

    python demo.py
    python demo.py --end-card "github.com/YOUR-USERNAME/sortbot"
    python demo.py --commands "sort the cubes by color" "stack them" --shuffle 4
    python demo.py --format square

Portrait 4:5 (1080 x 1350) fills more of a phone screen than a wide video and is not cropped in
the feed. Rendering takes a few minutes; the result is videos/sortbot_demo.mp4.
"""

import argparse
from pathlib import Path

from sortbot import Controller, Sim, Skills, execute, parse_command
from sortbot.video import Recorder, make_camera

ROOT = Path(__file__).resolve().parent
DEFAULT_COMMANDS = ["sort the cubes by color", "now build a tower: blue, green, red on top"]
FORMATS = {"portrait": (1080, 1350), "square": (1080, 1080)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commands", nargs="+", default=DEFAULT_COMMANDS)
    parser.add_argument("--end-card", help="text for the last 2.5 seconds, e.g. your GitHub URL")
    parser.add_argument("--format", choices=FORMATS, default="portrait", help="portrait 4:5 or square 1:1")
    parser.add_argument("--speed", type=float, default=1.25, help="playback speed (1.0 = real time)")
    parser.add_argument("--shuffle", type=int, metavar="SEED", help="start the cubes in random spots")
    parser.add_argument("--out", default=str(ROOT / "videos" / "sortbot_demo.mp4"))
    parser.add_argument("--gif", metavar="PATH", help="also save a short GIF, e.g. docs/demo.gif for the README")
    args = parser.parse_args()

    sim = Sim()
    if args.shuffle is not None:
        sim.reset(sim.random_layout(args.shuffle))
    width, height = FORMATS[args.format]
    camera = make_camera(lookat=(0.42, 0.02, 0.45), distance=1.7)   # leaves room for the captions
    rec = Recorder(sim.model, sim.data, args.out, width=width, height=height, camera=camera,
                   speed=args.speed, font_scale=0.95)
    ctrl = Controller(sim)

    def on_step():
        rec.status = ctrl.status or None    # caption = what the arm is doing right now
        rec()

    ctrl.on_step = on_step
    skills = Skills(ctrl, say=print)

    rec.hold(0.5)
    for text in args.commands:
        for i in range(1, len(text) + 1):          # typing effect, about 30 characters per second
            rec.command = text[:i] + "_"
            rec.hold(1 / 30)
        rec.command = text
        rec.hold(0.5)
        print(f"you> {text}")
        execute(parse_command(text), skills)
        ctrl.status = ""
        rec.hold(0.7)

    skills.home()
    ctrl.status = ""
    rec.hold(0.8)
    if args.end_card:
        rec.command, rec.status = None, args.end_card
        rec.hold(2.5)
    rec.close()
    if args.gif:
        make_gif(args.out, args.gif)


def make_gif(mp4_path, gif_path, start=0.0, seconds=8.0, fps=10, width=360):
    """Cut a small looping GIF out of the video (GitHub READMEs autoplay GIFs)."""
    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(mp4_path)
    src_fps = reader.get_meta_data()["fps"]
    frames, next_frame = [], start * src_fps
    for i, frame in enumerate(reader):
        if i > (start + seconds) * src_fps:
            break
        if i >= next_frame:
            image = Image.fromarray(frame)
            image.thumbnail((width, width))
            frames.append(image.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
            next_frame += src_fps / fps
    reader.close()
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    main()
