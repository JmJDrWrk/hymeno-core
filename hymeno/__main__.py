"""The brain's terminal.

    python -m hymeno check      # is the body there, and does the camera send photos?
    python -m hymeno floor      # one photo: where is the floor, which way to go?
    python -m hymeno floor --every 2.5   # the same every 2.5 s (0: as fast as it can), Ctrl+C stops
    python -m hymeno explore --speed 0.2 # walk where the floor is free, Ctrl+C stops
    python -m hymeno explore --every 2   # one decision, then 2 s walking it, and stop
    python -m hymeno explore --record    # the same, keeping every photo in data/runs/<date-time>/
    python -m hymeno explore --goal "shoe"       # find it (YOLO-World, short names) and walk there
    python -m hymeno replay data/runs/<date-time> --from 120 --to 300 --step
    python -m hymeno replay data/runs/<date-time> --step --select data/goals   # s copies a photo there
                                         # decide again on recorded photos, without the robot
    python -m hymeno calibrate           # click the tapes and screws on a photo: camera geometry
"""

import argparse
import glob
import io
import os
import sys
import time

from . import __version__, config
from .clients import Body, Head, HeadVideo


LOG = None  # while recording, everything said also goes to log.txt in the run


def say(message):
    line = "%s %s" % (time.strftime("%H:%M:%S"), message)
    print(line, flush=True)
    if LOG:
        LOG.write(line + "\n")
        LOG.flush()


def describe(decision):
    return "free floor: %.0f cm straight on, %.0f cm on the way chosen -> %s, steer %+.2f" % (
        decision["ahead_cm"], decision["clear_cm"], decision["direction"], decision["steer"])


def group_note(key, order):
    return "group %s: %s" % (key, "%s %+.2f" % (order["direction"].upper(), order["steer"]) if order else "pending")


def check(settings):
    """True when both the body and the camera answer."""
    body = Body(settings["body"]["url"], max_speed=settings["max_speed"])
    head = Head(settings["head"]["url"])
    ok = True
    try:
        say("body (%s): %s" % (body.url, body.state()))
    except Exception as e:
        say("body (%s) does not answer: %s" % (body.url, type(e).__name__))
        ok = False
    try:
        started = time.monotonic()
        jpeg = head.photo()
        path = os.path.join(settings["data_dir"], "check.jpg")
        os.makedirs(settings["data_dir"], exist_ok=True)
        with open(path, "wb") as f:
            f.write(jpeg)
        say("camera (%s): %d KB in %.0f ms, saved as %s" % (
            head.url, len(jpeg) // 1024, (time.monotonic() - started) * 1000, path))
    except Exception as e:
        say("camera (%s) does not answer: %s" % (head.url, type(e).__name__))
        ok = False
    return ok


def searching_view(image, frame, note):
    """The photo with only the searching state written on it."""
    from PIL import ImageDraw

    from .floor import write

    picture = image.copy()
    pen = ImageDraw.Draw(picture)
    write(pen, (8, 6), "SEARCHING", size=26)
    write(pen, (8, 42), note)
    write(pen, (picture.width - 8, 6), "frame %d" % frame, right=True)
    return picture


def floor(settings, every=None, speed=None, record=False, what=None, fastest=False):
    """One photo from the head (or one every `every` seconds until Ctrl+C,
    from its video), saved with the floor in green and an arrow. With a
    speed, the robot also walks that way (explore). With record, each photo
    of the loop is kept in data/runs/<date-time>/, numbered, for replay.
    With `what` (words), the vision model finds it and the robot walks there:
    standing (turning a little) until it is first seen, stopping on arrival.
    With fastest (loops only), no picture is drawn or saved (no view, no
    latest.jpg): to measure what that costs, and to run as fast as it can."""
    from PIL import Image, ImageOps

    from .floor import Floor, Goal, Groups, draw

    head = Head(settings["head"]["url"])
    say("loading the floor model...")
    eyes = Floor()
    folder = os.path.join(settings["data_dir"], "floor")
    os.makedirs(folder, exist_ok=True)
    video = HeadVideo(settings["head"]["url"]) if every is not None else None
    groups = Groups() if every is not None else None
    body = Body(settings["body"]["url"], max_speed=settings["max_speed"]) if speed else None
    view, goal, wanted = None, None, None
    if what and body:
        from .target import DETECTOR, SEARCH_TURN_S, DetectedTarget
        wanted = DetectedTarget(what)
        say("looking for %s with %s" % (what, DETECTOR))
    fastest = fastest and every is not None
    if every is not None and not fastest:
        from .view import View
        if not wanted:
            goal = Goal()  # clicked on the view; followed while it is seen
        view = View(on_goal=(lambda column: goal.set(column) if column is not None else goal.clear("goal cleared"))
                    if goal else None)
        say("watch it on http://localhost:%d" % view.server.server_port)
    run, frame = None, 0
    if record and every is not None:
        run = os.path.join(settings["data_dir"], "runs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(run)
        global LOG
        LOG = open(os.path.join(run, "log.txt"), "a")
        say("recording to %s (photos and log.txt)" % run)
    decision, number = None, 0
    try:
        while True:
            asked = time.monotonic()
            try:
                if video:
                    jpeg, number = video.photo(number)
                else:
                    jpeg = head.photo()
                image = Image.open(io.BytesIO(jpeg)).convert("RGB")
            except Exception as e:
                say("camera (%s) does not answer: %s" % (head.url, type(e).__name__))
                if body:
                    body.stop()
                if every is None:
                    return False
                time.sleep(1)
                continue
            frame += 1
            if run:
                with open(os.path.join(run, "%05d.jpg" % frame), "wb") as f:
                    f.write(jpeg)
            if settings["head"]["mirrored"]:
                image = ImageOps.mirror(image)
            started = time.monotonic()
            if wanted:
                wanted.offer(jpeg, image, frame)
                todo = wanted.check(frame)
            # While searching a goal in words the robot stands: no floor to look at,
            # which leaves the GPU to the vision model.
            searching = wanted is not None and wanted.state == "searching"
            if searching:
                mask, decision, order, note = None, None, None, wanted.note
            else:
                # Each photo decides on its own; its group decides what is carried out.
                mask, decision = eyes.look(image, groups.acting if groups else None, wanted or goal)
                if wanted:
                    decision["target"] = wanted.drawing()
                key, order = groups.add(frame, decision) if groups else (None, None)
                note = group_note(key, order) if groups else None
                if goal and goal.note != "no goal":
                    note += " | " + goal.note
                if wanted:
                    note += " | " + wanted.note
            if not fastest:
                # One fixed name in a loop, to keep it open while the robot moves.
                name = "latest" if every is not None else time.strftime("%Y%m%d-%H%M%S")
                path = os.path.join(folder, name + ".jpg")
                drawn = io.BytesIO()
                if searching:
                    searching_view(image, frame, note).save(drawn, "JPEG")
                else:
                    draw(image, mask, decision, "frame %d" % frame if groups else None, note).save(drawn, "JPEG")
                if view:
                    view.show(drawn.getvalue())
                for data, target in ((drawn.getvalue(), path), (jpeg, os.path.join(folder, name + "-raw.jpg"))):
                    with open(target + ".tmp", "wb") as f:
                        f.write(data)
                    os.replace(target + ".tmp", target)
            say("%s%s%s (photo %.0f ms, floor %.0f ms)" % (
                "frame %d: " % frame if groups else "", describe(decision) if decision else "standing",
                " | " + note if note else "", (started - asked) * 1000, (time.monotonic() - started) * 1000))
            if every is None:
                return True
            if not body:
                time.sleep(every)
            elif wanted and todo != "go":  # a goal in words, not being walked to now
                if todo == "turn":  # searching: turn a little, then ask again
                    body.move(w=speed, duration_s=SEARCH_TURN_S)
                    _, number = video.photo(number)  # skip the frame taken while turning
                elif todo in ("arrived", "gave up"):
                    body.stop()
                    say("%s: %s" % (todo, what))
                    return todo == "arrived"
                else:
                    body.stop()
            elif order and groups.fresh:  # a group just decided: carry its order out once
                # (while it is pending, keep doing the last)
                vx = speed if order["direction"] in ("ahead", "left", "right") else 0  # blocked: turn on the spot
                w = -order["steer"] * speed  # w > 0 turns left
                if order["direction"] == "wait":
                    body.stop()
                elif every:  # slow steps: carry it out for `every` s, then stop
                    body.move(vx=vx, w=w, duration_s=every)
                    _, number = video.photo(number)  # skip the frame taken while walking
                else:
                    body.drive(vx=vx, w=w)
    except KeyboardInterrupt:
        say("stopped")
        return True
    finally:
        if body:
            body.stop()


def read_key():
    """One key press, without Enter: "right", "left", "enter", "back" or the character."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        key = sys.stdin.read(1)
        if key == "\x1b":
            return {"[C": "right", "[D": "left"}.get(sys.stdin.read(2), "")
        return {"\n": "enter", "\r": "enter", "\x7f": "back", "\b": "back"}.get(key, key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def choose_frame(i, last, index_of, select=None):
    """Right arrow: next frame; left: the one before; a number and Enter: that
    frame. With select (a function of the frame index): s selects or unselects."""
    prompt = "  -> next, <- back, a number + Enter: go to that frame%s (Ctrl+C stops) " % (
        ", s: select" if select else "")
    print(prompt, end="", flush=True)
    typed = ""
    while True:
        key = read_key()
        if key == "s" and select and not typed:
            print("\n  " + select(i) + "\n" + prompt, end="", flush=True)
            continue
        if key == "right":
            if i >= last:
                print("\n  that was the last frame", end="", flush=True)
                continue
            i += 1
            break
        if key == "left":
            i = max(0, i - 1)
            break
        if key.isdigit():
            typed += key
            print(key, end="", flush=True)
        elif key == "back" and typed:
            typed = typed[:-1]
            print("\b \b", end="", flush=True)
        elif key == "enter" and typed:
            i = index_of(int(typed))
            break
    print()
    return i


WARM_UP = 5  # photos decided silently before a jump, so its groups start as they did


def replay(settings, folder, start=None, end=None, every=0, step=False, select_to=None):
    """Decides again on the photos recorded in folder (from --record), shown
    on the view and the console with their frame numbers. No robot needed.
    With step: Enter goes on, "b" goes back and a number jumps to that frame.
    Going back shows what was decided on the way forward (each decision
    depends on the photos before it); a jump forward starts WARM_UP before."""
    import bisect

    from PIL import Image, ImageOps

    from .floor import Floor, Groups, draw
    from .view import View

    frames = sorted((int(os.path.basename(p)[:-4]), p) for p in glob.glob(os.path.join(folder, "*.jpg"))
                    if os.path.basename(p)[:-4].isdigit())
    if not frames:
        say("no recorded photos in %s" % folder)
        return False
    numbers = [n for n, _ in frames]

    def index_of(number):
        return min(bisect.bisect_left(numbers, number), len(frames) - 1)

    say("loading the floor model...")
    eyes = Floor()
    view = View()
    say("watch it on http://localhost:%d" % view.server.server_port)
    shown = {}  # frame index -> (drawn JPEG, console line)
    timing = {"decide": [], "draw": []}  # seconds per photo, for the summary at the end
    state = {"groups": None, "at": None}  # the running sequence and its last index

    def decide_up_to(i):
        if i in shown:
            return
        at = state["at"]
        if at is None or i <= at or i - at > WARM_UP:
            state["groups"], first, warm = Groups(), max(0, i - WARM_UP), i
        else:
            first, warm = at + 1, at + 1
        for k in range(first, i + 1):
            frame, path = frames[k]
            image = Image.open(path).convert("RGB")
            if settings["head"]["mirrored"]:
                image = ImageOps.mirror(image)
            groups = state["groups"]
            started = time.monotonic()
            mask, decision = eyes.look(image, groups.acting)
            key, order = groups.add(frame, decision)
            decided = time.monotonic()
            if k >= warm:
                drawn = io.BytesIO()
                note = group_note(key, order)
                draw(image, mask, decision, "frame %d" % frame, note).save(drawn, "JPEG")
                drew = time.monotonic()
                timing["decide"].append(decided - started)
                timing["draw"].append(drew - decided)
                shown[k] = (drawn.getvalue(), "frame %d: %s | %s (decide %.0f ms, draw %.0f ms)" % (
                    frame, describe(decision), note, (decided - started) * 1000, (drew - decided) * 1000))
        state["at"] = i

    def select(k):
        """Copies photo k to select_to, or takes the copy away if it is there."""
        import shutil
        copy = os.path.join(select_to, os.path.basename(frames[k][1]))
        if os.path.exists(copy):
            os.remove(copy)
            return "frame %d unselected (removed from %s)" % (frames[k][0], select_to)
        os.makedirs(select_to, exist_ok=True)
        shutil.copy2(frames[k][1], copy)
        return "frame %d selected (copied to %s)" % (frames[k][0], select_to)

    i = index_of(numbers[0] if start is None else start)
    last = len(frames) - 1 if end is None else bisect.bisect_right(numbers, end) - 1
    try:
        while True:
            decide_up_to(i)
            jpeg, line = shown[i]
            view.show(jpeg)
            if select_to and os.path.exists(os.path.join(select_to, os.path.basename(frames[i][1]))):
                line += " | SELECTED"
            say(line)
            if step:
                i = choose_frame(i, last, index_of, select if select_to else None)
                continue
            if every:
                time.sleep(every)
            if i >= last:
                input("end of the photos; Enter quits ")
                break
            i += 1
    except (KeyboardInterrupt, EOFError):
        say("stopped")
    if timing["decide"]:
        decide, drawing = (sum(timing[k]) / len(timing[k]) * 1000 for k in ("decide", "draw"))
        say("%d photos: decide %.0f ms, draw %.0f ms a photo on average" % (len(timing["decide"]), decide, drawing))
    return True


def calibrate(settings, photo=None):
    """Camera calibration by clicks on a photo of the tapes and screws
    (a new one from the head, or the photo given)."""
    from . import calibrate as calibration
    from .view import PORT

    if photo:
        with open(photo, "rb") as f:
            jpeg = f.read()
    else:
        try:
            jpeg = Head(settings["head"]["url"]).photo()
        except Exception as e:
            say("camera (%s) does not answer: %s" % (settings["head"]["url"], type(e).__name__))
            return False
        os.makedirs(settings["data_dir"], exist_ok=True)
        photo = os.path.join(settings["data_dir"], "calibration.jpg")
        with open(photo, "wb") as f:
            f.write(jpeg)
        say("photo saved as %s" % photo)
    return calibration.run(jpeg, photo, say, PORT)


def main():
    parser = argparse.ArgumentParser(prog="hymeno", description="Remote brain for a robot.")
    parser.add_argument("command", choices=["check", "floor", "explore", "replay", "calibrate"])
    parser.add_argument("folder", nargs="?", help="replay: a folder of recorded photos (data/runs/<date-time>); "
                        "calibrate: a photo of the tapes (default: a new one from the head)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--every", type=float, help="floor: repeat every this many seconds (0: as fast as it can; Ctrl+C stops); "
                        "explore: walk each decision this many seconds, then stop")
    parser.add_argument("--speed", type=float, default=0.2, help="explore: walking speed, 0..1 (capped by max_speed)")
    parser.add_argument("--record", action="store_true", help="floor --every, explore: keep every photo for replay")
    parser.add_argument("--fastest", action="store_true",
                        help="floor --every, explore: draw and save no picture (no view, no latest.jpg)")
    parser.add_argument("--goal", help='explore: walk to this, found by the detector; short names work best (e.g. "shoe")')
    parser.add_argument("--from", dest="start", type=int, help="replay: first frame")
    parser.add_argument("--to", dest="end", type=int, help="replay: last frame")
    parser.add_argument("--step", action="store_true", help="replay: one frame at a time (arrows: next/back, a number + Enter: go there)")
    parser.add_argument("--select", "--moveselected", dest="select_to", metavar="FOLDER",
                        help="replay --step: s copies the photo shown to this folder (s again takes it away)")
    args = parser.parse_args()
    settings = config.load(args.config)
    say("hymeno-core %s" % __version__)
    if args.command == "floor":
        return 0 if floor(settings, args.every, record=args.record, fastest=args.fastest) else 1
    if args.command == "explore":
        return 0 if floor(settings, every=args.every or 0, speed=args.speed, record=args.record, what=args.goal,
                          fastest=args.fastest) else 1
    if args.command == "calibrate":
        return 0 if calibrate(settings, args.folder) else 1
    if args.command == "replay":
        if not args.folder:
            parser.error("replay needs a folder: data/runs/<date-time>")
        return 0 if replay(settings, args.folder, args.start, args.end, args.every or 0, args.step, args.select_to) else 1
    return 0 if check(settings) else 1


if __name__ == "__main__":
    sys.exit(main())
