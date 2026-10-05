"""Runs the brain: python -m hymeno [--config config.yaml] [--dry-run] [--once]

Every cycle: one photo from the head, one question to the vision model, at
most one short move of the body. Ctrl+C stops the robot and exits."""

import argparse
import sys
import time

from . import __version__, behavior, config, perception
from .clients import Body, Head, VisionModel
from .journal import Journal

ERROR_PAUSE_S = 3.0


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def main():
    parser = argparse.ArgumentParser(prog="hymeno", description="Remote brain for a robot.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="look and decide, but never move the robot")
    parser.add_argument("--once", action="store_true", help="one look, then exit")
    args = parser.parse_args()

    settings = config.load(args.config)
    b = settings["behavior"]
    head = Head(settings["head"]["url"])
    body = Body(settings["body"]["url"], max_speed=b["max_speed"])
    model = VisionModel(**{k: settings["model"][k] for k in ("url", "name", "timeout_s", "keep_alive", "max_answer_tokens")})
    journal = Journal(settings["data_dir"])
    target = settings["target"]

    log("hymeno-core %s: target '%s', model %s%s" % (
        __version__, target["label"], model.name, " (dry run: the robot will not move)" if args.dry_run else ""))
    if not args.dry_run:
        body.state()   # fail early if the robot is not there

    try:
        while True:
            pause = b["idle_s"]
            try:
                jpeg = head.photo()
                width, height = perception.photo_size(jpeg)
                answer, seconds = model.ask(jpeg, target["prompt"])
                boxes = perception.parse_boxes(answer, model.name, width, height)
                decision = behavior.face_target(boxes, b["center_tolerance"], settings["head"]["mirrored"])
                log("%.1fs  %d box%s  -> %s: %s" % (
                    seconds, len(boxes), "" if len(boxes) == 1 else "es", decision.action, decision.reason))
                journal.write(photo=[width, height], model=model.name, seconds=round(seconds, 2),
                              boxes=[vars(x) for x in boxes], action=decision.action, reason=decision.reason,
                              dry_run=args.dry_run)

                if decision.action in ("turn_left", "turn_right") and not args.dry_run:
                    w = b["turn_speed"] if decision.action == "turn_left" else -b["turn_speed"]
                    body.move(w=w, duration_s=b["turn_ms"] / 1000.0)
                    pause = b["settle_ms"] / 1000.0
            except Exception as e:   # one bad look must not end the brain
                log("error: %s" % e)
                pause = ERROR_PAUSE_S
                if not args.dry_run:
                    try:
                        body.stop()
                    except Exception:
                        pass
            if args.once:
                break
            time.sleep(pause)
    except KeyboardInterrupt:
        log("stopping")
    finally:
        if not args.dry_run:
            try:
                body.stop()
            except Exception as e:
                log("could not stop the robot: %s" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
