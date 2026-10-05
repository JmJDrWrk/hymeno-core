"""The brain's terminal: give it goals in plain words.

    python -m hymeno                      # asks for orders, one after another
    python -m hymeno "find the fan"       # one order, then exit
    python -m hymeno --dry-run            # looks and plans, never moves the robot

At the prompt, a goal in any language is worked on by the agent, which looks
and decides again at every step ("find your charging station: a square of
yellow tape on the floor"); "/plan goal" plans it once into verbs instead;
"/verb description name=value" runs one verb directly (e.g.
"/search a fan turn_direction=right"),
"/help" lists the verbs. Ctrl+C stops the robot and the current order; at the
prompt it exits."""

import argparse
import sys
import time

try:
    import readline  # noqa: F401  line editing that erases whole characters, and history with the arrows
except ImportError:
    pass

from . import __version__, config, detector as fast_eyes, planner
from .clients import Body, Head, VisionModel
from .journal import Journal
from .agent import Agent
from .skills import VERBS, Result, Skills


def say(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def run_step(skills, verb, args):
    say("  -> %s(%s)" % (verb, ", ".join("%s=%r" % kv for kv in args.items())))
    result = getattr(skills, verb)(**args)
    say("     %s: %s" % ("done" if result.ok else "failed", result.message))
    return result


def run_order(skills, model, order):
    steps, sentence, seconds = planner.plan(model, order)
    if sentence:
        say("%s  (planned in %.1fs)" % (sentence, seconds))
    if not steps:
        say("nothing to do")
        return
    for verb, args in steps:
        if not run_step(skills, verb, args).ok:
            break
    skills.stop()


def clean(line):
    """Drops half characters: without line editing, erasing an accented letter
    in the terminal can leave one of its bytes behind."""
    return line.encode("utf-8", "surrogateescape").decode("utf-8", "ignore").strip()


def direct(line):
    """'/verb words... name=value...' -> (verb, args), for trying a verb
    without the planner. Settings always go as name=value; the remaining
    words are the target, so a word describing it is never taken for one."""
    words = line[1:].split()
    verb = words[0] if words else ""
    if verb not in VERBS:
        raise ValueError("unknown verb %r; /help lists them" % verb)
    params = VERBS[verb][0]
    args, rest = {}, []
    for word in words[1:]:
        name, sep, value = word.partition("=")
        if sep and name in params:
            kind = params[name][0]
            if kind is bool:
                if value.lower() not in ("true", "false", "yes", "no", "1", "0"):
                    raise ValueError("%s must be true or false" % name)
                args[name] = value.lower() in ("true", "yes", "1")
            else:
                args[name] = value
        else:
            rest.append(word)
    if rest:
        free = "target" if "target" in params else "question" if "question" in params else None
        if free is None:
            raise ValueError("%s takes no description; settings go as name=value" % verb)
        args[free] = " ".join(rest)
    return planner.validate([{"verb": verb, "args": args}])[0]


def help_text():
    lines = ["A goal in plain words is worked on step by step by the agent (Ctrl+C stops it).",
             "/plan goal plans it once into these verbs instead.",
             "/rules shows the house rules it was taught; /world what it saw lately and where;",
             "/detect what the fast detector (YOLO) sees right now.",
             "/verb runs one directly: the description first, settings as name=value, e.g.",
             "  /search a large fan on the right turn_direction=left turn_step=small",
             "  /turn direction=right amount=large", ""]
    for verb, (args, summary) in VERBS.items():
        lines.append("  /%-9s %s" % (verb, summary))
        for name, (_, required, choices, text) in args.items():
            if name != "target":
                lines.append("  %11s%s=%s  %s" % ("", name, "|".join(choices) if choices else "true|false", text))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(prog="hymeno", description="Remote brain for a robot.")
    parser.add_argument("order", nargs="*", help="an order in plain words; without it, asks for orders")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="look and plan, but never move the robot")
    args = parser.parse_args()

    settings = config.load(args.config)
    head = Head(settings["head"]["url"])
    body = Body(settings["body"]["url"], max_speed=settings["max_speed"])
    model = VisionModel(**settings["model"])
    skills = Skills(head, body, model, Journal(settings["data_dir"]),
                    mirrored=settings["head"]["mirrored"], dry_run=args.dry_run, say=say,
                    detector=fast_eyes.load(settings, say=say))

    agent = Agent(skills, settings["data_dir"], say=say)

    say("hymeno-core %s, model %s%s" % (__version__, model.name,
                                        " (dry run: the robot will not move)" if args.dry_run else ""))
    if not args.dry_run:
        body.state()   # fail early if the robot is not there

    orders = [" ".join(args.order)] if args.order else None
    while True:
        try:
            line = clean(orders.pop(0) if orders else input("hymeno> "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(help_text())
            continue
        if line == "/rules":
            print(agent.rules.text())
            continue
        if line == "/world":
            print(agent.world.text())
            continue
        if line == "/detect":
            if skills.detector is None:
                print("no fast detector")
            else:
                started = time.monotonic()
                boxes = skills.detector.detect(head.photo())
                print("%d thing(s) in %.0f ms: %s" % (len(boxes), (time.monotonic() - started) * 1000,
                      ", ".join("%s at %.0f%%" % (b.label, b.center_x * 100) for b in boxes) or "-"))
            continue
        if line:
            try:
                if line.startswith("/plan "):
                    run_order(skills, model, line[len("/plan "):])
                elif line.startswith("/"):
                    run_step(skills, *direct(line))
                    skills.stop()
                else:
                    say("goal: %s" % line)
                    say(agent.run(line))
            except KeyboardInterrupt:
                say("interrupted")
            except Exception as e:   # one bad order must not end the brain
                say("error: %s" % e)
            finally:
                try:
                    skills.stop()
                except Exception as e:
                    say("could not stop the robot: %s" % e)
        if args.order:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
