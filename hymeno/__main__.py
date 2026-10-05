"""The brain's terminal: give it orders in plain words.

    python -m hymeno                      # asks for orders, one after another
    python -m hymeno "find the fan"       # one order, then exit
    python -m hymeno --dry-run            # looks and plans, never moves the robot

At the prompt, an order in any language is planned by the model into verbs;
"/verb arguments" runs one verb directly (e.g. "/face a yellow square"),
"/help" lists the verbs. Ctrl+C stops the robot and the current order; at the
prompt it exits."""

import argparse
import sys
import time

from . import __version__, config, planner
from .clients import Body, Head, VisionModel
from .journal import Journal
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


def direct(line):
    """'/verb words...' -> (verb, args), for trying a verb without the planner."""
    words = line[1:].split()
    verb, rest = words[0], words[1:]
    if verb not in VERBS:
        raise ValueError("unknown verb %r; /help lists them" % verb)
    params = VERBS[verb][0]
    if verb == "turn":
        args = {"direction": rest[0] if rest else "left"}
        if len(rest) > 1:
            args["amount"] = rest[1]
    elif verb == "forward":
        args = {"amount": rest[0]} if rest else {}
    elif "target" in params:
        if not rest:
            raise ValueError("%s needs something to look for" % verb)
        args = {"target": " ".join(rest)}
    else:
        args = {}
    return verb, args


def help_text():
    lines = ["Orders in plain words are planned into these verbs; /verb runs one directly:"]
    for verb, (args, summary) in VERBS.items():
        lines.append("  /%-9s %s" % (verb, summary))
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
                    mirrored=settings["head"]["mirrored"], dry_run=args.dry_run, say=say)

    say("hymeno-core %s, model %s%s" % (__version__, model.name,
                                        " (dry run: the robot will not move)" if args.dry_run else ""))
    if not args.dry_run:
        body.state()   # fail early if the robot is not there

    orders = [" ".join(args.order)] if args.order else None
    while True:
        try:
            line = orders.pop(0) if orders else input("hymeno> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(help_text())
            continue
        if line:
            try:
                if line.startswith("/"):
                    run_step(skills, *direct(line))
                    skills.stop()
                else:
                    run_order(skills, model, line)
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
