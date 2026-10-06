"""The brain's terminal.

    python -m hymeno check      # is the body there, and does the camera send photos?
"""

import argparse
import os
import sys
import time

from . import __version__, config
from .clients import Body, Head


def say(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


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


def main():
    parser = argparse.ArgumentParser(prog="hymeno", description="Remote brain for a robot.")
    parser.add_argument("command", choices=["check"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    settings = config.load(args.config)
    say("hymeno-core %s" % __version__)
    return 0 if check(settings) else 1


if __name__ == "__main__":
    sys.exit(main())
