"""The verbs the brain knows. Each is generic: what to look for is a text
argument, never a setting. Movements are short and always end stopped.

Every verb returns a Result; the planner chains verbs and stops at the first
one that fails."""

import threading
import time
from dataclasses import dataclass

from . import perception

# Turning steps: (command 0-1, milliseconds).
AMOUNTS = {"small": (0.5, 300), "medium": (0.5, 600), "large": (0.5, 1000)}
# Walking steps are longer: the robot needs about a quarter of a second to
# start its gait, and the walk moves one leg at a time, so a short command
# barely lifts a foot before it stops.
FORWARD_AMOUNTS = {"small": (0.5, 1000), "medium": (0.5, 2000), "large": (0.5, 3000)}
SETTLE_S = 0.4                 # after a move, so the next photo is sharp
CENTER_TOLERANCE = 0.12        # centred: within this share of the width from the middle
FACE_MAX_LOOKS = 10
FACE_TURN_MS_PER_OFFSET = 1500  # turn length for a target at the edge (offset 0.5) is half of this
FACE_TURN_MS = (200, 700)
SEARCH_MAX_STEPS = 12          # roughly one full turn with medium steps (to be measured)
SEARCH_FORWARD_EVERY = 4       # when not only turning: a small step forward every few turns
APPROACH_MAX_STEPS = 15
APPROACH_NEAR_WIDTH = 0.4      # close enough: the target fills this share of the width...
APPROACH_NEAR_BOTTOM = 0.95    # ...or reaches the bottom of the picture (things on the floor)
APPROACH_LOST_LOOKS = 2
# reach: walks without stopping while the eyes keep looking.
REACH_TICK_S = 0.25            # how often the command is sent (it also keeps the walk alive)
REACH_SPEED = 1.0              # forward command while heading to the target: full throttle
REACH_TURN_GAIN = 1.6          # turn command per unit of offset (target at the edge: 0.5 -> 0.8)
REACH_MAX_TURN = 0.6
REACH_SEARCH_TURN = 0.4        # turning on the spot while the target is not in sight
REACH_LOST_LOOKS = 2           # looks without the target before searching again
REACH_STALE_S = 3.0            # no new look for this long: stand still until one comes
REACH_MAX_S = 90.0
REACH_NEAR_WIDTH = 0.4         # close enough: the target fills this share of the width
REACH_ANSWER_TOKENS = 160


@dataclass
class Result:
    ok: bool
    message: str


def reach_prompt(target):
    """The same question as look (forcing a JSON object made the model miss the
    target), plus a last line about the way ahead."""
    return (look_prompt(target) + " Then, on a last line, write BLOCKED if the way straight ahead of the "
            "camera is blocked by an obstacle within about 30 cm, otherwise CLEAR.")


def look_prompt(target):
    return ("Locate %s only if present. Answer only with JSON: a list of "
            '{"label": ..., "bbox_2d": [x1, y1, x2, y2]}. If there is none, answer [].' % target)


class Skills:
    """`say` prints progress to the user; `journal` records every look and move."""

    def __init__(self, head, body, model, journal, mirrored=False, dry_run=False, say=print):
        self.head = head
        self.body = body
        self.model = model
        self.journal = journal
        self.mirrored = mirrored
        self.dry_run = dry_run
        self.say = say

    # ── Perceive ──

    def _find(self, target):
        """Biggest box of `target` in a new photo, or None."""
        jpeg = self.head.photo()
        width, height = perception.photo_size(jpeg)
        answer, seconds = self.model.ask(jpeg, look_prompt(target))
        boxes = perception.parse_boxes(answer, self.model.name, width, height)
        box = max(boxes, key=lambda b: b.width * b.height) if boxes else None
        self.journal.write(verb="look", target=target, seconds=round(seconds, 2),
                           boxes=[vars(b) for b in boxes], answer=answer[:400])
        return box, seconds

    def _offset(self, box):
        """Horizontal position of the box centre: -0.5 left edge, +0.5 right edge."""
        offset = box.center_x - 0.5
        return -offset if self.mirrored else offset

    @staticmethod
    def _where(offset, box):
        side = "centre" if abs(offset) <= CENTER_TOLERANCE else ("left" if offset < 0 else "right")
        return "%s (%.0f%% of the width)" % (side, box.width * 100)

    def look(self, target):
        box, seconds = self._find(target)
        if box is None:
            return Result(False, "%s not in sight (%.1fs)" % (target, seconds))
        return Result(True, "%s seen at the %s (%.1fs)" % (target, self._where(self._offset(box), box), seconds))

    # ── Move ──

    def _move(self, verb, vx=0.0, w=0.0, ms=500):
        self.journal.write(verb=verb, vx=vx, w=w, ms=ms, dry_run=self.dry_run)
        if self.dry_run:
            self.say("    (dry run: %s for %d ms)" % (verb, ms))
            time.sleep(ms / 1000.0)
        else:
            self.body.move(vx=vx, w=w, duration_s=ms / 1000.0)
        time.sleep(SETTLE_S)

    def turn(self, direction, amount="medium"):
        if direction not in ("left", "right") or amount not in AMOUNTS:
            return Result(False, "turn needs direction left/right and amount small/medium/large")
        speed, ms = AMOUNTS[amount]
        self._move("turn", w=speed if direction == "left" else -speed, ms=ms)
        return Result(True, "turned %s (%s)" % (direction, amount))

    def forward(self, amount="small"):
        if amount not in FORWARD_AMOUNTS:
            return Result(False, "forward needs amount small/medium/large")
        speed, ms = FORWARD_AMOUNTS[amount]
        self._move("forward", vx=speed, ms=ms)
        return Result(True, "walked forward (%s)" % amount)

    def stop(self):
        if not self.dry_run:
            self.body.stop()
        return Result(True, "stopped")

    def greet(self):
        if not self.dry_run:
            self.body.action("hello")
        return Result(True, "greeted")

    # ── Move with a goal ──

    def search(self, target, only_turning=True, turn_direction="left", turn_step="medium"):
        """Turns (and, unless only_turning, sometimes steps forward) until the
        target is in sight, or gives up after about a full turn."""
        if turn_direction not in ("left", "right") or turn_step not in AMOUNTS:
            return Result(False, "search needs turn_direction left/right and turn_step small/medium/large")
        for step in range(SEARCH_MAX_STEPS + 1):
            box, _ = self._find(target)
            if box is not None:
                return Result(True, "found %s at the %s" % (target, self._where(self._offset(box), box)))
            if step == SEARCH_MAX_STEPS:
                break
            self.say("    not in sight, turning (%d/%d)" % (step + 1, SEARCH_MAX_STEPS))
            if not only_turning and step and step % SEARCH_FORWARD_EVERY == 0:
                self.forward("small")
            self.turn(turn_direction, turn_step)
        return Result(False, "%s not found after a full search" % target)

    def face(self, target):
        """Turns until the target is centred."""
        for _ in range(FACE_MAX_LOOKS):
            box, _ = self._find(target)
            if box is None:
                return Result(False, "%s not in sight" % target)
            offset = self._offset(box)
            if abs(offset) <= CENTER_TOLERANCE:
                return Result(True, "facing %s" % target)
            ms = int(min(FACE_TURN_MS[1], max(FACE_TURN_MS[0], abs(offset) * FACE_TURN_MS_PER_OFFSET)))
            self.say("    %s at the %s, turning" % (target, self._where(offset, box)))
            self._move("turn", w=AMOUNTS["small"][0] * (1 if offset < 0 else -1), ms=ms)
        return Result(False, "could not centre %s" % target)

    def reach(self, target):
        """Walks towards the target without stopping: the eyes keep looking in
        the background and every new look corrects the heading. Stops when it
        is close, when the way is blocked, or after REACH_MAX_S. Searches by
        turning on the spot while it is not in sight; stands still whenever
        the eyes fall behind."""
        latest = {"look": None, "count": 0, "error": None}
        lock = threading.Lock()
        done = threading.Event()

        def eyes():
            while not done.is_set():
                try:
                    jpeg = self.head.photo()
                    width, height = perception.photo_size(jpeg)
                    answer, seconds = self.model.ask(jpeg, reach_prompt(target), max_tokens=REACH_ANSWER_TOKENS)
                    boxes = perception.parse_boxes(answer, self.model.name, width, height)
                    blocked = perception.says_blocked(answer)
                    box = max(boxes, key=lambda b: b.width * b.height) if boxes else None
                    self.journal.write(verb="reach-look", target=target, seconds=round(seconds, 2),
                                       boxes=[vars(b) for b in boxes], blocked=blocked, answer=answer[:400])
                    with lock:
                        latest["look"] = (time.monotonic(), box, blocked)
                        latest["count"] += 1
                except Exception as e:   # the driver stands still until looks come again
                    with lock:
                        latest["error"] = str(e)
                    time.sleep(1.0)

        thread = threading.Thread(target=eyes, daemon=True)
        thread.start()
        started = time.monotonic()
        seen_count, lost, last_seen = 0, 0, 0
        command = (0.0, 0.0)
        try:
            while time.monotonic() - started < REACH_MAX_S:
                with lock:
                    look, count = latest["look"], latest["count"]
                now = time.monotonic()
                if look is None or now - look[0] > REACH_STALE_S:
                    command = (0.0, 0.0)          # no fresh look: stand still
                elif count != seen_count:         # a new look: decide again
                    seen_count = count
                    _, box, blocked = look
                    if blocked:
                        return Result(False, "the way is blocked")
                    if box is not None:
                        lost, last_seen = 0, count
                        if box.width >= REACH_NEAR_WIDTH:
                            return Result(True, "reached %s" % target)
                        offset = self._offset(box)
                        w = max(-REACH_MAX_TURN, min(REACH_MAX_TURN, -offset * REACH_TURN_GAIN))
                        vx = REACH_SPEED * max(0.0, 1.0 - 2.0 * abs(offset))
                        command = (vx, w)
                        self.say("    %s at the %s: %s" % (target, self._where(offset, box),
                                 "straight on" if abs(offset) <= CENTER_TOLERANCE else
                                 "steering " + ("left" if w > 0 else "right")))
                    else:
                        lost += 1
                        if last_seen and lost < REACH_LOST_LOOKS:
                            self.say("    lost sight of %s, keeping the heading" % target)
                        else:
                            command = (0.0, REACH_SEARCH_TURN)
                            self.say("    %s not in sight, turning to look" % target)
                self._drive(*command)
                time.sleep(REACH_TICK_S)
            return Result(False, "gave up reaching %s" % target)
        finally:
            done.set()
            self.stop()

    def _drive(self, vx, w):
        if not self.dry_run and (vx or w):
            self.body.drive(vx=vx, w=w)
        elif not self.dry_run:
            self.body.stop()

    def approach(self, target):
        """Walks towards the target, keeping it centred, and stops when it is close."""
        lost = 0
        for _ in range(APPROACH_MAX_STEPS):
            box, _ = self._find(target)
            if box is None:
                lost += 1
                if lost >= APPROACH_LOST_LOOKS:
                    return Result(False, "lost sight of %s" % target)
                continue
            lost = 0
            if box.width >= APPROACH_NEAR_WIDTH or box.y2 >= APPROACH_NEAR_BOTTOM:
                return Result(True, "reached %s" % target)
            offset = self._offset(box)
            if abs(offset) > CENTER_TOLERANCE:
                self.say("    %s at the %s, turning" % (target, self._where(offset, box)))
                self.turn("left" if offset < 0 else "right", "small")
            else:
                self.say("    %s ahead (%.0f%% of the width), walking" % (target, box.width * 100))
                self.forward("small")
        return Result(False, "did not reach %s" % target)


# What the planner may use:
#   verb -> ({argument: (type, required, allowed values or None, description)}, summary)
# Argument names say what they are about (turn_step, not size), so a word that
# describes the target ("a large fan") is never taken for a setting.
SIDES = ("left", "right")
STEPS = tuple(AMOUNTS)
TARGET = (str, True, None, "what it is about, described in English")
VERBS = {
    "look": ({"target": TARGET}, "Check whether something is in sight and where."),
    "turn": ({"direction": (str, True, SIDES, "which way to turn"),
              "amount": (str, False, STEPS, "how far (default medium)")},
             "Turn on the spot."),
    "forward": ({"amount": (str, False, STEPS, "how far (default small)")},
                "Walk forward a little."),
    "stop": ({}, "Stop moving."),
    "greet": ({}, "Wave hello."),
    "search": ({"target": TARGET,
                "only_turning": (bool, False, None, "search by turning on the spot only (default true)"),
                "turn_direction": (str, False, SIDES, "which way to turn while searching (default left)"),
                "turn_step": (str, False, STEPS, "the size of each turning step (default medium)")},
               "Turn (and, unless only_turning, step forward) until the target is in sight."),
    "face": ({"target": TARGET}, "Turn until the target is straight ahead. It must be in sight."),
    "approach": ({"target": TARGET}, "Walk to the target in short steps, looking between them, and stop "
                                     "close to it. It must be in sight."),
    "reach": ({"target": TARGET}, "Walk to the target without stopping, steering as it goes, and stop close "
                                  "to it or if the way is blocked. Searches by turning if it is not in sight."),
}
