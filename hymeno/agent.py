"""The agent: works towards a goal given in plain words, deciding again at every
step, like a simple animal.

First the goal is classified with one short question (no photo): a question
about what it sees is answered at once, a house rule is remembered or
forgotten, and anything else is a task with a target described in English.

A task runs this cycle:

    observe (photo) -> interpret it with the goal, the rules, what was seen
    where, and what happened so far -> decide one action -> act -> remember

until the goal is reached (checked with a second look), it gives up, or it
runs out of steps or time. Interpreting and deciding are one short question to
the vision model, with the photo.

The model decides the strategy; the code does the precise work and keeps the
limits:
- when the model says the target is in view, the box-drawing look confirms it
  and gives its side and distance (the model's own word is not reliable);
- "center_on_goal" and "approach_goal" hand over to the face and reach verbs,
  which steer with those boxes;
- only a few short actions exist; it never walks forward into floor the model
  says is not free;
- when walking does not change the view, it is stuck: it backs off and turns;
- "done" is only accepted after a second look agrees."""

import json
import time
from dataclasses import dataclass, field

from . import perception
from .memory import Rules, WorldMemory

MOVES = ("look_around", "turn_left", "turn_right", "forward", "back", "strafe_left", "strafe_right")
ACTIONS = MOVES + ("center_on_goal", "approach_goal", "greet", "done", "give_up")
AMOUNTS = ("small", "medium", "large")
MAX_STEPS = 60
MAX_SECONDS = 600
MEMORY_STEPS = 8               # recent steps the model is shown
MAX_BAD_ANSWERS = 3            # unreadable answers in a row before stopping
DECISION_TOKENS = 90
INTENT_TOKENS = 120
CHECK_TOKENS = 60
MAX_REJECTED_DONE = 2          # "done" the second look disagreed with, before accepting anyway
STUCK_CHANGE = 4.0             # view change (0-255) below this after walking: it did not move
STUCK_LIMIT = 2                # stuck walks in a row before backing off by itself
# Rough size of each action, for the model to reason about (to be measured).
TURN_DEGREES = {"small": 10, "medium": 30, "large": 50}
STEP_CM = {"small": 10, "medium": 20, "large": 30}

PROMPT = """You are the mind of a small four-legged robot. The photo is what its camera sees right
now, from just above the floor, looking straight ahead. You cannot see behind you.

Goal, from the user: {goal}
Target: {target}

House rules the user taught you:
{rules}

Things seen lately, and where they are from where you face now (rough):
{world}

Turns are roughly {turns} degrees (small/medium/large); steps roughly {steps} cm.
So far for this goal: {totals}
Recent steps, oldest first:
{history}
{notes}
Actions: look_around (turn to look for the target, always the same way), turn_left, turn_right,
forward, back, strafe_left, strafe_right (all by "amount"), center_on_goal (face the target, it
must be in view), approach_goal (walk to the target and stop close, it must be in view), greet,
done (goal achieved), give_up (it cannot be achieved).

How to work:
- If the target was seen lately, turn towards where it was. Otherwise look_around again and
  again; after a full turn (about 360 degrees) without it, move to a new place, then look again.
- When the target is in view: approach_goal to go to it, or center_on_goal just to face it.
- Never go forward if the floor straight ahead is not free.
- Say the target is in view only if you can really see it in this photo.

Answer only with one JSON object, very short values:
{{"see": "main things in view, comma separated",
  "goal": "left" or "centre" or "right" or "no",
  "dist": "near" or "mid" or "far",
  "free": three letters for the floor left, centre, right: Y if free, N if not, e.g. "YYN",
  "do": one action,
  "amount": "small" or "medium" or "large"}}"""

INTENT = """A user gives an order to a small four-legged robot with a camera. Classify it.
Order: {goal}
Answer only with JSON: {{"kind": "question" or "remember" or "forget" or "task",
 "rule": "for remember or forget: only the rule itself, short, without words like 'remember that', in the order's language",
 "target": "for a task: the thing it is about, described in English precisely enough to find it
            in a photo (e.g. 'a square of yellow electrical tape on the floor'); else empty",
 "action": "for a task: find (look for it), face (turn towards it), reach (go to it) or other"}}
- question: asks what the robot sees or about its surroundings
- remember / forget: teaches or removes a rule about the house
- task: anything the robot has to do"""

CHECK = """You check the work of a small four-legged robot. The photo is what its camera sees now,
from just above the floor, looking straight ahead.
The robot's goal: {goal}
The robot believes the goal is achieved. Is it, judging by this photo? Answer "yes" or "no",
then a few words why."""


@dataclass
class Step:
    see: str
    goal: str
    dist: str
    action: str
    amount: str
    outcome: str = ""

    def line(self, n):
        return "%d. saw: %s | target in view: %s (%s) | did: %s %s%s" % (
            n, self.see, self.goal, self.dist, self.action, self.amount,
            " -> " + self.outcome if self.outcome else "")


@dataclass
class WorkingMemory:
    goal: str
    target: str = ""
    steps: list = field(default_factory=list)
    notes: list = field(default_factory=list)   # for the next step only
    turned: int = 0                             # degrees, left positive
    walked: int = 0                             # cm, forward positive
    stuck: int = 0
    rejected_done: int = 0

    def totals(self):
        return "turned about %d degrees to the %s, walked about %d cm" % (
            abs(self.turned), "left" if self.turned >= 0 else "right", self.walked)

    def history(self):
        recent = self.steps[-MEMORY_STEPS:]
        first = len(self.steps) - len(recent) + 1
        return "\n".join(s.line(first + i) for i, s in enumerate(recent)) or "none yet"


def parse_json_object(answer):
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        d = json.loads(answer[start:end + 1])
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def parse_decision(answer):
    """The decision, checked and filled in; None if unusable."""
    d = parse_json_object(answer)
    if d is None or d.get("do") not in ACTIONS:
        return None
    if d.get("amount") not in AMOUNTS:
        d["amount"] = "small"
    goal = str(d.get("goal", "no")).lower()
    d["goal"] = "centre" if goal == "center" else goal if goal in ("left", "centre", "right") else "no"
    if d.get("dist") not in ("near", "mid", "far"):
        d["dist"] = "far"
    letters = (str(d.get("free", "YYY")).upper().replace(" ", "") + "YYY")[:3]
    d["free"] = {side: letters[i] != "N" for i, side in enumerate(("left", "centre", "right"))}
    d["see"] = str(d.get("see") or "").strip()
    d["objects"] = [o.strip() for o in d["see"].split(",") if o.strip()][:8]
    return d


class Agent:
    """One per session: the world memory lasts across goals."""

    def __init__(self, skills, data_dir, say=print):
        self.skills = skills
        self.stuck_reflex = not skills.dry_run   # in a dry run the view never changes
        self.say = say
        self.rules = Rules(data_dir)
        self.world = WorldMemory()

    # ── Helpers ──

    def _turn(self, memory, side, amount):
        self.skills.turn(side, amount)
        degrees = TURN_DEGREES[amount] * (1 if side == "left" else -1)
        memory.turned += degrees
        self.world.turned(degrees)

    def _check_done(self, goal):
        jpeg = self.skills.head.photo()
        answer, seconds = self.skills.model.ask(jpeg, CHECK.format(goal=goal), max_tokens=CHECK_TOKENS)
        self.skills.journal.write(verb="agent-check", goal=goal, seconds=round(seconds, 2), answer=answer[:300])
        return answer.strip().lower().startswith("yes"), answer.strip()

    def _safe_forward_side(self, free):
        """Where to turn when the floor ahead is not free."""
        if free["left"] and not free["right"]:
            return "left"
        if free["right"] and not free["left"]:
            return "right"
        return None

    def _intent(self, goal):
        answer, _ = self.skills.model.ask_text(INTENT.format(goal=goal), json_only=True)
        d = parse_json_object(answer) or {}
        kind = d.get("kind") if d.get("kind") in ("question", "remember", "forget", "task") else "task"
        action = d.get("action") if d.get("action") in ("find", "face", "reach") else "other"
        return kind, str(d.get("rule") or "").strip(), str(d.get("target") or "").strip(), action

    def _confirm(self, memory, d):
        """The model's own word that the target is in view is not reliable: check
        it with the box-drawing look, and take the side and distance from the box."""
        box = self.skills.find(memory.target)
        if box is None:
            memory.notes.append("Note: you said the target was in view, but a closer look did not find it.")
            d["goal"] = "no"
            return
        offset = self.skills._offset(box)
        d["goal"] = "centre" if abs(offset) <= 0.12 else ("left" if offset < 0 else "right")
        d["dist"] = "near" if box.width >= 0.35 else "mid" if box.width >= 0.12 else "far"

    # ── The loop ──

    def _fast_task(self, goal, action, target, cls):
        """Find, face or reach a thing the detector knows: no step-by-step
        thinking needed; the vision model only checks the result."""
        self.say("  target: %s -> fast eyes (%s), %s" % (target, cls, action))
        found = self.skills.search(target)
        if not found.ok:
            return "gave up: %s" % found.message
        self.say("      %s" % found.message)
        done = self.skills.face(target) if action == "face" else self.skills.reach(target) if action == "reach" else found
        if done is not found:
            self.say("      %s" % done.message)
        if not done.ok:
            return "gave up: %s" % done.message
        ok, why = self._check_done(goal)
        return "done%s: %s" % ("" if ok else " (not confirmed)", why)

    def run(self, goal):
        kind, rule, target, action = self._intent(goal)
        if kind == "question":
            return "answer: %s" % self.skills.describe(goal).message
        if kind == "remember":
            return ("remembered: %s" if self.rules.add(rule or goal) else "already known: %s") % (rule or goal)
        if kind == "forget":
            removed = self.rules.remove(rule or goal)
            return "forgot: %s" % "; ".join(removed) if removed else "no rule matched: %s" % (rule or goal)
        cls = self.skills.fast_class(target) if target else None
        if cls and action in ("find", "face", "reach"):
            return self._fast_task(goal, action, target, cls)
        memory = WorkingMemory(goal, target=target)
        self.say("  target: %s" % (target or "(none)"))
        started = time.monotonic()
        bad = 0
        previous, moved = None, False
        result = "stopped after %d steps" % MAX_STEPS
        for n in range(1, MAX_STEPS + 1):
            if time.monotonic() - started > MAX_SECONDS:
                result = "ran out of time"
                break
            jpeg = self.skills.head.photo()

            # Reflex: walking that does not change the view means it is stuck.
            if self.stuck_reflex and previous is not None and moved:
                if perception.view_change(previous, jpeg) < STUCK_CHANGE:
                    memory.stuck += 1
                    memory.notes.append("Note: after the last move the view did not change; you seem stuck.")
                else:
                    memory.stuck = 0
            previous, moved = jpeg, False
            if memory.stuck >= STUCK_LIMIT:
                self.say("  %2d. stuck: backing off and turning" % n)
                self.skills.back("small")
                self._turn(memory, "left", "medium")
                memory.steps.append(Step("(stuck)", "no", "unknown", "back+turn_left", "small", "backed off"))
                memory.stuck, moved = 0, False
                continue

            prompt = PROMPT.format(goal=goal, target=memory.target or "(none)", rules=self.rules.text(),
                                   world=self.world.text(),
                                   turns="/".join(str(v) for v in TURN_DEGREES.values()),
                                   steps="/".join(str(v) for v in STEP_CM.values()),
                                   totals=memory.totals(), history=memory.history(),
                                   notes="\n".join(memory.notes) + "\n" if memory.notes else "")
            memory.notes = []
            answer, seconds = self.skills.model.ask(jpeg, prompt, max_tokens=DECISION_TOKENS)
            d = parse_decision(answer)
            self.skills.journal.write(verb="agent", goal=goal, step=n, seconds=round(seconds, 2),
                                      answer=answer[:1000], decision=d)
            if d is None:
                bad += 1
                self.say("  %2d. (could not read the answer, looking again)" % n)
                if bad >= MAX_BAD_ANSWERS:
                    result = "stopped: the model's answers could not be read"
                    break
                continue
            bad = 0

            if d["goal"] != "no" and memory.target:
                self._confirm(memory, d)
            self.world.saw(d["objects"])
            if d["goal"] != "no":
                self.world.saw([memory.target])
            action, amount = d["do"], d["amount"]
            if action in ("center_on_goal", "approach_goal") and d["goal"] == "no":
                action = "look_around"

            # Limits the model cannot override.
            if action == "forward" and not d["free"]["centre"]:
                side = self._safe_forward_side(d["free"])
                action, amount = ("turn_" + side, "small") if side else ("back", "small")
                memory.notes.append("Note: the floor ahead was not free, so you did %s instead of forward." % action)
            if action in ("center_on_goal", "approach_goal") and not memory.target:
                action = "look_around"

            free = "".join(side[0].upper() if ok else "-" for side, ok in d["free"].items())
            self.say("  %2d. %s | target: %s (%s) | floor L-C-R: %s -> %s%s  [%.1fs]" % (
                n, d["see"], d["goal"], d["dist"], free, action,
                " " + amount if action in MOVES else "", seconds))
            self.skills._checkpoint(jpeg, [], "decision above")

            step = Step(d["see"], d["goal"], d["dist"], action, amount if action in MOVES else "")
            memory.steps.append(step)

            if action == "give_up":
                result = "gave up"
                break
            if action == "done":
                ok, why = self._check_done(goal)
                if ok or memory.rejected_done >= MAX_REJECTED_DONE:
                    result = "done%s: %s" % ("" if ok else " (not confirmed)", why)
                    break
                memory.rejected_done += 1
                step.outcome = "a second look disagreed: " + why
                memory.notes.append("Note: you said done, but a second look disagreed: %s" % why)
                self.say("      second look disagrees: %s" % why)
                continue
            if action == "greet":
                self.skills.greet()
            elif action == "look_around":
                self._turn(memory, "left", amount if amount != "small" else "medium")
                moved = True
            elif action in ("turn_left", "turn_right"):
                self._turn(memory, action[5:], amount)
                moved = True
            elif action == "forward":
                self.skills.forward(amount)
                memory.walked += STEP_CM[amount]
                moved = True
            elif action == "back":
                self.skills.back(amount)
                memory.walked -= STEP_CM[amount]
                moved = True
            elif action in ("strafe_left", "strafe_right"):
                self.skills.strafe(action[7:], amount)
                moved = True
            elif action == "center_on_goal":
                r = self.skills.face(memory.target)
                step.outcome = r.message
                previous = None          # it turned an unknown amount
            elif action == "approach_goal":
                r = self.skills.reach(memory.target)
                step.outcome = r.message
                previous = None
            if step.outcome:
                self.say("      %s" % step.outcome)

        self.skills.stop()
        self.say("summary: %d steps, %s; target: %s" % (len(memory.steps), memory.totals(), memory.target or "-"))
        return result
