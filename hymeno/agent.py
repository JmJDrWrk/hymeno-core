"""The agent: works towards a goal given in plain words, deciding again at every
step, like a simple animal.

One cycle:

    observe (photo) -> interpret it with the goal, the rules, what was seen
    where, and what happened so far -> decide one action -> act -> remember

until the goal is reached (checked with a second look), it gives up, or it
runs out of steps or time. Interpreting and deciding are one question to the
vision model, with the photo.

The model decides the strategy; the code does the precise work and keeps the
limits:
- "center_on_goal" and "approach_goal" hand over to the face and reach verbs,
  which steer with the boxes the model draws (proven, and fast to react);
- only a few short actions exist; it never walks forward into a part of the
  floor the model says is not free;
- when walking does not change the view, it is stuck: it backs off and turns;
- "done" is only accepted after a second look agrees."""

import json
import time
from dataclasses import dataclass, field

from . import perception
from .memory import Rules, WorldMemory

MOVES = ("look_around", "turn_left", "turn_right", "forward", "back", "strafe_left", "strafe_right")
ACTIONS = MOVES + ("center_on_goal", "approach_goal", "greet", "remember", "forget", "answer", "done", "give_up")
AMOUNTS = ("small", "medium", "large")
MAX_STEPS = 60
MAX_SECONDS = 600
MEMORY_STEPS = 8               # recent steps the model is shown
MAX_BAD_ANSWERS = 3            # unreadable answers in a row before stopping
DECISION_TOKENS = 260
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

House rules the user taught you:
{rules}

Things seen lately, and where they are from where you face now (rough):
{world}

Your body: turns are roughly {turns} degrees (small/medium/large); steps roughly {steps} cm.
So far for this goal: {totals}
Recent steps, oldest first:
{history}
{notes}
Actions:
- look_around: turn to look for something (always the same way, to cover the room)
- turn_left, turn_right: turn by "amount"
- forward, back, strafe_left, strafe_right: walk by "amount"
- center_on_goal: turn until the target is straight ahead (it must be in view)
- approach_goal: walk to the target and stop close to it, steering by itself (it must be in view)
- greet: wave hello
- remember / forget: add or remove a house rule, given in "text"
- answer: reply to a question about what you see, with the reply in "text"; this ends the goal
- done: the goal is achieved; give_up: it cannot be achieved

How to work:
- A question about what you see: "answer" straight away. A rule to remember or forget:
  "remember" or "forget" it, then "done".
- To find something: if it was seen lately, turn towards where it was. Otherwise
  "look_around" again and again. After a full turn (about 360 degrees) without finding it,
  move to a new place (forward if the floor ahead is free), then look around again.
- When the target is in view: "approach_goal" to go to it, or "center_on_goal" just to face it.
- Never go forward if the floor straight ahead is not free; turn towards free floor instead.
- If a note says you seem stuck, go back and turn.

Answer only with one JSON object, short values:
{{"see": "what you see, a few words",
  "objects": ["main things in view, a few words each"],
  "target": "the thing the goal is about, described in English precisely enough to find it in a photo",
  "goal": "left" or "centre" or "right" or "no",
  "dist": "near" or "mid" or "far" or "unknown",
  "free": {{"left": true or false, "centre": true or false, "right": true or false}},
  "do": one of the actions,
  "amount": "small" or "medium" or "large",
  "text": "the rule or the answer, if any, in the language of the goal",
  "say": "a short sentence for the user about what you are doing, in the language of the goal"}}"""

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


def parse_decision(answer):
    """The first JSON object in the answer, checked and filled in; None if unusable."""
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        d = json.loads(answer[start:end + 1])
    except ValueError:
        return None
    if not isinstance(d, dict) or d.get("do") not in ACTIONS:
        return None
    if d.get("amount") not in AMOUNTS:
        d["amount"] = "small"
    if d.get("goal") not in ("left", "centre", "center", "right", "no"):
        d["goal"] = "no"
    d["goal"] = "centre" if d["goal"] == "center" else d["goal"]
    free = d.get("free") if isinstance(d.get("free"), dict) else {}
    d["free"] = {side: free.get(side) is not False for side in ("left", "centre", "right")}
    d["objects"] = [str(o) for o in d.get("objects", []) if o][:8] if isinstance(d.get("objects"), list) else []
    for key, default in (("see", ""), ("target", ""), ("dist", "unknown"), ("text", ""), ("say", "")):
        d[key] = str(d.get(key) or default).strip()
    return d


class Agent:
    """One per session: the world memory lasts across goals."""

    def __init__(self, skills, data_dir, say=print):
        self.skills = skills
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

    # ── The loop ──

    def run(self, goal):
        memory = WorkingMemory(goal)
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
            if previous is not None and moved:
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

            prompt = PROMPT.format(goal=goal, rules=self.rules.text(), world=self.world.text(),
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

            self.world.saw(d["objects"])
            if d["target"] and not memory.target:
                memory.target = d["target"]
            action, amount = d["do"], d["amount"]

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
            if d["say"]:
                self.say("      \"%s\"" % d["say"])

            step = Step(d["see"], d["goal"], d["dist"], action, amount if action in MOVES else "")
            memory.steps.append(step)

            if action == "answer":
                result = "answer: %s" % (d["text"] or d["see"])
                break
            if action == "give_up":
                result = "gave up: %s" % (d["say"] or "the goal cannot be achieved")
                break
            if action == "done":
                ok, why = self._check_done(goal)
                if ok or memory.rejected_done >= MAX_REJECTED_DONE:
                    result = "done: %s%s" % (d["say"] or "goal achieved", "" if ok else " (not confirmed)")
                    break
                memory.rejected_done += 1
                step.outcome = "a second look disagreed: " + why
                memory.notes.append("Note: you said done, but a second look disagreed: %s" % why)
                self.say("      second look disagrees: %s" % why)
                continue
            if action == "remember":
                step.outcome = "remembered" if self.rules.add(d["text"]) else "already known"
            elif action == "forget":
                removed = self.rules.remove(d["text"])
                step.outcome = "forgot %d rule(s)" % len(removed)
            elif action == "greet":
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
