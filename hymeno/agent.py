"""The agent: works towards a goal given in plain words, deciding again at every
step. One cycle:

    observe (photo) -> interpret it with the goal and what happened so far ->
    decide one short action -> act -> remember -> repeat

until the goal is reached, it gives up, or it runs out of steps. Observing,
interpreting and deciding are one question to the vision model, which sees
the photo together with the goal and the working memory (the recent steps,
and how far it has turned and walked). The body's limits stay in code: the
model can only choose among a few short actions, and never walks forward
when it says the way ahead is blocked."""

import json
import time
from dataclasses import dataclass, field

ACTIONS = ("turn_left", "turn_right", "forward", "greet", "done", "give_up")
AMOUNTS = ("small", "medium", "large")
MAX_STEPS = 40
MAX_SECONDS = 300
MEMORY_STEPS = 8               # recent steps the model is shown
MAX_BAD_ANSWERS = 3            # unreadable answers in a row before stopping
DECISION_TOKENS = 220
# Rough size of each action, for the model to reason about (to be measured).
TURN_DEGREES = {"small": 10, "medium": 30, "large": 50}
STEP_CM = {"small": 10, "medium": 20, "large": 30}

PROMPT = """You are the mind of a small four-legged robot. The photo is what its camera sees right
now, from just above the floor, looking straight ahead. You cannot see behind you.

Goal, from the user: {goal}

Your body: a small, medium or large turn is roughly {turns} degrees; a small, medium or
large step forward is roughly {steps} cm.

What you have done so far: {totals}
Recent steps, oldest first:
{history}

Decide the next single action. Guidance:
- To find something that is not in view, turn to look around, always the same way, so the
  whole room is covered. After a full turn (about 360 degrees) without seeing it, walk
  forward to a new place if the way is clear, then look around again.
- When it is in view, turn to bring it to the centre, then walk forward while the way is
  clear. Use smaller turns and steps as it gets near.
- Never walk forward when the way ahead is blocked.
- Use "done" as soon as the goal is achieved (to reach something: it is near and centred),
  and "give_up" if it cannot be achieved.

Answer only with one JSON object:
{{"see": "what you see, in a few words",
  "goal_in_view": "left" or "centre" or "right" or "no",
  "goal_distance": "near" or "medium" or "far" or "unknown",
  "way_ahead": "clear" or "blocked",
  "action": "turn_left" or "turn_right" or "forward" or "greet" or "done" or "give_up",
  "amount": "small" or "medium" or "large",
  "say": "one short sentence for the user about what you are doing, in the language of the goal"}}"""


@dataclass
class Step:
    see: str
    in_view: str
    distance: str
    ahead: str
    action: str
    amount: str

    def line(self, n):
        return "%d. saw: %s | goal in view: %s (%s) | way ahead: %s | did: %s %s" % (
            n, self.see, self.in_view, self.distance, self.ahead, self.action, self.amount)


@dataclass
class WorkingMemory:
    goal: str
    steps: list = field(default_factory=list)
    turned_degrees: int = 0        # left positive
    walked_cm: int = 0

    def totals(self):
        side = "left" if self.turned_degrees >= 0 else "right"
        return "turned about %d degrees to the %s in all, walked about %d cm" % (
            abs(self.turned_degrees), side, self.walked_cm)

    def history(self):
        recent = self.steps[-MEMORY_STEPS:]
        first = len(self.steps) - len(recent) + 1
        return "\n".join(s.line(first + i) for i, s in enumerate(recent)) or "none yet"


def parse_decision(answer):
    """The first JSON object in the answer, checked; None if unusable."""
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        d = json.loads(answer[start:end + 1])
    except ValueError:
        return None
    if not isinstance(d, dict) or d.get("action") not in ACTIONS:
        return None
    if d.get("amount") not in AMOUNTS:
        d["amount"] = "small"
    for key, default in (("see", ""), ("goal_in_view", "no"), ("goal_distance", "unknown"),
                         ("way_ahead", "clear"), ("say", "")):
        d[key] = str(d.get(key) or default)
    return d


class Agent:
    def __init__(self, skills, say=print):
        self.skills = skills
        self.say = say

    def run(self, goal):
        memory = WorkingMemory(goal)
        started = time.monotonic()
        bad = 0
        for n in range(1, MAX_STEPS + 1):
            if time.monotonic() - started > MAX_SECONDS:
                return "ran out of time"
            jpeg = self.skills.head.photo()
            prompt = PROMPT.format(goal=goal,
                                   turns="/".join(str(v) for v in TURN_DEGREES.values()),
                                   steps="/".join(str(v) for v in STEP_CM.values()),
                                   totals=memory.totals(), history=memory.history())
            answer, seconds = self.skills.model.ask(jpeg, prompt, max_tokens=DECISION_TOKENS)
            d = parse_decision(answer)
            self.skills.journal.write(verb="agent", goal=goal, step=n, seconds=round(seconds, 2),
                                      answer=answer[:800], decision=d)
            if d is None:
                bad += 1
                self.say("  %2d. (could not read the answer, looking again)" % n)
                if bad >= MAX_BAD_ANSWERS:
                    return "stopped: the model's answers could not be read"
                continue
            bad = 0

            action, amount = d["action"], d["amount"]
            if action == "forward" and d["way_ahead"] == "blocked":
                action, amount = "turn_left", "small"     # never walk into something
            self.say("  %2d. %s | goal: %s (%s) | ahead: %s -> %s %s  [%.1fs]" % (
                n, d["see"], d["goal_in_view"], d["goal_distance"], d["way_ahead"], action,
                amount if action in ("turn_left", "turn_right", "forward") else "", seconds))
            if d["say"]:
                self.say("      \"%s\"" % d["say"])

            memory.steps.append(Step(d["see"], d["goal_in_view"], d["goal_distance"], d["way_ahead"],
                                     action, amount))
            if action == "done":
                return "done: %s" % (d["say"] or "goal achieved")
            if action == "give_up":
                return "gave up: %s" % (d["say"] or "the goal cannot be achieved")
            if action in ("turn_left", "turn_right"):
                self.skills.turn("left" if action == "turn_left" else "right", amount)
                memory.turned_degrees += TURN_DEGREES[amount] * (1 if action == "turn_left" else -1)
            elif action == "forward":
                self.skills.forward(amount)
                memory.walked_cm += STEP_CM[amount]
            elif action == "greet":
                self.skills.greet()
        return "stopped after %d steps" % MAX_STEPS
