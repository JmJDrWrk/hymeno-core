"""Turns an order in plain words (any language) into a plan made only of the
verbs in skills.VERBS, using the same model server. The model chooses what to
do; the verbs decide how. A plan with anything else is rejected."""

import json
import re

from .skills import VERBS

INSTRUCTIONS = """You control a small four-legged walking robot that has a camera.
Turn the user's request into a short plan using ONLY these verbs:

%s

Rules:
- Questions about what the robot sees ("what do you see?", "is anyone there?")
  are answered with "describe", passing the question as the user wrote it.
  Never use "search" or "look" for them.
- Describe every target in English, precisely enough to recognise it in a photo
  (for example "an electric fan" or "a square of yellow electrical tape on the floor").
- "face" and "approach" need the target in sight: put "search" first unless the
  request says it is already in view. "reach" searches by itself.
- To go somewhere or to something, prefer "reach" (it walks without stopping).
- Keep plans short. Use only the arguments listed.
- If the request cannot be done with these verbs, return an empty plan and say why.

Answer only with JSON:
{"plan": [{"verb": "...", "args": {...}}], "say": "one short sentence for the user, in the user's language"}

Request: %s"""


def describe_verbs():
    lines = []
    for verb, (args, summary) in VERBS.items():
        params = ", ".join("%s%s: %s%s" % (name, "" if required else " (optional)", text,
                                           "; one of " + "/".join(choices) if choices else "")
                           for name, (_, required, choices, text) in args.items())
        lines.append("- %s(%s): %s" % (verb, params, summary))
    return "\n".join(lines)


def verb_name(written):
    """The verb as the model wrote it, tidied: "stop()", " Stop " -> "stop"."""
    match = re.match(r"^\s*([A-Za-z_]+)\s*(\(.*\))?\s*$", str(written))
    return match.group(1).lower() if match else ""


def validate(plan):
    """Raises ValueError unless every step is a known verb with valid arguments."""
    if not isinstance(plan, list):
        raise ValueError("the plan is not a list")
    steps = []
    for step in plan:
        verb = verb_name(step.get("verb")) if isinstance(step, dict) else ""
        if verb not in VERBS:
            raise ValueError("unknown step: %r" % (step,))
        args = step.get("args") or {}
        allowed, _ = VERBS[verb]
        for name, value in args.items():
            if name not in allowed:
                raise ValueError("%s has no argument %r" % (verb, name))
            kind, _, choices, _ = allowed[name]
            if not isinstance(value, kind):
                raise ValueError("%s: %s must be %s" % (verb, name, kind.__name__))
            if choices and value not in choices:
                raise ValueError("%s: %s must be one of %s" % (verb, name, "/".join(choices)))
        for name, (_, required, _, _) in allowed.items():
            if required and name not in args:
                raise ValueError("%s needs %s" % (verb, name))
        steps.append((verb, args))
    return steps


def plan(model, request):
    """Returns (steps, sentence for the user, seconds)."""
    answer, seconds = model.ask_text(INSTRUCTIONS % (describe_verbs(), request), json_only=True)
    try:
        data = json.loads(answer)
    except ValueError:
        raise ValueError("the model did not answer with JSON: %s" % answer[:200])
    return validate(data.get("plan", [])), str(data.get("say", "")), seconds
