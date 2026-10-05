"""What the brain keeps between steps and between goals.

- Rules: things the user teaches about the house ("rugs are not obstacles"),
  one per line in data/rules.txt, shown to the model at every step.
- World memory: what was seen and in which direction, worked out from the
  turns made since the brain started (dead reckoning: rough, and lost if the
  robot is moved by hand or the brain restarts). Lets it look first where it
  last saw something."""

import os
import time

WORLD_ITEMS_SHOWN = 10
WORLD_MAX_AGE_S = 30 * 60


class Rules:
    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "rules.txt")

    def all(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [line.strip() for line in f if line.strip()]

    def add(self, text):
        text = " ".join(text.split())
        if not text or text in self.all():
            return False
        with open(self.path, "a") as f:
            f.write(text + "\n")
        return True

    def remove(self, text):
        """Removes the rules that mention `text` (any case). Returns them."""
        needle = text.lower().strip()
        keep, removed = [], []
        for rule in self.all():
            (removed if needle and needle in rule.lower() else keep).append(rule)
        if removed:
            with open(self.path, "w") as f:
                f.writelines(rule + "\n" for rule in keep)
        return removed

    def text(self):
        rules = self.all()
        return "\n".join("- " + r for r in rules) if rules else "none"


def normalise(degrees):
    return (degrees + 180) % 360 - 180


class WorldMemory:
    """Heading 0 is where the robot faced when the brain started; left positive."""

    def __init__(self):
        self.heading = 0.0
        self.sightings = {}        # name -> (heading, time)

    def turned(self, degrees):
        self.heading = normalise(self.heading + degrees)

    def saw(self, names):
        now = time.time()
        for name in names:
            name = " ".join(str(name).lower().split())[:40]
            if name:
                self.sightings[name] = (self.heading, now)

    def text(self):
        """Things seen lately, and where they are from the current heading."""
        now = time.time()
        recent = sorted(((t, name, h) for name, (h, t) in self.sightings.items() if now - t < WORLD_MAX_AGE_S),
                        reverse=True)[:WORLD_ITEMS_SHOWN]
        if not recent:
            return "nothing yet"
        lines = []
        for t, name, h in recent:
            rel = normalise(h - self.heading)
            where = "straight ahead" if abs(rel) < 15 else "%d degrees to the %s" % (abs(rel), "left" if rel > 0 else "right")
            lines.append("- %s: %s (%d min ago)" % (name, where, (now - t) // 60))
        return "\n".join(lines)
