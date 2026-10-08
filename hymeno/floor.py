"""Floor eyes: SegFormer finds the floor in a photo; from it, how far the free
floor goes in each column of the photo (in cm), and where the robot fits.

free_distances() and decide() only need numpy, so they can be tested offline."""

import json
import os

import numpy as np

MODEL = "nvidia/segformer-b5-finetuned-ade-640-640"
FLOOR_WORDS = ("floor", "rug", "carpet")
FLOOR_P = 0.8       # floor only where the model is this sure: a door of the floor's colour gets 0.3-0.6

# Camera geometry, as fractions of the photo (rows from the top, columns from
# the left). Measured 2026-10-06: two tapes on the floor one robot-width apart,
# 1 m long, screws at 25 and 50 cm from the front legs.
HORIZON = 0.454     # row where the tapes meet (infinitely far)
CENTRE = 0.522      # column the robot walks towards (the camera looks a bit left)
CM_K = 2.40         # distance in cm = CM_K / (row - HORIZON)
HALF_WIDTH = 1.15   # half the robot's width, in photo widths, per (row - HORIZON)
# A calibration made with `python -m hymeno calibrate` replaces the four above
# (and the leg corners below).
CALIBRATION = os.path.join("data", "calibration.json")
_measured = {}
if os.path.exists(CALIBRATION):
    with open(CALIBRATION) as f:
        _measured = json.load(f)
if "horizon" in _measured:
    HORIZON, CENTRE, CM_K, HALF_WIDTH = (_measured[k] for k in ("horizon", "centre", "cm_k", "half_width"))
ROBOT_K = HALF_WIDTH * CM_K  # half the robot's width at d cm = ROBOT_K / d
SAFETY = 1.3        # the path checked is this much wider than the robot (legs slip)
PATH_K = SAFETY * ROBOT_K    # half the checked path's width at d cm = PATH_K / d

STOP_CM = 30        # nearer than this the robot does not walk
FAR_CM = 45 #100        # further than this counts as free (and is imprecise anyway)
TURN_CM = 30        # cm of free floor given up per photo width of turning: prefer straight on
KEEP_CM = 30        # same, per photo width away from the last heading: no zigzag

# Goals: a direction with far free floor (an opening) the robot keeps heading
# to, while the near floor (up to FAR_CM) keeps it from bumping into things.
SEE_CM = 200        # how far the floor is measured to find openings (rough, but enough to choose)
OPEN_CM = 80        # an opening: floor at least this far, over a stretch wider than the robot
GOAL_CM = 60        # cm of free floor given up per photo width away from the goal: its pull
TRACK = 0.15        # a goal moves at most this much (photo widths) between photos
LOST = 10           # photos without the goal's opening before it is given up

STRIPS = 64         # vertical strips the photo is cut into
EDGE = 0.07         # strips this close to the sides are ignored (the legs show there)
FLOOR_SHARE = 0.9   # a row of a strip is floor only if this much of it is floor
BOTTOM = 0.15       # a strip's floor may start this far up (the legs can show)
GAP = 0.008         # holes in the floor shorter than this (of the photo height) are noise
FAKE_TOL = 0.02     # "floor" this far above the horizon is a door or wall of the floor's colour
FAKE_SHARE = 0.2    # ... when it covers this much of a strip in some row
NEIGHBOURS = 0.15   # a fooled strip takes the nearest distance of trusted strips this close

LEGS = True         # ignore the bottom corners where the legs show (drawn in yellow)
LEG_IN = 0.2        # how far towards the centre each corner reaches along the bottom (photo widths)
LEG_UP = 0.35       # how far up the side each corner reaches (photo heights)
if "legs" in _measured:
    LEGS, LEG_IN, LEG_UP = (_measured[k] for k in ("legs", "leg_in", "leg_up"))

GROUP = 5           # photos per group (with 3: 1-3 are group 000001, 4-6 000002...): one carried-out order per group
MAJORITY = GROUP // 2 + 1  # blocked photos that make a group blocked (2 of 3, 3 of 5...)


class Floor:
    """The segmentation model, loaded once."""

    def __init__(self, model=MODEL):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SegformerImageProcessor.from_pretrained(model)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model).to(self.device).eval()
        self.half = self.device == "cuda"  # half precision: about half the GPU time, leaving room for the vision model
        if self.half:
            self.model = self.model.half()
        labels = self.model.config.id2label
        self.floor_ids = [int(i) for i, label in labels.items() if any(w in label.lower() for w in FLOOR_WORDS)]

    def mask(self, image):
        """PIL RGB image -> numpy bool array (height x width), True where there is floor."""
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        if self.half:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
        logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        floor_p = logits.softmax(dim=1)[0, self.floor_ids].sum(dim=0)
        return (floor_p >= FLOOR_P).cpu().numpy()

    def look(self, image, last=None, goal=None):
        """(floor mask, decision) for one photo; last is the order being
        carried out; goal (a Goal) is followed to its opening in this photo."""
        floor = self.mask(image)
        far = free_distances(floor, far=SEE_CM)
        found = openings(far)
        if goal:
            goal.update(found)
        near = [(column, min(cm, FAR_CM)) for column, cm in far]
        decision = decide(near, last, goal.column if goal else None)
        decision.update(openings=found, goal=goal.column if goal else None, goal_cm=goal.cm if goal else None)
        return floor, decision


def row_of(cm):
    """Row of the photo (fraction from the top) where the floor is cm away."""
    return HORIZON + CM_K / cm


def leg_corners(h, w):
    """The two bottom corners (triangles) where the legs show, as pixel points."""
    left = [(0, h), (LEG_IN * w, h), (0, h * (1 - LEG_UP))]
    right = [(w, h), (w * (1 - LEG_IN), h), (w, h * (1 - LEG_UP))]
    return left, right


def leg_zone(h, w):
    """Bool array (h x w), True inside the leg corners."""
    rows = np.arange(h)[:, None] / h
    columns = np.arange(w)[None, :] / w
    height = 1 - rows  # from the bottom
    side = np.minimum(columns, 1 - columns)  # from the nearest side
    return side * LEG_UP + height * LEG_IN <= LEG_IN * LEG_UP


def free_distances(floor, strips=STRIPS, far=FAR_CM):
    """(column, cm) for each strip, left to right: how far the floor goes
    without interruption from the bottom of the photo, up to `far` cm.
    Edge strips are left out.

    Floor can't be seen above the horizon: where the model sees some there,
    it is a door or wall of the floor's colour, and the strip can't be trusted
    further than its trusted neighbours."""
    h, w = floor.shape
    if LEGS:
        floor = floor | leg_zone(h, w)  # the legs are not obstacles
    bottom = max(1, int(BOTTOM * h))
    gap = max(1, int(GAP * h))
    sky = max(0, int((HORIZON - FAKE_TOL) * h))
    out, fooled = [], []
    for i in range(strips):
        column = (i + 0.5) / strips
        if column < EDGE or column > 1 - EDGE:
            continue
        strip = floor[:, i * w // strips:(i + 1) * w // strips]
        if sky and strip[:sky].mean(axis=1).max() >= FAKE_SHARE:
            fooled.append(len(out))
        rows = strip.mean(axis=1) >= FLOOR_SHARE
        rows = rows[::-1]  # bottom row first
        start = np.flatnonzero(rows[:bottom])
        if start.size == 0:
            out.append((column, 0.0))  # no floor at the bottom: something right in front
            continue
        top, missing = start[0], 0
        for r in range(start[0], h):
            if rows[r]:
                top, missing = r, 0
            else:
                missing += 1
                if missing > gap:
                    break
        above_horizon = (h - 1 - top) / h - HORIZON
        out.append((column, far if above_horizon <= CM_K / far else CM_K / above_horizon))
    trusted = [(c, cm) for k, (c, cm) in enumerate(out) if k not in fooled]
    for k in fooled:
        column, cm = out[k]
        near = [d for c, d in trusted if abs(c - column) <= NEIGHBOURS]
        out[k] = (column, min([cm] + near) if near else 0.0)
    return out


def openings(strips):
    """Directions where the floor goes far: stretches of strips at OPEN_CM
    or more, wider than the robot at their distance. Each one: its column
    (the middle), cm (the median) and width (photo widths)."""
    found, run = [], []
    for column, cm in list(strips) + [(None, 0.0)]:
        if cm >= OPEN_CM:
            run.append((column, cm))
            continue
        if run:
            cms = sorted(c for _, c in run)
            cm_mid = cms[len(cms) // 2]
            width = run[-1][0] - run[0][0] + 1 / STRIPS
            if width >= 2 * ROBOT_K / cm_mid:
                found.append({"column": (run[0][0] + run[-1][0]) / 2, "cm": cm_mid, "width": width})
            run = []
    return found


class Goal:
    """A direction the robot wants to reach: an opening, as a photo column.
    Without a compass, it is followed from photo to photo as the opening
    nearest to where it was; given up when not seen for LOST photos (also
    what happens once the robot gets there)."""

    def __init__(self):
        self.column, self.cm, self.missed, self.note = None, None, 0, "no goal"

    def set(self, column):
        self.column, self.cm, self.missed, self.note = column, None, 0, "goal set"

    def clear(self, why):
        self.column, self.cm, self.note = None, None, why

    def update(self, found):
        if self.column is None:
            return
        close = [o for o in found if abs(o["column"] - self.column) <= TRACK]
        if close:
            best = min(close, key=lambda o: abs(o["column"] - self.column))
            self.column, self.cm, self.missed, self.note = best["column"], best["cm"], 0, "following the goal"
        else:
            self.missed += 1
            self.note = "goal not seen (%d)" % self.missed
            if self.missed > LOST:
                self.clear("goal lost")


def nearest(strips, heading):
    """(cm, column) of the nearest strip inside the robot's path towards the
    photo column `heading`; (FAR_CM, None) if nothing is in the way.
    The path narrows in the photo with distance (perspective)."""
    half_strip = 0.5 / STRIPS
    found = (FAR_CM, None)
    for column, cm in strips:
        if cm < found[0] and (cm == 0 or abs(column - heading) - half_strip <= PATH_K / cm):
            found = (cm, column)
    return found


def clearance(strips, heading):
    """How far (cm) the robot can walk towards the photo column `heading`
    with its whole width."""
    return nearest(strips, heading)[0]


def decide(strips, last=None, goal=None):
    """Where to go. Tries headings across the photo (those where the robot's
    path fits in the photo at STOP_CM) and keeps the one with the most free
    floor, preferring straight on (or, with a goal column, the goal) and the
    last heading. Blocked, it turns towards the goal if there is one.

    Returns heading (photo column), steer (-1 left .. 1 right), ahead_cm
    (free floor straight on), clear_cm (on the best path), direction
    ("ahead", "left", "right" or "blocked"), path (the best path's column,
    also when blocked), limit (column of the strip that stops it, or None)
    and the strips."""
    margin = PATH_K / STOP_CM
    headings = np.linspace(margin, 1 - margin, 41)
    last_heading = last["heading"] if last and last["direction"] != "blocked" else None
    best = None
    for heading in headings:
        cm = clearance(strips, heading)
        if goal is None:
            score = cm - TURN_CM * abs(heading - CENTRE)
        else:
            score = cm - GOAL_CM * abs(heading - goal)
        if last_heading is not None:
            score -= KEEP_CM * abs(heading - last_heading)
        if best is None or score > best[0]:
            best = (score, heading, cm)
    _, heading, clear_cm = best
    decision = {"heading": float(heading), "ahead_cm": clearance(strips, CENTRE),
                "clear_cm": clear_cm, "path": float(heading), "limit": nearest(strips, heading)[1],
                "strips": strips}
    if clear_cm < STOP_CM:
        # Turn on the spot; keep turning the same way while blocked, so a
        # corner does not leave it dithering.
        if last and last["direction"] == "blocked":
            steer = last["steer"]
        elif goal is not None:
            steer = 1.0 if goal > CENTRE else -1.0
        else:
            left = [cm for column, cm in strips if column < CENTRE]
            right = [cm for column, cm in strips if column >= CENTRE]
            steer = 1.0 if np.mean(right or [0]) > np.mean(left or [0]) else -1.0
        decision.update(heading=CENTRE, steer=steer, direction="blocked")
        return decision
    steer = float(np.clip((heading - CENTRE) / 0.4, -1, 1))
    decision.update(steer=steer, direction=label(steer))
    return decision


def group_of(frame):
    """Key of the group of photo number `frame` (from 1): "000001" for 1-3..."""
    return "%06d" % ((frame - 1) // GROUP + 1)


def label(steer):
    return "ahead" if abs(steer) < 0.15 else ("left" if steer < 0 else "right")


def combine(decisions):
    """The order carried out for a group of per-photo decisions, oldest first
    (tests/test_offline.py CombineTableTest has the agreed cases):
    MAJORITY or more blocked: turn on the spot, the way most of them say;
    otherwise the blocked ones are ignored, and it walks with the median
    steer weighted by recency (the newest weighs most)."""
    # decisions: one per photo of the group, oldest first, each with "direction"
    # ("ahead", "left", "right" or "blocked"), "steer" (-1 all left .. +1 all
    # right) and "heading" (the photo column it aims at).
    # weighted: (weight, decision); the oldest photo weighs 1, the next 2...
    weighted = list(enumerate(decisions, start=1))
    blocked = [(w, d) for w, d in weighted if d["direction"] == "blocked"]

    # MAJORITY (3 of 5) or more blocked: turn on the spot. Their steers are -1
    # (turn left) or +1 (turn right); the side with more weight wins.
    if len(blocked) >= MAJORITY:
        steer = 1.0 if sum(w * d["steer"] for w, d in blocked) > 0 else -1.0
        return {"direction": "blocked", "steer": steer, "heading": CENTRE}

    # Fewer blocked: they are ignored. walking: the others, by steer, left first.
    walking = sorted(((w, d) for w, d in weighted if d["direction"] != "blocked"), key=lambda wd: wd[1]["steer"])
    if not walking:  # only in an incomplete group of blocked photos short of a majority
        return {"direction": "wait", "steer": 0.0, "heading": CENTRE}

    # The weighted median: going from left to right, the first steer that
    # gathers half the weight. It is always a steer some photo chose (an average
    # could invent "straight on" between going round an obstacle by the left and
    # by the right), odd photos are ignored, and recent photos count more.
    half, gathered = sum(w for w, _ in walking) / 2, 0
    for w, d in walking:
        gathered += w
        if gathered >= half:
            return {"direction": label(d["steer"]), "steer": d["steer"], "heading": d["heading"]}


class Groups:
    """Collects each photo's own decision into its group; when the group is
    complete (or MAJORITY of its photos already say blocked) it gives the
    order to carry out, once per group: `fresh` is True only on the photo
    that decided it. `acting` is the last order that moved the robot, which
    each photo's decision uses to prefer the same way."""

    def __init__(self):
        self.key, self.members, self.order, self.acting, self.fresh = None, [], None, None, False

    def add(self, frame, decision):
        """(group key, the group's order, or None while it is still undecided)."""
        key = group_of(frame)
        if key != self.key:
            self.key, self.members, self.order = key, [], None
        self.members.append(decision)
        self.fresh = False
        if self.order is not None:  # already decided (blocked early): not carried out again
            return key, self.order
        blocked = sum(d["direction"] == "blocked" for d in self.members)
        if frame % GROUP and blocked < MAJORITY:
            return key, None
        self.order, self.fresh = combine(self.members), True
        if self.order["direction"] != "wait":
            self.acting = self.order
        return key, self.order


def write(pen, xy, text, size=18, right=False):
    """Text in white on a black box, readable over any photo."""
    from PIL import ImageFont

    font = ImageFont.load_default(size=size)
    x, y = xy
    _, _, width, height = pen.textbbox((0, 0), text, font=font)
    if right:
        x -= width
    pen.rectangle([x - 4, y - 3, x + width + 4, y + height + 4], fill=(0, 0, 0))
    pen.text((x, y), text, fill=(255, 255, 255), font=font)


def draw(image, floor, decision, label=None, note=None):
    """A copy of the photo with the floor in green, where each strip's floor
    ends (white), the STOP_CM line (red), the robot's best path (cyan; red on
    the side it hits, and the strip that stops it), an arrow to where it would
    walk (or an X when blocked), this photo's decision (top left), a note
    under it and a label (top right)."""
    from PIL import Image, ImageDraw

    out = image.copy()
    green = Image.new("RGB", image.size, (0, 255, 0))
    out.paste(green, mask=Image.fromarray((floor * 140).astype("uint8")))
    if LEGS:
        yellow = Image.new("RGB", image.size, (255, 220, 0))
        out.paste(yellow, mask=Image.fromarray((leg_zone(*floor.shape) * 110).astype("uint8")))
    pen = ImageDraw.Draw(out)
    w, h = image.size
    write(pen, (8, 6), decision["direction"].upper(), size=26)
    if note:
        write(pen, (8, 42), note)
    if label:
        write(pen, (w - 8, 6), label, right=True)
    half = w / STRIPS / 2
    for column, cm in decision["strips"]:
        y = (row_of(cm) if cm > 0 else 1) * h
        pen.line([(column * w - half, y), (column * w + half, y)], fill=(255, 255, 255), width=3)
    y = row_of(STOP_CM) * h
    pen.line([(0, y), (w, y)], fill=(255, 0, 0), width=1)
    write(pen, (8, y + 6), "%d cm" % STOP_CM, size=14)
    # The robot's path on the best heading (also when blocked). The side where
    # it hits something is red, and so is the strip that stops it.
    path, limit = decision.get("path", decision["heading"]), decision.get("limit")
    hit = 0 if limit is None or abs(limit - path) <= 1 / STRIPS else (1 if limit > path else -1)
    distances = np.linspace(CM_K / (1 - HORIZON), FAR_CM, 40)
    for side in (-1, 1):
        colour = (255, 0, 0) if side == hit else (0, 255, 255)
        pen.line([((path + side * PATH_K / d) * w, row_of(d) * h) for d in distances], fill=colour, width=3)
    if limit is not None:
        cm = decision["clear_cm"]
        y = row_of(cm) * h if cm > 0 else h - 4
        pen.line([(limit * w - 2 * half, y), (limit * w + 2 * half, y)], fill=(255, 0, 0), width=7)
        write(pen, (limit * w - 2 * half, min(y + 8, h - 26)), "%.0f cm" % cm, size=14)
    # Openings (far free floor) as blue bars on the horizon; the goal as a yellow flag.
    y = HORIZON * h
    for o in decision.get("openings", []):
        pen.line([((o["column"] - o["width"] / 2) * w, y), ((o["column"] + o["width"] / 2) * w, y)],
                 fill=(60, 120, 255), width=6)
    if decision.get("goal") is not None:
        x = decision["goal"] * w
        pen.line([(x, y), (x, y - 60)], fill=(255, 220, 0), width=4)
        pen.polygon([(x, y - 60), (x + 28, y - 50), (x, y - 40)], fill=(255, 220, 0))
        cm = decision.get("goal_cm")
        write(pen, (x + 32, y - 62), "GOAL" + (" ~%.0f cm" % cm if cm else ""), size=14)
    # A goal in words (target.Target): the vision model's box around it, in magenta.
    target = decision.get("target")
    if target:
        x1, y1, x2, y2 = target["box"]
        pen.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=(255, 0, 255), width=4)
        write(pen, (x1 * w, max(0, y1 * h - 26)),
              target["label"] + (" ~%.0f cm" % target["cm"] if target["cm"] else ""), size=16)
    x0, y0 = CENTRE * w, h - 5
    if decision["direction"] == "blocked":
        s = h / 20
        pen.line([(x0 - s, y0 - 2 * s), (x0 + s, y0)], fill=(255, 0, 0), width=6)
        pen.line([(x0 - s, y0), (x0 + s, y0 - 2 * s)], fill=(255, 0, 0), width=6)
        write(pen, (x0 + 1.5 * s, y0 - 2 * s - 4), "turn " + ("right" if decision["steer"] > 0 else "left"))
        return out
    tip = (path * w, row_of(decision["clear_cm"]) * h)
    pen.line([(x0, y0), tip], fill=(255, 0, 0), width=6)
    pen.ellipse([tip[0] - 8, tip[1] - 8, tip[0] + 8, tip[1] + 8], fill=(255, 0, 0))
    write(pen, (tip[0] + 14, tip[1] - 10), "%.0f cm" % decision["clear_cm"])
    return out
