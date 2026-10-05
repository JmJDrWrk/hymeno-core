"""What to do about what was seen. Phase 1: face the target."""

from dataclasses import dataclass


@dataclass
class Decision:
    action: str          # "turn_left", "turn_right" or "wait"
    reason: str


def face_target(boxes, center_tolerance, mirrored=False):
    """Turns towards the biggest box until its centre is within
    center_tolerance (share of the picture width) of the middle."""
    if not boxes:
        return Decision("wait", "target not in sight")
    target = max(boxes, key=lambda b: b.width * b.height)
    offset = target.center_x - 0.5          # negative: left of the middle
    if mirrored:
        offset = -offset
    where = "%.0f%% %s of centre, %.0f%% of the width" % (
        abs(offset) * 100, "left" if offset < 0 else "right", target.width * 100)
    if abs(offset) <= center_tolerance:
        return Decision("wait", "target centred (%s)" % where)
    return Decision("turn_left" if offset < 0 else "turn_right", where)
