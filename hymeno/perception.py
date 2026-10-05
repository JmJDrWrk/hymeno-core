"""Turns the model's answer into boxes on the photo, as fractions of its size
(0..1), whatever scale the model answered in."""

import io
import json
import math
from dataclasses import dataclass

from PIL import Image

# Qwen2.5-VL answers in the pixels of the picture as Ollama resized it for the
# model: about this many pixels, sides in multiples of 28 (640x480 and 320x240
# photos both become 1036x784). Qwen3-VL answers on a 0-1000 scale.
QWEN25_PIXELS = 1036 * 784
QWEN25_STEP = 28
QWEN3_SCALE = 1000


@dataclass
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1


def photo_size(jpeg):
    with Image.open(io.BytesIO(jpeg)) as image:
        return image.size


def model_scale(model_name, width, height):
    """The width and height the model's coordinates refer to."""
    if "qwen3" in model_name.lower():
        return QWEN3_SCALE, QWEN3_SCALE
    h = max(QWEN25_STEP, round(math.sqrt(QWEN25_PIXELS * height / width) / QWEN25_STEP) * QWEN25_STEP)
    w = max(QWEN25_STEP, round(math.sqrt(QWEN25_PIXELS * width / height) / QWEN25_STEP) * QWEN25_STEP)
    return w, h


VIEW_SIZE = (32, 24)


def view_change(jpeg_a, jpeg_b):
    """How different two photos look, 0 (same) to 255: the mean difference of
    tiny greyscale copies. Small after walking means the robot did not move."""
    def tiny(jpeg):
        with Image.open(io.BytesIO(jpeg)) as image:
            return list(image.convert("L").resize(VIEW_SIZE).getdata())
    a, b = tiny(jpeg_a), tiny(jpeg_b)
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def says_blocked(answer):
    """True when the last word of the answer is BLOCKED."""
    words = answer.strip().split()
    return bool(words) and words[-1].strip(".`*").upper() == "BLOCKED"


def parse_boxes(answer, model_name, photo_width, photo_height):
    """The first JSON list in the answer (models like to wrap it in ``` fences)
    as boxes; anything else is ignored."""
    start, end = answer.find("["), answer.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(answer[start:end + 1])
    except ValueError:
        return []
    if not isinstance(items, list):
        return []
    scale_w, scale_h = model_scale(model_name, photo_width, photo_height)
    boxes = []
    for item in items:
        if not isinstance(item, dict):
            continue
        b = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            continue
        clamp = lambda v: min(1.0, max(0.0, v))
        boxes.append(Box(str(item.get("label", "")),
                         clamp(x1 / scale_w), clamp(y1 / scale_h), clamp(x2 / scale_w), clamp(y2 / scale_h)))
    return boxes
