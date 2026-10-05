"""Fast eyes: an object detector (YOLO) that finds the 80 everyday things of the
COCO set (person, cat, dog, chair, bottle, ...) in a few milliseconds on a GPU.

The vision model understands anything but takes seconds; the detector only
knows those 80 classes but answers many times a second. Verbs use it whenever
their target names one of its classes, and the vision model otherwise.

Optional: needs the ultralytics package (requirements-yolo.txt)."""

import io
import re

from PIL import Image

from .perception import Box

DEFAULT_WEIGHTS = "yolo11n.pt"     # downloaded on first use, about 5 MB
MIN_CONFIDENCE = 0.35


class Detector:
    def __init__(self, weights=DEFAULT_WEIGHTS, min_confidence=MIN_CONFIDENCE):
        from ultralytics import YOLO   # imported here: the brain also works without it
        self.model = YOLO(weights)
        self.min_confidence = min_confidence
        self.names = dict(self.model.names)
        # Longest names first, so "teddy bear" wins over "bear".
        self._classes = sorted(self.names.values(), key=len, reverse=True)

    def class_for(self, target):
        """The detector class a target description names ("the black cat" ->
        "cat"), or None."""
        text = " %s " % " ".join(str(target).lower().split())
        for name in self._classes:
            if re.search(r"\b%ss?\b" % re.escape(name), text):
                return name
        return None

    def detect(self, jpeg):
        """All boxes in the photo, as fractions of its size."""
        with Image.open(io.BytesIO(jpeg)) as image:
            result = self.model.predict(image.convert("RGB"), conf=self.min_confidence, verbose=False)[0]
        boxes = []
        for xyxy, cls in zip(result.boxes.xyxyn.tolist(), result.boxes.cls.tolist()):
            boxes.append(Box(self.names[int(cls)], *xyxy))
        return boxes


def load(settings, say=print):
    """The detector, or None when it is switched off or not installed."""
    options = settings.get("detector", {})
    if options is False or (isinstance(options, dict) and options.get("enabled") is False):
        return None
    options = options if isinstance(options, dict) else {}
    try:
        detector = Detector(options.get("weights", DEFAULT_WEIGHTS),
                            options.get("min_confidence", MIN_CONFIDENCE))
    except ImportError:
        say("no fast detector (pip install -r requirements-yolo.txt to add YOLO); using the vision model only")
        return None
    say("fast detector: %s, %d classes" % (options.get("weights", DEFAULT_WEIGHTS), len(detector.names)))
    return detector
