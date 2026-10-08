"""Goals given in words (`explore --goal "shoe"`): something finds them in the
photos, and the floor logic walks there.

- DetectedTarget (used): a fast open-vocabulary detector (YOLO-World, ~20 ms)
  on every photo, with a tracker so it keeps following the same one.
- AskedTarget (kept for later, for complex phrases): the vision model through
  Ollama, ~2-5 s an answer, asked again and again in a background thread."""

import threading
import time

from . import floor
from .perception import Box, parse_boxes, photo_size

ARRIVE_CM = 40      # arrived when its base is this near (more than floor.STOP_CM: it would block first)
SEARCH_TURN_S = 0.6  # while searching, turn on the spot this long between looks
SEARCH_LOOKS = 12   # looks without it, while searching, before giving up (about a full turn)

DETECTOR = "yolov8s-worldv2.pt"
FOUND = 0.25        # the detector's confidence to take a new one as the goal
TRACK_CONF = 0.1    # detections down to this go to the tracker, which keeps following a known one
DETECTOR_MISSES = 15  # photos in a row without it, while walking, before searching again (~2 s)

PROMPT = ('Is there {what} in this photo? If there is, answer only JSON for the clearest one: '
          '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{what}"}}]. If there is not, answer [].')
ASKED_MISSES = 3    # the same with the vision model, in answers


class Target:
    """What the robot is going to. Subclasses give answers (_answer: the
    photo number and the box, or None) through offer().

    state: "searching" (standing, turning a little between looks),
    "going" (walking towards its last box), "arrived" or "gave up".
    For floor.Floor.look it is a goal like floor.Goal: `column` is where it
    pulls (only while going) and update() does nothing."""

    misses_allowed = 1

    def __init__(self, what):
        self.what = what
        self.state, self.box, self.cm, self.misses, self.looks, self.turned_at = "searching", None, None, 0, 0, 0
        self.timing, self.error = "", None
        self._answer = None
        self._lock = threading.Lock()

    def offer(self, jpeg, image, frame):
        """The newest photo: as it came from the camera, as decided on
        (mirrored if the head is), and its number."""

    @property
    def column(self):
        return self.box.center_x if self.state == "going" and self.box else None

    def update(self, found):
        pass

    def check(self, frame):
        """Moves on with the newest answer. Returns what to do now: "wait",
        "turn" (searching: turn a little), "go", "arrived" or "gave up"."""
        with self._lock:
            answer, self._answer = self._answer, None
        if self.state == "searching":
            if answer is None or answer["frame"] <= self.turned_at:
                return "wait"  # no answer yet, or about a photo from before the last turn
            if answer["box"]:
                self.see(answer["box"])
                self.state, self.misses = "going", 0
                return "go"
            self.looks += 1
            if self.looks >= SEARCH_LOOKS:
                self.state = "gave up"
                return "gave up"
            self.turned_at = frame
            return "turn"
        if self.state == "going":
            if answer:
                if answer["box"]:
                    self.see(answer["box"])
                    self.misses = 0
                else:
                    self.misses += 1
            if self.misses >= self.misses_allowed:
                self.state, self.looks, self.turned_at = "searching", 0, frame
                return "wait"
            if self.cm is not None and self.cm < ARRIVE_CM:
                self.state = "arrived"
                return "arrived"
            return "go"
        return self.state

    def see(self, box):
        """Its box, and how far its base (where it touches the floor) is."""
        self.box = box
        below = box.y2 - floor.HORIZON
        self.cm = floor.CM_K / below if below > floor.CM_K / floor.SEE_CM else None

    @property
    def note(self):
        extra = self.timing + (", error: %s" % self.error if self.error else "")
        if self.state == "searching":
            return "searching %s, look %d%s" % (self.what, self.looks + 1, extra)
        if self.state == "going":
            far = "~%.0f cm" % self.cm if self.cm else "far"
            return "going to %s, %s%s" % (self.what, far, " (not seen %d)" % self.misses if self.misses else "") + extra
        return "%s: %s" % (self.state, self.what)

    def drawing(self):
        """For floor.draw: its box, label and distance."""
        if not self.box:
            return None
        b = self.box
        return {"box": (b.x1, b.y1, b.x2, b.y2), "label": self.what, "cm": self.cm}


class DetectedTarget(Target):
    """Found by YOLO-World on every photo.

    Searching: the detector alone (a tracker only shows a new object once it
    has seen it twice in a row, and the robot turns in between): the surest
    one, if sure enough. Going: a tracker (ByteTrack), started afresh, gives
    each object an id and the goal's is followed, even when the detector is
    less sure of it for a few photos; if the id is lost, the sure one nearest
    to where it was (never just the surest: that could be another one)."""

    misses_allowed = DETECTOR_MISSES
    JUMP = 0.3  # a re-found goal is at most this far (photo widths) from where it was

    def __init__(self, what, model=DETECTOR):
        super().__init__(what)
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(model)
        self.model.set_classes([what])
        self.track_id, self.tracking = None, False

    def detections(self, result, size):
        """[(confidence, id or None, Box)] of a result."""
        width, height = size
        if result is None or result.boxes is None or not len(result.boxes):
            return []
        ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        return [(c, i, Box(self.what, x1 / width, y1 / height, x2 / width, y2 / height))
                for c, i, (x1, y1, x2, y2) in zip(result.boxes.conf.tolist(), ids, result.boxes.xyxy.tolist())]

    def offer(self, jpeg, image, frame):
        started = time.monotonic()
        box = None
        try:
            if self.state != "going":
                self.tracking = False
                found = self.detections(self.model.predict(image, conf=FOUND, verbose=False)[0], image.size)
                if found:
                    box = max(found, key=lambda f: f[0])[2]
            else:
                fresh = not self.tracking  # start the tracker afresh when walking starts
                result = self.model.track(image, persist=not fresh, conf=TRACK_CONF, verbose=False)[0]
                self.tracking = True
                found = self.detections(result, image.size)
                same = [f for f in found if self.track_id is not None and f[1] == self.track_id and not fresh]
                if same:
                    box = same[0][2]
                else:
                    # Fresh or lost: the one nearest to where it was (sure, unless just started).
                    near = [f for f in found if (fresh or f[0] >= FOUND) and self.box
                            and abs(f[2].center_x - self.box.center_x) <= self.JUMP]
                    if near:
                        best = min(near, key=lambda f: abs(f[2].center_x - self.box.center_x))
                        box, self.track_id = best[2], best[1]
            self.error = None
        except Exception as e:
            self.error = type(e).__name__
        self.timing = " (detector %.0f ms)" % ((time.monotonic() - started) * 1000)
        with self._lock:
            self._answer = {"frame": frame, "box": box}


class AskedTarget(Target):
    """Found by asking the vision model (Ollama), again and again in a
    background thread, always with the newest photo."""

    misses_allowed = ASKED_MISSES

    def __init__(self, model, what, mirrored=False):
        super().__init__(what)
        self.model, self.mirrored = model, mirrored
        self._photo, self._new = None, threading.Event()
        threading.Thread(target=self._ask_again_and_again, daemon=True).start()

    def offer(self, jpeg, image, frame):
        with self._lock:
            self._photo = (jpeg, frame)
        self._new.set()

    def _ask_again_and_again(self):
        while True:
            self._new.wait()
            self._new.clear()
            with self._lock:
                jpeg, frame = self._photo
            try:
                answer, seconds = self.model.ask(jpeg, PROMPT.format(what=self.what))
                width, height = photo_size(jpeg)
                boxes = parse_boxes(answer, self.model.name, width, height)
                self.timing, self.error = " (model %.1f s)" % seconds, None
            except Exception as e:
                boxes, self.error = [], type(e).__name__
            box = boxes[0] if boxes else None
            if box and self.mirrored:  # the photo is mirrored before deciding: so is the box
                box.x1, box.x2 = 1 - box.x2, 1 - box.x1
            with self._lock:
                self._answer = {"frame": frame, "box": box}
