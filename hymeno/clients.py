"""The three things the brain talks to: the robot's head camera, the robot's
body and the vision model server. Each is just a URL."""

import base64
import threading
import time

from urllib.parse import urlsplit

import requests


class Head:
    """The camera: one JPEG per request."""

    def __init__(self, url, timeout_s=10):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def photo(self):
        r = requests.get(self.url + "/jpg", timeout=self.timeout_s)
        r.raise_for_status()
        if not r.content.startswith(b"\xff\xd8"):
            raise RuntimeError("the head did not send a JPEG")
        return r.content


class HeadVideo:
    """The camera's video (MJPEG on port 81, /stream). A background thread
    keeps only the newest frame, so each photo() is fresh and never a backlog.
    The head serves one video client at a time: a browser watching it stops this."""

    def __init__(self, url, timeout_s=10):
        parts = urlsplit(url)
        self.url = "%s://%s:81/stream" % (parts.scheme, parts.hostname)
        self.timeout_s = timeout_s
        self._frame, self._number, self._error = None, 0, None
        self._new = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _publish(self, frame=None, error=None):
        with self._new:
            if frame:
                self._frame, self._number = frame, self._number + 1
            self._error = error
            self._new.notify_all()

    def _run(self):
        while True:
            try:
                with requests.get(self.url, stream=True, timeout=self.timeout_s) as r:
                    r.raise_for_status()
                    buf = b""
                    for chunk in r.iter_content(16384):
                        buf += chunk
                        while True:  # cut out each JPEG, start to end marker
                            start = buf.find(b"\xff\xd8")
                            if start < 0:
                                buf = buf[-1:]
                                break
                            end = buf.find(b"\xff\xd9", start + 2)
                            if end < 0:
                                buf = buf[start:]
                                break
                            self._publish(buf[start:end + 2])
                            buf = buf[end + 2:]
            except Exception as e:
                self._publish(error=e)
                time.sleep(1)

    def photo(self, after=0):
        """(JPEG, number) of the first frame newer than frame number `after`."""
        with self._new:
            if not self._new.wait_for(lambda: self._number > after, self.timeout_s):
                raise RuntimeError("no video from the head (%s)" % (type(self._error).__name__ if self._error else "timeout"))
            return self._frame, self._number


class Body:
    """The robot's control API. A walk command lasts one second unless kept
    alive, so a brain that stops talking leaves the robot standing still."""

    KEEPALIVE_EVERY_S = 0.3

    def __init__(self, url, max_speed, timeout_s=3):
        self.url = url.rstrip("/")
        self.max_speed = max_speed
        self.timeout_s = timeout_s

    def _post(self, path, data=None):
        r = requests.post(self.url + path, data=data or {}, timeout=self.timeout_s)
        r.raise_for_status()

    def state(self):
        r = requests.get(self.url + "/api/v1/state", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json().get("data", {})

    def stop(self):
        self._post("/api/v1/stop")

    def action(self, name):
        """One of the robot's tricks, e.g. "hello"."""
        self._post("/api/v1/action", {"action": name})

    def _clamped(self, vx, vy, w):
        limit = self.max_speed
        return {k: "%.3f" % max(-limit, min(limit, v)) for k, v in (("vx", vx), ("vy", vy), ("w", w))}

    def drive(self, vx=0.0, vy=0.0, w=0.0):
        """Walks with this command for about a second; send it again (with the
        same or a new command) to keep going. For continuous driving."""
        self._post("/api/v1/walk", self._clamped(vx, vy, w))

    def move(self, vx=0.0, vy=0.0, w=0.0, duration_s=0.5):
        """Walks with (vx forward, vy left, w turn left) for duration_s, then stops."""
        self._post("/api/v1/walk", self._clamped(vx, vy, w))
        try:
            end = time.monotonic() + duration_s
            while time.monotonic() < end:
                time.sleep(min(self.KEEPALIVE_EVERY_S, max(0.0, end - time.monotonic())))
                if time.monotonic() < end:
                    self._post("/api/v1/keepalive")
        finally:
            self.stop()


class VisionModel:
    """A vision model served by Ollama."""

    def __init__(self, url, name, timeout_s=60, keep_alive="15m", max_answer_tokens=100):
        self.url = url.rstrip("/")
        self.name = name
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self.max_answer_tokens = max_answer_tokens

    def _generate(self, prompt, images=None, json_only=False, max_tokens=None):
        started = time.monotonic()
        request = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": max_tokens or self.max_answer_tokens},
        }
        if images:
            request["images"] = [base64.b64encode(i).decode() for i in images]
        if json_only:
            request["format"] = "json"
        r = requests.post(self.url + "/api/generate", json=request, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json().get("response", ""), time.monotonic() - started

    def ask(self, jpeg, prompt, json_only=False, max_tokens=None):
        """A question about a photo. Returns (answer text, seconds taken)."""
        return self._generate(prompt, images=[jpeg], json_only=json_only, max_tokens=max_tokens)

    PLAN_TOKENS = 400

    def ask_text(self, prompt, json_only=False):
        """A question without a photo (planning). Returns (answer text, seconds taken)."""
        return self._generate(prompt, json_only=json_only, max_tokens=self.PLAN_TOKENS)
