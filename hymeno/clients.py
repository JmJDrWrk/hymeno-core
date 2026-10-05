"""The three things the brain talks to: the robot's head camera, the robot's
body and the vision model server. Each is just a URL."""

import base64
import time

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

    def move(self, vx=0.0, vy=0.0, w=0.0, duration_s=0.5):
        """Walks with (vx forward, vy left, w turn left) for duration_s, then stops."""
        limit = self.max_speed
        command = {k: max(-limit, min(limit, v)) for k, v in (("vx", vx), ("vy", vy), ("w", w))}
        self._post("/api/v1/walk", {k: "%.3f" % v for k, v in command.items()})
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

    def ask(self, jpeg, prompt):
        """Returns (answer text, seconds taken)."""
        started = time.monotonic()
        r = requests.post(self.url + "/api/generate", json={
            "model": self.name,
            "prompt": prompt,
            "images": [base64.b64encode(jpeg).decode()],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": self.max_answer_tokens},
        }, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json().get("response", ""), time.monotonic() - started
