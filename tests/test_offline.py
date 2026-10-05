"""Checks the verbs and the planner without a robot or a model server:
python -m unittest discover tests"""

import io
import json
import tempfile
import unittest

from PIL import Image

from hymeno import planner
from hymeno.journal import Journal
from hymeno.skills import Skills


def jpeg(width=640, height=480):
    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, "JPEG")
    return buf.getvalue()


class FakeHead:
    def photo(self):
        return jpeg()


class FakeBody:
    def __init__(self):
        self.moves = []

    def move(self, vx=0.0, vy=0.0, w=0.0, duration_s=0.5):
        self.moves.append(("forward" if vx else "left" if w > 0 else "right", duration_s))

    def stop(self):
        pass

    def action(self, name):
        self.moves.append((name, 0))


class FakeModel:
    """Answers looks from a script of boxes (qwen2.5 scale, 1036 wide) and plans from a fixed plan."""
    name = "qwen2.5vl:7b"

    def __init__(self, looks=(), plan=None):
        self.looks = list(looks)
        self.plan = plan

    def ask(self, jpeg_bytes, prompt):
        box = self.looks.pop(0) if self.looks else None
        return (json.dumps([{"label": "x", "bbox_2d": box}]) if box else "[]"), 0.01

    def ask_text(self, prompt, json_only=False):
        return json.dumps(self.plan), 0.01


def skills(looks):
    body = FakeBody()
    s = Skills(FakeHead(), body, FakeModel(looks), Journal(tempfile.mkdtemp()), say=lambda m: None)
    return s, body


LEFT, CENTRE, RIGHT = [50, 300, 250, 500], [418, 300, 618, 500], [800, 300, 1000, 500]
NEAR = [300, 200, 800, 784]


class SkillsTest(unittest.TestCase):
    def setUp(self):
        import hymeno.skills as module
        module.SETTLE_S = 0   # no waiting in tests

    def test_face_turns_towards_the_target_then_stops(self):
        s, body = skills([LEFT, CENTRE])
        self.assertTrue(s.face("x").ok)
        self.assertEqual([m[0] for m in body.moves], ["left"])

    def test_face_fails_when_not_in_sight(self):
        s, body = skills([None])
        self.assertFalse(s.face("x").ok)
        self.assertEqual(body.moves, [])

    def test_search_turns_until_found(self):
        s, body = skills([None, None, RIGHT])
        result = s.search("x")
        self.assertTrue(result.ok)
        self.assertIn("right", result.message)
        self.assertEqual([m[0] for m in body.moves], ["left", "left"])

    def test_search_gives_up(self):
        s, body = skills([])
        self.assertFalse(s.search("x").ok)

    def test_approach_walks_until_near(self):
        s, body = skills([CENTRE, RIGHT, CENTRE, NEAR])
        self.assertTrue(s.approach("x").ok)
        self.assertEqual([m[0] for m in body.moves], ["forward", "right", "forward"])

    def test_mirrored_picture_turns_the_other_way(self):
        s, body = skills([LEFT, CENTRE])
        s.mirrored = True
        s.face("x")
        self.assertEqual([m[0] for m in body.moves], ["right"])


class PlannerTest(unittest.TestCase):
    def test_valid_plan(self):
        model = FakeModel(plan={"plan": [{"verb": "search", "args": {"target": "an electric fan", "only_turning": True}},
                                         {"verb": "face", "args": {"target": "an electric fan"}}], "say": "ok"})
        steps, sentence, _ = planner.plan(model, "busca el ventilador girando solo")
        self.assertEqual([v for v, _ in steps], ["search", "face"])

    def test_unknown_verb_is_rejected(self):
        model = FakeModel(plan={"plan": [{"verb": "jump", "args": {}}]})
        with self.assertRaises(ValueError):
            planner.plan(model, "jump")

    def test_missing_target_is_rejected(self):
        model = FakeModel(plan={"plan": [{"verb": "face", "args": {}}]})
        with self.assertRaises(ValueError):
            planner.plan(model, "face it")


if __name__ == "__main__":
    unittest.main()
