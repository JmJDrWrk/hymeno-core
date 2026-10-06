"""Checks the verbs and the planner without a robot or a model server:
python -m unittest discover tests"""

import io
import json
import tempfile
import time
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
        self.drives = []

    def move(self, vx=0.0, vy=0.0, w=0.0, duration_s=0.5):
        self.moves.append(("forward" if vx > 0 else "back" if vx < 0 else "left" if w > 0 else "right", duration_s))

    def stop(self):
        pass

    def drive(self, vx=0.0, vy=0.0, w=0.0):
        self.drives.append((round(vx, 2), round(w, 2)))

    def action(self, name):
        self.moves.append((name, 0))


class FakeModel:
    """Answers looks from a script of boxes (qwen2.5 scale, 1036 wide) and plans from a fixed plan."""
    name = "qwen2.5vl:7b"

    def __init__(self, looks=(), plan=None, blocked_after=None):
        self.looks = list(looks)
        self.plan = plan
        self.blocked_after = blocked_after
        self.asked = 0

    def ask(self, jpeg_bytes, prompt, json_only=False, max_tokens=None):
        self.asked += 1
        box = self.looks.pop(0) if self.looks else None
        boxes = [{"label": "x", "bbox_2d": box}] if box else []
        if "BLOCKED" in prompt:   # reach adds a last line about the way ahead
            time.sleep(0.05)
            blocked = self.blocked_after is not None and self.asked > self.blocked_after
            return "```json\n%s\n```\n%s" % (json.dumps(boxes), "BLOCKED" if blocked else "CLEAR"), 0.05
        return json.dumps(boxes), 0.01

    def ask_text(self, prompt, json_only=False):
        return json.dumps(self.plan), 0.01


def skills(looks, blocked_after=None):
    body = FakeBody()
    s = Skills(FakeHead(), body, FakeModel(looks, blocked_after=blocked_after), Journal(tempfile.mkdtemp()),
               say=lambda m: None)
    return s, body


LEFT, CENTRE, RIGHT = [50, 300, 250, 500], [418, 300, 618, 500], [800, 300, 1000, 500]
NEAR = [300, 200, 800, 784]


class SkillsTest(unittest.TestCase):
    def setUp(self):
        import hymeno.skills as module
        module.SETTLE_S = 0   # no waiting in tests
        module.REACH_TICK_S = 0.02

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

    def test_search_turns_the_way_asked(self):
        s, body = skills([None, CENTRE])
        self.assertTrue(s.search("x", turn_direction="right", turn_step="large").ok)
        self.assertEqual(body.moves, [("right", 1.0)])

    def test_reach_steers_without_stopping_and_stops_when_near(self):
        s, body = skills([RIGHT, RIGHT, CENTRE, CENTRE, NEAR])
        result = s.reach("x")
        self.assertTrue(result.ok)
        turns = [w for vx, w in body.drives]
        self.assertTrue(any(w < 0 for w in turns))                  # steered right
        self.assertTrue(any(vx > 0 and w == 0 for vx, w in body.drives))   # then straight on

    def test_reach_stops_when_blocked(self):
        s, body = skills([CENTRE] * 5, blocked_after=2)
        result = s.reach("x")
        self.assertFalse(result.ok)
        self.assertIn("blocked", result.message)

    def test_reach_searches_by_turning(self):
        s, body = skills([None, None, CENTRE, NEAR])
        self.assertTrue(s.reach("x").ok)
        self.assertIn((0.0, 0.4), body.drives)

    def test_describe_answers_with_the_model_text(self):
        s, body = skills([])
        s.model.ask = lambda jpeg, prompt, json_only=False, max_tokens=None: ("Veo una alfombra y una silla.", 0.5)
        result = s.describe("¿qué ves?")
        self.assertTrue(result.ok)
        self.assertIn("alfombra", result.message)
        self.assertEqual(body.moves, [])

    def test_mirrored_picture_turns_the_other_way(self):
        s, body = skills([LEFT, CENTRE])
        s.mirrored = True
        s.face("x")
        self.assertEqual([m[0] for m in body.moves], ["right"])


class VaryingHead:
    """A camera whose view changes a lot between photos (the robot is moving)."""
    def __init__(self):
        self.n = 0

    def photo(self):
        self.n += 1
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (self.n * 70 % 256, self.n * 130 % 256, 40)).save(buf, "JPEG")
        return buf.getvalue()


def decision(do, **kw):
    d = {"see": kw.pop("see", "a sofa, a rug"), "goal": kw.pop("goal", "no"), "dist": kw.pop("dist", "far"),
         "free": kw.pop("free", "YYY"), "do": do, "amount": kw.pop("amount", "small")}
    d.update(kw)
    return json.dumps(d)


class AgentTest(unittest.TestCase):
    def setUp(self):
        import hymeno.skills as module
        module.SETTLE_S = 0
        module.REACH_TICK_S = 0.02

    def agent_with(self, answers, check="yes, it is there", head=None, intent=None, box=CENTRE, dry_run=False):
        from hymeno.agent import Agent
        s, body = skills([])
        s.head = head or VaryingHead()
        s.dry_run = dry_run
        answers = list(answers)
        def ask(jpeg, prompt, json_only=False, max_tokens=None):
            if "believes the goal is achieved" in prompt:
                return check, 0.1
            if prompt.startswith("Locate"):          # the box-drawing look
                return (json.dumps([{"label": "x", "bbox_2d": box}]) if box else "[]"), 0.1
            if prompt.startswith("You are the eyes"):  # describe
                return "Veo un sofá.", 0.1
            return (answers.pop(0) if answers else "?"), 0.1
        s.model.ask = ask
        s.model.ask_text = lambda prompt, json_only=False: (
            json.dumps(intent or {"kind": "task", "target": "yellow tape"}), 0.1)
        return Agent(s, tempfile.mkdtemp(), say=lambda m: None), body

    def test_looks_around_then_walks_and_finishes_after_a_second_look(self):
        agent, body = self.agent_with([
            decision("look_around", amount="medium"),
            decision("turn_right", goal="right"),
            decision("forward", goal="centre"),
            decision("done", goal="centre", dist="near"),
        ])
        result = agent.run("find the yellow tape")
        self.assertTrue(result.startswith("done"), result)
        self.assertNotIn("not confirmed", result)
        self.assertEqual([m[0] for m in body.moves], ["left", "right", "forward"])

    def test_a_claimed_sighting_is_checked_with_a_box(self):
        agent, body = self.agent_with([decision("approach_goal", goal="left"), decision("give_up")], box=None)
        agent.run("go to the tape")
        self.assertEqual([m[0] for m in body.moves], ["left"])     # looked around instead of approaching

    def test_never_walks_into_floor_that_is_not_free(self):
        agent, body = self.agent_with([decision("forward", amount="large", free="NNY"), decision("give_up")])
        self.assertTrue(agent.run("go").startswith("gave up"))
        self.assertEqual([m[0] for m in body.moves], ["right"])

    def test_done_is_checked_by_a_second_look(self):
        agent, body = self.agent_with([decision("done"), decision("give_up")], check="no, nothing yellow here")
        self.assertTrue(agent.run("find the tape").startswith("gave up"))

    def test_backs_off_when_walking_does_not_change_the_view(self):
        agent, body = self.agent_with([decision("forward"), decision("forward"), decision("give_up")],
                                      head=FakeHead())
        agent.run("go")
        self.assertEqual([m[0] for m in body.moves], ["forward", "forward", "back", "left"])

    def test_no_stuck_reflex_in_a_dry_run(self):
        agent, body = self.agent_with([decision("forward"), decision("forward"), decision("give_up")],
                                      head=FakeHead(), dry_run=True)
        self.assertEqual(agent.run("go"), "gave up")

    def test_questions_and_rules_skip_the_loop(self):
        agent, body = self.agent_with([], intent={"kind": "question"})
        self.assertIn("Veo un sofá", agent.run("¿qué ves?"))
        agent, body = self.agent_with([], intent={"kind": "remember", "rule": "las alfombras no son obstáculos"})
        agent.run("recuerda que las alfombras no son obstáculos")
        self.assertIn("alfombras", agent.rules.text())
        self.assertEqual(body.moves, [])

    def test_world_memory_tracks_where_things_were_seen(self):
        agent, body = self.agent_with([decision("turn_left", amount="large", see="sofa"), decision("give_up")])
        agent.run("look")
        self.assertIn("sofa", agent.world.text())
        self.assertIn("50 degrees to the right", agent.world.text())

    def test_stops_after_unreadable_answers(self):
        agent, body = self.agent_with([])
        self.assertIn("could not be read", agent.run("go"))
        self.assertEqual(body.moves, [])


class FakeDetector:
    """Knows cats; sees what the script says, one list of (x1, x2, width-ish) per photo."""
    def __init__(self, frames):
        self.frames = list(frames)

    def class_for(self, target):
        return "cat" if "cat" in target else None

    def detect(self, jpeg):
        from hymeno.perception import Box
        spans = self.frames.pop(0) if self.frames else []
        return [Box("cat", x1, 0.4, x2, 0.6) for x1, x2 in spans]


C_LEFT, C_CENTRE, C_RIGHT, C_NEAR = (0.05, 0.2), (0.45, 0.55), (0.8, 0.95), (0.25, 0.75)


class FastEyesTest(unittest.TestCase):
    def setUp(self):
        import hymeno.skills as module
        module.FAST_MIN_PERIOD_S = 0
        module.SETTLE_S = 0

    def with_detector(self, frames):
        s, body = skills([])
        s.detector = FakeDetector(frames)
        return s, body

    def test_search_turns_until_the_detector_sees_it(self):
        s, body = self.with_detector([[], [], [C_RIGHT]])
        self.assertTrue(s.search("a black cat").ok)
        self.assertEqual(body.drives, [(0.0, 0.5), (0.0, 0.5)])
        self.assertEqual(s.model.asked, 0)             # the vision model was not needed

    def test_face_turns_until_centred(self):
        s, body = self.with_detector([[C_LEFT], [C_CENTRE], [C_CENTRE]])
        self.assertTrue(s.face("the cat").ok)
        self.assertTrue(body.drives[0][1] > 0)          # turned left towards it

    def test_reach_steers_and_stops_close(self):
        s, body = self.with_detector([[], [C_RIGHT], [C_CENTRE], [C_NEAR]])
        self.assertTrue(s.reach("the cat").ok)
        self.assertEqual(body.drives[0], (0.0, 0.4))   # searching
        self.assertTrue(body.drives[1][1] < 0)          # steering right
        self.assertTrue(body.drives[2][0] > 0 and body.drives[2][1] == 0)   # straight on

    def test_other_targets_still_use_the_vision_model(self):
        s, body = self.with_detector([])
        s.model.looks = [CENTRE]
        self.assertTrue(s.look("yellow tape").ok)
        self.assertEqual(s.model.asked, 1)

    def test_agent_takes_the_fast_path_for_known_things(self):
        from hymeno.agent import Agent
        s, body = self.with_detector([[C_RIGHT], [C_CENTRE], [C_NEAR]])
        s.model.ask_text = lambda prompt, json_only=False: (
            json.dumps({"kind": "task", "target": "a black cat", "action": "reach"}), 0.1)
        s.model.ask = lambda jpeg, prompt, json_only=False, max_tokens=None: ("yes, the cat is right there", 0.1)
        result = Agent(s, tempfile.mkdtemp(), say=lambda m: None).run("ve hacia el gato")
        self.assertTrue(result.startswith("done"), result)


class DirectTest(unittest.TestCase):
    def test_words_describe_the_target_settings_are_named(self):
        from hymeno.__main__ import direct
        verb, args = direct("/search a large fan on the right turn_direction=left turn_step=small")
        self.assertEqual(verb, "search")
        self.assertEqual(args, {"target": "a large fan on the right", "turn_direction": "left", "turn_step": "small"})

    def test_describe_takes_the_question(self):
        from hymeno.__main__ import direct
        self.assertEqual(direct("/describe is anyone there?"), ("describe", {"question": "is anyone there?"}))

    def test_half_characters_are_dropped(self):
        from hymeno.__main__ import clean
        self.assertEqual(clean("the bl\udcc3ack cat "), "the black cat")
        self.assertEqual(clean("¿qué ves?"), "¿qué ves?")

    def test_bad_value_is_rejected(self):
        from hymeno.__main__ import direct
        with self.assertRaises(ValueError):
            direct("/search a fan turn_step=huge")

    def test_turn_takes_no_description(self):
        from hymeno.__main__ import direct
        with self.assertRaises(ValueError):
            direct("/turn right")
        self.assertEqual(direct("/turn direction=right amount=large"), ("turn", {"direction": "right", "amount": "large"}))


class PlannerTest(unittest.TestCase):
    def test_valid_plan(self):
        model = FakeModel(plan={"plan": [{"verb": "search", "args": {"target": "an electric fan", "only_turning": True}},
                                         {"verb": "face", "args": {"target": "an electric fan"}}], "say": "ok"})
        steps, sentence, _ = planner.plan(model, "busca el ventilador girando solo")
        self.assertEqual([v for v, _ in steps], ["search", "face"])

    def test_verb_written_with_brackets_is_accepted(self):
        model = FakeModel(plan={"plan": [{"verb": "stop()", "args": {}}, {"verb": " Greet ", "args": {}}]})
        steps, _, _ = planner.plan(model, "para")
        self.assertEqual([v for v, _ in steps], ["stop", "greet"])

    def test_unknown_verb_is_rejected(self):
        model = FakeModel(plan={"plan": [{"verb": "jump", "args": {}}]})
        with self.assertRaises(ValueError):
            planner.plan(model, "jump")

    def test_missing_target_is_rejected(self):
        model = FakeModel(plan={"plan": [{"verb": "face", "args": {}}]})
        with self.assertRaises(ValueError):
            planner.plan(model, "face it")


class StepModeTest(unittest.TestCase):
    def test_look_saves_the_photo_and_waits_for_enter(self):
        import os
        from unittest import mock
        s, _ = skills([CENTRE])
        s.step = True
        with mock.patch("builtins.input", return_value="") as enter:
            self.assertTrue(s.look("x").ok)
        self.assertEqual(enter.call_count, 1)
        folder = os.path.join(os.path.dirname(s.journal.path), "steps")
        self.assertEqual(len(os.listdir(folder)), 1)

    def test_background_looks_pause_in_the_driving_loop(self):
        from unittest import mock
        s, _ = skills([RIGHT, NEAR])
        s.step = True
        with mock.patch("builtins.input", return_value="") as enter:
            self.assertTrue(s.reach("x").ok)
        self.assertGreaterEqual(enter.call_count, 2)    # a third look may race the end

    def test_off_by_default(self):
        from unittest import mock
        s, _ = skills([CENTRE])
        with mock.patch("builtins.input", side_effect=AssertionError("asked for Enter")):
            self.assertTrue(s.look("x").ok)


if __name__ == "__main__":
    unittest.main()
