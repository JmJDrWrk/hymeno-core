"""Offline tests: no robot, camera or model needed."""

import io
import unittest

import numpy as np
from PIL import Image

from hymeno import floor, perception


def jpeg(colour=(0, 0, 0), width=640, height=480):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, "JPEG")
    return buf.getvalue()


class PerceptionTest(unittest.TestCase):
    def test_qwen25_boxes_come_in_resized_pixels(self):
        answer = '```json\n[{"label": "tape", "bbox_2d": [518, 392, 1036, 784]}]\n```'
        [box] = perception.parse_boxes(answer, "qwen2.5vl:7b", 640, 480)
        self.assertAlmostEqual(box.x1, 0.5)
        self.assertAlmostEqual(box.y2, 1.0)

    def test_qwen3_boxes_come_on_a_thousand_scale(self):
        [box] = perception.parse_boxes('[{"bbox_2d": [250, 0, 750, 500]}]', "qwen3-vl:8b", 640, 480)
        self.assertAlmostEqual(box.center_x, 0.5)
        self.assertAlmostEqual(box.height, 0.5)

    def test_anything_else_is_no_box(self):
        self.assertEqual(perception.parse_boxes("I see no tape.", "qwen2.5vl:7b", 640, 480), [])
        self.assertEqual(perception.parse_boxes('[{"bbox_2d": [1, 2]}]', "qwen2.5vl:7b", 640, 480), [])

    def test_view_change(self):
        self.assertEqual(perception.view_change(jpeg(), jpeg()), 0)
        self.assertGreater(perception.view_change(jpeg(), jpeg((255, 255, 255))), 200)



def floor_mask(walls=(), width=640, height=480):
    """Floor up to the horizon, except where a wall (column from, column to,
    cm away) stops it."""
    mask = np.zeros((height, width), bool)
    mask[int(np.ceil(floor.HORIZON * height)):] = True
    for x0, x1, cm in walls:
        mask[:int(np.ceil(floor.row_of(cm) * height)), int(x0 * width):int(x1 * width)] = False
    return mask


def decide(walls=(), last=None):
    return floor.decide(floor.free_distances(floor_mask(walls)), last)


class FloorTest(unittest.TestCase):
    def test_open_floor_goes_straight_on(self):
        d = decide()
        self.assertEqual(d["direction"], "ahead")
        self.assertEqual(d["clear_cm"], floor.FAR_CM)

    def test_wall_close_is_blocked(self):
        self.assertEqual(decide([(0, 1, 15)])["direction"], "blocked")

    def test_a_thin_post_ahead_is_avoided(self):
        d = decide([(0.50, 0.55, 25)])
        self.assertIn(d["direction"], ("left", "right"))
        self.assertGreater(d["clear_cm"], floor.STOP_CM)

    def test_goes_through_a_wide_opening(self):
        d = decide([(0, 0.55, 20), (0.95, 1, 20)])
        self.assertEqual(d["direction"], "right")

    def test_does_not_try_a_narrow_opening(self):
        self.assertEqual(decide([(0, 0.65, 20), (0.85, 1, 20)])["direction"], "blocked")

    def test_a_door_of_the_floor_colour_does_not_fool_it(self):
        mask = floor_mask([(0, 1, 30)])
        mask[int(0.3 * mask.shape[0]):, int(0.4 * mask.shape[1]):int(0.6 * mask.shape[1])] = True
        d = floor.decide(floor.free_distances(mask))
        self.assertLess(d["ahead_cm"], 31)

    def test_keeps_turning_the_same_way_while_blocked(self):
        last = {"direction": "blocked", "steer": 1.0, "heading": floor.CENTRE}
        self.assertEqual(decide([(0, 1, 15)], last)["steer"], 1.0)



def photo(direction="ahead", steer=0.0):
    return {"direction": direction, "steer": steer, "heading": floor.CENTRE + steer * 0.4}


class GoalTest(unittest.TestCase):
    def far(self, walls):
        return floor.free_distances(floor_mask(walls), far=floor.SEE_CM)

    def test_an_opening_is_where_the_floor_goes_far(self):
        found = floor.openings(self.far([(0, 0.6, 40), (0.85, 1, 40)]))
        self.assertEqual(len(found), 1)
        self.assertTrue(0.6 < found[0]["column"] < 0.85)

    def test_a_goal_pulls_the_way_chosen(self):
        strips = floor.free_distances(floor_mask())
        self.assertEqual(floor.decide(strips)["direction"], "ahead")
        self.assertEqual(floor.decide(strips, goal=0.8)["direction"], "right")

    def test_a_goal_follows_its_opening_and_is_lost_without_it(self):
        goal = floor.Goal()
        goal.set(0.7)
        goal.update([{"column": 0.75, "cm": 150, "width": 0.2}])
        self.assertEqual(goal.column, 0.75)
        for _ in range(floor.LOST + 1):
            goal.update([])
        self.assertIsNone(goal.column)


class TargetTest(unittest.TestCase):
    """The search and walk of a goal in words, with the model's answers given by hand."""

    def target(self):
        from hymeno.target import Target
        t = Target("the chair")
        t.misses_allowed = 3
        return t

    def answer(self, target, frame, cm=None):
        box = None
        if cm is not None:
            bottom = floor.row_of(cm)
            box = perception.Box("the chair", 0.6, bottom - 0.2, 0.8, bottom)
        target._answer = {"frame": frame, "box": box, "seconds": 2.0, "error": None}

    def test_searches_standing_then_goes_when_seen(self):
        from hymeno import target as words
        t = self.target()
        self.assertEqual(t.check(1), "wait")
        self.answer(t, 1)
        self.assertEqual(t.check(2), "turn")
        self.answer(t, 2)  # about a photo from before the turn: ignored
        self.assertEqual(t.check(3), "wait")
        self.answer(t, 4, cm=150)
        self.assertEqual(t.check(5), "go")
        self.assertAlmostEqual(t.column, 0.7)
        self.assertAlmostEqual(t.cm, 150, places=0)
        self.answer(t, 6, cm=words.ARRIVE_CM - 5)
        self.assertEqual(t.check(7), "arrived")

    def test_searches_again_when_lost_while_going(self):
        t = self.target()
        self.answer(t, 1, cm=150)
        t.check(2)
        for frame in range(3, 3 + t.misses_allowed):
            self.answer(t, frame)
            todo = t.check(frame)
        self.assertEqual((todo, t.state), ("wait", "searching"))


class GroupTest(unittest.TestCase):
    """With groups of 3, whatever floor.GROUP is set to."""

    def setUp(self):
        self.saved = floor.GROUP, floor.MAJORITY
        floor.GROUP, floor.MAJORITY = 3, 2

    def tearDown(self):
        floor.GROUP, floor.MAJORITY = self.saved

    def test_keys_go_by_threes(self):
        self.assertEqual([floor.group_of(n) for n in (1, 3, 4, 9)], ["000001", "000001", "000002", "000003"])

    def test_two_blocked_turn_on_the_spot(self):
        order = floor.combine([photo("blocked", 1.0), photo("blocked", 1.0), photo()])
        self.assertEqual((order["direction"], order["steer"]), ("blocked", 1.0))

    def test_walks_with_the_median_steer(self):
        order = floor.combine([photo("right", 0.3), photo(), photo("right", 0.2)])
        self.assertEqual((order["direction"], order["steer"]), ("right", 0.2))

    def test_an_order_once_the_group_is_complete(self):
        groups = floor.Groups()
        self.assertEqual(groups.add(1, photo()), ("000001", None))
        self.assertEqual(groups.add(2, photo()), ("000001", None))
        self.assertEqual(groups.add(3, photo())[1]["direction"], "ahead")

    def test_two_blocked_stop_without_waiting_for_the_third(self):
        groups = floor.Groups()
        groups.add(4, photo("blocked", -1.0))
        self.assertEqual(groups.add(5, photo("blocked", -1.0))[1]["direction"], "blocked")

    def test_a_group_is_carried_out_once(self):
        groups = floor.Groups()
        groups.add(4, photo("blocked", -1.0))
        groups.add(5, photo("blocked", -1.0))
        self.assertTrue(groups.fresh)
        self.assertEqual(groups.add(6, photo())[1]["direction"], "blocked")
        self.assertFalse(groups.fresh)



def steering(*steers):
    """Per-photo decisions, oldest first: a number is a steer, "B<" / "B>" blocked turning left / right."""
    return [photo("blocked", -1.0 if s == "B<" else 1.0) if isinstance(s, str) else photo(floor.label(s), s)
            for s in steers]


class CombineTableTest(unittest.TestCase):
    """The behaviour agreed for a group of 5 (oldest photo first). Blocked: by
    majority of photos; a blocked minority is ignored. Steer: the median
    weighted by recency (newest weighs 5, oldest 1), always a steer some
    photo chose."""

    def setUp(self):
        self.saved = floor.GROUP, floor.MAJORITY
        floor.GROUP, floor.MAJORITY = 5, 3

    def tearDown(self):
        floor.GROUP, floor.MAJORITY = self.saved

    def order(self, *steers):
        o = floor.combine(steering(*steers))
        return o["direction"], round(o["steer"], 2)

    def test_1_all_agree(self):
        self.assertEqual(self.order(0.48, 0.48, 0.48, 0.48, 0.48), ("right", 0.48))

    def test_2_one_odd_photo_is_ignored(self):
        self.assertEqual(self.order(0.53, 0.48, 0.48, -0.28, 0.48), ("right", 0.48))

    def test_3_three_against_two(self):
        self.assertEqual(self.order(-0.04, -0.09, 0.53, 0.63, 0.63), ("right", 0.63))

    def test_4_one_blocked_is_ignored(self):
        self.assertEqual(self.order("B<", 0, 0, 0, 0), ("ahead", 0))

    def test_5_two_blocked_are_ignored(self):
        self.assertEqual(self.order("B<", "B<", 0, 0, 0), ("ahead", 0))

    def test_6_majority_blocked_turns_on_the_spot(self):
        self.assertEqual(self.order("B>", "B>", "B>", 0, 0), ("blocked", 1.0))

    def test_7_no_agreement_follows_the_newest(self):
        self.assertEqual(self.order(-0.8, -0.4, 0, 0.4, 0.8), ("right", 0.4))

    def test_8_blocked_turns_the_way_most_say(self):
        self.assertEqual(self.order("B<", "B>", "B<", 0, 0), ("blocked", -1.0))

    def test_9_majority_blocked_does_not_wait_for_the_rest(self):
        groups = floor.Groups()
        groups.add(1, steering("B<")[0])
        groups.add(2, steering("B<")[0])
        self.assertEqual(groups.add(3, steering("B<")[0])[1]["direction"], "blocked")


class CalibrateTest(unittest.TestCase):
    def test_clicks_give_back_the_geometry(self):
        from hymeno.calibrate import solve
        horizon, centre, cm_k, half_width, w, h = 0.45, 0.52, 2.4, 1.15, 640, 480

        def tape(side, row):
            return ((centre + side * half_width * (row - horizon)) * w, row * h)

        def screw(cm):
            return (centre * w, (horizon + cm_k / cm) * h)

        got = solve([tape(-1, 0.9), tape(-1, 0.55), tape(1, 0.9), tape(1, 0.55), screw(25), screw(50)], w, h)
        for key, value in (("horizon", horizon), ("centre", centre), ("cm_k", cm_k), ("half_width", half_width)):
            self.assertAlmostEqual(got[key], value, places=6)
        self.assertEqual(got["warnings"], [])

    def test_parallel_tapes_make_no_sense(self):
        from hymeno.calibrate import solve
        with self.assertRaises(ValueError):
            solve([(100, 400), (100, 300), (500, 400), (500, 300), (300, 350), (300, 300)], 640, 480)


if __name__ == "__main__":
    unittest.main()
