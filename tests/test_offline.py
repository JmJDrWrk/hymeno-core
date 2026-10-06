"""Offline tests: no robot, camera or model needed."""

import io
import unittest

from PIL import Image

from hymeno import perception


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


if __name__ == "__main__":
    unittest.main()
