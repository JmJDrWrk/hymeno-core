"""Depth eyes: a monocular depth model (Depth Anything V2, indoor metric) that
turns each photo into distances in metres, many times a second on a GPU.

Like an animal's peripheral vision it does not name things; it only says how
much room there is in each direction, which is all wandering needs.

Optional: needs the transformers package (requirements-depth.txt)."""

import io

from PIL import Image

DEFAULT_WEIGHTS = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"   # about 100 MB
COLUMNS = 7
# The rows that count: around the horizon. Lower rows show the floor itself,
# which is near but walkable; higher rows show what is above the robot.
BAND = (0.35, 0.70)
NEAREST_PERCENTILE = 10   # a column is as near as its nearest tenth, so thin legs still count
# Share of the photo's width and height kept around its centre: a wide-angle
# lens bends the edges, where the depth model (trained on normal lenses) errs.
CROP = 0.6


def column_clearances(depth, columns=COLUMNS, band=BAND, percentile=NEAREST_PERCENTILE):
    """Metres of room in each vertical column of a depth map, left to right."""
    import numpy as np
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    rows = depth[int(height * band[0]):max(int(height * band[1]), int(height * band[0]) + 1)]
    edges = np.linspace(0, width, columns + 1).astype(int)
    return [float(np.percentile(rows[:, a:max(b, a + 1)], percentile)) for a, b in zip(edges, edges[1:])]


class DepthEyes:
    def __init__(self, weights=DEFAULT_WEIGHTS):
        import torch                      # imported here: the brain also works without them
        from transformers import pipeline
        self.weights = weights
        self.pipe = pipeline("depth-estimation", model=weights, device=0 if torch.cuda.is_available() else -1)

    def depth(self, jpeg):
        """The depth map in metres, as a 2D numpy array."""
        with Image.open(io.BytesIO(jpeg)) as image:
            width, height = image.size
            margin_x, margin_y = width * (1 - CROP) / 2, height * (1 - CROP) / 2
            centre = image.convert("RGB").crop((int(margin_x), int(margin_y),
                                                int(width - margin_x), int(height - margin_y)))
            predicted = self.pipe(centre)["predicted_depth"]
        return predicted.squeeze().float().cpu().numpy()

    def clearances(self, jpeg):
        """Metres of room in each column of the photo, left to right."""
        return column_clearances(self.depth(jpeg))


def load(settings, say=print):
    """The depth eyes, or None when they are switched off or not installed."""
    options = settings.get("depth", {})
    if options is False or (isinstance(options, dict) and options.get("enabled") is False):
        return None
    options = options if isinstance(options, dict) else {}
    weights = options.get("weights", DEFAULT_WEIGHTS)
    try:
        eyes = DepthEyes(weights)
    except ImportError:
        say("no depth eyes (pip install -r requirements-depth.txt to add them); wander is not available")
        return None
    say("depth eyes: %s" % weights)
    return eyes
