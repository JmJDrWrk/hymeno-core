"""Try fast open-vocabulary detectors (you name the thing in words) on photos:
one sheet per photo in data/goals-test/, with each detector's boxes and time.

    python scripts/try_goals.py "black cat"                 # photos in data/goals/
    python scripts/try_goals.py "chair" data/runs/<date-time>
    python scripts/try_goals.py "shoe, bottle, yellow tape" data/goals   # several: each box says which

Put in the folder some photos where the thing shows (near, far, half hidden)
and some where it doesn't. The first run downloads each detector (and the text
model it needs); close Ollama first so the GPU is free."""

import glob
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hymeno.floor import write  # noqa: E402

DETECTORS = ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yoloe-11s-seg.pt"]
SHOW = 0.05   # boxes drawn from this confidence (thin), to see near misses
FOUND = 0.25  # a photo counts as "found" from this confidence (thick)
OUT = "data/goals-test"
TILE = (320, 240)


def load(name, names):
    if name.startswith("yoloe"):
        from ultralytics import YOLOE
        model = YOLOE(name)
        model.set_classes(names, model.get_text_pe(names))
    else:
        from ultralytics import YOLOWorld
        model = YOLOWorld(name)
        model.set_classes(names)
    return model


def boxes(model, image, names):
    """[(confidence, (x1, y1, x2, y2) in pixels, name)], best first."""
    result = model.predict(image, conf=SHOW, verbose=False)[0]
    found = [(float(c), tuple(float(v) for v in xyxy), names[int(k)])
             for c, xyxy, k in zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), result.boxes.cls.tolist())]
    return sorted(found, reverse=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    names = [n.strip() for n in sys.argv[1].split(",") if n.strip()]
    folder = sys.argv[2] if len(sys.argv) > 2 else "data/goals"
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        sys.exit("no photos in %s/" % folder)
    photos = [(os.path.splitext(os.path.basename(p))[0], Image.open(p).convert("RGB")) for p in paths]
    sheets = {name: [image.resize(TILE)] for name, image in photos}
    summary = {}
    for detector in DETECTORS:
        print("%s: loading..." % detector, flush=True)
        try:
            model = load(detector, names)
            boxes(model, photos[0][1], names)  # warm up: the first run is always slow
        except Exception as e:
            print("  can't use it: %s" % e)
            continue
        times, hits = [], 0
        for name, image in photos:
            started = time.monotonic()
            found = boxes(model, image, names)
            ms = (time.monotonic() - started) * 1000
            times.append(ms)
            hits += bool(found and found[0][0] >= FOUND)
            tile = image.copy()
            pen = ImageDraw.Draw(tile)
            for confidence, (x1, y1, x2, y2), label in found[:8]:
                pen.rectangle([x1, y1, x2, y2], outline=(255, 0, 255), width=5 if confidence >= FOUND else 1)
                write(pen, (x1 + 4, y1 + 4), ("%s " % label if len(names) > 1 else "") + "%.2f" % confidence, size=16)
            tile = tile.resize(TILE)
            write(ImageDraw.Draw(tile), (6, 4), "%s  %.0f ms" % (detector, ms), size=14)
            sheets[name].append(tile)
        summary[detector] = (np.mean(times), hits)
    os.makedirs(OUT, exist_ok=True)
    columns = 3
    for name, tiles in sheets.items():
        rows = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (TILE[0] * columns, TILE[1] * rows))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % columns) * TILE[0], (i // columns) * TILE[1]))
        sheet.save(os.path.join(OUT, name + ".jpg"), "JPEG")
    print("sheets in %s/ (thick box: confidence %.2f or more)" % (OUT, FOUND))
    for detector, (ms, hits) in summary.items():
        print("  %-22s %4.0f ms a photo, found in %d of %d photos" % (detector, ms, hits, len(photos)))


if __name__ == "__main__":
    main()
