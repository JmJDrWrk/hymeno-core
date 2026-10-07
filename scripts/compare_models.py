"""Compare floor models on hard photos. For each photo, a sheet in
data/compare/: the photo and what each model sees as floor (green, where it
is at least floor.FLOOR_P sure), with how long it took.

    python scripts/compare_models.py              # photos in data/hard/
    python scripts/compare_models.py some/folder

Close Ollama first (or unload its model): the GPU needs the memory.
The first run downloads each model (a few hundred MB each)."""

import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hymeno import floor  # noqa: E402

MODELS = [
    "nvidia/segformer-b2-finetuned-ade-512-512",  # the one in use
    "nvidia/segformer-b5-finetuned-ade-640-640",
    "facebook/mask2former-swin-tiny-ade-semantic",
    "shi-labs/oneformer_ade20k_swin_tiny",
]
OUT = "data/compare"
TILE = (320, 240)


def floor_ids(labels):
    return [int(i) for i, label in labels.items() if any(w in label.lower() for w in floor.FLOOR_WORDS)]


def load(name, device):
    """A function: PIL image -> numpy array (h x w) of how sure the model is
    that each pixel is floor (0..1)."""
    if "segformer" in name:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        processor = SegformerImageProcessor.from_pretrained(name)
        model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
        ids = floor_ids(model.config.id2label)

        def run(image):
            inputs = processor(images=image, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
            return logits.softmax(dim=1)[0, ids].sum(dim=0)
    else:
        # Mask2Former and OneFormer answer with masks and a class for each
        # mask; the floor probability of a pixel is the share of its masks'
        # weight that goes to floor classes.
        if "oneformer" in name:
            from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
            processor = OneFormerProcessor.from_pretrained(name)
            model = OneFormerForUniversalSegmentation.from_pretrained(name).to(device).eval()
            extra = {"task_inputs": ["semantic"]}
        else:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
            processor = AutoImageProcessor.from_pretrained(name)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(name).to(device).eval()
            extra = {}
        ids = floor_ids(model.config.id2label)

        def run(image):
            inputs = processor(images=image, return_tensors="pt", **extra).to(device)
            out = model(**inputs)
            classes = out.class_queries_logits.softmax(dim=-1)[..., :-1]  # the last one is "no object"
            masks = torch.nn.functional.interpolate(out.masks_queries_logits, size=image.size[::-1],
                                                    mode="bilinear", align_corners=False).sigmoid()
            weight = torch.einsum("bqc,bqhw->bchw", classes, masks)[0]
            return weight[ids].sum(dim=0) / weight.sum(dim=0).clamp_min(1e-6)

    def probability(image):
        with torch.no_grad():
            return run(image).cpu().numpy()

    return probability, model


def tile(image, title):
    t = image.resize(TILE)
    floor.write(ImageDraw.Draw(t), (6, 4), title, size=14)
    return t


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/hard"
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if not paths:
        sys.exit("no photos in %s/" % folder)
    photos = [(os.path.splitext(os.path.basename(p))[0], Image.open(p).convert("RGB")) for p in paths]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sheets = {name: [tile(image, name)] for name, image in photos}
    times = {}
    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print("%s: loading..." % short, flush=True)
        try:
            probability, model = load(model_name, device)
        except Exception as e:
            print("  can't load it: %s" % e)
            continue
        probability(photos[0][1])  # warm up: the first run is always slow
        times[short] = []
        for name, image in photos:
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.monotonic()
            sure = probability(image)
            ms = (time.monotonic() - started) * 1000
            times[short].append(ms)
            painted = image.copy()
            mask = Image.fromarray(((sure >= floor.FLOOR_P) * 140).astype("uint8"))
            painted.paste(Image.new("RGB", image.size, (0, 255, 0)), mask=mask)
            sheets[name].append(tile(painted, "%s  %.0f ms" % (short, ms)))
        del model, probability
        if device == "cuda":
            torch.cuda.empty_cache()
    os.makedirs(OUT, exist_ok=True)
    columns = 3
    for name, tiles in sheets.items():
        rows = (len(tiles) + columns - 1) // columns
        sheet = Image.new("RGB", (TILE[0] * columns, TILE[1] * rows))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % columns) * TILE[0], (i // columns) * TILE[1]))
        sheet.save(os.path.join(OUT, name + ".jpg"), "JPEG")
    print("sheets in %s/" % OUT)
    for short, ms in times.items():
        print("  %-45s %4.0f ms a photo" % (short, np.mean(ms)))


if __name__ == "__main__":
    main()
