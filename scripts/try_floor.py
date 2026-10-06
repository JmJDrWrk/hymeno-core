"""Try SegFormer (ADE20K) as floor eyes: paints the floor it finds in green.

From a folder of photos:   python scripts/try_floor.py data/steps
From the robot's camera:   python scripts/try_floor.py --head
  (takes a photo each time Enter is pressed; move the robot between photos, q to quit)

Results go to data/floor-test/. Needs transformers (pip install transformers)."""

import argparse
import glob
import io
import os
import sys
import time

import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
FLOOR_WORDS = ("floor", "rug", "carpet")
OUT = "data/floor-test"


def load_model(name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(name).to(device).eval()
    labels = model.config.id2label
    floor_ids = [int(i) for i, label in labels.items() if any(w in label.lower() for w in FLOOR_WORDS)]
    print("device %s; counted as floor: %s" % (device, ", ".join(labels[i] for i in floor_ids)))
    return processor, model, device, labels, floor_ids


def segment(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu()


def paint(image, classes, floor_ids, labels, name):
    floor = torch.isin(classes, torch.tensor(floor_ids))
    green = Image.new("RGB", image.size, (0, 255, 0))
    mask = Image.fromarray((floor.numpy() * 140).astype("uint8"))
    out = image.copy()
    out.paste(green, mask=mask)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    out.save(path, "JPEG")
    ids, counts = classes.unique(return_counts=True)
    top = sorted(zip(counts.tolist(), ids.tolist()), reverse=True)[:4]
    seen = ", ".join("%s %.0f%%" % (labels[i], 100 * c / classes.numel()) for c, i in top)
    print("%s: floor %.0f%% | most of the photo: %s" % (path, 100 * floor.float().mean(), seen))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", help="folder of .jpg photos")
    parser.add_argument("--head", action="store_true", help="take photos from the robot's camera (config.yaml)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if not args.folder and not args.head:
        parser.error("give a folder of photos or --head")

    processor, model, device, labels, floor_ids = load_model(args.model)

    def run(image, name):
        started = time.monotonic()
        classes = segment(image, processor, model, device)
        print("  segmented in %.0f ms" % ((time.monotonic() - started) * 1000))
        paint(image, classes, floor_ids, labels, name)

    if args.head:
        from hymeno import config
        from hymeno.clients import Head
        head = Head(config.load("config.yaml")["head"]["url"])
        n = 0
        while input("Enter takes a photo (q quits): ").strip().lower() != "q":
            n += 1
            run(Image.open(io.BytesIO(head.photo())).convert("RGB"), "head-%02d.jpg" % n)
    else:
        for path in sorted(glob.glob(os.path.join(args.folder, "*.jpg"))):
            run(Image.open(path).convert("RGB"), os.path.basename(path))


if __name__ == "__main__":
    main()
