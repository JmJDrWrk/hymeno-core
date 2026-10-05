"""A record of every look: what was seen, how long it took and what was done.
One JSON object per line in data/journal.jsonl; the seed of the brain's memory."""

import json
import os
import time


class Journal:
    def __init__(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "journal.jsonl")

    def write(self, **entry):
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
