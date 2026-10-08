# data/

Everything the brain records stays here, on this machine (only this README is
in git).

- `calibration.json` — camera geometry from `python -m hymeno calibrate`
  (horizon, centre, cm per row, robot width). `hymeno/floor.py` loads it at
  start; delete it to go back to the numbers in the code.
- `calibration.jpg` — the photo the last calibration was clicked on.
- `check.jpg` — the photo taken by `python -m hymeno check`.
- `floor/` — what `python -m hymeno floor` / `explore` saw:
  - `latest.jpg` / `latest-raw.jpg` — the newest photo with the drawing / as it
    came from the camera (overwritten in loops).
  - `<date-time>.jpg` / `-raw.jpg` — single photos from `floor` without `--every`.
- `runs/<date-time>/` — photos recorded with `--record` (`00001.jpg`, ...) and the console (`log.txt`),
  for `python -m hymeno replay`.
- `hard/` — hard photos picked from runs to compare models.
- `compare/` — sheets from `scripts/compare_models.py`: each hard photo as each
  model sees it.
- `goals/` — photos to try goal detectors on (`scripts/try_goals.py`).
- `goals-test/` — sheets from `scripts/try_goals.py`: each photo with each detector's boxes.
- `floor-test/` — results of `scripts/try_floor.py` (the first floor test).
- `journal.jsonl` — what the old vision-model agent saw (one JSON per line).
- `rules.txt` — left over from the old agent; empty.
