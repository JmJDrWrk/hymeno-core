# hymeno-core

A remote brain for a small robot. It looks through the robot's camera, asks a
vision model what is there, and steers the robot through its control API, one
short step at a time.

The robot itself stays simple: it walks, streams video and obeys. The thinking
happens here, on whatever machine has a GPU nearby. Some day a processor small
and efficient enough to run these models will fit inside the robot, and this
same program can move in with it without changes, because it only ever talks to
the robot through the same two interfaces: video and control.

## Why the name

*Hymenoepimecis argyraphaga* is a small wasp from Costa Rica whose larva lives
attached to the back of an orb-weaver spider, *Leucauge argyra*. For weeks the
spider goes about its life as usual. Then, on the night before the larva kills
it, the spider stops weaving its normal web and builds a completely different,
sturdy structure: exactly the shelter the larva will need to pupate. The larva
changes the spider's behaviour from the outside, by chemical signals, without
being the spider.

This project does the same to a spider robot: a separate mind, attached from
outside, deciding what the body does. Hence *hymeno*, and *core* because it is
the decision-making core of that mind.

## What it does

You give it a goal in plain words, in any language: "find your charging
station: a square of yellow tape on the floor", "look for a person and wave
hello". It then works on it step by step, like a simple animal:

1. **Observe**: one photo from the robot's camera.
2. **Interpret it with the goal and its working memory**: what it sees, whether
   the goal is in view (left, centre, right) and how far, whether the way
   ahead is clear, given what it has already done (the recent steps, how far
   it has turned and walked).
3. **Decide one short action**: turn left or right, walk forward, wave, or
   finish (done or give up).
4. **Act, remember, repeat.**

All of 1-3 is a single question to the vision model with the photo. The body's
limits stay in code: only those few short actions exist, it never walks
forward when the way is blocked, and it stops after 40 steps or 5 minutes.

### Fast eyes

The vision model understands anything but takes a second or two per look. For
the 80 everyday things of the COCO set (person, cat, dog, chair, bottle, ...)
an optional object detector, YOLO, finds them in a few milliseconds on a GPU.
Whenever a target names one of them, `search`, `face` and `reach` use the
detector instead and keep the robot moving while they check photos many times
a second; a goal like "go to the cat" skips the step-by-step loop entirely,
and the vision model only checks the result. Everything else (a square of
yellow tape, "the red box") still goes through the vision model.

```bash
pip install -r requirements-yolo.txt    # with a CUDA GPU, a few GB
```

`/detect` at the prompt shows what the detector sees and how long it takes.
To switch it off: `detector: false` in `config.yaml`.

The individual verbs are also available, to try one directly (`/verb`) or to
plan a goal once into verbs (`/plan`).

The verbs are generic; what to look for is always an argument in words, never
a setting:

| Verb | What it does |
|------|--------------|
| `describe(question)` | Answer a question about what the camera sees ("what do you see?"). |
| `look(target)` | Is it in sight, and where? |
| `turn(direction, amount)` | Turn on the spot, left or right, small/medium/large. |
| `forward(amount)` | Walk forward a little. |
| `stop()` | Stop. |
| `greet()` | Wave hello. |
| `search(target, only_turning, turn_direction, turn_step)` | Turn (and maybe step forward) until it is in sight. |
| `face(target)` | Turn until it is straight ahead. |
| `approach(target)` | Walk to it in short steps, looking between them, and stop when it is close. |
| `reach(target)` | Walk to it without stopping: the eyes keep looking in the background and each new look corrects the heading. Stops when close, when the way ahead is blocked, or when the eyes fall behind. |

Settings always go by name (`turn_step=large`); everything else is the
description of the target, so a word like "large" in "a large fan" is never
taken for a setting.

The model decides *what* to do; the verbs decide *how*, with plain geometry
(how far off centre the target is, how big it looks). A plan that uses
anything else is rejected.

## How it fits together

```
                ┌─────────────────── hymeno-core (Python) ───────────────────┐
                │  look → think → act, one short step per cycle               │
 head camera ◀──┤  GET  /jpg ........................... one photo            │
 model server ◀─┤  POST /api/generate (Ollama) ......... where is the target? │
 robot body ◀───┤  POST /api/v1/walk, /keepalive, /stop  short moves          │
                │  data/ ............................... journal of each look │
                └─────────────────────────────────────────────────────────────┘
 a browser ───── the robot's own control pages: manual driving, always available
```

Everything is reached by URL from `config.yaml`, so each piece can live on a
different machine: the brain on a Raspberry Pi, the model server on a PC with a
GPU, and so on.

### What the robot has to offer

- Camera: `GET /jpg` returns one JPEG.
- Body:
  - `POST /api/v1/walk` with `vx`, `vy`, `w` (forward, left, turn left; -1..1).
    The robot keeps walking for one second; `POST /api/v1/keepalive` extends it.
  - `POST /api/v1/stop`.
  - `GET /api/v1/state` (any JSON; used to check the robot is there).

This is the API of the robot it was written for, a small 8-servo quadruped
(ESP8266 body, ESP32-S3 camera head), but any robot with equivalent endpoints
will do.

## Safety

- Moves are short bursts. If this program stops, hangs or loses the network,
  the robot stops by itself within a second.
- Ctrl+C sends a stop before exiting.
- Speeds are capped in `config.yaml`.
- The robot's own control pages keep working at all times. (Making manual
  commands take priority over this program is a planned robot-side feature.)

## Setup

1. A model server: [Ollama](https://ollama.com) with a vision model.
   `scripts/setup-ollama.sh` installs it on Linux (including WSL) and pulls the
   model. Qwen2.5-VL 7B needs about 6 GB of GPU memory and answers in 1-2 s on
   an RTX 2060.
2. Python 3.10 or newer:

   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   pip install -r requirements.txt
   cp config.example.yaml config.yaml   # then edit the addresses
   ```

3. Run it and give it orders:

   ```
   $ python -m hymeno
   hymeno> busca tu estación de carga: un cuadrado de cinta amarilla en el suelo
      1. a sofa and a rug | goal: no (unknown) | ahead: clear -> turn_left medium
         "Giro para buscar la estación."
      2. yellow tape on the floor | goal: right (far) | ahead: clear -> turn_right small
      3. yellow tape on the floor | goal: centre (far) | ahead: clear -> forward medium
      ...
   done: He llegado a la estación.
   hymeno> /look a yellow square on the floor     # one verb, without the planner
   hymeno> /search a large fan turn_direction=right turn_step=large
   hymeno> /help
   ```

   `python -m hymeno --dry-run` looks and plans but never moves the robot;
   `python -m hymeno "an order"` runs one order and exits. Ctrl+C stops the
   robot and the current order (and exits at the prompt).

Every look and move is appended to `data/journal.jsonl`.

## Roadmap

1. Orders in plain words, planned into a few generic verbs (now).
2. Robot side: manual commands take priority, and a switch to allow the brain.
3. Measure the body: how many turning steps make a full turn, how big things
   look at a known distance; remembered, not configured.
4. Orders from the robot's control page or a chat, and progress shown there.
5. A fast tracker (OpenCV colour, then YOLO) between model calls; the model
   confirms what it is.
6. Memory: things taught by showing them, where things were last seen.
7. Docker: the brain and the model server in one `docker compose`.

## License

MIT, see [LICENSE](LICENSE).
