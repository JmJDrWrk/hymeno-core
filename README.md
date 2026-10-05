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

You give it orders in plain words, in any language: "find the fan, only
turning", "face the yellow square and walk up to it". The model turns each
order into a short plan made only of the verbs the brain knows, and the brain
carries it out: looking, moving a little, looking again.

The verbs are generic; what to look for is always an argument in words, never
a setting:

| Verb | What it does |
|------|--------------|
| `look(target)` | Is it in sight, and where? |
| `turn(direction, amount)` | Turn on the spot, left or right, small/medium/large. |
| `forward(amount)` | Walk forward a little. |
| `stop()` | Stop. |
| `greet()` | Wave hello. |
| `search(target, only_turning, turn_direction, turn_step)` | Turn (and maybe step forward) until it is in sight. |
| `face(target)` | Turn until it is straight ahead. |
| `approach(target)` | Walk to it, keeping it centred, and stop when it is close. |

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
   hymeno> busca el ventilador girando solo
     -> search(target='an electric fan', only_turning=True)
        done: found an electric fan at the right (18% of the width)
     -> face(target='an electric fan')
        done: facing an electric fan
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
