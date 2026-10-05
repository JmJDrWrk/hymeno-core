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

## What it does now

Phase 1, *face the target*: it looks for a target on the floor (a square of
yellow electrical tape, where a charging station will go) and turns the robot
towards it in small steps until it is centred. It does not walk towards it yet.

Every cycle:

1. takes a photo from the robot's head camera,
2. asks a vision model (Qwen-VL through Ollama) where the target is,
3. if the target is off to one side, turns a little towards it, stops, and
   looks again; if it is centred or not in sight, waits.

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

3. Run it:

   ```bash
   python -m hymeno --dry-run   # looks and decides, but does not move the robot
   python -m hymeno             # moves the robot
   python -m hymeno --once      # one look, then exit
   ```

Each look is printed and appended to `data/journal.jsonl`.

## Roadmap

1. Face the target (now).
2. Walk up to it and stop, using the size of its box as distance.
3. Search for it by turning when it is not in sight.
4. Robot side: manual commands take priority, and a switch to allow the brain.
5. A fast colour tracker (OpenCV) between model calls; the model only confirms.
6. Memory: a journal of what it saw and did, and things taught by showing them.
7. Docker: the brain and the model server in one `docker compose`.

## License

MIT, see [LICENSE](LICENSE).
