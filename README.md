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

It is being rebuilt from the basics, one tested stage at a time:

1. **Move and see** (now): check the robot and its camera answer
   (`python -m hymeno check`), and find where the floor is free with a
   segmentation model, SegFormer trained on ADE20K
   (`scripts/try_floor.py` paints it green on photos in `house_img/`).
2. **Explore**: walk wherever the floor is free, deciding again on every
   photo, so a slipping leg is corrected by the next look.
3. **Remember** where it has been.
4. **Goals**: find things (a vision model through Ollama), such as its
   charging station, a rectangle of yellow tape on the floor.

## How it fits together

```
                ┌─────────────────── hymeno-core (Python) ───────────────────┐
                │  look → decide → act, deciding again on every photo         │
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

3. Check the robot and its camera answer:

   ```bash
   python -m hymeno check
   ```

Every look and move is appended to `data/journal.jsonl`.

## Roadmap

See the stages in [What it does](#what-it-does). Later: manual commands from
the robot's control page take priority over this program, and Docker for the
brain and the model server together.

## License

MIT, see [LICENSE](LICENSE).
