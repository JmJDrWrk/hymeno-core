"""Settings: config.yaml over the defaults below (see config.example.yaml)."""

import copy

import yaml

DEFAULTS = {
    "body": {"url": "http://robot.local"},
    "head": {"url": "http://robot-head.local", "mirrored": False},
    "model": {
        "url": "http://localhost:11434",
        "name": "qwen2.5vl:7b",
        "timeout_s": 60,
        "keep_alive": "15m",
        "max_answer_tokens": 100,
    },
    "target": {"label": "dock", "prompt": ""},
    "behavior": {
        "center_tolerance": 0.12,
        "turn_speed": 0.5,
        "turn_ms": 500,
        "settle_ms": 400,
        "idle_s": 1.0,
        "max_speed": 0.6,
    },
    "data_dir": "data",
}


def _merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load(path):
    with open(path) as f:
        settings = _merge(copy.deepcopy(DEFAULTS), yaml.safe_load(f))
    if not settings["target"]["prompt"].strip():
        raise ValueError("target.prompt is empty in %s" % path)
    return settings
