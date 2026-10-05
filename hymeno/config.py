"""Settings: config.yaml over the defaults below (see config.example.yaml).
Only where things are and the safety limits; what to do comes as orders."""

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
    "max_speed": 0.6,
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
        return _merge(copy.deepcopy(DEFAULTS), yaml.safe_load(f))
