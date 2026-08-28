"""Minimal .env reader: no dependency, no os.environ mutation."""
import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def get_secret(name: str, env_path: str | Path = ".env") -> str:
    if name in os.environ and os.environ[name].strip():
        return os.environ[name].strip()
    env = load_env(env_path)
    if name in env and env[name]:
        return env[name]
    raise KeyError(f"{name} not set in environment or {env_path}")
