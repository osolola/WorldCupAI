import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


def load_config(path=CONFIG_PATH):
    with open(path, "rb") as f:
        return tomllib.load(f)


CONFIG = load_config()
