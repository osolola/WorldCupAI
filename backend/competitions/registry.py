from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from ..config import CONFIG
from ..data_sources.csv_source import CsvMatchResultSource

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Competition:
    key: str
    name: str
    format: str  # "knockout" | "league" (only knockout is implemented so far)
    data_source: object  # MatchResultSource
    elo_k_overrides: Dict[str, float] = field(default_factory=dict)


def _build_competition(key: str, cfg: dict) -> Competition:
    return Competition(
        key=key,
        name=cfg["name"],
        format=cfg["format"],
        data_source=CsvMatchResultSource(REPO_ROOT / cfg["results_csv"]),
        elo_k_overrides=cfg.get("elo_k_overrides", {}),
    )


COMPETITIONS = {
    key: _build_competition(key, cfg)
    for key, cfg in CONFIG["competitions"].items()
}


def get_competition(key: str) -> Competition:
    return COMPETITIONS[key]
