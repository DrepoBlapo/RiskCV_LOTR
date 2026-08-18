from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class MapMarks:
    """Índice inmutable de las marcas impresas descritas por map_texture_marcas.json."""

    def __init__(self, by_territory: dict[str, Counter[str]]) -> None:
        self.by_territory = by_territory
        self.totals = Counter()
        for counts in by_territory.values():
            self.totals.update(counts)

    @classmethod
    def empty(cls) -> "MapMarks":
        return cls({})

    @classmethod
    def load(
        cls,
        path: Path,
        valid_territories: set[str],
        type_to_color: dict[str, str] | None = None,
    ) -> "MapMarks":
        root: Any = json.loads(path.read_text(encoding="utf-8"))
        marks = root.get("marks") if isinstance(root, dict) else None
        if not isinstance(marks, list):
            raise ValueError("map_texture_marcas.json debe contener la lista 'marks'.")
        aliases = {str(key): str(value) for key, value in (type_to_color or {}).items()}
        by_territory: dict[str, Counter[str]] = defaultdict(Counter)
        for index, mark in enumerate(marks, start=1):
            if not isinstance(mark, dict):
                raise ValueError(f"La marca {index} no es un objeto JSON.")
            territory_id = mark.get("territory_id")
            if territory_id is None:
                continue
            territory_id = str(territory_id)
            if territory_id not in valid_territories:
                raise ValueError(f"La marca {index} usa un territorio desconocido: {territory_id}")
            raw_color = mark.get("color", mark.get("mark_color"))
            mark_type = str(mark.get("mark_type", "sin_tipo"))
            color = str(raw_color) if raw_color is not None else aliases.get(mark_type, mark_type)
            by_territory[territory_id][color] += 1
        return cls(dict(by_territory))

    def for_territory(self, territory_id: str) -> dict[str, int]:
        return dict(self.by_territory.get(territory_id, Counter()))
