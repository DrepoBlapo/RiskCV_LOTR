from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_io import read_image


class TerritoryIndex:
    """Consulta rápida de territorio a partir de un mapa RGB de identificadores."""

    def __init__(
        self,
        id_map_bgr: np.ndarray,
        territory_id_colors_rgb: dict[str, list[int]],
    ) -> None:
        if id_map_bgr.ndim != 3 or id_map_bgr.shape[2] < 3:
            raise ValueError("territory_id_map.png debe ser una imagen RGB/BGR.")
        self.id_map = id_map_bgr[:, :, :3].copy()
        self.height, self.width = self.id_map.shape[:2]
        blue = self.id_map[:, :, 0].astype(np.uint32)
        green = self.id_map[:, :, 1].astype(np.uint32)
        red = self.id_map[:, :, 2].astype(np.uint32)
        self.code_map = blue | (green << 8) | (red << 16)
        self.code_to_territory: dict[int, str] = {}
        for territory_id, rgb in territory_id_colors_rgb.items():
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                raise ValueError(f"Color de territorio inválido para {territory_id}: {rgb}")
            red_value, green_value, blue_value = (int(channel) for channel in rgb)
            code = blue_value | (green_value << 8) | (red_value << 16)
            if code in self.code_to_territory:
                raise ValueError("Dos territorios comparten el mismo color identificador.")
            self.code_to_territory[code] = str(territory_id)
        self.territory_ids = sorted(self.code_to_territory.values())

    @classmethod
    def load(cls, id_map_path: Path, territories_path: Path) -> "TerritoryIndex":
        with gzip.open(territories_path, "rt", encoding="utf-8") as stream:
            root: dict[str, Any] = json.load(stream)
        colors = root.get("territory_id_colors_rgb")
        if not isinstance(colors, dict) or not colors:
            raise ValueError(
                "territories.json.gz no contiene territory_id_colors_rgb. "
                "Regenera los assets con prepare_svg.py actualizado."
            )
        return cls(read_image(id_map_path), colors)

    def territory_at(
        self,
        x: float,
        y: float,
        search_radius: int = 8,
    ) -> str | None:
        center_x = int(round(x))
        center_y = int(round(y))
        if center_x < 0 or center_y < 0 or center_x >= self.width or center_y >= self.height:
            return None
        code = int(self.code_map[center_y, center_x])
        direct = self.code_to_territory.get(code)
        if direct is not None:
            return direct
        if search_radius <= 0:
            return None
        x0 = max(0, center_x - search_radius)
        x1 = min(self.width, center_x + search_radius + 1)
        y0 = max(0, center_y - search_radius)
        y1 = min(self.height, center_y + search_radius + 1)
        region = self.code_map[y0:y1, x0:x1]
        codes, counts = np.unique(region, return_counts=True)
        candidates: list[tuple[int, str]] = []
        for candidate_code, count in zip(codes.tolist(), counts.tolist()):
            territory_id = self.code_to_territory.get(int(candidate_code))
            if territory_id is not None:
                candidates.append((int(count), territory_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]

    def mask_for(self, territory_id: str) -> np.ndarray:
        for code, candidate in self.code_to_territory.items():
            if candidate == territory_id:
                return self.code_map == code
        return np.zeros((self.height, self.width), dtype=bool)

    def boundary_mask(self) -> np.ndarray:
        valid = np.isin(
            self.code_map,
            np.asarray(list(self.code_to_territory), dtype=np.uint32),
        ).astype(np.uint8)
        return cv2.morphologyEx(valid, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

