from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class MapCalibration:
    """Homografía entre el fotograma y el mapa canónico de territorios."""

    frame_width: int
    frame_height: int
    map_width: int
    map_height: int
    corners: list[tuple[float, float]]

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError("La calibración necesita cuatro esquinas.")
        if min(self.frame_width, self.frame_height, self.map_width, self.map_height) <= 1:
            raise ValueError("Las dimensiones de calibración no son válidas.")
        polygon = np.asarray(self.corners, dtype=np.float32)
        area = abs(float(cv2.contourArea(polygon)))
        if area < self.frame_width * self.frame_height * 0.01:
            raise ValueError("Las cuatro esquinas forman un área demasiado pequeña.")

    @property
    def matrix(self) -> np.ndarray:
        source = np.asarray(self.corners, dtype=np.float32)
        destination = np.asarray(
            [
                (0.0, 0.0),
                (self.map_width - 1.0, 0.0),
                (self.map_width - 1.0, self.map_height - 1.0),
                (0.0, self.map_height - 1.0),
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(source, destination)

    def transform_point(self, point: tuple[float, float]) -> tuple[float, float]:
        source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, self.matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def corners_for_frame(self, width: int, height: int) -> list[tuple[float, float]]:
        scale_x = width / self.frame_width
        scale_y = height / self.frame_height
        return [(x * scale_x, y * scale_y) for x, y in self.corners]

    def scaled_to_frame(self, width: int, height: int) -> "MapCalibration":
        return MapCalibration(
            width,
            height,
            self.map_width,
            self.map_height,
            self.corners_for_frame(width, height),
        )

    def save(self, path: Path) -> None:
        data = {
            "version": 1,
            "frame_size": [self.frame_width, self.frame_height],
            "map_size": [self.map_width, self.map_height],
            "corner_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "corners": [list(point) for point in self.corners],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "MapCalibration":
        data = json.loads(path.read_text(encoding="utf-8"))
        frame_size = data["frame_size"]
        map_size = data["map_size"]
        corners = [tuple(map(float, point)) for point in data["corners"]]
        return cls(
            int(frame_size[0]),
            int(frame_size[1]),
            int(map_size[0]),
            int(map_size[1]),
            corners,
        )

    @classmethod
    def from_clicks(
        cls,
        frame_size: tuple[int, int],
        map_size: tuple[int, int],
        corners: Iterable[tuple[float, float]],
    ) -> "MapCalibration":
        return cls(
            frame_size[0],
            frame_size[1],
            map_size[0],
            map_size[1],
            list(corners),
        )

