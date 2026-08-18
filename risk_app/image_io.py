from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"No se pudo abrir la imagen: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"OpenCV no pudo codificar la imagen: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))

