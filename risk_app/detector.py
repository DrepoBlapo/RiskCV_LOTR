from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .class_parser import parse_class_name
from .models import Detection


class YoloDetector:
    """Carga diferida del modelo: la interfaz abre sin ocupar la GPU."""

    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.3,
        iou: float = 0.55,
        imgsz: int = 1280,
        device: str = "0",
        half: bool = True,
        max_det: int = 300,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.max_det = max_det
        self._model: Any = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Falta ultralytics. Instálalo con: pip install ultralytics"
                ) from exc
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        inference_pass: str = "full_frame",
        offset_xy: tuple[int, int] = (0, 0),
        confidence: float | None = None,
    ) -> list[Detection]:
        model = self._ensure_loaded()
        height, width = frame_bgr.shape[:2]
        results = model.predict(
            source=frame_bgr,
            conf=self.confidence if confidence is None else confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(names[class_id] if isinstance(names, dict) else names[class_id])
            confidence = float(box.conf[0].item())
            coords = box.xyxy[0].detach().cpu().tolist()
            offset_x, offset_y = offset_xy
            coords = [
                coords[0] + offset_x,
                coords[1] + offset_y,
                coords[2] + offset_x,
                coords[3] + offset_y,
            ]
            color, pin_type = parse_class_name(class_name)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=tuple(float(value) for value in coords),
                    color=color,
                    pin_type=pin_type,
                    source_width=width,
                    source_height=height,
                    inference_pass=inference_pass,
                )
            )
        return detections

    def predict_crop(
        self,
        frame_bgr: np.ndarray,
        crop_xyxy: tuple[int, int, int, int],
        *,
        inference_pass: str,
        confidence: float | None = None,
    ) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        x0, y0, x1, y1 = crop_xyxy
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            return []
        detections = self.predict(
            frame_bgr[y0:y1, x0:x1],
            inference_pass=inference_pass,
            offset_xy=(x0, y0),
            confidence=confidence,
        )
        for detection in detections:
            detection.source_width = width
            detection.source_height = height
        return detections
