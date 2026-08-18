from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import MapCalibration
from .image_io import write_image
from .models import Detection, PhaseState


class PhaseStore:
    def __init__(self, output_root: Path, model_path: Path, config_path: Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_dir = output_root / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.model_path = model_path
        self.config_path = config_path
        self.phase_records: list[dict[str, Any]] = []
        self._write_session()

    def _write_session(self) -> None:
        data = {
            "version": 1,
            "started_at": self.session_dir.name.removeprefix("session_"),
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "phases": self.phase_records,
        }
        (self.session_dir / "session.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_phase(
        self,
        phase_number: int,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        rendered_map: np.ndarray,
        detections: list[Detection],
        state: PhaseState,
        metrics: dict[str, dict[str, float]],
        calibration: MapCalibration,
    ) -> Path:
        phase_dir = self.session_dir / f"phase_{phase_number:04d}"
        phase_dir.mkdir(parents=True, exist_ok=False)
        raw_path = phase_dir / "frame_raw.png"
        annotated_path = phase_dir / "frame_annotated.png"
        map_path = phase_dir / "territory_map.png"
        write_image(raw_path, raw_frame)
        write_image(annotated_path, annotated_frame)
        write_image(map_path, rendered_map)
        timestamp = datetime.now(timezone.utc).isoformat()
        data = {
            "version": 1,
            "phase_number": phase_number,
            "captured_at_utc": timestamp,
            "model_path": str(self.model_path),
            "images": {
                "raw": raw_path.name,
                "annotated": annotated_path.name,
                "territory_map": map_path.name,
            },
            "calibration": {
                "frame_size": [calibration.frame_width, calibration.frame_height],
                "map_size": [calibration.map_width, calibration.map_height],
                "corners": [list(point) for point in calibration.corners],
            },
            "detections": [detection.to_dict() for detection in detections],
            "state": state.to_dict(),
            "metrics": metrics,
        }
        json_path = phase_dir / "phase.json"
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self.phase_records.append(
            {
                "phase_number": phase_number,
                "captured_at_utc": timestamp,
                "directory": phase_dir.name,
                "occupied_territories": len(state.occupied),
                "detections": len(detections),
            }
        )
        self._write_session()
        return json_path
