from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore

from .calibration import MapCalibration
from .detector import YoloDetector
from .map_renderer import TerritoryMapRenderer
from .metrics import MetricContext, MetricEngine
from .models import Detection, PhaseState
from .state import assign_detections, build_phase_state
from .territory import TerritoryIndex
from .marks import MapMarks
from .regions import RegionIndex


@dataclass(slots=True)
class AnalysisResult:
    frame: np.ndarray
    calibration: MapCalibration
    detections: list[Detection]
    state: PhaseState
    metrics: dict[str, dict[str, float]]
    rendered_map: np.ndarray


class AnalysisThread(QtCore.QThread):
    analysis_ready = QtCore.Signal(object)
    analysis_error = QtCore.Signal(str)

    def __init__(
        self,
        detector: YoloDetector,
        frame: np.ndarray,
        calibration: MapCalibration,
        territory_index: TerritoryIndex,
        map_renderer: TerritoryMapRenderer,
        metric_engine: MetricEngine,
        previous_state: PhaseState | None,
        history: list[dict],
        phase_number: int,
        search_radius: int,
        contact_y_fraction: float,
        color_order: list[str],
        conflict_resolution: dict | None = None,
        marks: MapMarks | None = None,
        regions: RegionIndex | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.detector = detector
        self.frame = frame
        self.calibration = calibration
        self.territory_index = territory_index
        self.map_renderer = map_renderer
        self.metric_engine = metric_engine
        self.previous_state = previous_state
        self.history = history
        self.phase_number = phase_number
        self.search_radius = search_radius
        self.contact_y_fraction = contact_y_fraction
        self.color_order = color_order
        self.conflict_resolution = conflict_resolution or {}
        self.marks = marks or MapMarks.empty()
        self.regions = (
            regions
            if regions is not None
            else RegionIndex.empty()
        )

    def _refine_conflicts(
        self, detections: list[Detection], state: PhaseState
    ) -> list[Detection]:
        zoom = self.conflict_resolution.get("zoom", {})
        if not bool(zoom.get("enabled", False)):
            return detections
        conflict_ids = {
            territory_id for territory_id, status in state.territories.items()
            if status.conflict
        }
        if not conflict_ids:
            return detections
        padding = float(zoom.get("padding_fraction", 0.8))
        refined: list[Detection] = []
        for territory_id in sorted(conflict_ids):
            originals = [d for d in detections if d.territory_id == territory_id]
            if not originals:
                continue
            x0 = min(d.bbox_xyxy[0] for d in originals)
            y0 = min(d.bbox_xyxy[1] for d in originals)
            x1 = max(d.bbox_xyxy[2] for d in originals)
            y1 = max(d.bbox_xyxy[3] for d in originals)
            pad = max(x1 - x0, y1 - y0) * padding
            crop = (
                int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad)
            )
            crop_detections = self.detector.predict_crop(
                self.frame,
                crop,
                inference_pass=f"zoom:{territory_id}",
                confidence=float(zoom.get("confidence", self.detector.confidence)),
            )
            assigned = assign_detections(
                crop_detections, self.calibration, self.territory_index,
                search_radius=self.search_radius,
                contact_y_fraction=self.contact_y_fraction,
            )
            refined.extend(d for d in assigned if d.territory_id == territory_id)
        if not refined:
            return detections
        refined_ids = {d.territory_id for d in refined}
        mode = str(zoom.get("merge", "replace"))
        if mode == "append":
            return [*detections, *refined]
        return [d for d in detections if d.territory_id not in refined_ids] + refined

    def run(self) -> None:
        try:
            detections = self.detector.predict(self.frame)
            assigned = assign_detections(
                detections,
                self.calibration,
                self.territory_index,
                search_radius=self.search_radius,
                contact_y_fraction=self.contact_y_fraction,
            )
            state = build_phase_state(
                self.phase_number,
                assigned,
                self.territory_index.territory_ids,
                previous=self.previous_state,
                color_order=self.color_order,
                resolution=self.conflict_resolution,
                marks=self.marks,
            )
            state.controlled_regions = {
                color: [
                    region.region_id
                    for region in self.regions.controlled_by(
                        state,
                        color,
                    )
                ]
                for color in self.color_order
            }
            assigned = self._refine_conflicts(assigned, state)
            state = build_phase_state(
                self.phase_number,
                assigned,
                self.territory_index.territory_ids,
                previous=self.previous_state,
                color_order=self.color_order,
                resolution=self.conflict_resolution,
                marks=self.marks,
            )
            context = MetricContext(
                state=state,
                previous_state=self.previous_state,
                detections=assigned,
                history=self.history,
                regions=self.regions,
            )
            metrics = self.metric_engine.calculate(context)
            rendered_map = self.map_renderer.render(state)
            self.analysis_ready.emit(
                AnalysisResult(
                    self.frame,
                    self.calibration,
                    assigned,
                    state,
                    metrics,
                    rendered_map,
                )
            )
        except Exception as exc:  # La excepción cruza el hilo como texto.
            self.analysis_error.emit(f"{type(exc).__name__}: {exc}")
