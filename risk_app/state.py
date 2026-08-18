from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .calibration import MapCalibration
from .models import Detection, PhaseState, TerritoryStatus
from .territory import TerritoryIndex
from .marks import MapMarks


def assign_detections(
    detections: Iterable[Detection],
    calibration: MapCalibration,
    territories: TerritoryIndex,
    search_radius: int = 8,
    contact_y_fraction: float = 0.92,
) -> list[Detection]:
    assigned: list[Detection] = []
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_xyxy
        point = ((x0 + x1) * 0.5, y0 + (y1 - y0) * contact_y_fraction)
        map_point = calibration.transform_point(point)
        detection.contact_point = point
        detection.map_point = map_point
        detection.territory_id = territories.territory_at(
            map_point[0], map_point[1], search_radius=search_radius
        )
        assigned.append(detection)
    return assigned


def build_phase_state(
    phase_number: int,
    detections: Iterable[Detection],
    territory_ids: Iterable[str],
    previous: PhaseState | None = None,
    color_order: Iterable[str] = ("negro", "rojo", "amarillo", "verde"),
    resolution: dict | None = None,
    marks: MapMarks | None = None,
) -> PhaseState:
    detections = list(detections)
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pin_types: dict[str, Counter[str]] = defaultdict(Counter)
    unassigned = 0
    for detection in detections:
        if detection.territory_id is None or detection.color is None:
            unassigned += 1
            continue
        territory_id = detection.territory_id
        scores[territory_id][detection.color] += detection.confidence
        counts[territory_id][detection.color] += 1
        if detection.pin_type is not None:
            pin_types[territory_id][str(detection.pin_type)] += 1

    order = {color: index for index, color in enumerate(color_order)}
    resolution = resolution or {}
    strategy = str(resolution.get("strategy", "confidence_sum"))
    weights = resolution.get("weights", {})
    states: dict[str, TerritoryStatus] = {}
    for territory_id in territory_ids:
        territory_scores = dict(scores.get(territory_id, {}))
        color: str | None = None
        evidence: dict[str, dict[str, float]] = {}
        candidates = counts.get(territory_id, {})
        for candidate, count in candidates.items():
            confidences = [
                float(d.confidence) for d in detections
                if d.territory_id == territory_id and d.color == candidate
            ]
            total = float(sum(confidences))
            evidence[candidate] = {
                "count": float(count),
                "sum": total,
                "max": max(confidences, default=0.0),
                "mean": total / count if count else 0.0,
            }
        ranking_scores: dict[str, float] = {}
        for candidate, item in evidence.items():
            if strategy == "max_detection":
                ranking_scores[candidate] = item["max"]
            elif strategy == "evidence_weighted":
                ranking_scores[candidate] = (
                    float(weights.get("max", 1.0)) * item["max"]
                    + float(weights.get("mean", 0.0)) * item["mean"]
                    + float(weights.get("sum", 0.0)) * item["sum"]
                    + float(weights.get("count", 0.0)) * item["count"]
                )
            else:
                ranking_scores[candidate] = item["sum"]
        if ranking_scores:
            color = sorted(
                ranking_scores,
                key=lambda candidate: (
                    -ranking_scores[candidate],
                    order.get(candidate, 999),
                    candidate,
                ),
            )[0]
        previous_color = None
        if previous is not None and territory_id in previous.territories:
            previous_color = previous.territories[territory_id].color
        states[territory_id] = TerritoryStatus(
            territory_id=territory_id,
            color=color,
            changed=previous is not None and color != previous_color,
            conflict=len(territory_scores) > 1,
            detection_count=sum(counts.get(territory_id, {}).values()),
            confidence_sum=float(sum(territory_scores.values())),
            color_scores=territory_scores,
            pin_type_counts=dict(pin_types.get(territory_id, {})),
            color_evidence=evidence,
            resolution_strategy=strategy,
            resolution_reason=(
                f"{color} obtuvo {ranking_scores[color]:.4f} con {strategy}"
                if color is not None else "sin detecciones"
            ),
            marks_by_color=(marks.for_territory(territory_id) if marks else {}),
        )
    return PhaseState(
        phase_number,
        states,
        unassigned,
        dict(marks.totals) if marks else {},
    )
