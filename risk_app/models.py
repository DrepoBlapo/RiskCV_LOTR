from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    color: str | None
    pin_type: int | None
    source_width: int
    source_height: int
    contact_point: tuple[float, float] | None = None
    map_point: tuple[float, float] | None = None
    territory_id: str | None = None
    inference_pass: str = "full_frame"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("bbox_xyxy", "contact_point", "map_point"):
            if data.get(key) is not None:
                data[key] = list(data[key])
        return data


@dataclass(slots=True)
class TerritoryStatus:
    territory_id: str
    color: str | None = None
    changed: bool = False
    conflict: bool = False
    detection_count: int = 0
    confidence_sum: float = 0.0
    color_scores: dict[str, float] = field(default_factory=dict)
    pin_type_counts: dict[str, int] = field(default_factory=dict)
    color_evidence: dict[str, dict[str, float]] = field(default_factory=dict)
    resolution_strategy: str = "confidence_sum"
    resolution_reason: str = ""
    marks_by_color: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PhaseState:
    phase_number: int
    territories: dict[str, TerritoryStatus]
    unassigned_detections: int = 0
    marks_by_color: dict[str, int] = field(
        default_factory=dict
    )
    controlled_regions: dict[str, list[str]] = field(
        default_factory=dict
    )

    @property
    def occupied(self) -> dict[str, TerritoryStatus]:
        return {
            territory_id: status
            for territory_id, status in self.territories.items()
            if status.color is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_number": self.phase_number,
            "unassigned_detections": (
                self.unassigned_detections
            ),
            "marks_by_color": dict(
                self.marks_by_color
            ),
            "controlled_regions": {
                color: list(region_ids)
                for color, region_ids
                in self.controlled_regions.items()
            },
            "territories": {
                territory_id: status.to_dict()
                for territory_id, status
                in self.territories.items()
            },
        }
