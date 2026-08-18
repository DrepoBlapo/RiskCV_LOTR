from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

from risk_app.calibration import MapCalibration
from risk_app.class_parser import parse_class_name
from risk_app.map_renderer import TerritoryMapRenderer
from risk_app.metrics import MetricContext, MetricEngine
from risk_app.models import Detection
from risk_app.marks import MapMarks
from risk_app.state import assign_detections, build_phase_state
from risk_app.storage import PhaseStore
from risk_app.territory import TerritoryIndex


def detection(x: float, color: str, confidence: float = 0.9) -> Detection:
    return Detection(
        class_id=0,
        class_name=f"{color}_1",
        confidence=confidence,
        bbox_xyxy=(x - 2, 40, x + 2, 50),
        color=color,
        pin_type=1,
        source_width=100,
        source_height=100,
    )


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :49] = (3, 2, 1)
        image[:, 51:] = (6, 5, 4)
        self.index = TerritoryIndex(
            image,
            {"oeste": [1, 2, 3], "este": [4, 5, 6]},
        )
        self.calibration = MapCalibration(
            100,
            100,
            100,
            100,
            [(0, 0), (99, 0), (99, 99), (0, 99)],
        )

    def test_class_names(self) -> None:
        self.assertEqual(parse_class_name("n0"), ("negro", 0))
        self.assertEqual(parse_class_name("rojo_tipo_2"), ("rojo", 2))
        self.assertEqual(parse_class_name("type-3-green"), ("verde", 3))

    def test_calibration_and_nearest_territory(self) -> None:
        point = self.calibration.transform_point((25, 50))
        self.assertAlmostEqual(point[0], 25, places=3)
        self.assertEqual(self.index.territory_at(*point), "oeste")
        self.assertIn(self.index.territory_at(50, 50, search_radius=3), {"este", "oeste"})

    def test_assignment_and_conflict_resolution(self) -> None:
        items = [detection(25, "rojo", 0.9), detection(25, "negro", 0.2)]
        assign_detections(items, self.calibration, self.index)
        state = build_phase_state(1, items, self.index.territory_ids)
        self.assertEqual(state.territories["oeste"].color, "rojo")
        self.assertTrue(state.territories["oeste"].conflict)
        self.assertFalse(state.territories["oeste"].changed)
        next_state = build_phase_state(2, [detection(25, "negro")], self.index.territory_ids, state)
        self.assertTrue(next_state.territories["oeste"].changed)

    def test_max_detection_beats_confidence_sum(self) -> None:
        items = [
            detection(25, "amarillo", 0.90),
            detection(25, "verde", 0.55),
            detection(25, "verde", 0.50),
        ]
        assign_detections(items, self.calibration, self.index)
        by_sum = build_phase_state(
            1, items, self.index.territory_ids,
            resolution={"strategy": "confidence_sum"},
        )
        by_max = build_phase_state(
            1, items, self.index.territory_ids,
            resolution={"strategy": "max_detection"},
        )
        self.assertEqual(by_sum.territories["oeste"].color, "verde")
        self.assertEqual(by_max.territories["oeste"].color, "amarillo")

    def test_marks_are_saved_per_color_and_territory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map_texture_marcas.json"
            path.write_text(json.dumps({"marks": [
                {"territory_id": "oeste", "mark_type": "marca_1"},
                {"territory_id": "oeste", "mark_type": "marca_1"},
                {"territory_id": "este", "mark_type": "marca_2"},
            ]}), encoding="utf-8")
            marks = MapMarks.load(
                path, {"oeste", "este"},
                {"marca_1": "amarillo", "marca_2": "verde"},
            )
            state = build_phase_state(1, [], self.index.territory_ids, marks=marks)
            self.assertEqual(state.marks_by_color, {"amarillo": 2, "verde": 1})
            self.assertEqual(state.territories["oeste"].marks_by_color["amarillo"], 2)

    def test_metrics_and_formula(self) -> None:
        items = [detection(25, "rojo")]
        assign_detections(items, self.calibration, self.index)
        state = build_phase_state(1, items, self.index.territory_ids)
        engine = MetricEngine(
            [
                {"key": "territories"},
                {"key": "pins"},
                {"key": "double", "formula": "territories * 2 + pins"},
            ],
            ["rojo"],
        )
        result = engine.calculate(MetricContext(state, None, items, []))
        self.assertEqual(result["rojo"]["double"], 3)

    def test_changed_map_has_visible_border(self) -> None:
        items = [detection(25, "negro")]
        assign_detections(items, self.calibration, self.index)
        first = build_phase_state(1, [], self.index.territory_ids)
        state = build_phase_state(2, items, self.index.territory_ids, first)
        renderer = TerritoryMapRenderer(self.index, {"negro": "#151413"})
        rendered = renderer.render(state)
        self.assertGreater(int(rendered[:, 48:52].max()), 100)

    def test_phase_store_keeps_raw_data(self) -> None:
        items = [detection(25, "rojo")]
        assign_detections(items, self.calibration, self.index)
        state = build_phase_state(1, items, self.index.territory_ids)
        frame = np.zeros((100, 100, 3), np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "best.pt"
            config = root / "config.json"
            model.write_bytes(b"test")
            config.write_text("{}", encoding="utf-8")
            store = PhaseStore(root / "results", model, config)
            phase_json = store.save_phase(
                1,
                frame,
                frame,
                frame,
                items,
                state,
                {"rojo": {"territories": 1.0}},
                self.calibration,
            )
            text = phase_json.read_text(encoding="utf-8")
            self.assertIn('"territory_id": "oeste"', text)
            self.assertTrue((phase_json.parent / "frame_raw.png").is_file())


if __name__ == "__main__":
    unittest.main()
