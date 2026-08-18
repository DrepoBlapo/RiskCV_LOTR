from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..calibration import MapCalibration
from ..camera import CameraThread
from ..config import AppConfig
from ..detector import YoloDetector
from ..drawing import draw_detections
from ..image_io import read_image
from ..map_renderer import TerritoryMapRenderer, hex_to_bgr
from ..marks import MapMarks
from ..regions import RegionIndex
from ..metrics import MetricEngine
from ..models import Detection, PhaseState
from ..storage import PhaseStore
from ..territory import TerritoryIndex
from ..workers import AnalysisResult, AnalysisThread
from .image_widget import ImageWidget


CORNER_NAMES = ["superior izquierda", "superior derecha", "inferior derecha", "inferior izquierda"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.colors = config.colors
        self.color_order = config.color_order
        self.current_frame: np.ndarray | None = None
        self.last_detections: list[Detection] = []
        self.previous_state: PhaseState | None = None
        self.history: list[dict[str, Any]] = []
        self.phase_number = 0
        self.calibration: MapCalibration | None = None
        self.calibration_points: list[tuple[float, float]] = []
        self.calibrating = False
        self.analysis_thread: AnalysisThread | None = None
        self.camera_thread: CameraThread | None = None

        self.territories = TerritoryIndex.load(
            config.territory_id_map_path, config.territories_path
        )
        marks_path = config.map_marks_path
        marks_cfg = config.section("marks")
        self.map_marks = (
            MapMarks.load(
                marks_path,
                set(self.territories.territory_ids),
                marks_cfg.get("type_to_color", {}),
            )
            if marks_path is not None else MapMarks.empty()
        )

        regions_path = config.regions_path

        self.regions = (
            RegionIndex.load(
                regions_path,
                set(self.territories.territory_ids),
            )
            if regions_path is not None
            else RegionIndex.empty()
        )
        
        texture_path = config.map_texture_path
        texture = read_image(texture_path) if texture_path is not None else None

        orography_path = config.map_orography_overlay_path
        orography_overlay = (
            cv2.imread(str(orography_path), cv2.IMREAD_UNCHANGED)
            if orography_path is not None
            else None
        )

        if orography_path is not None and orography_overlay is None:
            raise RuntimeError(
                f"No se pudo cargar el PNG de orografía: {orography_path}"
            )

        display_cfg = config.section("display")

        orography_opacity = float(
            display_cfg.get("orography_opacity", 1.0)
        )

        self.map_renderer = TerritoryMapRenderer(
            self.territories,
            self.colors,
            texture,
            orography_overlay=orography_overlay,
            orography_opacity=float(
                display_cfg.get(
                    "orography_opacity",
                    1.0,
                )
            ),
            regions=self.regions,
            stripe_config=display_cfg.get(
                "region_stripes",
                {},
            ),
        )
        
        model_cfg = config.section("model")
        self.detector = YoloDetector(
            config.model_path,
            confidence=float(model_cfg.get("confidence", 0.3)),
            iou=float(model_cfg.get("iou", 0.55)),
            imgsz=int(model_cfg.get("imgsz", 1280)),
            device=str(model_cfg.get("device", "0")),
            half=bool(model_cfg.get("half", True)),
            max_det=int(model_cfg.get("max_det", 300)),
        )
        table_cfg = config.section("table")
        columns = table_cfg.get("columns", [{"key": "territories", "title": "Territorios"}])
        if not isinstance(columns, list) or not columns:
            raise ValueError("table.columns debe contener al menos una columna.")
        self.table_columns: list[dict[str, Any]] = columns
        self.metric_engine = MetricEngine(self.table_columns, self.color_order)
        self.store = PhaseStore(config.output_dir, config.model_path, config.path)

        self._build_ui()
        self._load_calibration()
        self._show_empty_map()
        self._start_camera()

    def _build_ui(self) -> None:
        self.setWindowTitle(
            "Risk · Control de fases con YOLO"
        )
        self.resize(1600, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # =====================================================
        # BARRA SUPERIOR
        # =====================================================

        controls = QtWidgets.QHBoxLayout()

        self.camera_button = QtWidgets.QPushButton(
            "Reiniciar cámara"
        )
        self.camera_button.clicked.connect(
            self._restart_camera
        )
        controls.addWidget(self.camera_button)

        self.calibrate_button = QtWidgets.QPushButton(
            "Calibrar mapa"
        )
        self.calibrate_button.clicked.connect(
            self._toggle_calibration
        )
        controls.addWidget(self.calibrate_button)

        controls.addWidget(
            QtWidgets.QLabel("Confianza:")
        )

        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setSingleStep(0.01)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(
            self.detector.confidence
        )
        self.confidence_spin.valueChanged.connect(
            self._set_confidence
        )
        controls.addWidget(self.confidence_spin)

        self.boxes_checkbox = QtWidgets.QCheckBox(
            "Mostrar cajas de la última fase"
        )
        self.boxes_checkbox.setChecked(True)
        controls.addWidget(self.boxes_checkbox)

        controls.addStretch(1)

        self.phase_button = QtWidgets.QPushButton(
            "CAMBIO DE FASE"
        )
        self.phase_button.setObjectName("phaseButton")
        self.phase_button.clicked.connect(
            self._analyze_phase
        )
        controls.addWidget(self.phase_button)

        root.addLayout(controls)

        # =====================================================
        # CONTENIDO PRINCIPAL
        # =====================================================

        splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal
        )
        splitter.setChildrenCollapsible(False)

        # -----------------------------------------------------
        # IZQUIERDA: cámara y tabla
        # -----------------------------------------------------

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.live_title = QtWidgets.QLabel(
            "CÁMARA EN DIRECTO · SIN FASE ANALIZADA"
        )
        self.live_title.setObjectName("panelTitle")
        left_layout.addWidget(self.live_title)

        self.video_widget = ImageWidget(clickable=True)
        self.video_widget.image_clicked.connect(
            self._on_video_click
        )
        self.video_widget.setMinimumSize(640, 360)
        left_layout.addWidget(
            self.video_widget,
            stretch=1,
        )

        table_title = QtWidgets.QLabel(
            "ESTADO Y ESTADÍSTICAS"
        )
        table_title.setObjectName("panelTitle")
        left_layout.addWidget(table_title)

        self.metrics_table = QtWidgets.QTableWidget()
        self.metrics_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.metrics_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.metrics_table.verticalHeader().setVisible(False)

        left_layout.addWidget(
            self.metrics_table,
            stretch=0,
        )

        # -----------------------------------------------------
        # DERECHA: mapa
        # -----------------------------------------------------

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        map_title = QtWidgets.QLabel(
            "MAPA TERRITORIAL · CAMBIOS CON BORDE"
        )
        map_title.setObjectName("panelTitle")
        right_layout.addWidget(map_title)

        self.map_widget = ImageWidget(clickable=False)
        self.map_widget.setMinimumSize(600, 500)

        right_layout.addWidget(
            self.map_widget,
            stretch=1,
        )

        # Proporción inicial entre ambas columnas.
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 48)
        splitter.setStretchFactor(1, 52)
        splitter.setSizes([700, 760])

        root.addWidget(
            splitter,
            stretch=1,
        )

        # =====================================================
        # BARRA INFERIOR
        # =====================================================

        footer = QtWidgets.QHBoxLayout()

        self.status_label = QtWidgets.QLabel(
            "Iniciando…"
        )
        footer.addWidget(
            self.status_label,
            stretch=1,
        )

        open_button = QtWidgets.QPushButton(
            "Abrir resultados"
        )
        open_button.clicked.connect(
            self._open_results
        )
        footer.addWidget(open_button)

        root.addLayout(footer)

        # La tabla debe configurarse después de crearla.
        self._configure_table()

        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #181c22;
                color: #edf1f7;
                font-size: 13px;
            }

            QPushButton {
                background: #303743;
                border: 1px solid #4b5563;
                border-radius: 5px;
                padding: 8px 12px;
            }

            QPushButton:hover {
                background: #3d4654;
            }

            QPushButton:disabled {
                color: #7f8996;
                background: #252a31;
            }

            QPushButton#phaseButton {
                background: #b81f32;
                border-color: #dc3d50;
                font-weight: 700;
                padding: 10px 22px;
            }

            QPushButton#phaseButton:hover {
                background: #d1263c;
            }

            QLabel#panelTitle {
                font-weight: 700;
                color: #c8d0db;
                padding: 3px 0;
            }

            QTableWidget {
                background: #11151a;
                gridline-color: #343a45;
                border: 1px solid #343a45;
            }

            QHeaderView::section {
                background: #252b34;
                color: #edf1f7;
                padding: 7px;
                border: 0;
            }

            QDoubleSpinBox {
                background: #11151a;
                padding: 5px;
                border: 1px solid #4b5563;
            }

            QSplitter::handle {
                background: #343a45;
                width: 3px;
            }

            QSplitter::handle:hover {
                background: #596273;
            }
            """
        )

    def _configure_table(self) -> None:
        headers = ["Color"] + [str(column.get("title", column["key"])) for column in self.table_columns]
        self.metrics_table.setColumnCount(len(headers))
        self.metrics_table.setHorizontalHeaderLabels(headers)
        self.metrics_table.setRowCount(len(self.color_order))
        for row, color in enumerate(self.color_order):
            item = QtWidgets.QTableWidgetItem(color.capitalize())
            item.setBackground(QtGui.QColor(self.colors.get(color, "#777777")))
            item.setForeground(
                QtGui.QColor("white" if sum(hex_to_bgr(self.colors.get(color, "#777777"))) < 390 else "black")
            )
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.metrics_table.setItem(row, 0, item)
            for column in range(1, len(headers)):
                value_item = QtWidgets.QTableWidgetItem("0")
                value_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.metrics_table.setItem(row, column, value_item)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setFixedHeight(55 + len(self.color_order) * 34)

    def _load_calibration(self) -> None:
        path = self.config.calibration_path
        if not path.is_file():
            self.status_label.setText("Cámara iniciando. Falta calibrar las cuatro esquinas del mapa.")
            return
        try:
            calibration = MapCalibration.load(path)
            if (calibration.map_width, calibration.map_height) != (
                self.territories.width,
                self.territories.height,
            ):
                raise ValueError("La calibración pertenece a otro territory_id_map.png.")
            self.calibration = calibration
            self.status_label.setText("Calibración cargada. Esperando cámara.")
        except Exception as exc:
            self.status_label.setText(f"No se usará la calibración guardada: {exc}")

    def _show_empty_map(self) -> None:
        empty = PhaseState(
            0,
            {
                territory_id: self._empty_status(territory_id)
                for territory_id in self.territories.territory_ids
            },
        )
        self.map_widget.set_cv_image(self.map_renderer.render(empty))

    @staticmethod
    def _empty_status(territory_id: str):
        from ..models import TerritoryStatus

        return TerritoryStatus(territory_id)

    def _start_camera(self) -> None:
        self.camera_thread = CameraThread(self.config.section("camera"), self)
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.camera_opened.connect(self._on_camera_opened)
        self.camera_thread.camera_error.connect(self._on_camera_error)
        self.camera_thread.start()

    @QtCore.Slot(object)
    def _on_frame(self, frame: np.ndarray) -> None:
        self.current_frame = frame.copy()
        display = frame.copy()
        if self.boxes_checkbox.isChecked() and self.last_detections:
            display = draw_detections(display, self.last_detections, self.colors)
        self._draw_calibration(display)
        self.video_widget.set_cv_image(display)

    def _draw_calibration(self, frame: np.ndarray) -> None:
        points: list[tuple[float, float]] = []
        if self.calibrating:
            points = self.calibration_points
        elif self.calibration is not None:
            height, width = frame.shape[:2]
            points = self.calibration.corners_for_frame(width, height)
        int_points = [(int(round(x)), int(round(y))) for x, y in points]
        for index, point in enumerate(int_points):
            cv2.circle(frame, point, 7, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, str(index + 1), (point[0] + 9, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        if len(int_points) >= 2:
            cv2.polylines(frame, [np.asarray(int_points, np.int32)], False, (0, 215, 255), 2, cv2.LINE_AA)
        if len(int_points) == 4:
            cv2.line(frame, int_points[-1], int_points[0], (0, 215, 255), 2, cv2.LINE_AA)

    @QtCore.Slot(int, int)
    def _on_camera_opened(self, width: int, height: int) -> None:
        calibration_text = "calibración lista" if self.calibration else "calibración pendiente"
        self.status_label.setText(f"Cámara {width}×{height} abierta · {calibration_text}.")

    @QtCore.Slot(str)
    def _on_camera_error(self, message: str) -> None:
        self.status_label.setText(message)
        QtWidgets.QMessageBox.critical(self, "Error de cámara", message)

    def _restart_camera(self) -> None:
        if self.camera_thread is not None:
            self.camera_thread.stop()
        self.current_frame = None
        self._start_camera()

    def _toggle_calibration(self) -> None:
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "Sin cámara", "Todavía no hay un fotograma de cámara.")
            return
        if self.calibrating:
            self.calibrating = False
            self.calibration_points.clear()
            self.calibrate_button.setText("Calibrar mapa")
            self.status_label.setText("Calibración cancelada; se conserva la anterior.")
            return
        self.calibrating = True
        self.calibration_points.clear()
        self.calibrate_button.setText("Cancelar calibración")
        self.status_label.setText(f"Haz clic en la esquina {CORNER_NAMES[0]} del mapa.")

    @QtCore.Slot(float, float)
    def _on_video_click(self, x: float, y: float) -> None:
        if not self.calibrating or self.current_frame is None:
            return
        self.calibration_points.append((x, y))
        if len(self.calibration_points) < 4:
            self.status_label.setText(
                f"Punto {len(self.calibration_points)} guardado. Haz clic en la esquina {CORNER_NAMES[len(self.calibration_points)]}."
            )
            return
        height, width = self.current_frame.shape[:2]
        try:
            calibration = MapCalibration.from_clicks(
                (width, height),
                (self.territories.width, self.territories.height),
                self.calibration_points,
            )
            calibration.save(self.config.calibration_path)
            self.calibration = calibration
            self.last_detections = []
            self.calibrating = False
            self.calibration_points.clear()
            self.calibrate_button.setText("Recalibrar mapa")
            self.status_label.setText("Calibración guardada. Las cajas anteriores se han ocultado.")
        except Exception as exc:
            self.calibration_points.clear()
            self.status_label.setText(f"Calibración inválida: {exc}. Vuelve a marcar las cuatro esquinas.")

    def _calibration_for_current_frame(self) -> MapCalibration | None:
        if self.calibration is None or self.current_frame is None:
            return None
        height, width = self.current_frame.shape[:2]
        if (width, height) == (self.calibration.frame_width, self.calibration.frame_height):
            return self.calibration
        return self.calibration.scaled_to_frame(width, height)

    def _analyze_phase(self) -> None:
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            return
        calibration = self._calibration_for_current_frame()
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "Sin imagen", "La cámara aún no ha entregado una imagen.")
            return
        if calibration is None:
            QtWidgets.QMessageBox.warning(self, "Sin calibración", "Pulsa 'Calibrar mapa' y marca las cuatro esquinas.")
            return
        if self.calibrating:
            QtWidgets.QMessageBox.warning(self, "Calibración incompleta", "Termina o cancela la calibración antes de analizar.")
            return
        self.phase_button.setEnabled(False)
        self.phase_button.setText("ANALIZANDO…")
        self.camera_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.status_label.setText(
            "Analizando una captura; el vídeo sigue en directo con el último resultado."
        )
        calibration_cfg = self.config.section("calibration")
        self.analysis_thread = AnalysisThread(
            self.detector,
            self.current_frame.copy(),
            calibration,
            self.territories,
            self.map_renderer,
            self.metric_engine,
            self.previous_state,
            self.history.copy(),
            self.phase_number + 1,
            int(calibration_cfg.get("lookup_radius_px", 8)),
            float(calibration_cfg.get("contact_y_fraction", 0.92)),
            self.color_order,
            conflict_resolution=self.config.section(
                "conflict_resolution"
            ),
            marks=self.map_marks,
            regions=self.regions,
            parent=self,
        )
        self.analysis_thread.analysis_ready.connect(self._on_analysis_ready)
        self.analysis_thread.analysis_error.connect(self._on_analysis_error)
        self.analysis_thread.finished.connect(self._analysis_finished)
        self.analysis_thread.start()

    @QtCore.Slot(object)
    def _on_analysis_ready(self, result: AnalysisResult) -> None:
        annotated = draw_detections(result.frame, result.detections, self.colors)
        try:
            phase_json = self.store.save_phase(
                result.state.phase_number,
                result.frame,
                annotated,
                result.rendered_map,
                result.detections,
                result.state,
                result.metrics,
                result.calibration,
            )
        except Exception as exc:
            self._on_analysis_error(f"No se pudo guardar la fase: {exc}")
            return
        self.phase_number = result.state.phase_number
        self.previous_state = result.state
        self.last_detections = result.detections
        self.history.append(
            {"phase_number": self.phase_number, "metrics": result.metrics}
        )
        self._update_table(result.metrics)
        self.map_widget.set_cv_image(result.rendered_map)
        self.live_title.setText(
            f"CÁMARA EN DIRECTO · CAJAS DE FASE {self.phase_number:04d}"
        )
        conflicts = sum(status.conflict for status in result.state.territories.values())
        self.status_label.setText(
            f"Fase {self.phase_number} guardada · {len(result.detections)} detecciones · "
            f"{len(result.state.occupied)} territorios ocupados · {conflicts} conflictos · {phase_json}"
        )

    @QtCore.Slot(str)
    def _on_analysis_error(self, message: str) -> None:
        self.status_label.setText(message)
        QtWidgets.QMessageBox.critical(self, "Error de análisis", message)

    @QtCore.Slot()
    def _analysis_finished(self) -> None:
        self.phase_button.setEnabled(True)
        self.phase_button.setText("CAMBIO DE FASE")
        self.camera_button.setEnabled(True)
        self.calibrate_button.setEnabled(True)

    def _update_table(self, metrics: dict[str, dict[str, float]]) -> None:
        for row, color in enumerate(self.color_order):
            values = metrics.get(color, {})
            for column_index, column in enumerate(self.table_columns, start=1):
                key = str(column["key"])
                value = values.get(key, 0.0)
                format_string = str(column.get("format", "{:.0f}"))
                try:
                    text = format_string.format(value)
                except (ValueError, KeyError):
                    text = str(value)
                self.metrics_table.item(row, column_index).setText(text)

    def _set_confidence(self, value: float) -> None:
        self.detector.confidence = float(value)

    def _open_results(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.store.session_dir)))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Análisis en curso",
                "Espera a que termine el análisis de la fase antes de cerrar.",
            )
            event.ignore()
            return
        if self.camera_thread is not None:
            self.camera_thread.stop()
        event.accept()
