from __future__ import annotations

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class ImageWidget(QtWidgets.QLabel):
    image_clicked = QtCore.Signal(float, float)

    def __init__(self, clickable: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.clickable = clickable
        self._source: QtGui.QImage | None = None
        self._source_size: tuple[int, int] | None = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setStyleSheet("background:#101318; border:1px solid #303640;")
        self.setText("Esperando imagen…")

    def set_cv_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QtGui.QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        self._source = image
        self._source_size = (width, height)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._source is None:
            return
        pixmap = QtGui.QPixmap.fromImage(self._source).scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)
        if not self.clickable or self._source_size is None or self.pixmap() is None:
            return
        pixmap = self.pixmap()
        offset_x = (self.width() - pixmap.width()) / 2.0
        offset_y = (self.height() - pixmap.height()) / 2.0
        point = event.position()
        relative_x = point.x() - offset_x
        relative_y = point.y() - offset_y
        if not (0 <= relative_x < pixmap.width() and 0 <= relative_y < pixmap.height()):
            return
        source_width, source_height = self._source_size
        x = relative_x / pixmap.width() * source_width
        y = relative_y / pixmap.height() * source_height
        self.image_clicked.emit(x, y)

