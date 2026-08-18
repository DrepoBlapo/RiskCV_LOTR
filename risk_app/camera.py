from __future__ import annotations

import platform
from typing import Any

import cv2
from PySide6 import QtCore


class CameraThread(QtCore.QThread):
    frame_ready = QtCore.Signal(object)
    camera_opened = QtCore.Signal(int, int)
    camera_error = QtCore.Signal(str)

    def __init__(self, settings: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.settings = settings

    def _backend(self) -> int:
        name = str(self.settings.get("backend", "auto")).lower()
        if name == "dshow":
            return cv2.CAP_DSHOW
        if name == "msmf":
            return cv2.CAP_MSMF
        if name == "v4l2":
            return cv2.CAP_V4L2
        if name == "auto" and platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY

    def run(self) -> None:
        source = self.settings.get("index", 0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        backend = self._backend()
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(source, cv2.CAP_ANY)
        if not capture.isOpened():
            self.camera_error.emit(f"No se pudo abrir la cámara {source!r}.")
            return
        width = int(self.settings.get("width", 1920))
        height = int(self.settings.get("height", 1080))
        fps = int(self.settings.get("fps", 30))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_opened.emit(actual_width, actual_height)
        failures = 0
        try:
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= 30:
                        self.camera_error.emit("La cámara dejó de entregar fotogramas.")
                        break
                    self.msleep(10)
                    continue
                failures = 0
                self.frame_ready.emit(frame)
        finally:
            capture.release()

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(3000)

