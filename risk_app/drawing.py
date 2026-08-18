from __future__ import annotations

import cv2
import numpy as np

from .map_renderer import hex_to_bgr
from .models import Detection


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    color_hex: dict[str, str],
    show_contact: bool = True,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for detection in detections:
        scale_x = width / max(detection.source_width, 1)
        scale_y = height / max(detection.source_height, 1)
        x0, y0, x1, y1 = detection.bbox_xyxy
        points = (
            int(round(x0 * scale_x)),
            int(round(y0 * scale_y)),
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
        )
        color = hex_to_bgr(color_hex.get(detection.color or "", "#20A4F3"))
        cv2.rectangle(output, points[:2], points[2:], color, 2, cv2.LINE_AA)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        if detection.territory_id:
            label += f" · {detection.territory_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(points[1], text_height + baseline + 4)
        cv2.rectangle(
            output,
            (points[0], label_y - text_height - baseline - 4),
            (points[0] + text_width + 6, label_y),
            color,
            -1,
        )
        text_color = (255, 255, 255) if sum(color) < 390 else (25, 25, 25)
        cv2.putText(
            output,
            label,
            (points[0] + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )
        if show_contact and detection.contact_point is not None:
            contact = (
                int(round(detection.contact_point[0] * scale_x)),
                int(round(detection.contact_point[1] * scale_y)),
            )
            cv2.drawMarker(output, contact, color, cv2.MARKER_CROSS, 10, 2)
    return output

