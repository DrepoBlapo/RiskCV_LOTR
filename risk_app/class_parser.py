from __future__ import annotations

import re
import unicodedata


COLOR_ALIASES = {
    "n": "negro",
    "black": "negro",
    "negro": "negro",
    "r": "rojo",
    "red": "rojo",
    "rojo": "rojo",
    "a": "amarillo",
    "y": "amarillo",
    "yellow": "amarillo",
    "amarillo": "amarillo",
    "v": "verde",
    "g": "verde",
    "green": "verde",
    "verde": "verde",
}


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return value.lower().strip()


def parse_class_name(class_name: str) -> tuple[str | None, int | None]:
    """Admite nombres como n0, n_0, negro_0, tipo_0_negro o red-type-2."""

    normalized = _normalize(class_name)
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]

    color: str | None = None
    pin_type: int | None = None

    for token in tokens:
        if token in COLOR_ALIASES:
            color = COLOR_ALIASES[token]
        match = re.fullmatch(r"(?:tipo|type)?([0-3])", token)
        if match:
            pin_type = int(match.group(1))

    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if color is None:
        for alias in sorted(COLOR_ALIASES, key=len, reverse=True):
            if compact.startswith(alias):
                color = COLOR_ALIASES[alias]
                break
            if compact.endswith(alias):
                color = COLOR_ALIASES[alias]
                break

    if pin_type is None:
        match = re.search(r"(?:tipo|type)?([0-3])(?:$|[^0-9])", normalized)
        if match:
            pin_type = int(match.group(1))
        else:
            digits = re.findall(r"[0-3]", compact)
            if digits:
                pin_type = int(digits[-1])

    return color, pin_type

