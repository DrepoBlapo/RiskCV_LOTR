from __future__ import annotations

from collections.abc import Callable
from typing import Any


MetricFunction = Callable[[Any, str], float]
REGISTRY: dict[str, MetricFunction] = {}


def register_metric(name: str) -> Callable[[MetricFunction], MetricFunction]:
    """Decorador para añadir una métrica por color sin tocar la interfaz."""

    def decorator(function: MetricFunction) -> MetricFunction:
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"Nombre de métrica inválido: {name!r}")
        REGISTRY[name] = function
        return function

    return decorator


def get_metric(name: str) -> MetricFunction:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"No existe la métrica registrada '{name}'.") from exc

