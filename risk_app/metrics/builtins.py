from __future__ import annotations

from .registry import register_metric


PIN_VALUES = {0: 0, 1: 1, 2: 3, 3: 5}


@register_metric("territories")
def territories(context, color: str) -> float:
    return float(
        sum(status.color == color for status in context.state.territories.values())
    )


@register_metric("pins")
def pins(context, color: str) -> float:
    return float(sum(detection.color == color for detection in context.detections))


@register_metric("pin_value")
def pin_value(context, color: str) -> float:
    return float(
        sum(
            PIN_VALUES.get(detection.pin_type, 0)
            for detection in context.detections
            if detection.color == color
        )
    )


@register_metric("changed_territories")
def changed_territories(context, color: str) -> float:
    return float(
        sum(
            status.color == color and status.changed
            for status in context.state.territories.values()
        )
    )


@register_metric("map_marks")
def map_marks(context, color: str) -> float:
    """Número fijo de marcas impresas de este color en todo el mapa."""
    return float(context.state.marks_by_color.get(color, 0))


@register_metric("marks_in_controlled_territories")
def marks_in_controlled_territories(context, color: str) -> float:
    """Marcas de cualquier color situadas en territorios controlados por `color`."""
    return float(sum(
        sum(status.marks_by_color.values())
        for status in context.state.territories.values()
        if status.color == color
    ))

@register_metric("controlled_marks_type_1")
def controlled_marks_type_1(
    context,
    color: str,
) -> float:
    """
    Cuenta las marcas tipo 1 situadas en territorios
    controlados actualmente por este color.
    """
    return float(
        sum(
            status.marks_by_color.get("marca_1", 0)
            for status in context.state.territories.values()
            if status.color == color
        )
    )


@register_metric("controlled_marks_type_2")
def controlled_marks_type_2(
    context,
    color: str,
) -> float:
    """
    Cuenta las marcas tipo 2 situadas en territorios
    controlados actualmente por este color.
    """
    return float(
        sum(
            status.marks_by_color.get("marca_2", 0)
            for status in context.state.territories.values()
            if status.color == color
        )
    )

@register_metric("conflicted_territories")
def conflicted_territories(context, color: str) -> float:
    return float(sum(
        status.color == color and status.conflict
        for status in context.state.territories.values()
    ))

@register_metric("control_region")
def control_region(
    context,
    color: str,
) -> float:
    """
    Suma el valor de las regiones completamente
    controladas por el color indicado.
    """
    if context.regions is None:
        return 0.0

    return context.regions.control_value(
        context.state,
        color,
    )

@register_metric(
    "changed_territories_with_mark_type_1"
)
def changed_territories_with_mark_type_1(
    context,
    color: str,
) -> float:
    """
    Cuenta los territorios recién adquiridos por el color
    que contenían al menos una marca tipo 1.

    Cada territorio cuenta como máximo una vez, aunque
    contenga varias marcas tipo 1.
    """
    if context.previous_state is None:
        return 0.0

    return float(
        sum(
            1
            for status
            in context.state.territories.values()
            if (
                status.color == color
                and status.changed
                and status.marks_by_color.get(
                    "marca_1",
                    0,
                ) > 0
            )
        )
    )