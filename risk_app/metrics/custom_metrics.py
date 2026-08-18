"""Añade aquí las estadísticas específicas del juego.

Ejemplo:

    from .registry import register_metric

    @register_metric("mi_estadistica")
    def mi_estadistica(context, color):
        territorios = [
            t for t in context.state.territories.values() if t.color == color
        ]
        return len(territorios) * 2

Después añade una columna con key="mi_estadistica" en config.json.

Datos disponibles en ``context``:
    context.state.marks_by_color
    context.state.territories[id].marks_by_color
    context.state.territories[id].color_evidence
    context.state.territories[id].conflict
    context.detections
    context.previous_state
    context.history
"""

from .registry import register_metric


@register_metric("example_marks_on_owned_territories")
def example_marks_on_owned_territories(context, color):
    """Plantilla: total de marcas en los territorios que controla el color."""
    return sum(
        sum(territory.marks_by_color.values())
        for territory in context.state.territories.values()
        if territory.color == color
    )
