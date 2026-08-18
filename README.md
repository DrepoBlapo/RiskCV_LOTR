# Risk Phase App

Aplicación local para ejecutar una inferencia YOLO únicamente cuando se pulsa **CAMBIO DE FASE**. La cámara permanece en directo entre fases y muestra las cajas del último análisis confirmado; las métricas y el mapa territorial tampoco cambian hasta el siguiente clic.

## Qué incluye

- Cámara en directo sin inferencia continua.
- Calibración manual por las cuatro esquinas del mapa.
- Una captura y una inferencia YOLO por cambio de fase.
- Cajas de la última fase superpuestas sobre el vídeo vivo.
- Asignación de cada chincheta al territorio situado bajo su punto de apoyo.
- Resolución configurable de conflictos por máximo individual, suma o evidencia ponderada.
- Segundo análisis ampliado opcional solo sobre los territorios conflictivos.
- Conteo de las marcas impresas de `map_texture_marcas.json` en cada fase.
- Conteo de territorios únicos por color.
- Mapa territorial coloreado; los cambios respecto de la fase anterior aparecen más saturados y con borde.
- Métricas ampliables en Python y columnas calculadas mediante fórmulas seguras.
- Historial por fase con captura original, captura anotada, mapa y JSON completo.

La primera fase es la línea base: no se resalta ningún cambio. A partir de la segunda fase se compara el propietario de cada territorio con la fase anterior.

## Instalación en Windows

Se recomienda Python 3.10–3.12 y un entorno virtual:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para usar la RTX 3050 Ti, instala una compilación de PyTorch compatible con el controlador/CUDA del equipo y comprueba:

```cmd
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Configuración

1. Copia `config.example.json` como `config.json`.
2. Cambia `model.path` para que apunte a tu `best.pt`.
3. Cambia las tres rutas de `assets` si los archivos no están en `prepared_assets`.
4. Si la cámara no es la predeterminada, cambia `camera.index` a `1`, `2`, etc.

Los archivos necesarios son:

- `best.pt`
- `territory_id_map.png`
- `territories.json.gz`, con `territory_id_colors_rgb`
- `map_texture.png` es opcional; solo sirve como fondo del mapa de estado.
- `map_texture_marcas.json`, con la lista `marks`, permite incorporar las marcas impresas.

Si el JSON usa nombres como `marca_1` y `marca_2`, indica qué color representa
cada tipo en `marks.type_to_color`. Si un registro ya tiene `color`, ese valor
tiene prioridad:

```json
"marks": {
  "type_to_color": {
    "marca_1": "amarillo",
    "marca_2": "verde"
  }
}
```

Inicia la aplicación con:

```cmd
python validate_setup.py --config config.json
python run_app.py --config config.json
```

También puedes hacer doble clic en `run_windows.bat` si ya existe `config.json`.

## Primer uso

1. Espera a ver la cámara en directo.
2. Pulsa **Calibrar mapa**.
3. Haz clic, en este orden, en las esquinas del mapa: superior izquierda, superior derecha, inferior derecha e inferior izquierda.
4. Pulsa **CAMBIO DE FASE**.

La calibración queda guardada en `app_data/calibration.json`. Si se mueve la cámara, recalibra. Al recalibrar se ocultan las cajas antiguas porque ya no coincidirían con el vídeo; el último estado territorial sí se conserva para comparar la siguiente fase.

## Resultados

Cada ejecución crea una carpeta independiente en `phase_results`:

```text
phase_results/
  session_YYYYMMDD_HHMMSS_micros/
    session.json
    phase_0001/
      frame_raw.png
      frame_annotated.png
      territory_map.png
      phase.json
```

`phase.json` guarda las detecciones completas, clases, confianza, cajas, puntos de apoyo, territorio asignado, conflictos, estado territorial y métricas. Eso permite recalcular estadísticas después sin volver a ejecutar YOLO.

## Personalizar la tabla

La tabla se define en `config.json`. La configuración inicial solo muestra el dato solicitado, el número de territorios de cada color:

```json
"columns": [
  {"key": "territories", "title": "Territorios", "format": "{:.0f}"}
]
```

Ya existen métricas reutilizables: `territories`, `pins`, `pin_value`,
`changed_territories`, `map_marks`, `marks_in_controlled_territories` y
`conflicted_territories`. Para mostrarlas, añade columnas:

```json
"columns": [
  {"key": "territories", "title": "Territorios", "format": "{:.0f}"},
  {"key": "pins", "title": "Chinchetas", "format": "{:.0f}"},
  {"key": "score", "title": "Mi cálculo", "formula": "territories * 2 + pins", "format": "{:.1f}"}
]
```

Una fórmula puede usar columnas anteriores y los operadores `+`, `-`, `*`, `/`, `//`, `%` y `**`. No ejecuta funciones ni código arbitrario.

Para una estadística con lógica propia, edita `risk_app/metrics/custom_metrics.py` y registra una función:

```python
from .registry import register_metric

@register_metric("mi_estadistica")
def mi_estadistica(context, color):
    territorios = [
        estado
        for estado in context.state.territories.values()
        if estado.color == color
    ]
    return len(territorios) * 2
```

Añade después `{"key": "mi_estadistica", "title": "Mi estadística"}` a `table.columns`. `context` da acceso a `state`, `previous_state`, `detections` e `history`; no es necesario otro lenguaje.

## Criterios de asignación

- Se proyecta el punto de apoyo de cada caja, no su centro, mediante la homografía de las cuatro esquinas.
- `calibration.contact_y_fraction` controla la altura dentro de la caja: `1.0` es el borde inferior y `0.5` es el centro.
- Si el punto cae justo en una frontera, `lookup_radius_px` busca el territorio válido predominante alrededor.
- Si un territorio contiene varios colores, `conflict_resolution.strategy`
  decide: `max_detection`, `confidence_sum` o `evidence_weighted`.
    `max_detection`: Gana el color de la detección individual con mayor confianza.
    `confidence_sum`: Suma las confianzas de todas las detecciones de cada color.
    `evidence_weighted`: puntuación =
    peso_max   × confianza máxima
  + peso_mean  × confianza media
  + peso_sum   × suma de confianzas
  + peso_count × número de detecciones
  
- Cada territorio guarda `color_evidence` con `max`, `mean`, `sum` y `count`.
- Con `zoom.enabled=true`, los conflictos se recortan, se vuelven a analizar y
  las cajas refinadas sustituyen a las del primer pase para ese territorio.

## Pruebas del núcleo

```cmd
python -m unittest discover -s tests -v
```
