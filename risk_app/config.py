from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_COLORS = {
    "negro": "#151413",
    "rojo": "#981A22",
    "amarillo": "#FAE81E",
    "verde": "#AFC839",
}


class ConfigError(ValueError):
    pass


@dataclass(slots=True)
class AppConfig:
    path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigError(f"No existe el archivo de configuración: {config_path}")
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"No se pudo leer {config_path}: {exc}") from exc
        config = cls(config_path, data)
        config.validate_shape()
        return config

    @property
    def root(self) -> Path:
        return self.path.parent

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"La sección '{name}' debe ser un objeto JSON.")
        return value

    def resolve(self, value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def required_path(self, section: str, key: str) -> Path:
        raw = self.section(section).get(key)
        path = self.resolve(raw)
        if path is None:
            raise ConfigError(f"Falta la ruta '{section}.{key}' en {self.path.name}.")
        if not path.is_file():
            raise ConfigError(f"No existe {section}.{key}: {path}")
        return path

    def validate_shape(self) -> None:
        for name in ("model", "camera", "assets", "calibration", "output", "display", "table"):
            self.section(name)

    @property
    def model_path(self) -> Path:
        return self.required_path("model", "path")

    @property
    def territory_id_map_path(self) -> Path:
        return self.required_path("assets", "territory_id_map")

    @property
    def territories_path(self) -> Path:
        return self.required_path("assets", "territories")

    @property
    def map_texture_path(self) -> Path | None:
        path = self.resolve(self.section("assets").get("map_texture"))
        if path is not None and not path.is_file():
            raise ConfigError(f"No existe assets.map_texture: {path}")
        return path

    @property
    def map_orography_overlay_path(self) -> Path | None:
        path = self.resolve(
            self.section("assets").get("map_orography_overlay")
        )
        if path is not None and not path.is_file():
            raise ConfigError(
                f"No existe assets.map_orography_overlay: {path}"
            )
        return path

    @property
    def map_marks_path(self) -> Path | None:
        path = self.resolve(self.section("assets").get("map_marks"))
        if path is not None and not path.is_file():
            raise ConfigError(f"No existe assets.map_marks: {path}")
        return path

    @property
    def regions_path(self) -> Path | None:
        path = self.resolve(
            self.section("assets").get("regions")
        )

        if path is not None and not path.is_file():
            raise ConfigError(
                f"No existe assets.regions: {path}"
            )

        return path

    @property
    def calibration_path(self) -> Path:
        raw = self.section("calibration").get("file", "./app_data/calibration.json")
        path = self.resolve(raw)
        assert path is not None
        return path

    @property
    def output_dir(self) -> Path:
        raw = self.section("output").get("directory", "./phase_results")
        path = self.resolve(raw)
        assert path is not None
        return path

    @property
    def colors(self) -> dict[str, str]:
        custom = self.section("display").get("colors", {})
        return {**DEFAULT_COLORS, **custom}

    @property
    def color_order(self) -> list[str]:
        value = self.section("table").get(
            "color_order", ["negro", "rojo", "amarillo", "verde"]
        )
        if not isinstance(value, list) or not value:
            raise ConfigError("table.color_order debe ser una lista no vacía.")
        return [str(item) for item in value]
