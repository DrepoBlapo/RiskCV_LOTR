from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .models import PhaseState


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """
    Definición inmutable de una región del mapa.
    """

    region_id: str
    name: str
    value: float
    territories: tuple[str, ...]


class RegionIndex:
    """
    Contiene las regiones y la relación entre cada territorio
    y su región.
    """

    def __init__(
        self,
        regions: list[RegionDefinition],
        territory_to_region: dict[str, str],
    ) -> None:
        self.regions = tuple(regions)
        self.by_id = {
            region.region_id: region
            for region in regions
        }
        self.territory_to_region = dict(
            territory_to_region
        )

    @classmethod
    def empty(cls) -> "RegionIndex":
        return cls([], {})

    @classmethod
    def load(
        cls,
        path: Path,
        valid_territories: set[str],
    ) -> "RegionIndex":
        try:
            root = json.loads(
                path.read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise ValueError(
                f"No se pudo leer el archivo de regiones: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"El archivo de regiones no es un JSON válido: "
                f"{path}. Línea {exc.lineno}, "
                f"columna {exc.colno}: {exc.msg}"
            ) from exc

        if not isinstance(root, dict):
            raise ValueError(
                "regions.json debe contener un objeto JSON."
            )

        raw_regions = root.get("regions")

        if not isinstance(raw_regions, list):
            raise ValueError(
                "regions.json debe contener la lista 'regions'."
            )

        if not raw_regions:
            raise ValueError(
                "La lista 'regions' no puede estar vacía."
            )

        regions: list[RegionDefinition] = []
        region_ids: set[str] = set()
        territory_to_region: dict[str, str] = {}

        for index, raw_region in enumerate(
            raw_regions,
            start=1,
        ):
            if not isinstance(raw_region, dict):
                raise ValueError(
                    f"La región número {index} no es "
                    "un objeto JSON."
                )

            region_id = str(
                raw_region.get("id", "")
            ).strip()

            if not region_id:
                raise ValueError(
                    f"La región número {index} no tiene "
                    "un campo 'id' válido."
                )

            if region_id in region_ids:
                raise ValueError(
                    f"El identificador de región está "
                    f"repetido: {region_id}"
                )

            region_ids.add(region_id)

            name = str(
                raw_region.get("name", region_id)
            ).strip()

            if not name:
                raise ValueError(
                    f"La región {region_id} no tiene "
                    "un nombre válido."
                )

            raw_value = raw_region.get("value")

            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
            ):
                raise ValueError(
                    f"El valor de la región {region_id} "
                    "debe ser numérico."
                )

            value = float(raw_value)

            if value < 0:
                raise ValueError(
                    f"El valor de la región {region_id} "
                    "no puede ser negativo."
                )

            raw_territories = raw_region.get(
                "territories"
            )

            if (
                not isinstance(raw_territories, list)
                or not raw_territories
            ):
                raise ValueError(
                    f"La región {region_id} debe contener "
                    "una lista no vacía de territorios."
                )

            territory_ids: list[str] = []

            for raw_territory_id in raw_territories:
                territory_id = str(
                    raw_territory_id
                ).strip()

                if not territory_id:
                    raise ValueError(
                        f"La región {region_id} contiene "
                        "un ID de territorio vacío."
                    )

                if territory_id not in valid_territories:
                    raise ValueError(
                        f"La región {region_id} contiene "
                        f"un territorio desconocido: "
                        f"{territory_id}"
                    )

                previous_region = (
                    territory_to_region.get(territory_id)
                )

                if previous_region is not None:
                    raise ValueError(
                        f"El territorio {territory_id} "
                        "aparece en más de una región: "
                        f"{previous_region} y {region_id}."
                    )

                territory_to_region[territory_id] = (
                    region_id
                )
                territory_ids.append(territory_id)

            regions.append(
                RegionDefinition(
                    region_id=region_id,
                    name=name,
                    value=value,
                    territories=tuple(territory_ids),
                )
            )

        missing_territories = (
            valid_territories
            - set(territory_to_region)
        )

        if missing_territories:
            missing_text = ", ".join(
                sorted(missing_territories)
            )

            raise ValueError(
                "Los siguientes territorios no pertenecen "
                f"a ninguna región: {missing_text}"
            )

        return cls(
            regions,
            territory_to_region,
        )

    def region_for_territory(
        self,
        territory_id: str,
    ) -> RegionDefinition | None:
        region_id = self.territory_to_region.get(
            territory_id
        )

        if region_id is None:
            return None

        return self.by_id[region_id]

    def controlled_by(
        self,
        state: "PhaseState",
        color: str,
    ) -> tuple[RegionDefinition, ...]:
        """
        Devuelve las regiones cuyos territorios están
        completamente controlados por un mismo color.
        """
        controlled: list[RegionDefinition] = []

        for region in self.regions:
            is_controlled = all(
                state.territories[territory_id].color
                == color
                for territory_id in region.territories
            )

            if is_controlled:
                controlled.append(region)

        return tuple(controlled)

    def control_value(
        self,
        state: "PhaseState",
        color: str,
    ) -> float:
        """
        Suma el valor de todas las regiones controladas
        por el color indicado.
        """
        return float(
            sum(
                region.value
                for region in self.controlled_by(
                    state,
                    color,
                )
            )
        )