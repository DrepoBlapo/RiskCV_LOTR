from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .models import PhaseState
from .territory import TerritoryIndex
from .regions import RegionIndex


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Color hexadecimal inválido: {value}")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def saturated_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = color
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
    saturation = min(1.0, saturation * 1.5 + 0.12)
    value = min(1.0, value * 1.18 + 0.06)
    out_red, out_green, out_blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(out_blue * 255), int(out_green * 255), int(out_red * 255)

def region_stripe_bgr(
    color: tuple[int, int, int],
) -> tuple[int, int, int]:
    """
    Genera una variante del color suficientemente visible
    sobre el relleno territorial.

    Los colores oscuros se aclaran y los claros se oscurecen.
    """
    blue, green, red = color

    luminance = (
        0.114 * blue
        + 0.587 * green
        + 0.299 * red
    )

    if luminance < 110:
        mix = 0.62

        return (
            int(blue + (255 - blue) * mix),
            int(green + (255 - green) * mix),
            int(red + (255 - red) * mix),
        )

    factor = 0.50

    return (
        int(blue * factor),
        int(green * factor),
        int(red * factor),
    )

def apply_bgra_overlay(
    base_bgr: np.ndarray,
    overlay_bgra: np.ndarray,
    opacity: float = 1.0,
) -> np.ndarray:
    if overlay_bgra.ndim != 3 or overlay_bgra.shape[2] != 4:
        raise ValueError(
            "El PNG de orografía debe tener cuatro canales BGRA."
        )

    if overlay_bgra.shape[:2] != base_bgr.shape[:2]:
        raise ValueError(
            "La orografía y el mapa deben tener las mismas dimensiones."
        )

    opacity = float(np.clip(opacity, 0.0, 1.0))

    overlay_bgr = overlay_bgra[:, :, :3].astype(np.float32)

    alpha = (
        overlay_bgra[:, :, 3].astype(np.float32) / 255.0
    )
    alpha = alpha * opacity
    alpha = alpha[:, :, np.newaxis]

    result = (
        overlay_bgr * alpha
        + base_bgr.astype(np.float32) * (1.0 - alpha)
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

class TerritoryMapRenderer:
    def __init__(
        self,
        territories: TerritoryIndex,
        colors: dict[str, str],
        base_texture: np.ndarray | None = None,
        orography_overlay: np.ndarray | None = None,
        orography_opacity: float = 1.0,
        regions: RegionIndex | None = None,
        stripe_config: dict | None = None,
    ) -> None:
        self.territories = territories
        self.colors = {
            name: hex_to_bgr(value)
            for name, value in colors.items()
        }
        self.regions = (
            regions
            if regions is not None
            else RegionIndex.empty()
        )

        stripe_config = stripe_config or {}

        self.stripes_enabled = bool(
            stripe_config.get("enabled", True)
        )

        self.stripe_spacing = max(
            2,
            int(
                stripe_config.get(
                    "spacing_px",
                    32,
                )
            ),
        )

        requested_thickness = int(
            stripe_config.get(
                "thickness_px",
                8,
            )
        )

        self.stripe_thickness = max(
            1,
            min(
                requested_thickness,
                self.stripe_spacing - 1,
            ),
        )

        self.stripe_opacity = float(
            np.clip(
                stripe_config.get("opacity", 0.65),
                0.0,
                1.0,
            )
        )

        if base_texture is None:
            self.base = np.full(
                (territories.height, territories.width, 3),
                238,
                dtype=np.uint8,
            )
        else:
            resized = cv2.resize(
                base_texture,
                (territories.width, territories.height),
                interpolation=cv2.INTER_AREA,
            )

            gray = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2GRAY,
            )

            self.base = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR,
            )

            self.base = cv2.addWeighted(
                self.base,
                0.42,
                np.full_like(self.base, 245),
                0.58,
                0,
            )

        self.orography_overlay = None
        self.orography_opacity = float(
            np.clip(orography_opacity, 0.0, 1.0)
        )

        if orography_overlay is not None:
            if (
                orography_overlay.ndim != 3
                or orography_overlay.shape[2] != 4
            ):
                raise ValueError(
                    "map_orography_overlay.png debe tener "
                    "cuatro canales y transparencia."
                )

            self.orography_overlay = cv2.resize(
                orography_overlay,
                (territories.width, territories.height),
                interpolation=cv2.INTER_AREA,
            )
        self.stripe_pattern = (
            self._create_stripe_pattern()
        )

    def _create_stripe_pattern(
        self,
    ) -> np.ndarray:
        """
        Crea una máscara global de líneas diagonales.

        Utilizar el mismo patrón para todo el mapa hace que
        las rayas continúen correctamente entre territorios
        contiguos de una misma región.
        """
        pattern = np.zeros(
            (
                self.territories.height,
                self.territories.width,
            ),
            dtype=np.uint8,
        )

        height = self.territories.height
        width = self.territories.width

        for offset in range(
            -height,
            width,
            self.stripe_spacing,
        ):
            cv2.line(
                pattern,
                (offset, 0),
                (offset + height, height),
                color=255,
                thickness=self.stripe_thickness,
                lineType=cv2.LINE_AA,
            )

        return pattern > 0   

    def _apply_controlled_region_stripes(
        self,
        output: np.ndarray,
        state: PhaseState,
    ) -> np.ndarray:
        """
        Aplica rayas únicamente sobre regiones controladas
        completamente por un color.
        """
        if not self.stripes_enabled:
            return output

        if not state.controlled_regions:
            return output

        result = output.copy()

        for color, region_ids in (
            state.controlled_regions.items()
        ):
            if color not in self.colors:
                continue

            region_mask = np.zeros(
                (
                    self.territories.height,
                    self.territories.width,
                ),
                dtype=bool,
            )

            for region_id in region_ids:
                region = self.regions.by_id.get(
                    region_id
                )

                if region is None:
                    continue

                for territory_id in region.territories:
                    region_mask |= (
                        self.territories.mask_for(
                            territory_id
                        )
                    )

            striped_area = (
                region_mask
                & self.stripe_pattern
            )

            if not np.any(striped_area):
                continue

            stripe_color = region_stripe_bgr(
                self.colors[color]
            )

            stripe_layer = np.empty_like(result)
            stripe_layer[:] = stripe_color

            blended = cv2.addWeighted(
                result,
                1.0 - self.stripe_opacity,
                stripe_layer,
                self.stripe_opacity,
                0,
            )

            result[striped_area] = (
                blended[striped_area]
            )

        return result
             
    def render(self, state: PhaseState) -> np.ndarray:
        output = self.base.copy()

        # Colorea los territorios según su propietario actual.
        for territory_id, status in state.territories.items():
            if (
                status.color is None
                or status.color not in self.colors
            ):
                continue

            mask = self.territories.mask_for(territory_id)
            color = self.colors[status.color]
            alpha = 0.84

            if status.changed:
                color = saturated_bgr(color)
                alpha = 0.96

            color_layer = np.empty_like(output)
            color_layer[:] = color

            blended = cv2.addWeighted(
                output,
                1.0 - alpha,
                color_layer,
                alpha,
                0,
            )

            output[mask] = blended[mask]

        # Regiones completamente controladas.
        output = self._apply_controlled_region_stripes(
            output,
            state,
        )

        # Montañas y ríos encima de colores y rayas.
        if self.orography_overlay is not None:
            output = apply_bgra_overlay(
                output,
                self.orography_overlay,
                opacity=self.orography_opacity,
            )

        # Fronteras territoriales.
        output[
            self.territories.boundary_mask()
        ] = (70, 70, 70)

        # Los bordes de cambio se dibujan al final para que
        # no queden ocultos por las fronteras generales.
        for territory_id, status in state.territories.items():
            if not status.changed:
                continue

            mask_u8 = (
                self.territories
                .mask_for(territory_id)
                .astype(np.uint8)
                * 255
            )

            border = cv2.morphologyEx(
                mask_u8,
                cv2.MORPH_GRADIENT,
                np.ones((5, 5), dtype=np.uint8),
            ) > 0

            if status.color is None:
                border_color = (0, 165, 255)
            elif status.color == "negro":
                border_color = (245, 245, 245)
            else:
                border_color = (20, 20, 20)

            output[border] = border_color

        return output
