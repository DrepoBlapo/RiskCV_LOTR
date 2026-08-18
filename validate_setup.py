from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from risk_app.config import AppConfig
from risk_app.territory import TerritoryIndex
from risk_app.marks import MapMarks
from risk_app.regions import RegionIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida los archivos de Risk Phase App.")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    config = AppConfig.load(args.config)
    index = TerritoryIndex.load(
        config.territory_id_map_path,
        config.territories_path,
    )
    
    regions_path = config.regions_path

    if regions_path is not None:
        regions = RegionIndex.load(
            regions_path,
            set(index.territory_ids),
        )

        print(
            f"Regiones: {len(regions.regions)}"
        )

        for region in regions.regions:
            print(
                f"  - {region.name} "
                f"({region.region_id}): "
                f"{len(region.territories)} territorios, "
                f"valor {region.value:g}"
            )
    else:
        print("Regiones: no configuradas")

    print(f"Configuración: {config.path}")
    print(f"Modelo: {config.model_path}")
    print(f"Mapa de IDs: {index.width}x{index.height}")
    print(f"Territorios: {len(index.territory_ids)}")
    texture = config.map_texture_path
    print(f"Textura: {texture if texture is not None else 'no configurada'}")
    orography = config.map_orography_overlay_path

    if orography is not None:
        overlay = cv2.imread(
            str(orography),
            cv2.IMREAD_UNCHANGED,
        )

        if overlay is None:
            raise RuntimeError(
                f"No se pudo abrir la orografía: {orography}"
            )

        if overlay.ndim != 3 or overlay.shape[2] != 4:
            raise ValueError(
                "El PNG de orografía no tiene canal alfa."
            )

        overlay_height, overlay_width = overlay.shape[:2]
        
        overlay_ratio = overlay_width / overlay_height
        map_ratio = index.width / index.height
        
        relative_ratio_error = abs(
            overlay_ratio - map_ratio
        ) / map_ratio
        
        if relative_ratio_error > 0.005:
            raise ValueError(
                "La orografía tiene una proporción diferente "
                "a la del mapa de territorios. "
                f"Orografía: {overlay_width}x{overlay_height} "
                f"(proporción {overlay_ratio:.6f}); "
                f"mapa: {index.width}x{index.height} "
                f"(proporción {map_ratio:.6f})."
            )
        
        if (
            overlay_width != index.width
            or overlay_height != index.height
        ):
            print(
                "AVISO: la orografía tiene una resolución "
                "diferente a la del mapa y será redimensionada "
                "automáticamente por la aplicación. "
                f"Orografía: {overlay_width}x{overlay_height}; "
                f"mapa: {index.width}x{index.height}; "
                f"diferencia proporcional: "
                f"{relative_ratio_error * 100:.3f}%."
            )

        print(
            f"Orografía: {orography} "
            f"({overlay.shape[1]}x{overlay.shape[0]})"
        )
    else:
        print("Orografía: no configurada")
    marks_path = config.map_marks_path
    if marks_path is not None:
        marks = MapMarks.load(
            marks_path,
            set(index.territory_ids),
            config.section("marks").get("type_to_color", {}),
        )
        print(f"Marcas: {sum(marks.totals.values())} · {dict(marks.totals)}")
    else:
        print("Marcas: no configuradas")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"CUDA: {cuda}")
        if cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: no instalado todavía")
    print("VALIDACIÓN CORRECTA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
