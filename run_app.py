from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6 import QtWidgets

from risk_app.config import AppConfig
from risk_app.ui.main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control local de fases del tablero Risk mediante YOLO."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Configuración de la app (por defecto: config.json).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Risk Phase App")
    application.setStyle("Fusion")
    try:
        config = AppConfig.load(args.config)
        window = MainWindow(config)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(
            None,
            "No se pudo iniciar Risk Phase App",
            f"{type(exc).__name__}: {exc}",
        )
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

