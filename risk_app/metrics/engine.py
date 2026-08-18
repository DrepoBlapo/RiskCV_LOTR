from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from ..models import Detection, PhaseState
from . import builtins as _builtins  # noqa: F401
from . import custom_metrics as _custom_metrics  # noqa: F401
from .registry import get_metric
from ..regions import RegionIndex


@dataclass(slots=True)
class MetricContext:
    state: PhaseState
    previous_state: PhaseState | None
    detections: list[Detection]
    history: list[dict[str, Any]]
    regions: RegionIndex | None = None


class FormulaError(ValueError):
    pass


def evaluate_formula(expression: str, values: dict[str, float]) -> float:
    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise FormulaError(f"La fórmula usa una métrica desconocida: {node.id}")
            return float(values[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            operators = {
                ast.Add: lambda: left + right,
                ast.Sub: lambda: left - right,
                ast.Mult: lambda: left * right,
                ast.Div: lambda: left / right,
                ast.FloorDiv: lambda: left // right,
                ast.Mod: lambda: left % right,
                ast.Pow: lambda: left**right,
            }
            for operator_type, operation in operators.items():
                if isinstance(node.op, operator_type):
                    return float(operation())
        raise FormulaError("La fórmula contiene una operación no permitida.")

    return evaluate(tree)


class MetricEngine:
    def __init__(self, columns: list[dict[str, Any]], colors: list[str]) -> None:
        self.columns = columns
        self.colors = colors

    def calculate(self, context: MetricContext) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {color: {} for color in self.colors}
        for column in self.columns:
            key = str(column.get("key", "")).strip()
            if not key:
                raise ValueError("Cada columna de table.columns necesita 'key'.")
            formula = column.get("formula")
            for color in self.colors:
                if formula:
                    value = evaluate_formula(str(formula), results[color])
                else:
                    value = float(get_metric(key)(context, color))
                results[color][key] = value
        return results

