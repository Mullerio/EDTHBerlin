from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class Vertex:
    grid: np.ndarray  # [row, col]
    meters: np.ndarray  # [x, y]


@dataclass
class CellPosition:
    grid: Optional[np.ndarray]
    meters: Optional[np.ndarray]


@dataclass
class Attacker:
    name: str
    start: CellPosition
    target: CellPosition


@dataclass
class GridData:
    matrix: np.ndarray
    cell_size_m: float
    centroid_grid: Optional[np.ndarray]
    centroid_meters: Optional[np.ndarray]
    shape_vertices: list[Vertex]
    attackers: list[Attacker]


def _vector_from_dict(data: Optional[dict[str, float]], keys: Iterable[str]) -> Optional[np.ndarray]:
    if not data:
        return None
    try:
        values = [float(data[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return None
    return np.array(values, dtype=float)


def _vertex_from_dict(entry: dict) -> Vertex:
    grid_vec = _vector_from_dict(entry.get("grid"), ("row", "col"))
    meter_vec = _vector_from_dict(entry.get("meters"), ("x", "y"))
    if grid_vec is None:
        grid_vec = np.zeros(2, dtype=float)
    if meter_vec is None:
        meter_vec = np.zeros(2, dtype=float)
    return Vertex(grid=grid_vec, meters=meter_vec)


def _cell_position_from_dict(entry: Optional[dict]) -> CellPosition:
    if not entry:
        return CellPosition(grid=None, meters=None)
    grid_vec = _vector_from_dict(entry.get("grid"), ("row", "col"))
    meter_vec = _vector_from_dict(entry.get("meters"), ("x", "y"))
    return CellPosition(grid=grid_vec, meters=meter_vec)


def load_grid_payload(path: str | Path) -> GridData:
    """Load a generated grid JSON file into convenient numpy objects."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    matrix = np.array(raw["matrix"], dtype=int)
    cell_size_m = float(raw.get("cellSizeMeters", 1.0))

    centroid = raw.get("centroid")
    centroid_grid = _vector_from_dict(centroid.get("grid") if centroid else None, ("row", "col"))
    centroid_meters = _vector_from_dict(centroid.get("meters") if centroid else None, ("x", "y"))

    vertices = [_vertex_from_dict(entry) for entry in raw.get("shapeVertices", []) or []]

    attackers: list[Attacker] = []
    for attacker in raw.get("attackers", []):
        attackers.append(
            Attacker(
                name=attacker.get("name", ""),
                start=_cell_position_from_dict(attacker.get("start")),
                target=_cell_position_from_dict(attacker.get("target")),
            )
        )

    return GridData(
        matrix=matrix,
        cell_size_m=cell_size_m,
        centroid_grid=centroid_grid,
        centroid_meters=centroid_meters,
        shape_vertices=vertices,
        attackers=attackers,
    )


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Load a grid JSON file and print summary info.")
    # parser.add_argument("json_path", type=Path, help="Path to the downloaded grid JSON file")
    # args = parser.parse_args()

    json_path = r'utils/grid-12x12(2).json'

    grid = load_grid_payload(json_path)
    print(f"Matrix shape: {grid.matrix.shape}, dtype={grid.matrix.dtype}")
    print(f"Cell size (m): {grid.cell_size_m}")
    print(f"Centroid grid coords: {grid.centroid_grid}")
    print(f"Centroid meters: {grid.centroid_meters}")
    print(f"Vertices: {len(grid.shape_vertices)}")
    print(f"Attackers: {len(grid.attackers)}")
    for attacker in grid.attackers:
        print(f"Attacker: {attacker.name}, Start: {attacker.start}, Target: {attacker.target}")
