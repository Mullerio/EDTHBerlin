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
class Shape:
    id: int
    centroid_grid: Optional[np.ndarray]
    centroid_meters: Optional[np.ndarray]
    vertices: list[Vertex]


@dataclass
class GridData:
    matrix: np.ndarray
    cell_size_m: float
    centroid_grid: Optional[np.ndarray]
    centroid_meters: Optional[np.ndarray]
    shape_vertices: list[Vertex]
    shapes: list[Shape]
    attackers: list[Attacker]


def _apply_lower_left_origin(grid_data: GridData) -> GridData:
    height = grid_data.matrix.shape[0]
    if height == 0:
        return grid_data

    max_row_index = height - 1
    max_meter_y = max_row_index * grid_data.cell_size_m

    def flip_grid(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None or vec.size == 0:
            return vec
        flipped = vec.copy()
        flipped[0] = max_row_index - flipped[0]
        return flipped

    def flip_meters(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None or vec.size < 2:
            return vec
        flipped = vec.copy()
        flipped[1] = max_meter_y - flipped[1]
        return flipped

    converted_vertex_ids: set[int] = set()

    def flip_vertex(vertex: Vertex) -> None:
        if vertex is None:
            return
        vertex_id = id(vertex)
        if vertex_id in converted_vertex_ids:
            return
        converted_vertex_ids.add(vertex_id)
        vertex.grid = flip_grid(vertex.grid)
        vertex.meters = flip_meters(vertex.meters)

    def flip_cell_position(cell: Optional[CellPosition]) -> Optional[CellPosition]:
        if cell is None:
            return None
        cell.grid = flip_grid(cell.grid)
        cell.meters = flip_meters(cell.meters)
        return cell

    grid_data.centroid_grid = flip_grid(grid_data.centroid_grid)
    grid_data.centroid_meters = flip_meters(grid_data.centroid_meters)

    for vertex in grid_data.shape_vertices:
        flip_vertex(vertex)

    for shape in grid_data.shapes:
        shape.centroid_grid = flip_grid(shape.centroid_grid)
        shape.centroid_meters = flip_meters(shape.centroid_meters)
        for vertex in shape.vertices:
            flip_vertex(vertex)

    for attacker in grid_data.attackers:
        attacker.start = flip_cell_position(attacker.start)
        attacker.target = flip_cell_position(attacker.target)

    return grid_data


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

    shapes_data = raw.get("shapes")
    shapes: list[Shape] = []
    if shapes_data:
        for index, entry in enumerate(shapes_data, start=1):
            centroid_entry = entry.get("centroid") or {}
            shape_centroid_grid = _vector_from_dict(centroid_entry.get("grid"), ("row", "col"))
            shape_centroid_meters = _vector_from_dict(centroid_entry.get("meters"), ("x", "y"))
            vertices = [_vertex_from_dict(vertex) for vertex in entry.get("vertices", []) or []]
            shape_id = int(entry.get("id", index))
            shapes.append(
                Shape(
                    id=shape_id,
                    centroid_grid=shape_centroid_grid,
                    centroid_meters=shape_centroid_meters,
                    vertices=vertices,
                )
            )
        if shapes:
            centroid_grid = shapes[0].centroid_grid
            centroid_meters = shapes[0].centroid_meters
            vertices = shapes[0].vertices
        else:
            vertices = []
    else:
        vertices = [_vertex_from_dict(entry) for entry in raw.get("shapeVertices", []) or []]
        if vertices:
            shapes.append(
                Shape(
                    id=1,
                    centroid_grid=centroid_grid,
                    centroid_meters=centroid_meters,
                    vertices=vertices,
                )
            )

    attackers: list[Attacker] = []
    for attacker in raw.get("attackers", []):
        attackers.append(
            Attacker(
                name=attacker.get("name", ""),
                start=_cell_position_from_dict(attacker.get("start")),
                target=_cell_position_from_dict(attacker.get("target")),
            )
        )

    grid_data = GridData(
        matrix=matrix,
        cell_size_m=cell_size_m,
        centroid_grid=centroid_grid,
        centroid_meters=centroid_meters,
        shape_vertices=vertices,
        shapes=shapes,
        attackers=attackers,
    )
    return _apply_lower_left_origin(grid_data)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Load a grid JSON file and print summary info.")
    # parser.add_argument("json_path", type=Path, help="Path to the downloaded grid JSON file")
    # args = parser.parse_args()

    json_path = r'utils/grid-12x12(6).json'

    grid = load_grid_payload(json_path)
    print(f"Matrix shape: {grid.matrix.shape}, dtype={grid.matrix.dtype}")
    print(f"Cell size (m): {grid.cell_size_m}")
    print(f"Centroid grid coords: {grid.centroid_grid}")
    print(f"Centroid meters: {grid.centroid_meters}")
    print(f"Shapes: {grid.shapes}")
    print(f"Vertices (first shape): {len(grid.shape_vertices)}")
    print(f"Attackers: {len(grid.attackers)}")
    for attacker in grid.attackers:
        print(f"Attacker: {attacker.name}, Start: {attacker.start}, Target: {attacker.target}")

    print(grid.matrix)
