from __future__ import annotations

from dataclasses import dataclass
import math
import random
import tempfile
import time
from typing import Dict, Iterable, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


COVERAGE_DRAW_SCALE = 2  # render overlay at higher resolution for smoother edges

CANVAS_PX = 640
GRID_CELLS_PER_SIDE = 330
GRID_CELL_SIZE_KM = 0.1  # 100 m × 100 m cells
SECTOR_KM = GRID_CELLS_PER_SIDE * GRID_CELL_SIZE_KM  # modeled 33 km square sector
KM_PER_PX = SECTOR_KM / CANVAS_PX
GRID_CELL_SPACING_PX = CANVAS_PX / GRID_CELLS_PER_SIDE
GRID_CELL_CENTERS_PX = tuple(
    (idx + 0.5) * GRID_CELL_SPACING_PX for idx in range(GRID_CELLS_PER_SIDE)
)

SUPPORT_RADAR_DEFAULT_RANGE_KM = 0.35
SUPPORT_RADAR_STANDOFF_KM = 0.8
SUPPORT_RADAR_MAX = 200
SUPPORT_ANIMATION_FRAMES = 18
SUPPORT_ANIMATION_DELAY_S = 0.08
ANGLE_MERGE_THRESHOLD = math.radians(3)
MIN_SCATTER_DISTANCE_KM = 8.0
OBSTACLE_BASE_WIDTH_KM = 0.3
SUPPORT_LAYER_OVERLAP = 0.65
SUPPORT_ANGULAR_OVERLAP = 0.75
SUPPORT_COVERAGE_SCALE = 3.0
SUPPORT_GRID_STEP_KM = SUPPORT_RADAR_DEFAULT_RANGE_KM * SUPPORT_COVERAGE_SCALE * 0.9
SUPPORT_GRID_MARGIN_KM = SUPPORT_GRID_STEP_KM * 0.5

try:  # Pillow>=9 exposes enums under Image.Resampling
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - fallback for older Pillow
    RESAMPLING_FILTER = Image.LANCZOS


@dataclass(frozen=True)
class RadarPreset:
    name: str
    range_km: int
    description: str


RADAR_PRESETS: Dict[str, RadarPreset] = {
    "R42": RadarPreset(
        name="R42",
        range_km=15,
        description="Short-range R42 guard radar providing near-sector coverage.",
    ),
    "SUPER-R": RadarPreset(
        name="SUPER-R",
        range_km=30,
        description="Extended-range SUPER-R sentry covering the entire azimuth arc.",
    ),
}


def _make_sector_background(size: int = CANVAS_PX) -> Image.Image:
    """Generate a simple stylised sector map with grid lines."""

    background = Image.new("RGB", (size, size), color=(4, 15, 29))
    draw = ImageDraw.Draw(background)

    cell_spacing_px = size / GRID_CELLS_PER_SIDE
    base_grid_color = (17, 54, 92)
    accent_grid_color = (26, 92, 125)

    def grid_positions():
        prev_coord = -1
        for idx in range(GRID_CELLS_PER_SIDE):
            coord = min(int(round(idx * cell_spacing_px)), size - 1)
            if coord == prev_coord:
                continue
            prev_coord = coord
            color = accent_grid_color if idx % 2 == 0 else base_grid_color
            yield coord, color

    for x, color in grid_positions():
        draw.line([(x, 0), (x, size)], fill=color, width=1)

    for y, color in grid_positions():
        draw.line([(0, y), (size, y)], fill=color, width=1)

    border_color = (53, 140, 199)
    draw.rectangle([(1, 1), (size - 2, size - 2)], outline=border_color, width=4)
    draw.text((16, 12), "N", fill=(120, 199, 255))

    return background


BASE_SECTOR_IMAGE = _make_sector_background()
BASE_SECTOR_ARRAY = np.array(BASE_SECTOR_IMAGE)


def _compute_wedge_points(
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
    center_angle: float,
    angle_span: float,
    steps: int = 48,
):
    half_span = angle_span / 2
    if steps < 2:
        steps = 2

    start_angle = center_angle - half_span
    end_angle = center_angle + half_span
    angle_range = end_angle - start_angle

    points = []
    for i in range(steps + 1):
        angle = start_angle + (angle_range * i / steps)
        points.append(
            (
                center_x + outer_radius * math.cos(angle),
                center_y - outer_radius * math.sin(angle),
            )
        )

    for i in range(steps, -1, -1):
        angle = start_angle + (angle_range * i / steps)
        points.append(
            (
                center_x + inner_radius * math.cos(angle),
                center_y - inner_radius * math.sin(angle),
            )
        )

    return points


def _summarize_obstacles(
    radar_x: int,
    radar_y: int,
    obstacles: Iterable[Dict[str, float]],
) -> list[Dict[str, float]]:
    summary = []
    for obstacle in obstacles:
        obstacle_x = obstacle["x"]
        obstacle_y = obstacle["y"]
        dx_px = obstacle_x - radar_x
        dy_px = obstacle_y - radar_y
        distance_px = math.hypot(dx_px, dy_px)
        if distance_px < 1:
            continue
        distance_km = distance_px * KM_PER_PX
        width_km = obstacle["width_km"] * obstacle.get("spread_factor", 1.0)
        half_angle = math.atan2(max(width_km, 0.01) / 2, max(distance_km, 0.01))
        angle_span = max(half_angle * 2, math.radians(1))
        center_angle = math.atan2(-dy_px, dx_px)
        bearing = (math.degrees(center_angle) + 360) % 360
        summary.append(
            {
                "x": obstacle_x,
                "y": obstacle_y,
                "width_km": width_km,
                "distance_px": distance_px,
                "distance_km": distance_km,
                "dx_px": dx_px,
                "dy_px": dy_px,
                "angle_span": angle_span,
                "center_angle": center_angle,
                "bearing": bearing,
            }
        )
    return summary


def _compute_dead_zones(
    position: Optional[Tuple[int, int]],
    radar_range_km: float,
    obstacles: Iterable[Dict[str, float]],
) -> list[Dict[str, float]]:
    if position is None:
        return []

    radar_x, radar_y = position
    obstacle_vectors = _summarize_obstacles(radar_x, radar_y, obstacles)
    windows = []
    for obstacle in obstacle_vectors:
        expanded_span = min(obstacle["angle_span"] * 1.2, math.radians(90))
        clearance_km = obstacle["distance_km"] + obstacle["width_km"] / 2
        for start, end in _angle_windows(obstacle["center_angle"], expanded_span):
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "distance_km": clearance_km,
                    "width_km": obstacle["width_km"],
                }
            )

    merged_windows = _merge_angle_intervals(windows)
    zones: list[Dict[str, float]] = []
    for window in merged_windows:
        span = window["end"] - window["start"]
        inner_radius = window["distance_km"] + SUPPORT_RADAR_STANDOFF_KM
        outer_radius = max(inner_radius + SUPPORT_RADAR_DEFAULT_RANGE_KM, radar_range_km)
        area = 0.5 * max(outer_radius**2 - inner_radius**2, 0) * span
        zones.append(
            {
                "start": window["start"],
                "end": window["end"],
                "distance_km": window["distance_km"],
                "inner_radius": inner_radius,
                "outer_radius": outer_radius,
                "area": area,
            }
        )
    if not zones:
        return []

    max_outer_radius = max(zone["outer_radius"] for zone in zones) + SUPPORT_GRID_STEP_KM
    grid_cells = _build_sector_grid_metadata(radar_x, radar_y, max_outer_radius)
    for zone in zones:
        zone_cells = _collect_zone_grid_cells(zone, grid_cells)
        if not zone_cells:
            zone_cells = [_synthesize_zone_cell(zone, radar_x, radar_y)]
        zone["grid_cells"] = zone_cells
    return zones


def _build_coverage_matrix(
    position: Tuple[int, int],
    range_km: float,
    obstacles: Optional[Iterable[Dict[str, float]]] = None,
    dead_zones: Optional[Iterable[Dict[str, float]]] = None,
) -> np.ndarray:
    if position is None:
        raise ValueError("Radar position required to compute coverage matrix.")

    radar_x, radar_y = position
    obstacle_list = list(obstacles or [])
    zones = list(dead_zones or []) or _compute_dead_zones(position, range_km, obstacle_list)
    simplified_zones = [
        {
            "start": zone["start"],
            "end": zone["end"],
            "distance_km": zone["distance_km"],
        }
        for zone in zones
    ]

    matrix = np.zeros((GRID_CELLS_PER_SIDE, GRID_CELLS_PER_SIDE), dtype=np.uint8)
    for row, cell_center_y in enumerate(GRID_CELL_CENTERS_PX):
        dy_px = cell_center_y - radar_y
        for col, cell_center_x in enumerate(GRID_CELL_CENTERS_PX):
            dx_px = cell_center_x - radar_x
            distance_km = math.hypot(dx_px, dy_px) * KM_PER_PX
            if distance_km > range_km:
                matrix[row, col] = 1
                continue
            angle = math.atan2(-dy_px, dx_px)
            blocked = False
            for zone in simplified_zones:
                if distance_km < zone["distance_km"]:
                    continue
                if _angle_in_interval(angle, zone["start"], zone["end"]):
                    blocked = True
                    break
            if not blocked:
                matrix[row, col] = 1
    return matrix


def _build_sector_grid_metadata(
    radar_x: int, radar_y: int, max_distance_km: float
) -> list[Dict[str, float]]:
    step_px = max(int(round(SUPPORT_GRID_STEP_KM / KM_PER_PX)), 1)
    start_offset = max(step_px // 2, 0)
    cells: list[Dict[str, float]] = []
    for y in range(start_offset, CANVAS_PX, step_px):
        for x in range(start_offset, CANVAS_PX, step_px):
            cx = min(x, CANVAS_PX - 1)
            cy = min(y, CANVAS_PX - 1)
            dx_px = cx - radar_x
            dy_px = cy - radar_y
            distance_km = math.hypot(dx_px, dy_px) * KM_PER_PX
            if max_distance_km and distance_km > max_distance_km:
                continue
            angle = math.atan2(-dy_px, dx_px)
            cells.append(
                {
                    "pixel_x": cx,
                    "pixel_y": cy,
                    "distance_km": distance_km,
                    "angle": angle % (2 * math.pi),
                    "bearing": (math.degrees(angle) + 360) % 360,
                    "cell_key": f"{cx}_{cy}",
                }
            )
    return cells


def _angle_in_interval(angle: float, start: float, end: float) -> bool:
    two_pi = 2 * math.pi
    angle = angle % two_pi
    if start <= angle <= end:
        return True
    angle_shifted = angle + two_pi
    return start <= angle_shifted <= end


def _collect_zone_grid_cells(
    zone: Dict[str, float], grid_cells: Iterable[Dict[str, float]]
) -> list[Dict[str, float]]:
    zone_cells: list[Dict[str, float]] = []
    start = zone["start"]
    end = zone["end"]
    inner_limit = max(zone["distance_km"], zone["inner_radius"])
    outer_limit = zone["outer_radius"] + SUPPORT_GRID_MARGIN_KM
    for cell in grid_cells:
        if cell["distance_km"] + SUPPORT_GRID_MARGIN_KM < inner_limit:
            continue
        if cell["distance_km"] > outer_limit:
            continue
        if not _angle_in_interval(cell["angle"], start, end):
            continue
        zone_cells.append(
            {
                "pixel_x": cell["pixel_x"],
                "pixel_y": cell["pixel_y"],
                "distance_km": cell["distance_km"],
                "bearing": cell["bearing"],
                "cell_key": cell["cell_key"],
            }
        )
    return zone_cells


def _synthesize_zone_cell(zone: Dict[str, float], radar_x: int, radar_y: int) -> Dict[str, float]:
    center_angle = (zone["start"] + (zone["end"] - zone["start"]) / 2) % (2 * math.pi)
    radius = (zone["inner_radius"] + zone["outer_radius"]) / 2
    delta_px = radius / KM_PER_PX
    target_x = int(round(radar_x + delta_px * math.cos(center_angle)))
    target_y = int(round(radar_y - delta_px * math.sin(center_angle)))
    target_x = max(0, min(CANVAS_PX - 1, target_x))
    target_y = max(0, min(CANVAS_PX - 1, target_y))
    distance_km = math.hypot(target_x - radar_x, target_y - radar_y) * KM_PER_PX
    return {
        "pixel_x": target_x,
        "pixel_y": target_y,
        "distance_km": distance_km,
        "bearing": (math.degrees(center_angle) + 360) % 360,
        "cell_key": f"{target_x}_{target_y}",
    }

def _render_coverage_at(
    x: int,
    y: int,
    radar_name: str,
    range_km: float,
    obstacles: Optional[Iterable[Dict[str, float]]] = None,
    support_origin: Optional[Tuple[int, int]] = None,
):
    radius_px = int(max(range_km, 2.5) / KM_PER_PX)
    bbox = (x - radius_px, y - radius_px, x + radius_px, y + radius_px)

    base = Image.fromarray(BASE_SECTOR_ARRAY).convert("RGBA")
    overlay_size = (base.width * COVERAGE_DRAW_SCALE, base.height * COVERAGE_DRAW_SCALE)
    overlay = Image.new("RGBA", overlay_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    coverage_fill = (24, 200, 255, 70)
    coverage_outline = (79, 185, 255, 255)

    scaled_bbox = tuple(coord * COVERAGE_DRAW_SCALE for coord in bbox)
    draw.ellipse(scaled_bbox, fill=coverage_fill, outline=coverage_outline, width=4)

    obstacle_list = list(obstacles) if obstacles else []
    obstacle_vectors = _summarize_obstacles(x, y, obstacle_list)
    if obstacle_vectors:
        dead_zone_color = (255, 120, 120, 140)
        scaled_center_x = x * COVERAGE_DRAW_SCALE
        scaled_center_y = y * COVERAGE_DRAW_SCALE
        scaled_radius_px = radius_px * COVERAGE_DRAW_SCALE

        for obstacle in obstacle_vectors:
            if obstacle["distance_km"] >= range_km:
                continue
            start_radius = obstacle["distance_px"] * COVERAGE_DRAW_SCALE
            wedge_points = _compute_wedge_points(
                center_x=scaled_center_x,
                center_y=scaled_center_y,
                inner_radius=start_radius,
                outer_radius=scaled_radius_px,
                center_angle=obstacle["center_angle"],
                angle_span=obstacle["angle_span"],
                steps=max(12, int(obstacle["angle_span"] / math.radians(3))),
            )
            draw.polygon(wedge_points, fill=dead_zone_color)

    if support_origin:
        origin_x, origin_y = support_origin
        truck_width = 36 * COVERAGE_DRAW_SCALE
        truck_height = 18 * COVERAGE_DRAW_SCALE
        cab_length = int(truck_width * 0.35)
        scaled_origin_x = origin_x * COVERAGE_DRAW_SCALE
        scaled_origin_y = origin_y * COVERAGE_DRAW_SCALE
        body_box = (
            scaled_origin_x - truck_width // 2,
            scaled_origin_y - truck_height // 2,
            scaled_origin_x + truck_width // 2,
            scaled_origin_y + truck_height // 2,
        )
        cab_box = (
            body_box[0],
            body_box[1],
            body_box[0] + cab_length,
            body_box[3],
        )
        draw.rectangle(body_box, fill=(40, 60, 96, 255), outline=(140, 210, 255, 255), width=3)
        draw.rectangle(cab_box, fill=(30, 45, 80, 255))
        wheel_radius = 4 * COVERAGE_DRAW_SCALE
        for offset in (-truck_width // 3, truck_width // 4):
            wheel_cx = scaled_origin_x + offset
            wheel_cy = scaled_origin_y + truck_height // 2 + wheel_radius
            draw.ellipse(
                (
                    wheel_cx - wheel_radius,
                    wheel_cy - wheel_radius,
                    wheel_cx + wheel_radius,
                    wheel_cy + wheel_radius,
                ),
                fill=(15, 15, 20, 255),
                outline=(110, 190, 255, 255),
                width=1,
            )

    radar_center_color = (255, 92, 92, 255)
    dot_radius = 6 * COVERAGE_DRAW_SCALE
    draw.ellipse(
        (
            x * COVERAGE_DRAW_SCALE - dot_radius,
            y * COVERAGE_DRAW_SCALE - dot_radius,
            x * COVERAGE_DRAW_SCALE + dot_radius,
            y * COVERAGE_DRAW_SCALE + dot_radius,
        ),
        fill=radar_center_color,
    )

    overlay = overlay.resize(base.size, RESAMPLING_FILTER)

    combined = Image.alpha_composite(base, overlay)
    obstacle_note = ""
    if obstacle_list:
        obstacle_note = f" · {len(obstacle_list)} obstacle{'s' if len(obstacle_list) != 1 else ''}"
    info = (
        f"{radar_name} deployed at pixel ({x}, {y}) → coverage radius ~{int(range_km)} km"
        f"{obstacle_note}"
    )

    return np.array(combined.convert("RGB")), info


def place_radar(
    radar_name: str,
    range_km: float,
    interaction_mode: str,
    position: Optional[Tuple[int, int]],
    obstacles: Optional[Iterable[Dict[str, float]]],
    support_origin: Optional[Tuple[int, int]],
    dead_zones: Optional[Iterable[Dict[str, float]]],
    evt: gr.SelectData,
):
    if evt is None or evt.index is None:
        raise gr.Error("Click inside the sector to position the radar or obstacle.")

    obstacles_list = list(obstacles or [])
    origin = support_origin

    if interaction_mode == "obstacle":
        if position is None:
            raise gr.Error("Place the radar before defining obstacles.")
        x, y = evt.index
        dx_px = x - position[0]
        dy_px = y - position[1]
        distance_km = math.hypot(dx_px, dy_px) * KM_PER_PX
        if distance_km == 0:
            raise gr.Error("Click away from the radar to insert an obstacle.")

        width_km = random.uniform(OBSTACLE_BASE_WIDTH_KM, OBSTACLE_BASE_WIDTH_KM * 5)
        spread_factor = random.uniform(1.0, 3.0)
        obstacles_list.append(
            {"x": x, "y": y, "width_km": width_km, "spread_factor": spread_factor}
        )
        image, coverage_info = _render_coverage_at(
            position[0], position[1], radar_name, range_km, obstacles_list, origin
        )
        bearing = (math.degrees(math.atan2(-(dy_px), dx_px)) + 360) % 360
        placement_msg = (
            f"{coverage_info}\n"
            f"Obstacle logged {distance_km:.1f} km out "
            f"(bearing {bearing:.0f}°, width≈{width_km * 1000:.0f} m)."
        )
        dead_zones = _compute_dead_zones(position, range_km, obstacles_list)
        return image, placement_msg, position, "radar", obstacles_list, origin, dead_zones

    if interaction_mode == "origin":
        if position is None:
            raise gr.Error("Place the radar before choosing the BaseTruck location.")
        origin = evt.index
        image, coverage_info = _render_coverage_at(
            position[0], position[1], radar_name, range_km, obstacles_list, origin
        )
        placement_msg = f"{coverage_info}\nBaseTruck waypoint anchored at pixel {origin}."
        dead_zones = _compute_dead_zones(position, range_km, obstacles_list)
        return image, placement_msg, position, "radar", obstacles_list, origin, dead_zones

    x, y = evt.index
    image, info = _render_coverage_at(x, y, radar_name, range_km, obstacles_list, origin)
    if origin is None:
        origin = (x, y)
    dead_zones = _compute_dead_zones((x, y), range_km, obstacles_list)
    return image, info, (x, y), interaction_mode, obstacles_list, origin, dead_zones


def redraw_radar(
    radar_name: str,
    range_km: float,
    position: Optional[Tuple[int, int]],
    obstacles: Optional[Iterable[Dict[str, float]]],
    support_origin: Optional[Tuple[int, int]],
):
    if position is None:
        return BASE_SECTOR_ARRAY, "Select a location to visualize coverage."

    x, y = position
    return _render_coverage_at(x, y, radar_name, range_km, obstacles, support_origin)


def set_radar_defaults(radar_name: str):
    preset = RADAR_PRESETS[radar_name]
    info = (
        f"{preset.description}\n\n"
        f"Default radius: {preset.range_km} km"
    )
    return preset.range_km, info


def activate_obstacle_mode(
    position: Optional[Tuple[int, int]],
):
    if position is None:
        raise gr.Error("Place the radar before adding obstacles.")

    info = (
        f"Obstacle placement mode: click the map to drop a ~{OBSTACLE_BASE_WIDTH_KM * 1000:.0f}–"
        f"{OBSTACLE_BASE_WIDTH_KM * 5 * 1000:.0f} m blocker."
        " Coverage redraws once placed."
    )
    return "obstacle", info


def activate_origin_mode(position: Optional[Tuple[int, int]]):
    if position is None:
        raise gr.Error("Place the radar before selecting a BaseTruck origin.")

    return "origin", "Click on the map to set the BaseTruck launch point."


def scatter_random_obstacles(
    radar_name: str,
    range_km: float,
    scatter_count: int,
    position: Optional[Tuple[int, int]],
    obstacles: Optional[Iterable[Dict[str, float]]],
    support_origin: Optional[Tuple[int, int]],
):
    if position is None:
        raise gr.Error("Place the radar before scattering obstacles.")

    count = max(int(scatter_count), 0)
    if count <= 0:
        raise gr.Error("Set a scatter count above zero.")

    obstacles_list: list[Dict[str, float]] = []
    radar_x, radar_y = position

    min_distance_km = MIN_SCATTER_DISTANCE_KM
    max_distance_km = max(range_km - 0.5, min_distance_km + 0.1)

    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        distance_km = random.uniform(min_distance_km, max_distance_km)
        dx_px = math.cos(angle) * (distance_km / KM_PER_PX)
        dy_px = math.sin(angle) * (distance_km / KM_PER_PX)
        obstacle_x = int(round(radar_x + dx_px))
        obstacle_y = int(round(radar_y - dy_px))
        obstacle_x = max(0, min(CANVAS_PX - 1, obstacle_x))
        obstacle_y = max(0, min(CANVAS_PX - 1, obstacle_y))
        width_km = random.uniform(OBSTACLE_BASE_WIDTH_KM, OBSTACLE_BASE_WIDTH_KM * 5)
        spread_factor = random.uniform(1.0, 3.0)
        obstacles_list.append(
            {
                "x": obstacle_x,
                "y": obstacle_y,
                "width_km": width_km,
                "spread_factor": spread_factor,
            }
        )

    image, coverage_info = _render_coverage_at(
        radar_x,
        radar_y,
        radar_name,
        range_km,
        obstacles_list,
        support_origin,
    )
    placement_msg = (
        f"{coverage_info}\n"
        f"Scattered {count} obstacle{'s' if count != 1 else ''} "
        f"(widths vary ~{OBSTACLE_BASE_WIDTH_KM * 1000:.0f}–{OBSTACLE_BASE_WIDTH_KM * 5 * 1000:.0f} m)."
    )
    dead_zones = _compute_dead_zones(position, range_km, obstacles_list)
    return image, placement_msg, obstacles_list, dead_zones


def clear_obstacles(
    radar_name: str,
    range_km: float,
    position: Optional[Tuple[int, int]],
    support_origin: Optional[Tuple[int, int]],
):
    if position is None:
        return BASE_SECTOR_ARRAY, "Obstacles cleared. Place the radar to resume.", [], []

    image, info = _render_coverage_at(
        position[0],
        position[1],
        radar_name,
        range_km,
        [],
        support_origin,
    )
    return image, f"Obstacles cleared. {info}", [], []


def export_coverage_data(
    radar_name: str,
    range_km: float,
    position: Optional[Tuple[int, int]],
    obstacles: Optional[Iterable[Dict[str, float]]],
    dead_zones: Optional[Iterable[Dict[str, float]]],
):
    if position is None:
        raise gr.Error("Place the radar before exporting coverage data.")

    matrix = _build_coverage_matrix(position, range_km, obstacles, dead_zones)
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv",
        prefix="coverage_matrix_",
        mode="w",
    )
    with temp_file:
        np.savetxt(temp_file, matrix, fmt="%d", delimiter=",")
    return gr.update(value=temp_file.name, visible=True)


def _angle_windows(center_angle: float, angle_span: float) -> list[Tuple[float, float]]:
    two_pi = 2 * math.pi
    span = min(angle_span, two_pi)
    start = (center_angle - span / 2) % two_pi
    end = start + span
    if end <= two_pi:
        return [(start, end)]
    return [(start, two_pi), (0.0, end - two_pi)]


def _merge_angle_intervals(intervals: list[Dict[str, float]]) -> list[Dict[str, float]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda item: item["start"])
    merged = [sorted_intervals[0].copy()]
    for interval in sorted_intervals[1:]:
        current = merged[-1]
        if interval["start"] <= current["end"] + ANGLE_MERGE_THRESHOLD:
            current["end"] = max(current["end"], interval["end"])
            current["distance_km"] = max(current["distance_km"], interval["distance_km"])
            current["width_km"] = max(current["width_km"], interval["width_km"])
        else:
            merged.append(interval.copy())

    two_pi = 2 * math.pi
    if len(merged) >= 2:
        first = merged[0]
        last = merged[-1]
        if first["start"] <= ANGLE_MERGE_THRESHOLD and (two_pi - last["end"]) <= ANGLE_MERGE_THRESHOLD:
            first["start"] = (last["start"] + first["start"]) - two_pi
            first["distance_km"] = max(first["distance_km"], last["distance_km"])
            first["width_km"] = max(first["width_km"], last["width_km"])
            merged.pop()

    # Normalise back into [0, 2π)
    for interval in merged:
        interval["start"] %= two_pi
        interval["end"] %= two_pi
        if interval["end"] <= interval["start"]:
            interval["end"] += two_pi
    return merged


def _plan_support_radar_positions(
    position: Tuple[int, int],
    radar_range_km: float,
    support_range_km: float,
    obstacles: Iterable[Dict[str, float]],
    dead_zones: Optional[Iterable[Dict[str, float]]] = None,
) -> list[Dict[str, float]]:
    radar_x, radar_y = position
    if dead_zones is None or not dead_zones:
        dead_zones = _compute_dead_zones(position, radar_range_km, obstacles)

    support_radars: list[Dict[str, float]] = []
    seen_cells: set[str] = set()

    for zone in dead_zones:
        if len(support_radars) >= SUPPORT_RADAR_MAX:
            break
        zone_cells = list(zone.get("grid_cells", []))
        zone_cells.sort(key=lambda cell: (cell["distance_km"], cell["bearing"]))
        for cell in zone_cells:
            if len(support_radars) >= SUPPORT_RADAR_MAX:
                break
            cell_key = cell.get("cell_key") or f"{cell['pixel_x']}_{cell['pixel_y']}"
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)
            support_radars.append(
                {
                    "pixel_x": cell["pixel_x"],
                    "pixel_y": cell["pixel_y"],
                    "bearing": cell["bearing"],
                    "distance_km": cell["distance_km"],
                }
            )

    if support_radars:
        return support_radars

    return _legacy_support_radar_plan(position, radar_range_km, support_range_km, obstacles, dead_zones)


def _legacy_support_radar_plan(
    position: Tuple[int, int],
    radar_range_km: float,
    support_range_km: float,
    obstacles: Iterable[Dict[str, float]],
    dead_zones: Optional[Iterable[Dict[str, float]]],
) -> list[Dict[str, float]]:
    """Fallback tiling based on the original layered polar layout."""

    radar_x, radar_y = position
    if dead_zones is None or not dead_zones:
        dead_zones = _compute_dead_zones(position, radar_range_km, obstacles)

    support_area = math.pi * (support_range_km * SUPPORT_COVERAGE_SCALE) ** 2 * 0.65
    support_radars: list[Dict[str, float]] = []
    two_pi = 2 * math.pi

    for zone in dead_zones:
        zone_span = zone["end"] - zone["start"]
        inner_radius = zone["inner_radius"]
        outer_radius = zone["outer_radius"]
        zone_area = zone["area"]
        required_units = max(3, int(math.ceil(zone_area / max(support_area, 1e-6))))
        added = 0

        radial_depth = max(outer_radius - inner_radius, support_range_km)
        layer_spacing = support_range_km * SUPPORT_LAYER_OVERLAP
        num_layers = max(3, int(math.ceil(radial_depth / layer_spacing)))

        for layer_idx in range(num_layers):
            if len(support_radars) >= SUPPORT_RADAR_MAX or added >= required_units:
                break

            radius = inner_radius + layer_spacing * layer_idx
            radius = min(radius, outer_radius)
            radius = max(radius, inner_radius + support_range_km * 0.1)

            coverage_angle = 2 * math.asin(min(1.0, support_range_km / max(radius, support_range_km)))
            spacing_angle = max(coverage_angle * SUPPORT_ANGULAR_OVERLAP, math.radians(4))
            per_layer = max(3, int(math.ceil(zone_span / spacing_angle)))

            for idx in range(per_layer):
                if len(support_radars) >= SUPPORT_RADAR_MAX or added >= required_units:
                    break
                frac = (idx + 0.5) / per_layer
                center_angle = (zone["start"] + zone_span * frac) % two_pi
                delta_px = radius / KM_PER_PX
                target_x = int(round(radar_x + delta_px * math.cos(center_angle)))
                target_y = int(round(radar_y - delta_px * math.sin(center_angle)))
                target_x = max(0, min(CANVAS_PX - 1, target_x))
                target_y = max(0, min(CANVAS_PX - 1, target_y))
                distance_km = math.hypot(target_x - radar_x, target_y - radar_y) * KM_PER_PX
                support_radars.append(
                    {
                        "pixel_x": target_x,
                        "pixel_y": target_y,
                        "bearing": (math.degrees(center_angle) + 360) % 360,
                        "distance_km": distance_km,
                    }
                )
                added += 1

    return support_radars


def _build_support_animation_frames(
    base_image: np.ndarray,
    launch_position: Tuple[int, int],
    support_radars: Iterable[Dict[str, float]],
    support_range_km: float,
) -> list[np.ndarray]:
    launch_x, launch_y = launch_position
    frames: list[np.ndarray] = []
    support_stroke = (62, 192, 126, 255)
    support_fill = (60, 255, 180, 24)
    support_outline = (44, 180, 120, 120)

    for frame_idx in range(SUPPORT_ANIMATION_FRAMES):
        progress = min(1.0, (frame_idx + 1) / SUPPORT_ANIMATION_FRAMES)
        base = Image.fromarray(base_image).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        for support in support_radars:
            target_x = support["pixel_x"]
            target_y = support["pixel_y"]
            current_x = launch_x + (target_x - launch_x) * progress
            current_y = launch_y + (target_y - launch_y) * progress
            dot_radius = 2
            draw.ellipse(
                (
                    current_x - dot_radius,
                    current_y - dot_radius,
                    current_x + dot_radius,
                    current_y + dot_radius,
                ),
                fill=support_stroke,
                outline=support_outline,
                width=2,
            )
            if progress >= 1.0:
                radius_px = int(max(support_range_km * SUPPORT_COVERAGE_SCALE, 0.1) / KM_PER_PX)
                draw.ellipse(
                    (
                        target_x - radius_px,
                        target_y - radius_px,
                        target_x + radius_px,
                        target_y + radius_px,
                    ),
                    fill=support_fill,
                    outline=support_outline,
                    width=3,
                )
        composed = Image.alpha_composite(base, overlay)
        frames.append(np.array(composed.convert("RGB")))

    return frames


def deploy_support_radars(
    radar_name: str,
    range_km: float,
    position: Optional[Tuple[int, int]],
    obstacles: Optional[Iterable[Dict[str, float]]],
    support_origin: Optional[Tuple[int, int]],
    dead_zones: Optional[Iterable[Dict[str, float]]],
):
    if position is None:
        raise gr.Error("Place the primary radar before deploying support units.")

    obstacles_list = list(obstacles or [])
    if not obstacles_list:
        raise gr.Error("Log at least one obstacle to identify blind spots.")

    radar_x, radar_y = position
    support_range_km = SUPPORT_RADAR_DEFAULT_RANGE_KM
    launch_point = support_origin or position
    support_targets = _plan_support_radar_positions(
        (radar_x, radar_y), range_km, support_range_km, obstacles_list, dead_zones
    )
    if not support_targets:
        raise gr.Error("No blind sectors found within the active coverage radius.")
    if support_origin is None:
        raise gr.Error("Select the BaseTruck location before deploying support radars.")

    base_image, _ = _render_coverage_at(
        radar_x,
        radar_y,
        radar_name,
        range_km,
        obstacles_list,
        support_origin,
    )
    animation_frames = _build_support_animation_frames(
        base_image, launch_point, support_targets, support_range_km
    )

    summary_lines = [
        f"Deploy {len(support_targets)} support radar{'s' if len(support_targets) != 1 else ''} "
        f"({support_range_km:.1f} km coverage each)."
    ]
    for idx, target in enumerate(support_targets, 1):
        summary_lines.append(
            f"{idx}. Pixel ({target['pixel_x']}, {target['pixel_y']}) "
            f"· {target['distance_km']:.1f} km from HQ · bearing {target['bearing']:.0f}°"
        )
    summary_lines.append("Re-run after changing obstacles or primary radar settings.")
    plan_text = "Support deployment plan:\n" + "\n".join(summary_lines)

    for frame in animation_frames:
        yield frame, plan_text
        time.sleep(SUPPORT_ANIMATION_DELAY_S)


def build_interface() -> gr.Blocks:
    default_range, default_info = set_radar_defaults("R42")

    with gr.Blocks(title="Radar Coverage Sandbox") as demo:
        default_radar_position = (CANVAS_PX // 2, CANVAS_PX // 2)
        base_radar_name = "R42"
        initial_map, initial_message = _render_coverage_at(
            default_radar_position[0],
            default_radar_position[1],
            base_radar_name,
            default_range,
        )

        placement_state = gr.State(value=default_radar_position)
        obstacles_state = gr.State(value=[])
        interaction_mode = gr.State(value="radar")
        support_origin_state = gr.State(value=None)
        dead_zones_state = gr.State(
            value=_compute_dead_zones(default_radar_position, default_range, [])
        )

        gr.Markdown(
            """
            ### Sector Watch - Radar Coverage Prototype
            1. Pick a radar preset below, adjust its radius, and then click inside the sector map.  
            2. Use "Add obstacle marker" to log buildings or terrain that block coverage beyond a given distance.  
            3. Each radar click shades omnidirectional coverage; obstacle clicks carve out blind sectors beyond the obstruction.
            4. Use "Scatter random obstacles" to auto-place multiple blockers for quick what-if runs.  
            5. Lock in the BaseTruck waypoint before deploying support radars so the convoy launches from the intended spot.  
            6. When blind spots exist, click "Deploy support radars" to animate the recommended covering maneuver.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                radar_dropdown = gr.Dropdown(
                    choices=list(RADAR_PRESETS), value="R42", label="Radar preset"
                )
                range_slider = gr.Slider(
                    minimum=5,
                    maximum=80,
                    value=default_range,
                    step=1,
                    label="Coverage radius (km)",
                )
                preset_info = gr.Markdown(value=default_info)
                scatter_count_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=8,
                    step=1,
                    label="Random obstacles per scatter",
                )
                obstacle_button = gr.Button("Add obstacle marker", variant="secondary")
                scatter_button = gr.Button("Scatter random obstacles", variant="secondary")
                clear_button = gr.Button("Clear obstacles", variant="secondary")
                export_button = gr.Button("Export coverage matrix", variant="secondary")
                export_file = gr.File(
                    label="Coverage matrix CSV",
                    interactive=False,
                    visible=False,
                    file_count="single",
                )
                origin_button = gr.Button("Set BaseTruck location", variant="secondary")
                deploy_button = gr.Button("Deploy support radars", variant="primary")

            with gr.Column(scale=2):
                placement_info = gr.Markdown(
                    f"{initial_message}\n\nClick anywhere in the sector to reposition."
                )
                map_image = gr.Image(
                    value=initial_map,
                    label="Operational sector (~33 km × ~33 km)",
                    interactive=True,
                )
                support_plan_info = gr.Markdown(
                    "Support deployment plan pending simulation."
                )
                map_image.select(
                    fn=place_radar,
                    inputs=[
                        radar_dropdown,
                        range_slider,
                        interaction_mode,
                        placement_state,
                        obstacles_state,
                        support_origin_state,
                        dead_zones_state,
                    ],
                    outputs=[
                        map_image,
                        placement_info,
                        placement_state,
                        interaction_mode,
                        obstacles_state,
                        support_origin_state,
                        dead_zones_state,
                    ],
                )

        def handle_radar_change(
            radar_name: str,
            position: Optional[Tuple[int, int]],
            obstacles: Optional[Iterable[Dict[str, float]]],
            support_origin: Optional[Tuple[int, int]],
        ):
            range_km, info = set_radar_defaults(radar_name)
            image, placement_msg = redraw_radar(
                radar_name, range_km, position, obstacles, support_origin
            )
            dead_zones = _compute_dead_zones(position, range_km, obstacles or [])
            return range_km, info, image, placement_msg, dead_zones

        radar_dropdown.change(
            fn=handle_radar_change,
            inputs=[radar_dropdown, placement_state, obstacles_state, support_origin_state],
            outputs=[range_slider, preset_info, map_image, placement_info, dead_zones_state],
        )

        obstacle_button.click(
            fn=activate_obstacle_mode,
            inputs=[placement_state],
            outputs=[interaction_mode, placement_info],
        )

        scatter_button.click(
            fn=scatter_random_obstacles,
            inputs=[
                radar_dropdown,
                range_slider,
                scatter_count_slider,
                placement_state,
                obstacles_state,
                support_origin_state,
            ],
            outputs=[map_image, placement_info, obstacles_state, dead_zones_state],
        )

        clear_button.click(
            fn=clear_obstacles,
            inputs=[radar_dropdown, range_slider, placement_state, support_origin_state],
            outputs=[map_image, placement_info, obstacles_state, dead_zones_state],
        )

        export_button.click(
            fn=export_coverage_data,
            inputs=[radar_dropdown, range_slider, placement_state, obstacles_state, dead_zones_state],
            outputs=[export_file],
        )

        origin_button.click(
            fn=activate_origin_mode,
            inputs=[placement_state],
            outputs=[interaction_mode, placement_info],
        )

        deploy_button.click(
            fn=deploy_support_radars,
            inputs=[
                radar_dropdown,
                range_slider,
                placement_state,
                obstacles_state,
                support_origin_state,
                dead_zones_state,
            ],
            outputs=[map_image, support_plan_info],
        )

        def handle_range_change(
            radar_name: str,
            range_km: float,
            position: Optional[Tuple[int, int]],
            obstacles: Optional[Iterable[Dict[str, float]]],
            support_origin: Optional[Tuple[int, int]],
        ):
            image, placement_msg = redraw_radar(
                radar_name, range_km, position, obstacles, support_origin
            )
            dead_zones = _compute_dead_zones(position, range_km, obstacles or [])
            return image, placement_msg, dead_zones

        range_slider.change(
            fn=handle_range_change,
            inputs=[radar_dropdown, range_slider, placement_state, obstacles_state, support_origin_state],
            outputs=[map_image, placement_info, dead_zones_state],
        )

    return demo


def main():
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
