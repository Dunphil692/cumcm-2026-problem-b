"""楔形交会定位与多边形直径（半平面交，问题四复用）。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from typing import Iterable

Point = tuple[float, float]
HalfPlane = tuple[float, float, float]


def _cross(origin: Point, first: Point, second: Point) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _angular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _build_halfplanes(
    measurements: list[tuple[float, float, float]], error_deg: float
) -> list[HalfPlane]:
    halfplanes: list[HalfPlane] = []
    for x, y, angle in measurements:
        lower = math.radians(angle - error_deg)
        upper = math.radians(angle + error_deg)

        # cross(lower_direction, P-S) >= 0
        lower_x, lower_y = math.cos(lower), math.sin(lower)
        halfplanes.append(
            (-lower_y, lower_x, lower_y * x - lower_x * y)
        )

        # cross(upper_direction, P-S) <= 0
        upper_x, upper_y = math.cos(upper), math.sin(upper)
        halfplanes.append(
            (upper_y, -upper_x, -upper_y * x + upper_x * y)
        )
    return halfplanes


def _line_intersection(
    first: HalfPlane, second: HalfPlane, eps: float
) -> Point | None:
    a1, b1, c1 = first
    a2, b2, c2 = second
    determinant = a1 * b2 - a2 * b1
    parallel_tolerance = max(16.0 * math.ulp(1.0), eps * eps)
    if abs(determinant) <= parallel_tolerance:
        return None

    return (
        (b1 * c2 - b2 * c1) / determinant,
        (c1 * a2 - c2 * a1) / determinant,
    )


def _inside_all(point: Point, halfplanes: Iterable[HalfPlane], eps: float) -> bool:
    x, y = point
    for a, b, c in halfplanes:
        value = a * x + b * y + c
        tolerance = eps * max(1.0, abs(x), abs(y), abs(c))
        if value < -tolerance:
            return False
    return True


def _deduplicate(points: Iterable[Point], eps: float) -> list[Point]:
    unique: list[Point] = []
    tolerance = max(1e-7, eps * 100.0)
    for point in points:
        if not any(
            math.dist(point, existing) <= tolerance
            for existing in unique
        ):
            unique.append(point)
    return unique


def _convex_hull(points: list[Point], eps: float) -> list[Point]:
    points = sorted(points)
    if len(points) <= 1:
        return points

    def build_half(ordered: Iterable[Point]) -> list[Point]:
        half: list[Point] = []
        for point in ordered:
            while len(half) >= 2 and _cross(half[-2], half[-1], point) <= eps:
                half.pop()
            half.append(point)
        return half

    lower = build_half(points)
    upper = build_half(reversed(points))
    return lower[:-1] + upper[:-1]


def _has_common_recession_direction(
    angles: list[float], error_deg: float, eps: float
) -> bool:
    angle_tolerance = math.degrees(max(16.0 * math.ulp(1.0), eps * eps))
    boundary_angles = [
        (angle + offset) % 360.0
        for angle in angles
        for offset in (-error_deg, error_deg)
    ]
    return any(
        all(
            _angular_distance(candidate, angle) <= error_deg + angle_tolerance
            for angle in angles
        )
        for candidate in boundary_angles
    )


def _has_reflex_angle(vertices: list[Point], eps: float) -> bool:
    if len(vertices) < 3:
        return False
    return any(
        _cross(vertices[index - 1], vertices[index], vertices[(index + 1) % len(vertices)])
        < -eps
        for index in range(len(vertices))
    )


def _diameter(vertices: list[Point], eps: float) -> tuple[float, list[tuple[Point, Point]]]:
    if len(vertices) == 1:
        return 0.0, [(vertices[0], vertices[0])]

    squared_distances = [
        (
            (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2,
            first,
            second,
        )
        for first, second in itertools.combinations(vertices, 2)
    ]
    maximum_squared = max(item[0] for item in squared_distances)
    tolerance = eps * max(1.0, maximum_squared)
    pairs = [
        (first, second)
        for squared, first, second in squared_distances
        if abs(squared - maximum_squared) <= tolerance
    ]
    return math.sqrt(maximum_squared), pairs


def _circle_covers(
    vertices: list[Point], first: Point, second: Point, eps: float
) -> bool:
    center = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
    radius_squared = (
        (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2
    ) / 4.0
    tolerance = eps * max(1.0, radius_squared)
    return all(
        (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
        <= radius_squared + tolerance
        for point in vertices
    )


def analyze_localization(
    measurements: list[dict[str, float]],
    error_deg: float = 1.0,
    eps: float = 1e-9,
) -> dict[str, object]:
    """分析多个 2° 测向楔形的公共定位区域。"""
    if not measurements:
        raise ValueError("measurements 不能为空")
    if not math.isfinite(error_deg) or error_deg <= 0.0 or error_deg >= 90.0:
        raise ValueError("error_deg 必须在 0 到 90 度之间")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps 必须是正有限数")

    parsed: list[tuple[float, float, float]] = []
    for index, item in enumerate(measurements):
        try:
            x = float(item["x"])
            y = float(item["y"])
            angle = float(item["angle_deg"]) % 360.0
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"第 {index + 1} 个测量必须包含数值 x、y、angle_deg"
            ) from exc
        if not all(math.isfinite(value) for value in (x, y, angle)):
            raise ValueError(f"第 {index + 1} 个测量包含非有限数值")
        parsed.append((x, y, angle))

    halfplanes = _build_halfplanes(parsed, error_deg)
    intersections = []
    for first, second in itertools.combinations(halfplanes, 2):
        point = _line_intersection(first, second, eps)
        if point is not None and _inside_all(point, halfplanes, eps):
            intersections.append(point)

    vertices = _convex_hull(_deduplicate(intersections, eps), eps)
    if not vertices:
        return {
            "status": "empty",
            "vertices": [],
            "diameter": None,
            "diameter_endpoints": None,
            "diameter_circle_covers": None,
            "has_reflex_angle": False,
        }

    if _has_common_recession_direction(
        [angle for _, _, angle in parsed], error_deg, eps
    ):
        return {
            "status": "unbounded",
            "vertices": [[x, y] for x, y in vertices],
            "diameter": "Infinity",
            "diameter_endpoints": None,
            "diameter_circle_covers": None,
            "has_reflex_angle": False,
        }

    diameter, diameter_pairs = _diameter(vertices, eps)
    covering_pair = next(
        (
            pair
            for pair in diameter_pairs
            if _circle_covers(vertices, pair[0], pair[1], eps)
        ),
        None,
    )
    selected_pair = covering_pair or diameter_pairs[0]

    return {
        "status": "bounded",
        "vertices": [[x, y] for x, y in vertices],
        "diameter": diameter,
        "diameter_endpoints": [
            [selected_pair[0][0], selected_pair[0][1]],
            [selected_pair[1][0], selected_pair[1][1]],
        ],
        "diameter_circle_covers": covering_pair is not None,
        "has_reflex_angle": _has_reflex_angle(vertices, eps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据检测点坐标和示向度计算定位区域"
    )
    parser.add_argument("input_json", help="包含 measurements 数组的 JSON 文件")
    args = parser.parse_args()

    try:
        with open(args.input_json, encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("measurements"), list
        ):
            raise ValueError("JSON 顶层必须包含 measurements 数组")
        result = analyze_localization(
            payload["measurements"],
            error_deg=float(payload.get("error_deg", 1.0)),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
