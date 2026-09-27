"""Pareto frontier filtering and hypervolume calculation for multi-objective optimization."""
from __future__ import annotations

from typing import Any

from aloe.optimization.objectives import MultiObjectivePoint


def is_dominated(
    a_vals: dict[str, float],
    b_vals: dict[str, float],
    directions: dict[str, bool],
) -> bool:
    """Check if point `a` is strictly dominated by point `b`.

    Args:
        a_vals: Objective dictionary for point a.
        b_vals: Objective dictionary for point b.
        directions: Mapping of objective name to bool (True = minimize, False = maximize).

    Returns:
        True if b dominates a (b is no worse than a everywhere, and strictly better somewhere).
    """
    not_worse_anywhere = True
    strictly_better_somewhere = False

    for obj_name, minimize in directions.items():
        if obj_name not in a_vals or obj_name not in b_vals:
            continue
        val_a = a_vals[obj_name]
        val_b = b_vals[obj_name]

        if minimize:
            # b should be <= a
            if val_b > val_a:
                not_worse_anywhere = False
                break
            if val_b < val_a:
                strictly_better_somewhere = True
        else:
            # b should be >= a (maximizing)
            if val_b < val_a:
                not_worse_anywhere = False
                break
            if val_b > val_a:
                strictly_better_somewhere = True

    return not_worse_anywhere and strictly_better_somewhere


def extract_pareto_frontier(
    points: list[MultiObjectivePoint],
    directions: dict[str, bool] | None = None,
) -> list[MultiObjectivePoint]:
    """Filter a set of multi-objective points to return only the non-dominated Pareto frontier.

    Args:
        points: List of evaluated points.
        directions: Optional mapping of objective name to bool (True = minimize, False = maximize).
            If None, defaults to: tool_accuracy: False (maximize),
            trainable_parameters: True (minimize), peak_process_rss_mb: True (minimize).

    Returns:
        List of non-dominated MultiObjectivePoint instances.
    """
    if not points:
        return []

    if directions is None:
        directions = {
            "tool_accuracy": False,  # Maximize
            "trainable_parameters": True,  # Minimize
            "peak_process_rss_mb": True,  # Minimize
        }

    pareto_frontier: list[MultiObjectivePoint] = []

    for i, candidate in enumerate(points):
        dominated = False
        for j, opponent in enumerate(points):
            if i == j:
                continue
            if is_dominated(candidate.objectives, opponent.objectives, directions):
                dominated = True
                break
        if not dominated:
            pareto_frontier.append(candidate)

    return pareto_frontier


def calculate_hypervolume_2d(
    points: list[tuple[float, float]],
    reference_point: tuple[float, float],
) -> float:
    """Calculate exact 2D hypervolume dominated by a set of points above a reference point.

    Assumes both objectives are to be MAXIMIZED relative to `reference_point = (r_x, r_y)`.
    If an objective was originally to be minimized, transform it prior to calling
    (e.g., negated or subtracted from an upper bound: `y_transformed = Y_max - y`).

    Args:
        points: List of (x, y) tuples.
        reference_point: (r_x, r_y) reference coordinates (nadir).

    Returns:
        Total 2D hypervolume (area).
    """
    r_x, r_y = reference_point
    # Filter points that do not strictly exceed the reference point
    valid_points = [(x, y) for (x, y) in points if x > r_x and y > r_y]
    if not valid_points:
        return 0.0

    # Sort descending by x
    sorted_points = sorted(valid_points, key=lambda p: (p[0], p[1]), reverse=True)

    # Filter dominated points in 2D (as x decreases, y must strictly increase)
    non_dominated: list[tuple[float, float]] = []
    max_y_seen = r_y
    for x, y in sorted_points:
        if y > max_y_seen:
            non_dominated.append((x, y))
            max_y_seen = y

    # Sort ascending by x
    non_dominated.sort(key=lambda p: p[0])

    hypervolume = 0.0
    for i, (x_curr, y_curr) in enumerate(non_dominated):
        x_prev = reference_point[0] if i == 0 else non_dominated[i - 1][0]
        # Width between current x and previous x
        width = x_curr - x_prev
        # Height relative to reference point
        height = y_curr - reference_point[1]
        hypervolume += width * height

    return round(hypervolume, 6)
