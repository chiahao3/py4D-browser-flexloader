from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Mapping, Sequence

import numpy as np


ROLE_FIXED = "Fixed Index"
ROLE_SCAN_Y = "Scan Y"
ROLE_SCAN_X = "Scan X"
ROLE_FLATTENED_SCAN = "Flattened Scan"
ROLE_DETECTOR_Y = "Detector Y"
ROLE_DETECTOR_X = "Detector X"

DATACUBE_ROLES = (ROLE_SCAN_Y, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X)
ALL_ROLES = (ROLE_FIXED, ROLE_FLATTENED_SCAN, *DATACUBE_ROLES)


@dataclass(frozen=True)
class DimensionSelection:
    roles: tuple[str, ...]
    fixed_indices: tuple[int, ...]
    scan_shapes: tuple[tuple[int, int] | None, ...] = ()


@dataclass(frozen=True)
class PreviewArrays:
    diffraction: np.ndarray
    axial_bf: np.ndarray
    description: str


def default_roles(ndim: int) -> list[str]:
    roles = [ROLE_FIXED] * ndim
    if ndim >= 4:
        start = ndim - 4
        for offset, role in enumerate(DATACUBE_ROLES):
            roles[start + offset] = role
    elif ndim == 3:
        roles = [ROLE_FLATTENED_SCAN, ROLE_DETECTOR_Y, ROLE_DETECTOR_X]
    return roles


def default_scan_shape(size: int) -> tuple[int, int]:
    for scan_y in range(isqrt(size), 0, -1):
        if size % scan_y == 0:
            return scan_y, size // scan_y
    return 1, size


def normalize_scan_shapes(
    ndim: int, scan_shapes: Sequence[tuple[int, int] | None]
) -> tuple[tuple[int, int] | None, ...]:
    if not scan_shapes:
        return tuple(None for _ in range(ndim))
    if len(scan_shapes) != ndim:
        raise ValueError("Scan-shape count must match dataset dimensionality.")
    return tuple(scan_shapes)


def validate_selection(shape: Sequence[int], selection: DimensionSelection) -> None:
    if len(shape) != len(selection.roles):
        raise ValueError("Role count must match dataset dimensionality.")
    if len(shape) != len(selection.fixed_indices):
        raise ValueError("Fixed-index count must match dataset dimensionality.")
    scan_shapes = normalize_scan_shapes(len(shape), selection.scan_shapes)

    assigned = [role for role in selection.roles if role != ROLE_FIXED]
    has_flattened_scan = ROLE_FLATTENED_SCAN in assigned
    required_roles = (
        (ROLE_FLATTENED_SCAN, ROLE_DETECTOR_Y, ROLE_DETECTOR_X)
        if has_flattened_scan
        else DATACUBE_ROLES
    )
    missing = [role for role in required_roles if role not in assigned]
    duplicated = sorted({role for role in assigned if assigned.count(role) > 1})

    if missing:
        raise ValueError(f"Missing required dimension role(s): {', '.join(missing)}.")
    if duplicated:
        raise ValueError(f"Duplicate dimension role(s): {', '.join(duplicated)}.")
    if has_flattened_scan and (
        ROLE_SCAN_Y in assigned or ROLE_SCAN_X in assigned
    ):
        raise ValueError(
            "Use either Flattened Scan or separate Scan Y/Scan X roles, not both."
        )

    for axis, (size, role, fixed_index) in enumerate(
        zip(shape, selection.roles, selection.fixed_indices)
    ):
        if role not in ALL_ROLES:
            raise ValueError(f"Unsupported role for axis {axis}: {role}.")
        if role == ROLE_FIXED and not 0 <= fixed_index < size:
            raise ValueError(
                f"Fixed index {fixed_index} is outside axis {axis} with size {size}."
            )
        if role == ROLE_FLATTENED_SCAN:
            scan_shape = scan_shapes[axis]
            if scan_shape is None:
                raise ValueError("Flattened Scan requires Scan Y and Scan X factors.")
            scan_y, scan_x = scan_shape
            if scan_y <= 0 or scan_x <= 0:
                raise ValueError("Scan Y and Scan X factors must be positive.")
            if scan_y * scan_x != size:
                raise ValueError(
                    f"Scan Y x Scan X must equal flattened scan size {size}."
                )


def load_datacube_array(array, selection: DimensionSelection) -> np.ndarray:
    shape = tuple(int(size) for size in array.shape)
    validate_selection(shape, selection)
    scan_shapes = normalize_scan_shapes(len(shape), selection.scan_shapes)

    indexer = []
    kept_roles = []
    flattened_axis = None
    flattened_scan_shape = None
    for axis, (role, fixed_index) in enumerate(
        zip(selection.roles, selection.fixed_indices)
    ):
        if role == ROLE_FIXED:
            indexer.append(int(fixed_index))
        else:
            indexer.append(slice(None))
            if role == ROLE_FLATTENED_SCAN:
                flattened_axis = len(kept_roles)
                flattened_scan_shape = scan_shapes[axis]
                kept_roles.extend((ROLE_SCAN_Y, ROLE_SCAN_X))
            else:
                kept_roles.append(role)

    subset = np.asarray(array[tuple(indexer)])
    if flattened_axis is not None:
        scan_y, scan_x = flattened_scan_shape
        subset = np.reshape(
            subset,
            (
                *subset.shape[:flattened_axis],
                scan_y,
                scan_x,
                *subset.shape[flattened_axis + 1 :],
            ),
        )
    permutation = [kept_roles.index(role) for role in DATACUBE_ROLES]
    if permutation != list(range(4)):
        subset = np.transpose(subset, permutation)
    return np.asarray(subset)


def load_preview_arrays(array, selection: DimensionSelection) -> PreviewArrays:
    shape = tuple(int(size) for size in array.shape)
    validate_selection(shape, selection)
    scan_shape = _mapped_datacube_shape(shape, selection)
    scan_y, scan_x, detector_y, detector_x = scan_shape

    scan_y_center = scan_y // 2
    scan_x_center = scan_x // 2
    detector_y_center = detector_y // 2
    detector_x_center = detector_x // 2

    diffraction = _read_mapped_slice(
        array,
        selection,
        {
            ROLE_SCAN_Y: scan_y_center,
            ROLE_SCAN_X: scan_x_center,
            ROLE_DETECTOR_Y: slice(None),
            ROLE_DETECTOR_X: slice(None),
        },
        output_roles=(ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
    )
    axial_bf = _read_mapped_slice(
        array,
        selection,
        {
            ROLE_SCAN_Y: slice(None),
            ROLE_SCAN_X: slice(None),
            ROLE_DETECTOR_Y: detector_y_center,
            ROLE_DETECTOR_X: detector_x_center,
        },
        output_roles=(ROLE_SCAN_Y, ROLE_SCAN_X),
    )

    description = (
        f"DP: scan center ({scan_y_center}, {scan_x_center}); "
        f"Axial BF: detector center ({detector_y_center}, {detector_x_center})"
    )
    return PreviewArrays(
        diffraction=np.asarray(diffraction),
        axial_bf=np.asarray(axial_bf),
        description=description,
    )


def _mapped_datacube_shape(
    shape: Sequence[int], selection: DimensionSelection
) -> tuple[int, int, int, int]:
    scan_shapes = normalize_scan_shapes(len(shape), selection.scan_shapes)
    role_sizes: dict[str, int] = {}
    for axis, (size, role) in enumerate(zip(shape, selection.roles)):
        if role == ROLE_FLATTENED_SCAN:
            scan_shape = scan_shapes[axis]
            role_sizes[ROLE_SCAN_Y], role_sizes[ROLE_SCAN_X] = scan_shape
        elif role != ROLE_FIXED:
            role_sizes[role] = int(size)
    return tuple(role_sizes[role] for role in DATACUBE_ROLES)


def _read_mapped_slice(
    array,
    selection: DimensionSelection,
    target_by_role: Mapping[str, int | slice],
    output_roles: Sequence[str],
) -> np.ndarray:
    shape = tuple(int(size) for size in array.shape)
    scan_shapes = normalize_scan_shapes(len(shape), selection.scan_shapes)

    indexer = []
    kept_roles = []
    flattened_axis = None
    flattened_scan_shape = None

    for axis, (role, fixed_index) in enumerate(
        zip(selection.roles, selection.fixed_indices)
    ):
        if role == ROLE_FIXED:
            indexer.append(int(fixed_index))
        elif role == ROLE_FLATTENED_SCAN:
            scan_y_target = target_by_role[ROLE_SCAN_Y]
            scan_x_target = target_by_role[ROLE_SCAN_X]
            _, scan_x = scan_shapes[axis]
            if isinstance(scan_y_target, slice) and isinstance(scan_x_target, slice):
                indexer.append(slice(None))
                flattened_axis = len(kept_roles)
                flattened_scan_shape = scan_shapes[axis]
                kept_roles.extend((ROLE_SCAN_Y, ROLE_SCAN_X))
            elif isinstance(scan_y_target, int) and isinstance(scan_x_target, int):
                indexer.append(scan_y_target * scan_x + scan_x_target)
            else:
                raise ValueError(
                    "Flattened Scan preview requires both scan axes to be sliced "
                    "or both scan axes to be fixed."
                )
        else:
            target = target_by_role[role]
            indexer.append(target)
            if isinstance(target, slice):
                kept_roles.append(role)

    subset = np.asarray(array[tuple(indexer)])
    if flattened_axis is not None:
        scan_y, scan_x = flattened_scan_shape
        subset = np.reshape(
            subset,
            (
                *subset.shape[:flattened_axis],
                scan_y,
                scan_x,
                *subset.shape[flattened_axis + 1 :],
            ),
        )

    permutation = [kept_roles.index(role) for role in output_roles]
    if permutation != list(range(len(output_roles))):
        subset = np.transpose(subset, permutation)
    return np.asarray(subset)


def selection_from_mapping(
    ndim: int,
    role_by_axis: Mapping[int, str],
    fixed_by_axis: Mapping[int, int],
    scan_shape_by_axis: Mapping[int, tuple[int, int]] | None = None,
) -> DimensionSelection:
    roles = []
    fixed_indices = []
    scan_shapes = []
    scan_shape_by_axis = scan_shape_by_axis or {}
    for axis in range(ndim):
        roles.append(role_by_axis.get(axis, ROLE_FIXED))
        fixed_indices.append(int(fixed_by_axis.get(axis, 0)))
        scan_shapes.append(scan_shape_by_axis.get(axis))
    return DimensionSelection(tuple(roles), tuple(fixed_indices), tuple(scan_shapes))
