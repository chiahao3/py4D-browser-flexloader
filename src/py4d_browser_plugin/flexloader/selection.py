from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ROLE_FIXED = "Fixed Index"
ROLE_SCAN_Y = "Scan Y"
ROLE_SCAN_X = "Scan X"
ROLE_DETECTOR_Y = "Detector Y"
ROLE_DETECTOR_X = "Detector X"

DATACUBE_ROLES = (ROLE_SCAN_Y, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X)
ALL_ROLES = (ROLE_FIXED, *DATACUBE_ROLES)


@dataclass(frozen=True)
class DimensionSelection:
    roles: tuple[str, ...]
    fixed_indices: tuple[int, ...]


def default_roles(ndim: int) -> list[str]:
    roles = [ROLE_FIXED] * ndim
    if ndim >= 4:
        start = ndim - 4
        for offset, role in enumerate(DATACUBE_ROLES):
            roles[start + offset] = role
    return roles


def validate_selection(shape: Sequence[int], selection: DimensionSelection) -> None:
    if len(shape) != len(selection.roles):
        raise ValueError("Role count must match dataset dimensionality.")
    if len(shape) != len(selection.fixed_indices):
        raise ValueError("Fixed-index count must match dataset dimensionality.")

    assigned = [role for role in selection.roles if role != ROLE_FIXED]
    missing = [role for role in DATACUBE_ROLES if role not in assigned]
    duplicated = sorted({role for role in assigned if assigned.count(role) > 1})

    if missing:
        raise ValueError(f"Missing required dimension role(s): {', '.join(missing)}.")
    if duplicated:
        raise ValueError(f"Duplicate dimension role(s): {', '.join(duplicated)}.")

    for axis, (size, role, fixed_index) in enumerate(
        zip(shape, selection.roles, selection.fixed_indices)
    ):
        if role not in ALL_ROLES:
            raise ValueError(f"Unsupported role for axis {axis}: {role}.")
        if role == ROLE_FIXED and not 0 <= fixed_index < size:
            raise ValueError(
                f"Fixed index {fixed_index} is outside axis {axis} with size {size}."
            )


def load_datacube_array(array, selection: DimensionSelection) -> np.ndarray:
    shape = tuple(int(size) for size in array.shape)
    validate_selection(shape, selection)

    indexer = []
    kept_roles = []
    for role, fixed_index in zip(selection.roles, selection.fixed_indices):
        if role == ROLE_FIXED:
            indexer.append(int(fixed_index))
        else:
            indexer.append(slice(None))
            kept_roles.append(role)

    subset = np.asarray(array[tuple(indexer)])
    permutation = [kept_roles.index(role) for role in DATACUBE_ROLES]
    if permutation != list(range(4)):
        subset = np.transpose(subset, permutation)
    return np.asarray(subset)


def selection_from_mapping(
    ndim: int, role_by_axis: Mapping[int, str], fixed_by_axis: Mapping[int, int]
) -> DimensionSelection:
    roles = []
    fixed_indices = []
    for axis in range(ndim):
        roles.append(role_by_axis.get(axis, ROLE_FIXED))
        fixed_indices.append(int(fixed_by_axis.get(axis, 0)))
    return DimensionSelection(tuple(roles), tuple(fixed_indices))
