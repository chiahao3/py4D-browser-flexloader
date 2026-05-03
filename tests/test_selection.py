import numpy as np
import pytest

from py4d_browser_plugin.flexloader.selection import (
    DimensionSelection,
    ROLE_DETECTOR_X,
    ROLE_DETECTOR_Y,
    ROLE_FIXED,
    ROLE_SCAN_X,
    ROLE_SCAN_Y,
    default_roles,
    load_datacube_array,
)


def test_default_roles_use_last_four_axes():
    assert default_roles(5) == [
        ROLE_FIXED,
        ROLE_SCAN_Y,
        ROLE_SCAN_X,
        ROLE_DETECTOR_Y,
        ROLE_DETECTOR_X,
    ]


def test_load_plain_4d_dataset():
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    selection = DimensionSelection(
        (ROLE_SCAN_Y, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0, 0),
    )
    result = load_datacube_array(array, selection)
    np.testing.assert_array_equal(result, array)


def test_load_5d_dataset_with_fixed_index():
    array = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    selection = DimensionSelection(
        (ROLE_FIXED, ROLE_SCAN_Y, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (1, 0, 0, 0, 0),
    )
    result = load_datacube_array(array, selection)
    np.testing.assert_array_equal(result, array[1])


def test_arbitrary_axis_assignment_is_transposed_to_datacube_order():
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    selection = DimensionSelection(
        (ROLE_DETECTOR_X, ROLE_SCAN_Y, ROLE_DETECTOR_Y, ROLE_SCAN_X),
        (0, 0, 0, 0),
    )
    result = load_datacube_array(array, selection)
    np.testing.assert_array_equal(result, np.transpose(array, (1, 3, 2, 0)))


def test_duplicate_role_is_rejected():
    array = np.zeros((2, 3, 4, 5))
    selection = DimensionSelection(
        (ROLE_SCAN_Y, ROLE_SCAN_Y, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="Missing|required|Duplicate"):
        load_datacube_array(array, selection)
