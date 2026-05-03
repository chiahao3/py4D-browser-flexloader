import numpy as np
import pytest

from py4d_browser_plugin.flexloader.selection import (
    DimensionSelection,
    ROLE_DETECTOR_X,
    ROLE_DETECTOR_Y,
    ROLE_FLATTENED_SCAN,
    ROLE_FIXED,
    ROLE_SCAN_X,
    ROLE_SCAN_Y,
    default_roles,
    default_scan_shape,
    load_datacube_array,
    load_preview_arrays,
)


def test_default_roles_use_last_four_axes():
    assert default_roles(5) == [
        ROLE_FIXED,
        ROLE_SCAN_Y,
        ROLE_SCAN_X,
        ROLE_DETECTOR_Y,
        ROLE_DETECTOR_X,
    ]


def test_default_roles_use_flattened_scan_for_3d_data():
    assert default_roles(3) == [
        ROLE_FLATTENED_SCAN,
        ROLE_DETECTOR_Y,
        ROLE_DETECTOR_X,
    ]


def test_default_scan_shape_prefers_near_square_factors():
    assert default_scan_shape(12) == (3, 4)


def test_load_plain_4d_dataset():
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    selection = DimensionSelection(
        (ROLE_SCAN_Y, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0, 0),
    )
    result = load_datacube_array(array, selection)
    np.testing.assert_array_equal(result, array)


def test_preview_plain_4d_dataset_reads_central_dp_and_axial_bf():
    array = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    selection = DimensionSelection(
        (ROLE_SCAN_Y, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0, 0),
    )
    previews = load_preview_arrays(array, selection)
    np.testing.assert_array_equal(previews.diffraction, array[1, 1])
    np.testing.assert_array_equal(previews.axial_bf, array[:, :, 2, 2])
    assert "scan center" in previews.description


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


def test_preview_arbitrary_axis_assignment_transposes_preview_outputs():
    array = np.arange(5 * 2 * 4 * 3).reshape(5, 2, 4, 3)
    selection = DimensionSelection(
        (ROLE_DETECTOR_X, ROLE_SCAN_Y, ROLE_DETECTOR_Y, ROLE_SCAN_X),
        (0, 0, 0, 0),
    )
    previews = load_preview_arrays(array, selection)
    np.testing.assert_array_equal(previews.diffraction, array[:, 1, :, 1].T)
    np.testing.assert_array_equal(previews.axial_bf, array[2, :, 2, :])


def test_flattened_scan_axis_is_factored_into_scan_y_and_scan_x():
    array = np.arange(6 * 4 * 5).reshape(6, 4, 5)
    selection = DimensionSelection(
        (ROLE_FLATTENED_SCAN, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0),
        ((2, 3), None, None),
    )
    result = load_datacube_array(array, selection)
    np.testing.assert_array_equal(result, array.reshape(2, 3, 4, 5))


def test_preview_flattened_scan_reads_single_dp_and_axial_bf():
    array = np.arange(6 * 4 * 5).reshape(6, 4, 5)
    selection = DimensionSelection(
        (ROLE_FLATTENED_SCAN, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0),
        ((2, 3), None, None),
    )
    previews = load_preview_arrays(array, selection)
    np.testing.assert_array_equal(previews.diffraction, array[4])
    np.testing.assert_array_equal(previews.axial_bf, array[:, 2, 2].reshape(2, 3))


def test_flattened_scan_axis_can_follow_fixed_dimensions():
    array = np.arange(2 * 6 * 4 * 5).reshape(2, 6, 4, 5)
    selection = DimensionSelection(
        (ROLE_FIXED, ROLE_FLATTENED_SCAN, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (1, 0, 0, 0),
        (None, (2, 3), None, None),
    )
    result = load_datacube_array(array, selection)
    np.testing.assert_array_equal(result, array[1].reshape(2, 3, 4, 5))


def test_flattened_scan_factor_product_must_match_axis_size():
    array = np.zeros((6, 4, 5))
    selection = DimensionSelection(
        (ROLE_FLATTENED_SCAN, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0),
        ((2, 4), None, None),
    )
    with pytest.raises(ValueError, match="must equal"):
        load_datacube_array(array, selection)


def test_flattened_scan_cannot_mix_with_separate_scan_axes():
    array = np.zeros((6, 2, 4, 5))
    selection = DimensionSelection(
        (ROLE_FLATTENED_SCAN, ROLE_SCAN_X, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0, 0),
        ((2, 3), None, None, None),
    )
    with pytest.raises(ValueError, match="either Flattened Scan"):
        load_datacube_array(array, selection)


def test_duplicate_role_is_rejected():
    array = np.zeros((2, 3, 4, 5))
    selection = DimensionSelection(
        (ROLE_SCAN_Y, ROLE_SCAN_Y, ROLE_DETECTOR_Y, ROLE_DETECTOR_X),
        (0, 0, 0, 0),
    )
    with pytest.raises(ValueError, match="Missing|required|Duplicate"):
        load_datacube_array(array, selection)
