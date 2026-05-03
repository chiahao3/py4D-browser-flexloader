import numpy as np

from py4d_browser_plugin.flexloader.backends import HDF5Source


def test_hdf5_tree_scanning(tmp_path):
    import h5py

    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        group = f.create_group("entry")
        group.attrs["kind"] = "root"
        dset = group.create_dataset(
            "data",
            data=np.zeros((2, 3, 4, 5), dtype=np.float32),
            chunks=(1, 3, 4, 5),
        )
        dset.attrs["units"] = "counts"

    source = HDF5Source(str(path))
    try:
        root = source.scan()
        entry = root.children[0]
        data = entry.children[0]
        assert entry.path == "/entry"
        assert data.path == "/entry/data"
        assert data.shape == (2, 3, 4, 5)
        assert data.dtype == "float32"
        assert data.chunks == (1, 3, 4, 5)
        assert data.attrs["units"] == "'counts'"
    finally:
        source.close()


def test_zarr_tree_scanning_when_zarr_is_available(tmp_path):
    zarr = __import__("pytest").importorskip("zarr")

    path = tmp_path / "sample.zarr"
    root = zarr.open_group(str(path), mode="w")
    group = root.create_group("entry")
    group.attrs["kind"] = "root"
    group.create_array(
        "data",
        data=np.zeros((2, 3, 4, 5), dtype=np.float32),
        chunks=(1, 3, 4, 5),
    )

    from py4d_browser_plugin.flexloader.backends import ZarrSource

    source = ZarrSource(str(path))
    root_node = source.scan()
    entry = root_node.children[0]
    data = entry.children[0]
    assert entry.path == "/entry"
    assert data.path == "/entry/data"
    assert data.shape == (2, 3, 4, 5)
    assert data.dtype == "float32"


def test_zarr_root_array_scanning_when_zarr_is_available(tmp_path):
    zarr = __import__("pytest").importorskip("zarr")

    path = tmp_path / "array.zarr"
    root = zarr.open_array(
        str(path),
        mode="w",
        shape=(2, 3, 4, 5),
        chunks=(1, 3, 4, 5),
        dtype="float32",
    )
    root.attrs["units"] = "counts"

    from py4d_browser_plugin.flexloader.backends import ZarrSource

    source = ZarrSource(str(path))
    root_node = source.scan()
    assert root_node.path == "/"
    assert root_node.name == "array.zarr"
    assert root_node.is_array
    assert root_node.shape == (2, 3, 4, 5)
    assert root_node.dtype == "float32"
    assert source.get_array("/") is not None
