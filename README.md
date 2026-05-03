# py4D-browser-flexloader

`py4D-browser-flexloader` is a plugin for
[py4D-browser](https://github.com/sezelt/py4D-browser) that adds a flexible
loader for HDF5 and Zarr datasets. It lets you browse datasets inside a file or
Zarr store, choose the data array to load, map arbitrary source dimensions into
py4DGUI's expected 4D datacube order, and preview the selected mapping before
loading.

This is useful for datasets that are not organized in the standard py4DSTEM file
layout, including files with multiple arrays, higher-dimensional arrays, and
flattened scan dimensions such as `(Nscans, ky, kx)`.

## Installation

You can install `py4D-browser-flexloader` directly from PyPI:

```bash
pip install py4d-browser-flexloader
```

> **Note:**
> - If you install into a fresh Python environment, `py4D-browser` and
>   `py4DSTEM` will be installed automatically as dependencies.
> - If you already have `py4D-browser` installed, install this plugin into the
>   same Python environment.

A step-by-step installation using conda looks like this:

```bash
conda create -n py4dgui python=3.12
conda activate py4dgui
python -m pip install --upgrade pip
python -m pip cache purge
pip install py4d-browser-flexloader
```

## Usage

Start py4DGUI after activating the environment:

```bash
py4dgui
```

After installing this plugin, you should see a **"Flex Loader"** submenu under
the **"Plugins"** menu. From there, choose the loader that matches your data
source:

- **Load HDF5 File...** for `.h5`, `.hdf5`, `.emd`, `.mat`, and `.py4dstem`
  files.
- **Load Zarr Directory...** for directory-backed `.zarr` stores.
- **Load Zarr Zip File...** for Zarr stores saved as `.zip` or `.zarr.zip`
  files.

The loader dialog shows the internal hierarchy of the selected file or store,
including array paths, shapes, dtypes, chunks, and attributes. Once an array is
selected, assign its dimensions to:

- `Scan Y`
- `Scan X`
- `Detector Y`
- `Detector X`
- `Fixed Index`
- `Flattened Scan`

Use `Flattened Scan` for datasets stored as `(Nscans, ky, kx)` or
`(..., Nscans, ky, kx)`. The loader will factor the flattened scan axis into
`Scan Y` and `Scan X` before loading.

Click **Update Preview** to inspect a lightweight preview before loading. The
preview reads only two 2D slices from the selected HDF5/Zarr dataset:

- a sqrt-scaled diffraction pattern from the central scan position
- an axial bright-field image from the central diffraction pixel

Click **Load** to create a `py4DSTEM.DataCube` from the selected 4D subset and
send it to py4DGUI. The raw source file is not modified. Calibration can be set
after loading using py4DGUI's calibration tools.

Both grouped Zarr stores and root-level Zarr arrays are supported.

## License

GNU GPLv3

**py4D-browser-flexloader** is open source software distributed under a GPLv3
license. It is free to use, alter, or build on, provided that any work derived
from **py4D-browser-flexloader** is also kept free and open.
