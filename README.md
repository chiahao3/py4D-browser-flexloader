# py4D-browser-flexloader

`py4D-browser-flexloader` is a py4D-browser plugin that loads arbitrary HDF5
and Zarr datasets into py4DGUI as 4D-STEM datacubes.

After installation, launch `py4DGUI` and open **Plugins > Flex Loader > Load
File...** for HDF5-like files or **Plugins > Flex Loader > Load Zarr Store...**
for directory-backed Zarr stores. The loader lets you browse the internal file
hierarchy, inspect array shape and metadata, assign source dimensions to
`Scan Y`, `Scan X`, `Detector Y`, and `Detector X`, and fix any extra dimensions
at a selected index. Datasets stored as `(Nscans, ky, kx)` or
`(..., Nscans, ky, kx)` can use the `Flattened Scan` role, which factors one
source axis into `Scan Y` and `Scan X` directly in the loader.

Both grouped Zarr stores and root-level Zarr arrays are supported.

The first implementation reads the selected 4D subset into RAM before creating
a `py4DSTEM.DataCube`. This is intentionally conservative because py4DGUI's
current viewer code expects NumPy-like array behavior for repeated slicing and
reductions.

## Installation

```bash
pip install -e .
```

Install this plugin into the same Python environment as `py4DGUI`.

## Development

```bash
pip install -e ".[test]"
pytest
```
