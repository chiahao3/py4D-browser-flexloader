from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py


@dataclass
class DataNode:
    path: str
    name: str
    kind: str
    backend: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    chunks: tuple[int, ...] | None = None
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["DataNode"] = field(default_factory=list)

    @property
    def is_array(self) -> bool:
        return self.kind == "array"


class DataSource:
    backend = "unknown"

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._arrays: dict[str, Any] = {}

    def scan(self) -> DataNode:
        raise NotImplementedError

    def get_array(self, path: str):
        return self._arrays[path]

    def close(self) -> None:
        pass


def open_source(filepath: str) -> DataSource:
    path = Path(filepath)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.is_dir() or ".zarr" in suffixes or suffixes[-2:] == [".zarr", ".zip"]:
        return ZarrSource(filepath)
    else:
        return HDF5Source(filepath)


class HDF5Source(DataSource):
    backend = "HDF5"

    def __init__(self, filepath: str):
        super().__init__(filepath)
        self._file = h5py.File(filepath, "r")
        self._root: DataNode | None = None

    def scan(self) -> DataNode:
        self._arrays.clear()
        self._root = self._scan_group(self._file, "/")
        return self._root

    def close(self) -> None:
        self._file.close()

    def _scan_group(self, group: h5py.Group, path: str) -> DataNode:
        node = DataNode(
            path=path,
            name="/" if path == "/" else path.rsplit("/", 1)[-1],
            kind="group",
            backend=self.backend,
            attrs=_stringify_attrs(group.attrs),
        )
        for name in group.keys():
            child = group[name]
            child_path = f"/{name}" if path == "/" else f"{path}/{name}"
            if isinstance(child, h5py.Dataset):
                self._arrays[child_path] = child
                node.children.append(
                    DataNode(
                        path=child_path,
                        name=name,
                        kind="array",
                        backend=self.backend,
                        shape=tuple(int(size) for size in child.shape),
                        dtype=str(child.dtype),
                        chunks=tuple(child.chunks) if child.chunks is not None else None,
                        attrs=_stringify_attrs(child.attrs),
                    )
                )
            elif isinstance(child, h5py.Group):
                node.children.append(self._scan_group(child, child_path))
        return node


class ZarrSource(DataSource):
    backend = "Zarr"

    def __init__(self, filepath: str):
        super().__init__(filepath)
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("The flexloader Zarr backend requires zarr.") from exc

        self._zarr = zarr
        self._store = None
        suffixes = [suffix.lower() for suffix in Path(filepath).suffixes]
        if suffixes[-2:] == [".zarr", ".zip"] or suffixes[-1:] == [".zip"]:
            try:
                self._store = zarr.storage.ZipStore(filepath, mode="r")
                self._root = zarr.open_group(store=self._store, mode="r")
            except Exception:
                self._store = None
                self._root = zarr.open_group(filepath, mode="r")
        else:
            self._root = zarr.open_group(filepath, mode="r")
        self._root_node: DataNode | None = None

    def scan(self) -> DataNode:
        self._arrays.clear()
        self._root_node = self._scan_group(self._root, "/")
        return self._root_node

    def close(self) -> None:
        if self._store is not None:
            close = getattr(self._store, "close", None)
            if close is not None:
                close()

    def _scan_group(self, group, path: str) -> DataNode:
        node = DataNode(
            path=path,
            name="/" if path == "/" else path.rsplit("/", 1)[-1],
            kind="group",
            backend=self.backend,
            attrs=_stringify_attrs(getattr(group, "attrs", {})),
        )
        for name, child in _iter_zarr_children(group):
            child_path = f"/{name}" if path == "/" else f"{path}/{name}"
            if _is_zarr_array(child):
                self._arrays[child_path] = child
                chunks = getattr(child, "chunks", None)
                node.children.append(
                    DataNode(
                        path=child_path,
                        name=name,
                        kind="array",
                        backend=self.backend,
                        shape=tuple(int(size) for size in child.shape),
                        dtype=str(child.dtype),
                        chunks=tuple(chunks) if chunks is not None else None,
                        attrs=_stringify_attrs(getattr(child, "attrs", {})),
                    )
                )
            else:
                node.children.append(self._scan_group(child, child_path))
        return node


def _iter_zarr_children(group) -> Iterable[tuple[str, Any]]:
    try:
        yield from group.items()
        return
    except Exception:
        pass

    try:
        for name in group.keys():
            yield name, group[name]
    except Exception:
        return


def _is_zarr_array(obj) -> bool:
    return hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "keys")


def _stringify_attrs(attrs) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = attrs.items()
    except Exception:
        return out

    for key, value in items:
        text = repr(_to_plain_value(value))
        if len(text) > 300:
            text = text[:297] + "..."
        out[str(key)] = text
    return out


def _to_plain_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
