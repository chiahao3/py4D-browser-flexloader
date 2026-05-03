from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtWidgets import QAction, QMessageBox, QWidget

from .dialog import FlexLoaderDialog
from .selection import load_datacube_array

if TYPE_CHECKING:
    from py4D_browser import DataViewer


class FlexLoaderPlugin(QWidget):
    plugin_id = "chiahao3.flexloader"
    display_name = "Flex Loader"
    uses_plugin_menu = True

    def __init__(self, parent: "DataViewer", plugin_menu, **kwargs):
        super().__init__()
        self.parent = parent
        self.flexloader_menu = plugin_menu

        self.load_action = QAction("Load File...", self)
        self.load_action.triggered.connect(self.launch_loader)
        self.flexloader_menu.addAction(self.load_action)

    def close(self):
        pass

    def launch_loader(self) -> None:
        filepath = FlexLoaderDialog.select_file(self.parent)
        if not filepath:
            return

        dialog = FlexLoaderDialog(filepath, parent=self.parent)
        if dialog.exec_() != dialog.Accepted:
            dialog.close_source()
            return

        node = dialog.selected_node
        source = dialog.source
        if node is None or source is None or not node.is_array:
            dialog.close_source()
            return

        try:
            import py4DSTEM

            raw_array = source.get_array(node.path)
            array = load_datacube_array(raw_array, dialog.get_selection())
            array = np.asarray(array)
            datacube = py4DSTEM.DataCube(array)
            _set_default_pixel_calibration(datacube)
            _install_datacube(self.parent, datacube, f"{filepath}:{node.path}")
        except Exception as exc:
            QMessageBox.critical(self.parent, "Could not load dataset", str(exc))
            raise
        finally:
            dialog.close_source()


def _set_default_pixel_calibration(datacube) -> None:
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return
    setters = [
        ("set_R_pixel_size", 1.0),
        ("set_R_pixel_units", "pixels"),
        ("set_Q_pixel_size", 1.0),
        ("set_Q_pixel_units", "pixels"),
    ]
    for method_name, value in setters:
        method = getattr(calibration, method_name, None)
        if method is not None:
            method(value)


def _install_datacube(parent: "DataViewer", datacube, window_title: str) -> None:
    if hasattr(parent, "set_datacube"):
        parent.set_datacube(datacube, window_title)
        return

    parent.datacube = datacube
    if hasattr(parent, "update_scalebars"):
        parent.update_scalebars()
    parent.update_diffraction_space_view(reset=True)
    parent.update_real_space_view(reset=True)
    parent.setWindowTitle(window_title)
    if hasattr(parent, "signal_datacube_changed"):
        parent.signal_datacube_changed.emit()


# TODO: Re-evaluate lazy HDF5/Zarr-backed loading after confirming that the
# current py4DSTEM.DataCube and py4DGUI view code fully support non-ndarray
# data through repeated slicing, summation, max projection, and transpose paths.
