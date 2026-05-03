from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .backends import DataNode, DataSource, open_source
from .selection import ALL_ROLES, DATACUBE_ROLES, DimensionSelection, default_roles
from .selection import ROLE_FIXED, ROLE_FLATTENED_SCAN, default_scan_shape
from .selection import validate_selection

if TYPE_CHECKING:
    from py4D_browser import DataViewer


class FlexLoaderDialog(QDialog):
    def __init__(self, filepath: str, parent: "DataViewer"):
        super().__init__(parent)
        self.parent_viewer = parent
        self.filepath = filepath
        self.source: DataSource | None = None
        self.root_node: DataNode | None = None
        self.selected_node: DataNode | None = None
        self.role_boxes: list[QComboBox] = []
        self.index_boxes: list[QSpinBox] = []
        self.scan_y_boxes: list[QSpinBox] = []
        self.scan_x_boxes: list[QSpinBox] = []

        self.setWindowTitle(f"Flex Loader - {Path(filepath).name}")
        self.resize(900, 600)
        self._build_layout()
        self._open_source(filepath)

    @classmethod
    def select_hdf5_file(cls, parent) -> str | None:
        filename, _ = QFileDialog.getOpenFileName(
            parent,
            "Open HDF5 Data",
            "",
            "HDF5 (*.h5 *.hdf5 *.emd *.mat *.py4dstem);;Any file (*)",
        )
        return filename or None

    @classmethod
    def select_zarr_directory(cls, parent) -> str | None:
        directory = QFileDialog.getExistingDirectory(parent, "Open Zarr Directory", "")
        return directory or None

    @classmethod
    def select_zarr_zip_file(cls, parent) -> str | None:
        filename, _ = QFileDialog.getOpenFileName(
            parent,
            "Open Zarr Zip File",
            "",
            "Zarr Zip (*.zip *.zarr.zip);;Any file (*)",
        )
        return filename or None

    def closeEvent(self, event):
        self.close_source()
        super().closeEvent(event)

    def close_source(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Dataset", "Shape", "Type"])
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.currentItemChanged.connect(self._tree_selection_changed)
        splitter.addWidget(self.tree)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        right_layout.addWidget(self.details, stretch=1)

        self.mapping_box = QGroupBox("Dimension Mapping")
        self.mapping_layout = QGridLayout(self.mapping_box)
        right_layout.addWidget(self.mapping_box)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.load_button = QPushButton("Load")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.accept)
        button_layout.addWidget(self.load_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        splitter.setSizes([360, 540])

    def _open_source(self, filepath: str) -> None:
        try:
            self.source = open_source(filepath)
            self.root_node = self.source.scan()
        except Exception as exc:
            QMessageBox.critical(self, "Could not open file", str(exc))
            self.reject()
            return
        self._populate_tree()

    def _populate_tree(self) -> None:
        self.tree.clear()
        if self.root_node is None:
            return
        root_item = self._make_tree_item(self.root_node)
        self.tree.addTopLevelItem(root_item)
        self._add_tree_children(root_item, self.root_node)
        self.tree.expandItem(root_item)

    def _add_tree_children(self, parent_item: QTreeWidgetItem, parent_node: DataNode):
        for child in parent_node.children:
            item = self._make_tree_item(child)
            parent_item.addChild(item)
            self._add_tree_children(item, child)

    def _make_tree_item(self, node: DataNode) -> QTreeWidgetItem:
        shape = "" if node.shape is None else " x ".join(str(size) for size in node.shape)
        dtype = node.kind if node.dtype is None else node.dtype
        item = QTreeWidgetItem([node.name, shape, dtype])
        item.setData(0, Qt.UserRole, node)
        item.setToolTip(0, node.path)
        return item

    def _tree_selection_changed(self, current: QTreeWidgetItem, _previous) -> None:
        node = current.data(0, Qt.UserRole) if current is not None else None
        self.selected_node = node if isinstance(node, DataNode) else None
        self._update_details()
        self._build_mapping_controls()
        self._update_load_enabled()

    def _update_details(self) -> None:
        node = self.selected_node
        if node is None:
            self.details.clear()
            return

        lines = [
            f"Path: {node.path}",
            f"Backend: {node.backend}",
            f"Kind: {node.kind}",
        ]
        if node.shape is not None:
            lines.append(f"Shape: {node.shape}")
        if node.dtype is not None:
            lines.append(f"Dtype: {node.dtype}")
        if node.chunks is not None:
            lines.append(f"Chunks: {node.chunks}")
        if node.attrs:
            lines.append("")
            lines.append("Attributes:")
            for key, value in sorted(node.attrs.items()):
                lines.append(f"  {key}: {value}")
        self.details.setPlainText("\n".join(lines))

    def _build_mapping_controls(self) -> None:
        self._clear_mapping_controls()
        node = self.selected_node
        if node is None or node.shape is None:
            return

        headers = ["Axis", "Size", "Role", "Fixed Index", "Scan Y", "Scan X"]
        for col, label in enumerate(headers):
            self.mapping_layout.addWidget(QLabel(label), 0, col)

        defaults = default_roles(len(node.shape))
        for axis, size in enumerate(node.shape):
            axis_label = QLabel(str(axis))
            size_label = QLabel(str(size))

            role_box = QComboBox()
            role_box.addItems(ALL_ROLES)
            role_box.setCurrentText(defaults[axis])
            role_box.currentTextChanged.connect(self._update_load_enabled)
            role_box.currentTextChanged.connect(self._update_axis_control_states)
            self.role_boxes.append(role_box)

            index_box = QSpinBox()
            index_box.setRange(0, max(0, int(size) - 1))
            index_box.setEnabled(defaults[axis] == ROLE_FIXED)
            self.index_boxes.append(index_box)

            scan_y, scan_x = default_scan_shape(int(size))
            scan_y_box = QSpinBox()
            scan_y_box.setRange(1, max(1, int(size)))
            scan_y_box.setValue(scan_y)
            scan_y_box.setEnabled(defaults[axis] == ROLE_FLATTENED_SCAN)
            scan_y_box.valueChanged.connect(self._update_load_enabled)
            self.scan_y_boxes.append(scan_y_box)

            scan_x_box = QSpinBox()
            scan_x_box.setRange(1, max(1, int(size)))
            scan_x_box.setValue(scan_x)
            scan_x_box.setEnabled(defaults[axis] == ROLE_FLATTENED_SCAN)
            scan_x_box.valueChanged.connect(self._update_load_enabled)
            self.scan_x_boxes.append(scan_x_box)

            row = axis + 1
            self.mapping_layout.addWidget(axis_label, row, 0)
            self.mapping_layout.addWidget(size_label, row, 1)
            self.mapping_layout.addWidget(role_box, row, 2)
            self.mapping_layout.addWidget(index_box, row, 3)
            self.mapping_layout.addWidget(scan_y_box, row, 4)
            self.mapping_layout.addWidget(scan_x_box, row, 5)

    def _clear_mapping_controls(self) -> None:
        while self.mapping_layout.count():
            item = self.mapping_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.role_boxes.clear()
        self.index_boxes.clear()
        self.scan_y_boxes.clear()
        self.scan_x_boxes.clear()

    def _update_axis_control_states(self) -> None:
        for role_box, index_box, scan_y_box, scan_x_box in zip(
            self.role_boxes, self.index_boxes, self.scan_y_boxes, self.scan_x_boxes
        ):
            role = role_box.currentText()
            index_box.setEnabled(role == ROLE_FIXED)
            scan_y_box.setEnabled(role == ROLE_FLATTENED_SCAN)
            scan_x_box.setEnabled(role == ROLE_FLATTENED_SCAN)

    def _update_load_enabled(self) -> None:
        node = self.selected_node
        if node is None or not node.is_array or node.shape is None:
            self.load_button.setEnabled(False)
            return
        try:
            validate_selection(node.shape, self.get_selection())
            ok = True
        except ValueError:
            ok = False
        self.load_button.setEnabled(ok)

    def get_selection(self) -> DimensionSelection:
        scan_shapes = []
        for role_box, scan_y_box, scan_x_box in zip(
            self.role_boxes, self.scan_y_boxes, self.scan_x_boxes
        ):
            if role_box.currentText() == ROLE_FLATTENED_SCAN:
                scan_shapes.append((scan_y_box.value(), scan_x_box.value()))
            else:
                scan_shapes.append(None)
        return DimensionSelection(
            tuple(box.currentText() for box in self.role_boxes),
            tuple(box.value() for box in self.index_boxes),
            tuple(scan_shapes),
        )
