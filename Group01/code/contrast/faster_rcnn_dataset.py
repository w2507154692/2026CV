from __future__ import annotations

import csv
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

_SPLIT_NAMES = {"train", "val", "test"}


class FasterRCNNDatasetBase(Dataset):
	"""Faster R-CNN 检测数据集基类：从 annotations/boxes.csv 读取边界框标注。

	数据目录结构（与 CoNSeP_box / MoNuSAC_box 一致）：
	- images/{split}/*.png
	- annotations/boxes.csv
	- metadata/classes.csv
	"""

	def __init__(
		self,
		data_root: str | Path,
		split: str,
		annotations_csv: str | Path | None = None,
	) -> None:
		self.data_root = Path(data_root)
		self.split = split.strip().lower()
		if self.split not in _SPLIT_NAMES:
			raise ValueError(f"不支持的 split: {split}")
		self.annotations_csv = (
			Path(annotations_csv)
			if annotations_csv is not None
			else self.data_root / "annotations" / "boxes.csv"
		)
		if not self.annotations_csv.is_file():
			raise ValueError(f"标注文件不存在: {self.annotations_csv}")

		self.samples = self._load_samples()
		if not self.samples:
			raise ValueError(f"split={self.split} 没有可用样本，请检查 {self.annotations_csv}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int):
		sample = self.samples[index]
		image = Image.open(sample["image_full_path"]).convert("RGB")
		image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float32)

		boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
		labels = torch.tensor(sample["labels"], dtype=torch.int64)
		area = torch.tensor(sample["area"], dtype=torch.float32)
		iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64)

		target = {
			"boxes": boxes,
			"labels": labels,
			"image_id": torch.tensor([index], dtype=torch.int64),
			"area": area,
			"iscrowd": iscrowd,
		}
		return image_tensor, target

	@abstractmethod
	def _is_label_allowed(self, label_id: int) -> bool:
		"""子类决定是否保留某个前景 label_id。"""

	def _load_samples(self) -> list[dict]:
		grouped: OrderedDict[str, dict] = OrderedDict()
		with self.annotations_csv.open("r", encoding="utf-8", newline="") as handle:
			reader = csv.DictReader(handle)
			for row in reader:
				if row["split"] != self.split:
					continue

				image_path = row["image_path"].replace("\\", "/")
				sample = grouped.setdefault(
					image_path,
					{
						"image_path": image_path,
						"image_full_path": self.data_root / image_path,
						"boxes": [],
						"labels": [],
						"area": [],
					},
				)

				if int(row["is_negative"]):
					continue

				label_id = int(row["label_id"])
				if not self._is_label_allowed(label_id):
					continue

				xmin = float(row["xmin"])
				ymin = float(row["ymin"])
				xmax = float(row["xmax"])
				ymax = float(row["ymax"])
				sample["boxes"].append([xmin, ymin, xmax, ymax])
				sample["labels"].append(label_id)
				sample["area"].append(max(0.0, xmax - xmin) * max(0.0, ymax - ymin))

		formatted_samples: list[dict] = []
		for sample in grouped.values():
			if not sample["boxes"]:
				sample["boxes"] = []
				sample["labels"] = []
				sample["area"] = []
			formatted_samples.append(sample)
		return formatted_samples


class CoNSePBoxDataset(FasterRCNNDatasetBase):
	"""读取 CoNSeP_box：3 类前景（inflammatory / epithelial / stromal）。"""

	def _is_label_allowed(self, label_id: int) -> bool:
		return label_id >= 1


class MoNuSACBoxDataset(FasterRCNNDatasetBase):
	"""读取 MoNuSAC_box：仅保留前两类 epithelial / lymphocyte。"""

	_ACTIVE_LABEL_IDS = frozenset({1, 2})

	def _is_label_allowed(self, label_id: int) -> bool:
		return label_id in self._ACTIVE_LABEL_IDS


def detection_collate_fn(batch):
	images, targets = zip(*batch)
	return list(images), list(targets)


def build_faster_rcnn_dataset(
	dataset_type: str,
	data_root: str | Path,
	split: str,
	annotations_csv: str | Path | None = None,
):
	dataset_type_normalized = dataset_type.strip().lower()
	dataset_registry = {
		"consep": CoNSePBoxDataset,
		"monusac": MoNuSACBoxDataset,
	}
	if dataset_type_normalized not in dataset_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(dataset_registry.keys()))}"
		)
	return dataset_registry[dataset_type_normalized](
		data_root=data_root,
		split=split,
		annotations_csv=annotations_csv,
	)


def get_faster_rcnn_num_classes(dataset_type: str) -> int:
	dataset_type_normalized = dataset_type.strip().lower()
	class_registry = {
		"consep": 4,
		"monusac": 3,
	}
	if dataset_type_normalized not in class_registry:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(class_registry.keys()))}"
		)
	return class_registry[dataset_type_normalized]


_FASTER_RCNN_CLASS_VIZ = {
	"consep": {
		"names": ["inflammatory", "epithelial", "stromal"],
		"gt_colors": ["blue", "red", "yellow"],
		"pred_colors": ["blue", "red", "yellow"],
	},
	"monusac": {
		"names": ["epithelial", "lymphocyte"],
		"gt_colors": ["red", "lime"],
		"pred_colors": ["red", "lime"],
	},
}


def get_faster_rcnn_class_names(dataset_type: str) -> list[str]:
	dataset_type_normalized = dataset_type.strip().lower()
	if dataset_type_normalized not in _FASTER_RCNN_CLASS_VIZ:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(_FASTER_RCNN_CLASS_VIZ.keys()))}"
		)
	return list(_FASTER_RCNN_CLASS_VIZ[dataset_type_normalized]["names"])


def get_faster_rcnn_label_id_to_name(dataset_type: str) -> dict[int, str]:
	dataset_type_normalized = dataset_type.strip().lower()
	label_id_to_name = {0: "background"}
	for class_index, class_name in enumerate(get_faster_rcnn_class_names(dataset_type_normalized), start=1):
		label_id_to_name[class_index] = class_name
	return label_id_to_name


def get_faster_rcnn_gt_colors(dataset_type: str) -> list[str]:
	dataset_type_normalized = dataset_type.strip().lower()
	if dataset_type_normalized not in _FASTER_RCNN_CLASS_VIZ:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(_FASTER_RCNN_CLASS_VIZ.keys()))}"
		)
	return list(_FASTER_RCNN_CLASS_VIZ[dataset_type_normalized]["gt_colors"])


def get_faster_rcnn_pred_colors(dataset_type: str) -> list[str]:
	dataset_type_normalized = dataset_type.strip().lower()
	if dataset_type_normalized not in _FASTER_RCNN_CLASS_VIZ:
		raise ValueError(
			f"不支持的数据集类型: {dataset_type}。当前支持: {', '.join(sorted(_FASTER_RCNN_CLASS_VIZ.keys()))}"
		)
	return list(_FASTER_RCNN_CLASS_VIZ[dataset_type_normalized]["pred_colors"])


def get_num_classes(classes_csv: str | Path) -> int:
	"""从 metadata/classes.csv 读取类别数（含背景）。保留供兼容旧逻辑使用。"""
	classes_csv = Path(classes_csv)
	with classes_csv.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		label_ids = [int(row["label_id"]) for row in reader]
	if not label_ids:
		raise ValueError(f"未能从 {classes_csv} 读取到类别信息")
	return max(label_ids) + 1
