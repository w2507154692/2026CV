from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import ImageDraw
from torchvision.transforms import functional as F


def box_iou(box1, box2):
	"""计算两个边界框的 IoU。"""

	x1 = max(float(box1[0]), float(box2[0]))
	y1 = max(float(box1[1]), float(box2[1]))
	x2 = min(float(box1[2]), float(box2[2]))
	y2 = min(float(box1[3]), float(box2[3]))

	inter_w = max(0.0, x2 - x1)
	inter_h = max(0.0, y2 - y1)
	inter_area = inter_w * inter_h

	area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
	area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
	union_area = area1 + area2 - inter_area

	if union_area <= 0:
		return 0.0
	return inter_area / union_area


def match_detection_counts(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold):
	"""基于 IoU 和类别匹配，返回 TP / FP / FN 数量。

	该接口只关心 boxes、labels 和阈值，不关心模型来源。
	"""

	matched_gt_indices: set[int] = set()
	tp = 0
	fp = 0

	if pred_boxes.numel() == 0:
		return 0, 0, int(gt_boxes.shape[0])

	for pred_index in range(pred_boxes.shape[0]):
		best_iou = 0.0
		best_gt_index = -1

		for gt_index in range(gt_boxes.shape[0]):
			if gt_index in matched_gt_indices:
				continue
			if int(pred_labels[pred_index].item()) != int(gt_labels[gt_index].item()):
				continue

			iou = box_iou(pred_boxes[pred_index], gt_boxes[gt_index])
			if iou > best_iou:
				best_iou = iou
				best_gt_index = gt_index

		if best_gt_index >= 0 and best_iou >= iou_threshold:
			matched_gt_indices.add(best_gt_index)
			tp += 1
		else:
			fp += 1

	fn = int(gt_boxes.shape[0]) - len(matched_gt_indices)
	return tp, fp, fn


def compute_precision_recall_f1(tp, fp, fn):
	"""根据 TP / FP / FN 计算 precision、recall、f1。"""

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	return precision, recall, f1


def filter_predictions_by_score(pred_boxes, pred_labels, pred_scores, score_threshold):
	"""按分数阈值过滤预测结果。"""

	keep = pred_scores >= score_threshold
	return pred_boxes[keep], pred_labels[keep], pred_scores[keep]


def boxes_to_centers(boxes) -> "torch.Tensor":
	"""将边界框转换为中心点坐标 (N, 2)。"""

	if boxes.numel() == 0:
		return boxes.new_zeros((0, 2))
	return torch.stack(
		[
			(boxes[:, 0] + boxes[:, 2]) / 2.0,
			(boxes[:, 1] + boxes[:, 3]) / 2.0,
		],
		dim=1,
	)


def visualize_detection_points(
	image_tensor,
	gt_save_path,
	pred_save_path,
	gt_boxes,
	gt_labels,
	pred_boxes,
	pred_labels,
	gt_colors,
	pred_colors,
):
	"""将 GT / 预测边界框中心可视化为 5x5 色块，并分别保存。"""

	gt_points = boxes_to_centers(gt_boxes)
	pred_points = boxes_to_centers(pred_boxes)

	gt_image = F.to_pil_image(image_tensor.cpu())
	gt_draw = ImageDraw.Draw(gt_image)
	for point, label_id in zip(gt_points.tolist(), gt_labels.tolist()):
		color = _resolve_class_color(label_id, gt_colors)
		if color is not None:
			_draw_point_square(gt_draw, point, color)

	pred_image = F.to_pil_image(image_tensor.cpu())
	pred_draw = ImageDraw.Draw(pred_image)
	for point, label_id in zip(pred_points.tolist(), pred_labels.tolist()):
		color = _resolve_class_color(label_id, pred_colors)
		if color is not None:
			_draw_point_square(pred_draw, point, color)

	gt_save_path = Path(gt_save_path)
	pred_save_path = Path(pred_save_path)
	gt_save_path.parent.mkdir(parents=True, exist_ok=True)
	pred_save_path.parent.mkdir(parents=True, exist_ok=True)
	gt_image.save(gt_save_path)
	pred_image.save(pred_save_path)


def visualize_detection_result(
	image_tensor,
	save_path,
	gt_boxes,
	gt_labels,
	pred_boxes,
	pred_labels,
	pred_scores,
	label_id_to_name=None,
	gt_color="lime",
	pred_color="red",
):
	"""可视化单张图的 GT 和预测框（保留供兼容旧逻辑使用）。"""

	image = F.to_pil_image(image_tensor.cpu())
	draw = ImageDraw.Draw(image)

	for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
		label_text = _label_to_text(label, label_id_to_name)
		_draw_box(draw, box, f"GT:{label_text}", gt_color)

	for box, label, score in zip(pred_boxes.tolist(), pred_labels.tolist(), pred_scores.tolist()):
		label_text = _label_to_text(label, label_id_to_name)
		_draw_box(draw, box, f"Pred:{label_text} {score:.2f}", pred_color)

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)
	image.save(save_path)


def build_label_id_to_name(rows: Iterable[dict]) -> dict[int, str]:
	"""根据类别表构建 label_id -> label_name 映射。"""

	return {int(row["label_id"]): row["label"] for row in rows}


def _label_to_text(label_id, label_id_to_name):
	if label_id_to_name is None:
		return str(label_id)
	return label_id_to_name.get(int(label_id), str(label_id))


def _draw_box(draw, box, text, color):
	xmin, ymin, xmax, ymax = [float(value) for value in box]
	draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=2)
	text_y = max(0.0, ymin - 12.0)
	draw.text((xmin, text_y), text, fill=color)


def _resolve_class_color(label_id, colors: list[str]) -> str | None:
	color_index = int(label_id) - 1
	if color_index < 0 or color_index >= len(colors):
		return None
	return colors[color_index]


def _draw_point_square(draw, point, color):
	x_coord, y_coord = int(round(float(point[0]))), int(round(float(point[1])))
	draw.rectangle((x_coord - 2, y_coord - 2, x_coord + 2, y_coord + 2), fill=color, outline=color)