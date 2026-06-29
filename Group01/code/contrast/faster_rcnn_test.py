from __future__ import annotations

import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from faster_rcnn_dataset import (
	build_faster_rcnn_dataset,
	detection_collate_fn,
	get_faster_rcnn_gt_colors,
	get_faster_rcnn_label_id_to_name,
	get_faster_rcnn_num_classes,
	get_faster_rcnn_pred_colors,
)
from faster_rcnn import FasterRCNNConfig, build_faster_rcnn
from faster_rcnn_utils import (
	compute_precision_recall_f1,
	filter_predictions_by_score,
	match_detection_counts,
	visualize_detection_points,
)


DATA_ROOT = Path("data") / "CoNSeP_box_patch"
DATASET_TYPE = "consep"
SPLIT = "test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_BATCH_SIZE = 2
NUM_WORKERS = 0 if platform.system() == "Windows" else 16
SCORE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.3
RESULTS_FILE_NAME = "test_results.txt"
VISUALIZATION_DIR_NAME = "visualizations"
GT_VIS_SUFFIX = "_gt.png"
PRED_VIS_SUFFIX = "_pred.png"


def faster_rcnn_test(out_dir, pth_file_path):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	results_file = out_dir / RESULTS_FILE_NAME
	visualization_dir = out_dir / VISUALIZATION_DIR_NAME

	checkpoint = torch.load(pth_file_path, map_location="cpu")
	label_id_to_name = get_faster_rcnn_label_id_to_name(DATASET_TYPE)
	gt_colors = get_faster_rcnn_gt_colors(DATASET_TYPE)
	pred_colors = get_faster_rcnn_pred_colors(DATASET_TYPE)

	test_dataset = build_faster_rcnn_dataset(DATASET_TYPE, DATA_ROOT, split=SPLIT)
	test_loader = DataLoader(
		test_dataset,
		batch_size=TEST_BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available(),
		collate_fn=detection_collate_fn,
	)

	model_config = _load_model_config(checkpoint)
	model = build_faster_rcnn(model_config)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(DEVICE)
	model.eval()
	class_ids = sorted(label_id_to_name.keys())
	foreground_class_ids = [class_id for class_id in class_ids if class_id != 0]

	summary = {
		"num_images": len(test_dataset),
		"num_gt_boxes": 0,
		"num_pred_boxes": 0,
		"tp": 0,
		"fp": 0,
		"fn": 0,
		"score_sum": 0.0,
		"score_count": 0,
	}
	per_class_summary = {
		class_id: {
			"tp": 0,
			"fp": 0,
			"fn": 0,
		}
		for class_id in class_ids
	}
	detection_summary = {
		"tp": 0,
		"fp": 0,
		"fn": 0,
	}
	per_image_lines: list[str] = []

	with torch.inference_mode():
		progress_bar = tqdm(test_loader, desc="Test", leave=False)
		for images, targets in progress_bar:
			images_on_device = [image.to(DEVICE) for image in images]
			outputs = model(images_on_device)

			for image_tensor, target, output in zip(images, targets, outputs):
				image_index = int(target["image_id"].item())
				image_path = test_dataset.samples[image_index]["image_path"]

				gt_boxes = target["boxes"].cpu()
				gt_labels = target["labels"].cpu()

				pred_boxes = output["boxes"].detach().cpu()
				pred_labels = output["labels"].detach().cpu()
				pred_scores = output["scores"].detach().cpu()

				pred_boxes, pred_labels, pred_scores = filter_predictions_by_score(
					pred_boxes=pred_boxes,
					pred_labels=pred_labels,
					pred_scores=pred_scores,
					score_threshold=SCORE_THRESHOLD,
				)

				tp, fp, fn = match_detection_counts(
					pred_boxes=pred_boxes,
					pred_labels=pred_labels,
					gt_boxes=gt_boxes,
					gt_labels=gt_labels,
					iou_threshold=IOU_THRESHOLD,
				)

				for class_id in class_ids:
					pred_mask = pred_labels == class_id
					gt_mask = gt_labels == class_id
					class_tp, class_fp, class_fn = match_detection_counts(
						pred_boxes=pred_boxes[pred_mask],
						pred_labels=pred_labels[pred_mask],
						gt_boxes=gt_boxes[gt_mask],
						gt_labels=gt_labels[gt_mask],
						iou_threshold=IOU_THRESHOLD,
					)
					per_class_summary[class_id]["tp"] += class_tp
					per_class_summary[class_id]["fp"] += class_fp
					per_class_summary[class_id]["fn"] += class_fn

				detection_pred_labels = torch.ones((pred_boxes.shape[0],), dtype=torch.int64)
				detection_gt_labels = torch.ones((gt_boxes.shape[0],), dtype=torch.int64)
				detection_tp, detection_fp, detection_fn = match_detection_counts(
					pred_boxes=pred_boxes,
					pred_labels=detection_pred_labels,
					gt_boxes=gt_boxes,
					gt_labels=detection_gt_labels,
					iou_threshold=IOU_THRESHOLD,
				)

				image_stem = Path(image_path).stem
				gt_visualization_path = visualization_dir / f"{image_stem}{GT_VIS_SUFFIX}"
				pred_visualization_path = visualization_dir / f"{image_stem}{PRED_VIS_SUFFIX}"
				visualize_detection_points(
					image_tensor=image_tensor,
					gt_save_path=gt_visualization_path,
					pred_save_path=pred_visualization_path,
					gt_boxes=gt_boxes,
					gt_labels=gt_labels,
					pred_boxes=pred_boxes,
					pred_labels=pred_labels,
					gt_colors=gt_colors,
					pred_colors=pred_colors,
				)

				summary["num_gt_boxes"] += int(gt_boxes.shape[0])
				summary["num_pred_boxes"] += int(pred_boxes.shape[0])
				summary["tp"] += tp
				summary["fp"] += fp
				summary["fn"] += fn
				detection_summary["tp"] += detection_tp
				detection_summary["fp"] += detection_fp
				detection_summary["fn"] += detection_fn
				summary["score_sum"] += float(pred_scores.sum().item())
				summary["score_count"] += int(pred_scores.numel())

				top_scores = ", ".join(f"{score:.4f}" for score in pred_scores[:5].tolist())
				if not top_scores:
					top_scores = "None"
				per_image_lines.append(
					f"image={image_path} | gt={gt_boxes.shape[0]} | pred={pred_boxes.shape[0]} | "
					f"tp={tp} | fp={fp} | fn={fn} | top_scores=[{top_scores}] | "
					f"gt_vis={gt_visualization_path.as_posix()} | pred_vis={pred_visualization_path.as_posix()}"
				)

	precision, recall, f1 = compute_precision_recall_f1(summary["tp"], summary["fp"], summary["fn"])
	detection_precision, detection_recall, detection_f1 = compute_precision_recall_f1(
		detection_summary["tp"],
		detection_summary["fp"],
		detection_summary["fn"],
	)
	average_score = (
		summary["score_sum"] / summary["score_count"] if summary["score_count"] > 0 else 0.0
	)
	per_class_lines: list[str] = []
	per_class_f1_values: list[float] = []
	for class_id in foreground_class_ids:
		class_name = label_id_to_name.get(class_id, str(class_id))
		class_summary = per_class_summary[class_id]
		class_precision, class_recall, class_f1 = compute_precision_recall_f1(
			class_summary["tp"],
			class_summary["fp"],
			class_summary["fn"],
		)
		per_class_f1_values.append(class_f1)
		per_class_lines.extend(
			[
				f"class={class_name} (label_id={class_id})",
				f"  tp: {class_summary['tp']}",
				f"  fp: {class_summary['fp']}",
				f"  fn: {class_summary['fn']}",
				f"  precision: {class_precision:.6f}",
				f"  recall: {class_recall:.6f}",
				f"  f1: {class_f1:.6f}",
			]
		)
	mean_class_f1 = sum(per_class_f1_values) / len(per_class_f1_values) if per_class_f1_values else 0.0

	lines = [
		f"checkpoint_path: {pth_file_path}",
		f"device: {DEVICE}",
		f"score_threshold: {SCORE_THRESHOLD}",
		f"iou_threshold: {IOU_THRESHOLD}",
		f"visualization_dir: {visualization_dir}",
		f"num_images: {summary['num_images']}",
		f"num_gt_boxes: {summary['num_gt_boxes']}",
		f"num_pred_boxes: {summary['num_pred_boxes']}",
		"classification_aware_results:",
		f"tp: {summary['tp']}",
		f"fp: {summary['fp']}",
		f"fn: {summary['fn']}",
		f"precision: {precision:.6f}",
		f"recall: {recall:.6f}",
		f"f1: {f1:.6f}",
		f"mean_class_f1_foreground_only: {mean_class_f1:.6f}",
		"",
		"per_class_results:",
	]
	lines.extend(per_class_lines)
	lines.extend(
		[
			"",
			"detection_results_ignore_class:",
			f"tp: {detection_summary['tp']}",
			f"fp: {detection_summary['fp']}",
			f"fn: {detection_summary['fn']}",
			f"precision: {detection_precision:.6f}",
			f"recall: {detection_recall:.6f}",
			f"f1: {detection_f1:.6f}",
			"",
			f"average_score: {average_score:.6f}",
			"",
			"per_image_results:",
		]
	)
	lines.extend(per_image_lines)

	results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"测试完成，结果已保存到: {results_file}")


def _load_model_config(checkpoint):
	model_config_dict = checkpoint.get("model_config")
	if model_config_dict is None:
		return FasterRCNNConfig(num_classes=get_faster_rcnn_num_classes(DATASET_TYPE))
	return FasterRCNNConfig(**model_config_dict)