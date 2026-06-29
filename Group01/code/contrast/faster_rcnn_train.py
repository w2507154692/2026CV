from __future__ import annotations

import platform
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from faster_rcnn_dataset import (
	build_faster_rcnn_dataset,
	detection_collate_fn,
	get_faster_rcnn_num_classes,
)
from faster_rcnn import FasterRCNNConfig, build_faster_rcnn
from faster_rcnn_utils import (
	compute_precision_recall_f1,
	filter_predictions_by_score,
	match_detection_counts,
)


# DATA_ROOT = Path("data") / "MoNuSAC_box"
# DATASET_TYPE = "monusac"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NUM_EPOCHS = 100
# BATCH_SIZE = 2
# NUM_WORKERS = 0 if platform.system() == "Windows" else 16
# LEARNING_RATE = 0.0005
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.0005
# LR_STEP_SIZE = 30
# LR_GAMMA = 0.1
# EVAL_INTERVAL_EPOCHS = 1
# PRINT_FREQ = 10
# GRAD_CLIP_NORM = 10.0

# SCORE_THRESHOLD = 0.15
# IOU_THRESHOLD = 0.3

# PRETRAINED_DETECTOR = True
# PRETRAINED_BACKBONE = True
# TRAINABLE_BACKBONE_LAYERS = 5
# MIN_SIZE = 512
# MAX_SIZE = 512
# BOX_SCORE_THRESH = 0.05
# BOX_NMS_THRESH = 0.3
# BOX_DETECTIONS_PER_IMG = 500
# RPN_ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
# RPN_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5
#------------------------------------------------------------------
DATA_ROOT = Path("data") / "CoNSeP_box_patch"
DATASET_TYPE = "consep"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 300
BATCH_SIZE = 4
NUM_WORKERS = 0 if platform.system() == "Windows" else 16
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 50
LR_GAMMA = 0.1
EVAL_INTERVAL_EPOCHS = 1
PRINT_FREQ = 10
GRAD_CLIP_NORM = 10.0

SCORE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.3

PRETRAINED_DETECTOR = True
PRETRAINED_BACKBONE = True
TRAINABLE_BACKBONE_LAYERS = 5
MIN_SIZE = 500
MAX_SIZE = 500
BOX_SCORE_THRESH = 0.05
BOX_NMS_THRESH = 0.5
BOX_DETECTIONS_PER_IMG = 600
RPN_ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
RPN_ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5


def faster_rcnn_train(checkpoints_save_dir, logger):
    checkpoints_dir = Path(checkpoints_save_dir)
    train_dataset = build_faster_rcnn_dataset(DATASET_TYPE, DATA_ROOT, split="train")
    val_dataset = build_faster_rcnn_dataset(DATASET_TYPE, DATA_ROOT, split="val")
    num_classes = get_faster_rcnn_num_classes(DATASET_TYPE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )

    model_config = FasterRCNNConfig(
        num_classes=num_classes,
        pretrained_detector=PRETRAINED_DETECTOR,
        pretrained_backbone=PRETRAINED_BACKBONE,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        box_score_thresh=BOX_SCORE_THRESH,
        box_nms_thresh=BOX_NMS_THRESH,
        box_detections_per_img=BOX_DETECTIONS_PER_IMG,
        rpn_anchor_sizes=RPN_ANCHOR_SIZES,
        rpn_aspect_ratios=RPN_ASPECT_RATIOS,
    )
    model = build_faster_rcnn(model_config).to(DEVICE)

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_checkpoint_path: Path | None = None

    logger.info(f"训练设备: {DEVICE}")
    logger.info(f"训练集样本数: {len(train_dataset)} | 验证集样本数: {len(val_dataset)}")
    logger.info(f"类别数(含背景): {num_classes}")
    logger.info(f"训练配置: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    logger.info(
        "模型配置: "
        f"trainable_backbone_layers={TRAINABLE_BACKBONE_LAYERS}, "
        f"pretrained_detector={PRETRAINED_DETECTOR}, "
        f"pretrained_backbone={PRETRAINED_BACKBONE}, "
        f"min_size={MIN_SIZE}, max_size={MAX_SIZE}, "
        f"box_detections_per_img={BOX_DETECTIONS_PER_IMG}",
    )
    logger.info(f"验证 F1 阈值: score={SCORE_THRESHOLD}, iou={IOU_THRESHOLD}")
    logger.info(f"梯度裁剪阈值: {GRAD_CLIP_NORM}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_metrics = _train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            logger=logger,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{NUM_EPOCHS} 训练结束 | train_loss={train_loss:.6f} | lr={current_lr:.8f}",
        )
        logger.info(f"Epoch {epoch} 训练损失明细: {_format_metrics(train_metrics)}")

        if epoch % EVAL_INTERVAL_EPOCHS != 0 and epoch != NUM_EPOCHS:
            continue

        val_loss, val_metrics, val_f1, val_precision, val_recall = _validate_one_epoch(
            model=model,
            data_loader=val_loader,
            device=DEVICE,
            epoch=epoch,
            logger=logger,
        )
        logger.info(f"Epoch {epoch} 验证损失明细: {_format_metrics(val_metrics)}")
        logger.info(
            f"Epoch {epoch} 验证分类感知指标: "
            f"precision={val_precision:.6f} | recall={val_recall:.6f} | f1={val_f1:.6f}",
        )

        if val_f1 > best_val_f1:
            previous_best_f1 = best_val_f1
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_checkpoint_path = _save_best_checkpoint(
                checkpoints_dir=checkpoints_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_f1=best_val_f1,
                best_val_loss=best_val_loss,
                previous_best_checkpoint_path=best_checkpoint_path,
                model_config=model_config,
            )
            logger.info(
                f"Epoch {epoch} 获得更优验证指标 | val_f1={val_f1:.6f} | "
                f"上一最佳={previous_best_f1:.6f} | 对应val_loss={val_loss:.6f}",
            )
            logger.info(f"已保留最佳权重: {best_checkpoint_path}")
        else:
            logger.info(
                f"Epoch {epoch} 未刷新最佳验证指标 | val_f1={val_f1:.6f} | "
                f"best_val_f1={best_val_f1:.6f} | val_loss={val_loss:.6f}",
            )

    logger.info(f"训练完成，最佳验证 F1: {best_val_f1:.6f} | 对应验证损失: {best_val_loss:.6f}")
    if best_checkpoint_path is not None:
        logger.info(f"最佳权重文件: {best_checkpoint_path}")


def _train_one_epoch(model, data_loader, optimizer, device, epoch, logger):
    model.train()
    running_total_loss = 0.0
    running_metrics: dict[str, float] = {}
    valid_steps = 0

    progress_bar = tqdm(data_loader, desc=f"Train Epoch {epoch}", leave=False)
    for step, (images, targets) in enumerate(progress_bar, start=1):
        images = [image.to(device) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(total_loss):
            logger.warning(
                f"Epoch {epoch} Step {step}/{len(data_loader)} 出现非有限 loss，跳过该 batch: "
                f"{_format_metrics({name: value.detach().item() for name, value in loss_dict.items()})}"
            )
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        valid_steps += 1
        running_total_loss += total_loss.item()
        for name, value in loss_dict.items():
            running_metrics[name] = running_metrics.get(name, 0.0) + value.item()

        avg_loss = running_total_loss / valid_steps
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        if step % PRINT_FREQ == 0 or step == len(data_loader):
            logger.info(
                f"Epoch {epoch} Step {step}/{len(data_loader)} | train_loss={avg_loss:.6f}",
            )

    if valid_steps == 0:
        raise RuntimeError("整个训练 epoch 都出现了非有限 loss，请检查数据和训练配置")

    averaged_metrics = {name: value / valid_steps for name, value in running_metrics.items()}
    return running_total_loss / valid_steps, averaged_metrics


def _validate_one_epoch(model, data_loader, device, epoch, logger):
    was_training = model.training

    running_total_loss = 0.0
    running_metrics: dict[str, float] = {}
    valid_steps = 0
    val_tp = 0
    val_fp = 0
    val_fn = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Val Epoch {epoch}", leave=False)
        for step, (images, targets) in enumerate(progress_bar, start=1):
            images_on_device = [image.to(device) for image in images]
            targets_on_device = [_move_target_to_device(target, device) for target in targets]

            model.train()
            loss_dict = model(images_on_device, targets_on_device)
            total_loss = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(total_loss):
                logger.warning(
                    f"Epoch {epoch} 验证阶段出现非有限 loss，跳过该 batch: "
                    f"{_format_metrics({name: value.detach().item() for name, value in loss_dict.items()})}"
                )
                continue

            valid_steps += 1
            running_total_loss += total_loss.item()
            for name, value in loss_dict.items():
                running_metrics[name] = running_metrics.get(name, 0.0) + value.item()

            model.eval()
            outputs = model(images_on_device)
            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()

                pred_boxes = output["boxes"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()

                pred_boxes, pred_labels, _ = filter_predictions_by_score(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    pred_scores=pred_scores,
                    score_threshold=SCORE_THRESHOLD,
                )

                batch_tp, batch_fp, batch_fn = match_detection_counts(
                    pred_boxes=pred_boxes,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels,
                    iou_threshold=IOU_THRESHOLD,
                )
                val_tp += batch_tp
                val_fp += batch_fp
                val_fn += batch_fn

            avg_loss = running_total_loss / valid_steps
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    if was_training:
        model.train()
    else:
        model.eval()

    if valid_steps == 0:
        raise RuntimeError("整个验证 epoch 都出现了非有限 loss，请检查数据和训练配置")

    averaged_metrics = {name: value / valid_steps for name, value in running_metrics.items()}
    val_loss = running_total_loss / valid_steps
    val_precision, val_recall, val_f1 = compute_precision_recall_f1(val_tp, val_fp, val_fn)
    logger.info(
        f"Epoch {epoch}/{NUM_EPOCHS} 验证结束 | val_loss={val_loss:.6f} | "
        f"val_f1={val_f1:.6f} | val_precision={val_precision:.6f} | val_recall={val_recall:.6f}",
    )
    return val_loss, averaged_metrics, val_f1, val_precision, val_recall


def _move_target_to_device(target, device):
    return {key: value.to(device) for key, value in target.items()}


def _save_best_checkpoint(
    checkpoints_dir,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_f1,
    best_val_loss,
    previous_best_checkpoint_path,
    model_config,
):
    if previous_best_checkpoint_path is not None and previous_best_checkpoint_path.exists():
        previous_best_checkpoint_path.unlink()

    checkpoint_path = checkpoints_dir / f"epoch{epoch}_f1{best_val_f1:.6f}.pth"
    torch.save(
        {
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": asdict(model_config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def _format_metrics(metrics):
    return ", ".join(f"{name}={value:.6f}" for name, value in sorted(metrics.items()))
