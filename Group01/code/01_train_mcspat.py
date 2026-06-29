import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import shutil
from tqdm import tqdm as tqdm
import math
import skimage.io as io
import cv2
from skimage import filters
from skimage.measure import label, moments
import glob

from model_arch import UnetVggMultihead
from my_dataloader_w_kfunc import CellsDataset
from my_dataloader import CellsDataset as CellsDataset_simple
from cluster_helper import *

# checkpoints_root_dir = './exp' # 所有训练输出的根目录。
# checkpoints_folder_name = 'exp12_brcam2c_class_det' # 当前训练实例的输出文件夹名称，将创建在 <checkpoints_root_dir> 下。
# model_param_path        = None;  # 用于继续训练的历史 checkpoint 路径。
# clustering_pseudo_gt_root = './MCSpatNet_epoch_subclasses' # 这个是干什么的？
# train_data_root = './data/BRCA-M2C'
# test_data_root = './data/BRCA-M2C'   # 这是验证集，不是测试集！
# train_split_filepath = './data_splits/brca-m2c/train_split.txt'
# test_split_filepath = './data_splits/brca-m2c/val_split.txt'
# epochs  = 450 # 训练轮数。对于 CoNSeP 数据集建议使用 300。
# ------------------------------------------------------
checkpoints_root_dir = "./exp"  # 所有训练输出的根目录。
checkpoints_folder_name = (
    "exp18_consep"  # 当前训练实例的输出文件夹名称，将创建在 <checkpoints_root_dir> 下。
)
model_param_path = None
# 用于继续训练的历史 checkpoint 路径。
clustering_pseudo_gt_root = "./MCSpatNet_epoch_subclasses"  # 这个是干什么的？
train_data_root = "./data/CoNSeP_train"
test_data_root = "./data/CoNSeP_train"  # 这是验证集，不是测试集！
train_split_filepath = "./data_splits/consep/train_split.txt"
test_split_filepath = "./data_splits/consep/val_split.txt"
epochs = 300  # 训练轮数。对于 CoNSeP 数据集建议使用 300。
# ------------------------------------------------------
# checkpoints_root_dir = "./exp"  # 所有训练输出的根目录。
# checkpoints_folder_name = (
#     "exp12_brcam2c_class_det"  # 当前训练实例的输出文件夹名称，将创建在 <checkpoints_root_dir> 下。
# )
# model_param_path = None
# # 用于继续训练的历史 checkpoint 路径。
# clustering_pseudo_gt_root = "./MCSpatNet_epoch_subclasses"  # 这个是干什么的？
# train_data_root = "./data/MoNuSAC_MCSpatNet_point/train"
# test_data_root = "./data/MoNuSAC_MCSpatNet_point/val"  # 这是验证集，不是测试集！
# train_split_filepath = None
# test_split_filepath = None
# epochs = 300  # 训练轮数。对于 CoNSeP 数据集建议使用 300。


use_k_function_loss = True
use_subclass_loss = False

# 检测/分类主分支损失：Dice 与 Focal 互斥切换（子类分支仍用 Dice）。
use_dice_loss = True
use_focal_loss = False
focal_alpha = 0.25
focal_gamma = 2.0
lamda_focal_det = 1.0
lamda_focal_cls = 1.0

# U-CE：与 Focal 互斥；启用 Focal 时须关闭 U-CE。
use_uce_detection_loss = False
use_uce_classification_loss = False
uce_mc_samples = 5
uce_alpha = 1.0
lamda_uce_det = 1.0
lamda_uce_cls = 1.0

# 点热图一致性：gt_dots 高斯热图监督 + 检测/分类热图一致性。
use_point_heatmap_loss = False
use_det_cls_heatmap_consistency_loss = False
point_heatmap_sigma = 2.0
lamda_point_heatmap_det = 1.0
lamda_point_heatmap_cls = 1.0
lamda_det_cls_heatmap_consistency = 0.5

# 分阶段训练：前若干 epoch 仅优化检测分支，从 epoch_start_classification 起加入分类损失。
epoch_start_classification = 50


cell_code = {1: "lymphocyte", 2: "tumor", 3: "stromal"}
# cell_code = {1: "epithelial", 2: "lymphocyte"}

feature_code = {"decoder": 0, "cell-detect": 1, "class": 2, "subclass": 3, "k-cell": 4}

seed = 42  # 固定随机种子，保证训练过程中的数据增强、参数初始化和采样顺序可复现


def backup_python_files(experiment_dir):
    project_root = os.path.dirname(os.path.abspath(__file__))
    code_backup_dir = os.path.join(experiment_dir, "code")
    os.makedirs(code_backup_dir, exist_ok=True)
    for py_filepath in glob.glob(os.path.join(project_root, "*.py")):
        shutil.copy2(
            py_filepath, os.path.join(code_backup_dir, os.path.basename(py_filepath))
        )


def _uncertainty_weights(uncertainty, alpha):
    return (1.0 + uncertainty).pow(alpha)


def _estimate_mc_uncertainties(model, img, mc_samples):
    """MC-Dropout 估计像素级预测不确定性（激活概率在多次采样间的标准差）。"""
    was_training = model.training
    model.train()
    det_samples = []
    cls_samples = []
    with torch.no_grad():
        for _ in range(mc_samples):
            et_dmap_lst = model(img)
            det_logits = et_dmap_lst[0][:, :, 2:-2, 2:-2]
            cls_logits = et_dmap_lst[1][:, :, 2:-2, 2:-2]
            det_samples.append(torch.sigmoid(det_logits).squeeze(1))
            cls_prob = torch.softmax(cls_logits, dim=1)
            cls_samples.append(cls_prob.max(dim=1).values)
    if was_training:
        model.train()
    else:
        model.eval()
    det_uncertainty = torch.stack(det_samples, dim=0).std(dim=0)
    cls_uncertainty = torch.stack(cls_samples, dim=0).std(dim=0)
    return det_uncertainty, cls_uncertainty


def _bce_uce_loss(pred_logits, gt_mask, uncertainty_weights, eps=1e-7):
    if gt_mask.ndim == 4:
        gt_mask = gt_mask.squeeze(1)
    if pred_logits.ndim == 4:
        pred_logits = pred_logits.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, gt_mask.float(), reduction="none"
    )
    return (bce * uncertainty_weights).mean()


def _multiclass_uce_loss(pred_logits, gt_multichannel, uncertainty_weights, eps=1e-7):
    prob = torch.softmax(pred_logits, dim=1)
    gt = gt_multichannel.float()
    ce = -(gt * torch.log(prob + eps) + (1.0 - gt) * torch.log(1.0 - prob + eps))
    ce = ce.sum(dim=1)
    return (ce * uncertainty_weights).mean()


def _compute_dice_losses(et_dmap_all, et_dmap_class, gt_dmap_all, gt_dmap):
    et_all_sig = torch.sigmoid(et_dmap_all)
    et_class_sig = torch.softmax(et_dmap_class, dim=1)

    intersection = (et_class_sig * gt_dmap).sum()
    union = (et_class_sig**2).sum() + (gt_dmap**2).sum()
    loss_dice_class = 1 - ((2 * intersection + 1) / (union + 1))

    intersection = (et_all_sig * gt_dmap_all.unsqueeze(1)).sum()
    union = (et_all_sig**2).sum() + (gt_dmap_all.unsqueeze(1) ** 2).sum()
    loss_dice_all = 1 - ((2 * intersection + 1) / (union + 1))
    return loss_dice_class, loss_dice_all


def _binary_focal_bce_with_logits(logits, targets, alpha, gamma):
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = targets * prob + (1.0 - targets) * (1.0 - prob)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def _multiclass_focal_ce_loss(logits, gt_multichannel, alpha, gamma):
    fg_mask = gt_multichannel.sum(dim=1) > 0
    if not fg_mask.any():
        return logits.new_tensor(0.0)

    target = gt_multichannel.argmax(dim=1)
    logits_fg = logits.permute(0, 2, 3, 1)[fg_mask]
    target_fg = target[fg_mask]
    ce = F.cross_entropy(logits_fg, target_fg, reduction="none")
    pt = torch.exp(-ce)
    return (alpha * (1.0 - pt).pow(gamma) * ce).mean()


def _compute_focal_losses(et_dmap_all, et_dmap_class, gt_dmap_all, gt_dmap):
    loss_focal_det = _binary_focal_bce_with_logits(
        et_dmap_all, gt_dmap_all, focal_alpha, focal_gamma
    )
    loss_focal_cls = _multiclass_focal_ce_loss(
        et_dmap_class, gt_dmap, focal_alpha, focal_gamma
    )
    return loss_focal_det, loss_focal_cls


def _validate_det_cls_loss_config():
    if use_dice_loss and use_focal_loss:
        raise ValueError("use_dice_loss and use_focal_loss are mutually exclusive")
    if not use_dice_loss and not use_focal_loss:
        raise ValueError("enable either use_dice_loss or use_focal_loss")
    if use_focal_loss and (use_uce_detection_loss or use_uce_classification_loss):
        raise ValueError("use_focal_loss is mutually exclusive with U-CE losses")


def _compute_det_cls_loss(
    model,
    img,
    et_dmap_all,
    et_dmap_class,
    gt_dmap_all,
    gt_dmap,
    lamda_dice_weight,
    train_classification=True,
):
    loss_dice_class = None
    loss_dice_all = None
    loss_focal_det = None
    loss_focal_cls = None
    loss = et_dmap_all.new_tensor(0.0)

    if use_dice_loss:
        loss_dice_class, loss_dice_all = _compute_dice_losses(
            et_dmap_all, et_dmap_class, gt_dmap_all, gt_dmap
        )
        loss = loss + lamda_dice_weight * loss_dice_all
        if train_classification:
            loss = loss + lamda_dice_weight * loss_dice_class
    elif use_focal_loss:
        loss_focal_det, loss_focal_cls = _compute_focal_losses(
            et_dmap_all, et_dmap_class, gt_dmap_all, gt_dmap
        )
        loss = loss + lamda_focal_det * loss_focal_det
        if train_classification:
            loss = loss + lamda_focal_cls * loss_focal_cls

    loss_uce_det = None
    loss_uce_cls = None
    if not use_focal_loss:
        if use_uce_detection_loss or (
            train_classification and use_uce_classification_loss
        ):
            unc_det, unc_cls = _estimate_mc_uncertainties(model, img, uce_mc_samples)
        if use_uce_detection_loss:
            loss_uce_det = _bce_uce_loss(
                et_dmap_all,
                gt_dmap_all,
                _uncertainty_weights(unc_det, uce_alpha),
            )
            loss = loss + lamda_uce_det * loss_uce_det
        if train_classification and use_uce_classification_loss:
            loss_uce_cls = _multiclass_uce_loss(
                et_dmap_class,
                gt_dmap,
                _uncertainty_weights(unc_cls, uce_alpha),
            )
            loss = loss + lamda_uce_cls * loss_uce_cls

    return (
        loss,
        loss_dice_class,
        loss_dice_all,
        loss_uce_det,
        loss_uce_cls,
        loss_focal_det,
        loss_focal_cls,
    )


def _build_gaussian_kernel(sigma, device, dtype):
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.max().clamp_min(1e-6)
    return kernel.view(1, 1, kernel.shape[0], kernel.shape[1])


def _dots_to_gaussian_heatmap(dot_maps, sigma):
    """将二值点图渲染为高斯热图，dot_maps 形状为 B×C×H×W 或 B×H×W。"""
    if dot_maps.ndim == 3:
        dot_maps = dot_maps.unsqueeze(1)

    batch_size, num_channels, height, width = dot_maps.shape
    kernel = _build_gaussian_kernel(sigma, dot_maps.device, dot_maps.dtype)
    padding = kernel.shape[-1] // 2
    heatmaps = F.conv2d(
        dot_maps.float(),
        kernel.expand(num_channels, 1, -1, -1),
        padding=padding,
        groups=num_channels,
    )
    return heatmaps.clamp(0.0, 1.0)


def _compute_point_heatmap_consistency_loss(
    et_dmap_all,
    et_dmap_class,
    gt_dots,
    gt_dots_all,
    train_classification=True,
):
    """点热图 MSE 监督 + 检测/分类热图一致性（sigmoid(det) vs max(softmax(cls))）。"""
    loss_point_det = None
    loss_point_cls = None
    loss_consistency = None
    total_loss = et_dmap_all.new_tensor(0.0)

    det_prob = torch.sigmoid(et_dmap_all)
    cls_prob = torch.softmax(et_dmap_class, dim=1)

    if use_point_heatmap_loss:
        # gt_dots / gt_dmap 与裁剪后的 et 同为输入分辨率，勿再 2:-2 裁剪。
        target_det = _dots_to_gaussian_heatmap(
            gt_dots_all.unsqueeze(1), point_heatmap_sigma
        )
        loss_point_det = F.mse_loss(det_prob, target_det)
        total_loss = total_loss + lamda_point_heatmap_det * loss_point_det
        if train_classification:
            target_cls = _dots_to_gaussian_heatmap(gt_dots, point_heatmap_sigma)
            loss_point_cls = F.mse_loss(cls_prob, target_cls)
            total_loss = total_loss + lamda_point_heatmap_cls * loss_point_cls

    if train_classification and use_det_cls_heatmap_consistency_loss:
        cls_union_prob = cls_prob.max(dim=1, keepdim=True).values
        loss_consistency = F.l1_loss(det_prob, cls_union_prob)
        total_loss = total_loss + lamda_det_cls_heatmap_consistency * loss_consistency

    return total_loss, loss_point_det, loss_point_cls, loss_consistency


if __name__ == "__main__":
    _validate_det_cls_loss_config()

    # checkpoints_save_path：当前训练过程中各轮 checkpoint 的保存路径。
    checkpoints_save_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    cluster_tmp_out = os.path.join(clustering_pseudo_gt_root, checkpoints_folder_name)

    if not os.path.exists(checkpoints_root_dir):
        os.mkdir(checkpoints_root_dir)

    if not os.path.exists(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)

    if not os.path.exists(clustering_pseudo_gt_root):
        os.mkdir(clustering_pseudo_gt_root)

    if not os.path.exists(cluster_tmp_out):
        os.mkdir(cluster_tmp_out)

    # 训练开始前备份项目根目录下的 Python 源码，便于回溯当前实验配置和实现版本。
    backup_python_files(checkpoints_save_path)

    # # log_file_path：训练日志文件的保存路径。
    # i=1
    # while(True):
    #     log_file_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name, f'train_log_{i}.txt')
    #     if(not os.path.exists(log_file_path)):
    #         break
    #     i +=1

    # log_file_path：训练日志文件的保存路径。
    log_file_path = os.path.join(
        checkpoints_root_dir, checkpoints_folder_name, f"train_log.txt"
    )

    start_epoch = (
        0  # 如果从 model_param_path 加载历史模型继续训练，可在这里指定起始轮次。
    )
    epoch_start_eval_prec = 1  # 从该轮开始，在验证集上评估预测结果的 F-score。
    gpu_or_cpu = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 自动选择设备，优先使用 CUDA，不可用时回退到 CPU。
    device = torch.device(gpu_or_cpu)
    # print_frequency         = 1  # 每个 epoch 的打印频率（当前未启用）。

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 初始化日志文件。
    log_file = open(log_file_path, "a+")
    log_file.write(
        "staged_training_config: "
        f"epoch_start_classification={epoch_start_classification}\n"
    )
    log_file.write(
        "det_cls_loss_config: "
        f"use_dice_loss={use_dice_loss}, use_focal_loss={use_focal_loss}, "
        f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
        f"lamda_focal_det={lamda_focal_det}, lamda_focal_cls={lamda_focal_cls}\n"
    )
    log_file.write(
        "uce_config: "
        f"use_uce_detection_loss={use_uce_detection_loss}, "
        f"use_uce_classification_loss={use_uce_classification_loss}, "
        f"uce_mc_samples={uce_mc_samples}, uce_alpha={uce_alpha}, "
        f"lamda_uce_det={lamda_uce_det}, lamda_uce_cls={lamda_uce_cls}\n"
    )
    log_file.write(
        "point_heatmap_config: "
        f"use_point_heatmap_loss={use_point_heatmap_loss}, "
        f"use_det_cls_heatmap_consistency_loss={use_det_cls_heatmap_consistency_loss}, "
        f"point_heatmap_sigma={point_heatmap_sigma}, "
        f"lamda_point_heatmap_det={lamda_point_heatmap_det}, "
        f"lamda_point_heatmap_cls={lamda_point_heatmap_cls}, "
        f"lamda_det_cls_heatmap_consistency={lamda_det_cls_heatmap_consistency}\n"
    )
    log_file.flush()

    # 配置训练集路径。
    train_image_root = os.path.join(train_data_root, "images")
    train_dmap_root = os.path.join(train_data_root, "gt_custom")
    train_dots_root = os.path.join(train_data_root, "gt_custom")
    train_dmap_subclasses_root = cluster_tmp_out
    train_dots_subclasses_root = train_dmap_subclasses_root
    train_kmap_root = os.path.join(train_data_root, "k_func_maps")

    # 配置验证集路径。
    test_image_root = os.path.join(test_data_root, "images")
    test_dmap_root = os.path.join(test_data_root, "gt_custom")
    test_dots_root = os.path.join(test_data_root, "gt_custom")
    test_dmap_subclasses_root = cluster_tmp_out
    test_dots_subclasses_root = test_dmap_subclasses_root
    test_kmap_root = os.path.join(test_data_root, "k_func_maps")

    dropout_prob = 0.2
    initial_pad = (
        126  # 由于卷积未使用 same padding，这里预先补边以保证最终输出尺寸与输入一致。
    )
    interpolate = "False"
    conv_init = "he"

    n_channels = 3
    n_classes = 3  # 细胞类别数（lymphocytes、tumor、stromal）。
    n_classes_out = n_classes + 1  # 输出类别数 = 细胞分类通道数 + 1 个细胞检测通道。
    class_indx = "1,2,3"  # 真值标签中类别通道对应的索引。
    n_clusters = 5  # 每个类别内部进一步聚类得到的簇数。
    n_classes2 = n_clusters * (n_classes)  # 细胞子类/聚类分类头的输出通道数。

    lr = 0.00005  # 学习率。
    batch_size = 8
    prints_per_epoch = 1  # 每个 epoch 保存/打印样例结果的频率控制参数。

    # 初始化 K function 的半径采样范围。每个类别都会在这些半径上计算空间统计特征。
    r_step = 15
    r_range = range(0, 100, r_step)
    r_arr = np.array([*r_range])
    r_classes = len(r_range)  # 单个类别对应的 K function 输出通道数。
    r_classes_all = r_classes * (
        n_classes
    )  # 所有类别合并后的 K function 输出通道总数。

    k_norm_factor = 100  # K 值归一化上限，即半径 r 内邻近细胞数的最大参考值，用于将 K function 归一化到 [0,1]。
    lamda_dice = 1
    # 主输出头（细胞检测 + 细胞分类）的 Dice 损失权重。
    lamda_subclasses = 1  # 子类/聚类分类输出头的 Dice 损失权重。
    lamda_k = 1  # K function 回归分支的 L1 损失权重。

    model = UnetVggMultihead(
        kwargs={
            "dropout_prob": dropout_prob,
            "initial_pad": initial_pad,
            "interpolate": interpolate,
            "conv_init": conv_init,
            "n_classes": n_classes,
            "n_channels": n_channels,
            "n_heads": 4,
            "head_classes": [1, n_classes, n_classes2, r_classes_all],
        }
    )
    if not (model_param_path is None):
        model.load_state_dict(torch.load(model_param_path), strict=False)
        log_file.write("model loaded \n")
        log_file.flush()
    model.to(device)

    # 初始化细胞检测分支使用的 Sigmoid 激活层。
    criterion_sig = nn.Sigmoid()
    # 初始化细胞分类分支使用的 Softmax 激活层。
    criterion_softmax = nn.Softmax(dim=1)
    # 初始化 K function 回归分支使用的 L1 损失。
    criterion_l1_sum = nn.L1Loss(reduction="sum")

    # 初始化优化器。这里联合优化编码器、瓶颈层、解码器和各个输出头。
    optimizer = torch.optim.Adam(
        list(model.final_layers_lst.parameters())
        + list(model.decoder.parameters())
        + list(model.bottleneck.parameters())
        + list(model.encoder.parameters()),
        lr,
    )

    # 初始化训练集 DataLoader。
    train_dataset = CellsDataset(
        train_image_root,
        train_dmap_root,
        train_dots_root,
        class_indx,
        train_dmap_subclasses_root,
        train_dots_subclasses_root,
        train_kmap_root,
        split_filepath=train_split_filepath,
        phase="train",
        fixed_size=448,
        max_scale=16,
    )
    # 这里的fixed_size>0时返回对应尺寸的随机裁剪块
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # 初始化验证集 DataLoader。
    test_dataset = CellsDataset(
        test_image_root,
        test_dmap_root,
        test_dots_root,
        class_indx,
        test_dmap_subclasses_root,
        test_dots_subclasses_root,
        test_kmap_root,
        split_filepath=test_split_filepath,
        phase="test",
        fixed_size=-1,
        max_scale=16,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初始化聚类阶段使用的训练集 DataLoader。
    # 该加载器不做随机打乱，用于在每个 epoch 开始前抽取特征并生成新的伪子类标签。
    simple_train_dataset = CellsDataset_simple(
        train_image_root,
        train_dmap_root,
        train_dots_root,
        class_indx,
        phase="test",
        fixed_size=-1,
        max_scale=16,
        return_padding=True,
    )
    # 这里设置fixed_size>0，导致同一batch内图像尺寸有可能不一致
    # 固定为 batch_size=1 避免默认 collate 的 stack 报错。
    simple_train_loader = torch.utils.data.DataLoader(
        simple_train_dataset, batch_size=1, shuffle=False
    )

    # 根据 prints_per_epoch 计算验证阶段的样例结果保存频率。
    # print_frequency = len(train_loader)//prints_per_epoch;
    print_frequency_test = len(test_loader) // prints_per_epoch

    best_epoch_filepath = None
    best_epoch = None
    best_f1_mean = 0
    best_prec_recall_diff = math.inf

    centroids = None
    for epoch in range(start_epoch, epochs):
        # 如果当前 epoch 的模型文件已存在，则跳过，避免重复训练。
        epoch_files = glob.glob(
            os.path.join(checkpoints_save_path, "mcspat_epoch_" + str(epoch) + "_*.pth")
        )
        if len(epoch_files) > 0:
            continue
        # 每个 epoch 开始前先做一次特征聚类，用于更新子类伪标签。
        print("epoch", epoch, "start clustering")
        # KEY：K-means聚类
        centroids = perform_clustering(
            model,
            simple_train_loader,
            n_clusters,
            n_classes,
            [feature_code["k-cell"], feature_code["subclass"]],
            train_dmap_subclasses_root,
            centroids,
        )
        print("epoch", epoch, "end clustering")

        train_classification = epoch >= epoch_start_classification
        phase_msg = "det+cls" if train_classification else "det_only"
        print("epoch", epoch, "training phase:", phase_msg)
        log_file.write(f"epoch= {epoch} training_phase= {phase_msg}\n")
        log_file.flush()

        # 训练阶段。
        model.train()
        log_file.write("epoch= " + str(epoch) + "\n")
        log_file.flush()

        # 初始化当前 epoch 的累计损失统计变量。
        epoch_loss = 0
        train_count = 0
        # train_loss_k = 0
        # train_loss_dice = 0
        # train_count_k = 0

        for i, (
            img,
            gt_dmap,
            gt_dots,
            gt_dmap_subclasses,
            gt_dots_subclasses,
            gt_kmap,
            img_name,
        ) in enumerate(tqdm(train_loader)):
            """
            img：输入图像。
            gt_dmap：细胞类别（lymphocytes、epithelial/tumor、stromal）的真值图，使用膨胀后的点标注表示。
                     它可以是二值掩码（本实验都是二值膨胀掩码），也可以是密度图；若为密度图，后续会转换成二值掩码。它标记一块连续的区域
            gt_dots：细胞类别对应的真值二值点图。它标记细胞的中心，一个细胞只有一个掩码
            gt_dmap_subclasses：细胞聚类子类的真值图，同样使用膨胀后的点表示。
                                它可以是二值掩码，也可以是密度图；若为密度图，后续会转换成二值掩码。
            gt_dots_subclasses：细胞聚类子类的真值二值点图。
            gt_kmap：真值 k-function 图。在每个细胞中心位置，保存以该细胞为中心的 cross k-functions。
            img_name：图像文件名。
            """
            gt_kmap /= k_norm_factor  # 对真值 K function 做归一化。
            img_name = img_name[0]
            train_count += 1

            img = img.to(device)
            gt_dmap = gt_dmap > 0  # 将真值图转换为二值掩码，兼容输入为密度图的情况。
            gt_dmap_subclasses = gt_dmap_subclasses > 0
            # 由分类真值图合并得到检测真值图。
            gt_dmap_all = gt_dmap.max(1)[
                0
            ]  # 按通道维取最大值，返回一个二元组（最大值，最大值的索引）。这里相当于看一下是否每个类别都为0，只要有一个类别不为0，那么就存在细胞，掩码为1
            gt_dots_all = gt_dots.max(1)[0]
            gt_dots = (gt_dots > 0).type(torch.FloatTensor)
            gt_dots_all = (gt_dots_all > 0).type(torch.FloatTensor)
            # 设置张量类型并移动到 GPU/计算设备。
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
            gt_dmap_subclasses = gt_dmap_subclasses.type(torch.FloatTensor)
            gt_kmap = gt_kmap.type(torch.FloatTensor)
            gt_dmap = gt_dmap.to(device)
            gt_dmap_all = gt_dmap_all.to(device)
            gt_dmap_subclasses = gt_dmap_subclasses.to(device)
            gt_kmap = gt_kmap.to(device)
            gt_dots = gt_dots.to(device)
            gt_dots_all = gt_dots_all.to(device)

            # KEY：前向传播，模型会输出检测、分类、子类分类和 K function 回归四个分支。
            et_dmap_lst = model(img)
            et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]  # 细胞检测分支的预测结果。
            et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]  # 细胞分类分支的预测结果。
            et_dmap_subclasses = et_dmap_lst[2][
                :, :, 2:-2, 2:-2
            ]  # 细胞聚类子类分支的预测结果。
            et_kmap = (
                et_dmap_lst[3][:, :, 2:-2, 2:-2] ** 2
            )  # cross K-functions 的回归结果，平方后保证非负。
            # 上面每个预测图都裁剪掉边界的2个像素，为了避免干扰？

            # K function 损失只在检测掩码区域上计算，避免背景区域干扰。
            loss_l1_k = et_dmap_all.new_tensor(0.0)
            if use_k_function_loss:
                k_loss_mask = gt_dmap_all.clone().unsqueeze(1)
                loss_l1_k = criterion_l1_sum(
                    et_kmap * k_loss_mask, gt_kmap * k_loss_mask
                ) / (k_loss_mask.sum() * r_classes_all)

            et_subclasses_sig = criterion_softmax(
                et_dmap_subclasses
            )  # 子分类结果（softmax化）

            intersection = (et_subclasses_sig * gt_dmap_subclasses).sum()
            union = (et_subclasses_sig**2).sum() + (gt_dmap_subclasses**2).sum()
            loss_dice_subclass = 1 - ((2 * intersection + 1) / (union + 1))

            (
                det_cls_loss,
                loss_dice_class,
                loss_dice_all,
                loss_uce_det,
                loss_uce_cls,
                loss_focal_det,
                loss_focal_cls,
            ) = _compute_det_cls_loss(
                model,
                img,
                et_dmap_all,
                et_dmap_class,
                gt_dmap_all,
                gt_dmap,
                lamda_dice,
                train_classification=train_classification,
            )

            loss_point_det = None
            loss_point_cls = None
            loss_heatmap_consistency = None
            point_heatmap_loss = et_dmap_all.new_tensor(0.0)
            if use_point_heatmap_loss or use_det_cls_heatmap_consistency_loss:
                (
                    point_heatmap_loss,
                    loss_point_det,
                    loss_point_cls,
                    loss_heatmap_consistency,
                ) = _compute_point_heatmap_consistency_loss(
                    et_dmap_all,
                    et_dmap_class,
                    gt_dots,
                    gt_dots_all,
                    train_classification=train_classification,
                )

            # KEY：汇总loss
            if use_k_function_loss and use_subclass_loss:
                loss = (
                    det_cls_loss
                    + point_heatmap_loss
                    + lamda_dice * lamda_subclasses * loss_dice_subclass
                )
                if not math.isnan(loss_l1_k.item()):
                    loss += loss_l1_k * lamda_k
            elif use_k_function_loss:
                loss = det_cls_loss + point_heatmap_loss
                if not math.isnan(loss_l1_k.item()):
                    loss += loss_l1_k * lamda_k
            elif use_subclass_loss:
                raise Exception("Subclass must be with K-Function!")
            else:
                loss = det_cls_loss + point_heatmap_loss

            # 反向传播并更新参数。
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 按当前训练配置自适应记录可用损失项，避免引用未定义变量。
            log_items = [
                f"epoch: {epoch}",
                f"i: {i}",
                f"loss_total: {loss.item()}",
            ]
            if (
                use_dice_loss
                and loss_dice_class is not None
                and loss_dice_all is not None
            ):
                log_items.append(f"loss_dice_all: {loss_dice_all.item()}")
                if train_classification:
                    log_items.append(f"loss_dice_class: {loss_dice_class.item()}")
            if use_focal_loss and loss_focal_det is not None:
                log_items.append(f"loss_focal_det: {loss_focal_det.item()}")
                if train_classification and loss_focal_cls is not None:
                    log_items.append(f"loss_focal_cls: {loss_focal_cls.item()}")
            if use_subclass_loss:
                log_items.append(f"loss_dice_subclass: {loss_dice_subclass.item()}")
            if use_uce_detection_loss and loss_uce_det is not None:
                log_items.append(f"loss_uce_det: {loss_uce_det.item()}")
            if (
                train_classification
                and use_uce_classification_loss
                and loss_uce_cls is not None
            ):
                log_items.append(f"loss_uce_cls: {loss_uce_cls.item()}")
            if use_point_heatmap_loss and loss_point_det is not None:
                log_items.append(f"loss_point_heatmap_det: {loss_point_det.item()}")
            if (
                train_classification
                and use_point_heatmap_loss
                and loss_point_cls is not None
            ):
                log_items.append(f"loss_point_heatmap_cls: {loss_point_cls.item()}")
            if (
                train_classification
                and use_det_cls_heatmap_consistency_loss
                and loss_heatmap_consistency is not None
            ):
                log_items.append(
                    f"loss_det_cls_heatmap_consistency: {loss_heatmap_consistency.item()}"
                )

            loss_l1_k_val = loss_l1_k.item()
            if use_k_function_loss:
                if math.isnan(loss_l1_k_val):
                    log_items.append("loss_l1_k: nan(skipped)")
                else:
                    log_items.append(f"loss_l1_k: {loss_l1_k_val}")

            log_file.write("   ".join(log_items) + "\n")
            log_file.flush()

        log_file.write(
            "epoch: "
            + str(epoch)
            + " train loss: "
            + str(epoch_loss / train_count)
            + "\n"
        )
        log_file.flush()
        epoch_loss = epoch_loss / train_count

        # break

        # KEY：验证
        # 在验证集上进行测试/评估。
        model.eval()
        err = np.array([0 for s in range(n_classes_out)])
        loss_val = 0
        loss_val_k_wo_nan = 0
        loss_val_k = 0
        loss_val_dice = 0
        loss_val_focal = 0
        loss_val_dice2 = 0
        tp_count_all = np.zeros((n_classes_out))
        fp_count_all = np.zeros((n_classes_out))
        fn_count_all = np.zeros((n_classes_out))
        test_count_k = 0
        for i, (
            img,
            gt_dmap,
            gt_dots,
            gt_dmap_subclasses,
            gt_dots_subclasses,
            gt_kmap,
            img_name,
        ) in enumerate(tqdm(test_loader)):
            """
            img：输入图像。
            gt_dmap：细胞类别（lymphocytes、epithelial/tumor、stromal）的真值图，使用膨胀后的点标注表示。
                     它可以是二值掩码，也可以是密度图；若为密度图，后续会转换成二值掩码。
            gt_dots：细胞类别对应的真值二值点图。
            gt_dmap_subclasses：细胞聚类子类的真值图，同样使用膨胀后的点表示。
                                它可以是二值掩码，也可以是密度图；若为密度图，后续会转换成二值掩码。
            gt_dots_subclasses：细胞聚类子类的真值二值点图。
            gt_kmap：真值 k-function 图。在每个细胞中心位置，保存以该细胞为中心的 cross k-functions。
            img_name：图像文件名。
            """
            gt_kmap /= k_norm_factor  # 对真值 K function 做归一化。
            img_name = img_name[0]
            img = img.to(device)
            # 将真值图转换为二值掩码，兼容输入为密度图的情况。
            gt_dmap = gt_dmap > 0
            # 由分类真值图合并得到检测真值图。
            gt_dmap_all = gt_dmap.max(1)[0]
            gt_dots_all = gt_dots.max(1)[0]
            # 设置张量类型并移动到 GPU/计算设备。
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
            gt_kmap = gt_kmap.type(torch.FloatTensor)
            gt_kmap = gt_kmap.to(device)
            gt_dmap_t = gt_dmap.to(device)
            gt_dmap_all_t = gt_dmap_all.to(device)
            # unsqueeze(1) 将 (B,H,W) 扩展为 (B,1,H,W)，使其能与 et_kmap (B,C,H,W) 正确广播
            k_loss_mask = (
                gt_dmap_all.clone().to(device).unsqueeze(1)
            )  # K function 损失仅在膨胀点掩码区域上计算。

            # 将真值图转为 numpy，便于后续基于连通域和点匹配的评估逻辑处理。
            gt_dots = gt_dots.detach().cpu().numpy()
            gt_dots_all = gt_dots_all.detach().cpu().numpy()
            gt_dmap = gt_dmap.detach().cpu().numpy()
            gt_dmap_all = gt_dmap_all.detach().cpu().numpy()

            # 前向传播，得到四个预测分支的输出。
            et_dmap_lst = model(img)
            et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]  # 细胞检测分支的预测结果。
            et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]  # 细胞分类分支的预测结果。
            et_dmap_subclasses = et_dmap_lst[2][
                :, :, 2:-2, 2:-2
            ]  # 细胞聚类子类分支的预测结果。
            et_kmap = (
                et_dmap_lst[3][:, :, 2:-2, 2:-2] ** 2
            )  # cross K-functions 的回归结果，平方后保证非负。

            loss_dice_class = None
            loss_dice_all = None
            loss_focal_det = None
            loss_focal_cls = None
            if use_dice_loss:
                loss_dice_class, loss_dice_all = _compute_dice_losses(
                    et_dmap_all, et_dmap_class, gt_dmap_all_t, gt_dmap_t
                )
                if train_classification:
                    loss_val_main = (loss_dice_class + loss_dice_all).item()
                else:
                    loss_val_main = loss_dice_all.item()
                loss_val_dice += loss_val_main
            elif use_focal_loss:
                loss_focal_det, loss_focal_cls = _compute_focal_losses(
                    et_dmap_all, et_dmap_class, gt_dmap_all_t, gt_dmap_t
                )
                if train_classification:
                    loss_val_main = (
                        lamda_focal_det * loss_focal_det
                        + lamda_focal_cls * loss_focal_cls
                    ).item()
                else:
                    loss_val_main = (lamda_focal_det * loss_focal_det).item()
                loss_val_focal += loss_val_main

            # 对检测分支应用 Sigmoid，对分类分支应用 Softmax。
            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            # K function 损失只在检测掩码区域上计算。
            loss_l1_k = criterion_l1_sum(
                et_kmap * (k_loss_mask), gt_kmap * (k_loss_mask)
            ) / (k_loss_mask.sum() * r_classes_all)

            # 按设定频率保存部分验证样例的预测结果，便于人工查看训练过程。
            if i % print_frequency_test == 0:
                io.imsave(
                    os.path.join(
                        checkpoints_save_path,
                        "test" + "_indx" + str(i) + "_img" + ".png",
                    ),
                    (img.squeeze().detach().cpu().numpy() * 255)
                    .transpose(1, 2, 0)
                    .astype(np.uint8),
                )
                for s in range(n_classes):
                    io.imsave(
                        os.path.join(
                            checkpoints_save_path,
                            "epoch"
                            + str(epoch)
                            + "_test"
                            + "_indx"
                            + str(i)
                            + "_likelihood"
                            + "_s"
                            + str(s)
                            + ".png",
                        ),
                        (et_class_sig[:, s, :, :] * 255).squeeze().astype(np.uint8),
                    )
                    io.imsave(
                        os.path.join(
                            checkpoints_save_path,
                            "test" + "_indx" + str(i) + "_gt" + "_s" + str(s) + ".png",
                        ),
                        (gt_dmap[:, s, :, :] * 255).squeeze().astype(np.uint8),
                    )
                io.imsave(
                    os.path.join(
                        checkpoints_save_path,
                        "epoch"
                        + str(epoch)
                        + "_test"
                        + "_indx"
                        + str(i)
                        + "_likelihood"
                        + "_all"
                        + ".png",
                    ),
                    (et_all_sig * 255).squeeze().astype(np.uint8),
                )
                io.imsave(
                    os.path.join(
                        checkpoints_save_path,
                        "test" + "_indx" + str(i) + "_gt" + "_all" + ".png",
                    ),
                    (gt_dmap_all * 255).squeeze().astype(np.uint8),
                )

            # 累计验证阶段的 K function 损失。
            loss_val_k += loss_l1_k.item()
            if not math.isnan(loss_l1_k.item()):
                loss_val_k_wo_nan += loss_l1_k.item()
                test_count_k += 1

            if use_k_function_loss:
                if math.isnan(loss_l1_k.item()):
                    val_log_items = ["loss_l1_k: nan"]
                else:
                    val_log_items = [f"loss_l1_k: {loss_l1_k.item()}"]
            else:
                val_log_items = []
            if use_dice_loss:
                val_log_items.append(f"loss_dice: {loss_val_main}")
                if not train_classification and loss_dice_all is not None:
                    val_log_items.append(f"loss_dice_all: {loss_dice_all.item()}")
            elif use_focal_loss:
                val_log_items.append(f"loss_focal_det: {loss_focal_det.item()}")
                if train_classification and loss_focal_cls is not None:
                    val_log_items.append(f"loss_focal_cls: {loss_focal_cls.item()}")
                val_log_items.append(f"loss_focal: {loss_val_main}")
            print_msg = "epoch {} test {} {}".format(epoch, i, " ".join(val_log_items))
            print(print_msg)
            log_file.write(print_msg + "\n")
            log_file.flush()

            # KEY：从指定轮次开始，基于点匹配统计 TP/FP/FN，并计算 F-score。
            """
                检测：对检测结果做sigmoid后二值化，得到检测图（连通域），看每个连通域是否和真实的细胞中心点相交，如果相交则记为TP，然后删除该真实中心点（避免一个真实点重复匹配多个预测结果）。如果该预测连通域没有匹配任何真实点，则记为FP。如果某个真实点没有被任何预测结果所匹配，则记为FN。
                分类：对分类结果做softmax后argmax得到类别编号，然后转换为某个类别通道的二值掩码（连通域）。为了将分类的连通域转换成单个点，首先先把检测结果的连通域寻找中心点，然后将该中心点与分类掩码相乘，得到最终的分类掩码（单点），与真实掩码计算相关指标。
            """
            if epoch >= epoch_start_eval_prec:
                # 对检测输出施加 0.5 阈值，并转成二值掩码。
                e_hard = filters.apply_hysteresis_threshold(
                    et_all_sig.squeeze(), 0.5, 0.5
                )
                e_hard2 = (e_hard > 0).astype(np.uint8)  # [H, W]
                e_hard2_all = e_hard2.copy()

                # 在二值检测图上寻找轮廓，并用轮廓中心作为预测细胞中心点。
                e_dot = np.zeros(
                    (img.shape[-2], img.shape[-1])
                )  # [H, W]，用预测的检测掩码寻找连通域中心
                contours, hierarchy = cv2.findContours(
                    e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                for idx in range(len(contours)):
                    contour_i = contours[idx]
                    M = cv2.moments(contour_i)
                    if M["m00"] == 0:
                        continue
                    cx = round(M["m10"] / M["m00"])
                    cy = round(M["m01"] / M["m00"])
                    e_dot[cy, cx] = 1
                e_dot_all = e_dot.copy()

                tp_count = 0  # 当前图像中的真正例数量。
                fp_count = 0  # 当前图像中的假正例数量。
                fn_count = 0  # 当前图像中的假负例数量。
                # g_dot_vis 初始保存该图像中的所有检测真值点，匹配成功后会被逐个移除。
                g_dot_vis = gt_dots_all.copy().squeeze()
                # 对预测的检测二值图做连通域分析。
                e_hard2_comp = label(e_hard2)
                e_hard2_comp_all = e_hard2_comp.copy()
                # 遍历每个连通域：
                # 如果该连通域与任一真值点相交，则记为 TP；否则记为 FP。
                # 一旦匹配为 TP，会从 g_dot_vis 中删除对应真值点，避免重复匹配。
                # 注意：即使一个连通域覆盖多个真值点，也只记作一个 TP。
                for l in range(1, e_hard2_comp.max() + 1):
                    e_hard2_comp_l = e_hard2_comp == l
                    M = moments(e_hard2_comp_l)
                    (y, x) = int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0])
                    if (e_hard2_comp_l * g_dot_vis).sum() > 0:  # 真正例。
                        tp_count += 1
                        (yg, xg) = np.where((e_hard2_comp_l * g_dot_vis) > 0)
                        yg = yg[0]
                        xg = xg[0]
                        g_dot_vis[yg, xg] = 0
                    else:  # ((e_hard2_comp_l * g_dot_vis).sum()==0): # 假正例。
                        fp_count += 1
                # g_dot_vis 中剩余未匹配到的真值点即为假负例。
                fn_points = np.where(g_dot_vis > 0)
                fn_count = len(fn_points[0])

                # 将当前图像的检测 TP/FP/FN 累加到整体统计量中。
                tp_count_all[-1] = tp_count_all[-1] + tp_count
                fp_count_all[-1] = fp_count_all[-1] + fp_count
                fn_count_all[-1] = fn_count_all[-1] + fn_count

                # 取分类分支的 argmax 作为每个像素的预测细胞类别。
                et_class_argmax = et_class_sig.squeeze().argmax(
                    axis=0
                )  # [H, W]，每个位置记录类别编号
                e_hard2_all = e_hard2.copy()
                # 对每个细胞类别分别统计 TP、FP、FN，逻辑与检测分支类似。
                for s in range(n_classes):
                    g_count = gt_dots[0, s, :, :].sum()  # 统计类别s的细胞数量

                    e_hard2 = et_class_argmax == s  # 记录哪些位置和真实类别一致

                    e_dot = (
                        e_hard2 * e_dot_all
                    )  # 和预测的检测图相乘，避免出现某个位置有分类结果而没有检测结果

                    g_dot = gt_dots[
                        0, s, :, :
                    ].squeeze()  # [H, W]，每个位置取0/1，表示类别s的二值点图

                    tp_count = 0
                    fp_count = 0
                    fn_count = 0
                    g_dot_vis = g_dot.copy()  # [H, W]，类别s的真实二值点图（细胞中心）
                    e_dots_tuple = np.where(
                        e_dot > 0
                    )  # [H, W]，类别s的预测二值点图（连通域中心）
                    for idx in range(len(e_dots_tuple[0])):
                        cy = e_dots_tuple[0][idx]
                        cx = e_dots_tuple[1][idx]
                        l = e_hard2_comp_all[cy, cx]
                        e_hard2_comp_l = e_hard2_comp == l
                        if (e_hard2_comp_l * g_dot_vis).sum() > 0:  # 真正例。
                            tp_count += 1
                            (yg, xg) = np.where((e_hard2_comp_l * g_dot_vis) > 0)
                            yg = yg[0]
                            xg = xg[0]
                            g_dot_vis[yg, xg] = 0
                        else:  # ((e_hard2_comp_l * g_dot_vis).sum()==0): # 假正例。
                            fp_count += 1
                    fn_points = np.where(g_dot_vis > 0)
                    fn_count = len(fn_points[0])

                    tp_count_all[s] = tp_count_all[s] + tp_count
                    fp_count_all[s] = fp_count_all[s] + fp_count
                    fn_count_all[s] = fn_count_all[s] + fn_count

            del (
                img,
                gt_dmap,
                gt_dmap_all,
                gt_dmap_subclasses,
                gt_kmap,
                et_dmap_all,
                et_dmap_class,
                et_kmap,
                gt_dots,
            )

        saved = False

        val_count = max(1, len(test_loader))
        if use_dice_loss:
            val_loss_msg = (
                f"epoch {epoch} val loss_dice mean: {loss_val_dice / val_count}"
            )
        elif use_focal_loss:
            val_loss_msg = (
                f"epoch {epoch} val loss_focal mean: {loss_val_focal / val_count}"
            )
        else:
            val_loss_msg = None
        if val_loss_msg is not None:
            print(val_loss_msg)
            log_file.write(val_loss_msg + "\n")
            log_file.flush()

        precision_all = np.zeros((n_classes_out))
        recall_all = np.zeros((n_classes_out))
        f1_all = np.zeros((n_classes_out))
        if epoch >= epoch_start_eval_prec:
            count_all = tp_count_all.sum() + fn_count_all.sum()
            for s in range(n_classes_out):
                if tp_count_all[s] + fp_count_all[s] == 0:
                    precision_all[s] = 1
                else:
                    precision_all[s] = tp_count_all[s] / (
                        tp_count_all[s] + fp_count_all[s]
                    )
                if tp_count_all[s] + fn_count_all[s] == 0:
                    recall_all[s] = 1
                else:
                    recall_all[s] = tp_count_all[s] / (
                        tp_count_all[s] + fn_count_all[s]
                    )
                if precision_all[s] + recall_all[s] == 0:
                    f1_all[s] = 0
                else:
                    f1_all[s] = (
                        2
                        * (precision_all[s] * recall_all[s])
                        / (precision_all[s] + recall_all[s])
                    )
                print_msg = f"epoch {epoch} s {s} precision_all {precision_all[s]} recall_all {recall_all[s]} f1_all {f1_all[s]}"
                print(print_msg)
                log_file.write(print_msg + "\n")
                log_file.flush()
            print_msg = f"epoch {epoch} all precision_all {precision_all.mean()} recall_all {recall_all.mean()} f1_all {f1_all.mean()}"
            print(print_msg)
            log_file.write(print_msg + "\n")
            log_file.flush()
            print_msg = f"epoch {epoch} classes precision_all {precision_all[:-1].mean()} recall_all {recall_all[:-1].mean()} f1_all {f1_all[:-1].mean()}"
            print(print_msg)
            log_file.write(print_msg + "\n")
            log_file.flush()

        # KEY：权重保存
        # 根据验证集上的 F-score 判断当前是否为最佳 epoch。
        model_save_postfix = ""
        is_best_epoch = False
        # if (f1_all.mean() > best_f1_mean):
        if f1_all.mean() - best_f1_mean >= 0.005:
            model_save_postfix += "_f1"
            best_f1_mean = f1_all.mean()
            best_prec_recall_diff = abs(recall_all.mean() - precision_all.mean())
            is_best_epoch = True
        elif (
            abs(f1_all.mean() - best_f1_mean) < 0.005
        ) and abs(  # F-score 略低，但 precision 与 recall 的差距更小。
            recall_all.mean() - precision_all.mean()
        ) < best_prec_recall_diff:
            model_save_postfix += "_pr-diff"
            best_f1_mean = f1_all.mean()
            best_prec_recall_diff = abs(recall_all.mean() - precision_all.mean())
            is_best_epoch = True
        # if (recall_all.mean() > best_recall_mean):
        #     model_save_postfix += '_rec'
        #     best_recall_mean = recall_all.mean()
        #     is_best_epoch = True

        # 如果当前 epoch 达到目前最佳表现，则保存 checkpoint 与聚类中心。
        if (saved == False) and (model_save_postfix != ""):
            print("epoch", epoch, "saving")
            new_epoch_filepath = os.path.join(
                checkpoints_save_path,
                "mcspat_epoch_" + str(epoch) + model_save_postfix + ".pth",
            )
            torch.save(
                model.state_dict(), new_epoch_filepath
            )  # 仅在验证指标改善时保存模型。
            # 仅保留当前最优权重，删除上一个最优权重文件以节省磁盘空间。
            if (
                (best_epoch_filepath is not None)
                and (best_epoch_filepath != new_epoch_filepath)
                and os.path.exists(best_epoch_filepath)
            ):
                os.remove(best_epoch_filepath)
            centroids.dump(
                os.path.join(
                    checkpoints_save_path, "epoch{}_centroids.npy".format(epoch)
                )
            )
            saved = True
            print_msg = f"epoch {epoch} saved."
            print(print_msg)
            log_file.write(print_msg + "\n")
            log_file.flush()
            if is_best_epoch:
                best_epoch_filepath = new_epoch_filepath
                best_epoch = epoch

    log_file.close()
