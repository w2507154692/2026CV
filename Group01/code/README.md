# MCSpatNet

基于空间统计（K-function）的多任务细胞检测与分类网络，并提供 **E2ECR**、**Faster R-CNN**、**HoVer-Net** 等对比实验代码。

本仓库主要包含三条实验/应用线：

| 实验线 | 目录 | 任务形式 | 典型数据集 |
|--------|------|----------|------------|
| **MCSpatNet（主方法）** | 项目根目录 | 像素级检测 + 分类 + K-function 回归 | CoNSeP、MoNuSAC、BRCA-M2C |
| **对比实验** | `contrast/` | 点检测（E2ECR）、框检测（Faster R-CNN）、实例分割（HoVer-Net） | CoNSeP、MoNuSAC、BRCA-M2C |
| **Web 交互平台** | `src/src/` | 浏览器上传 patch，在线检测与分类可视化 | 单张 CoNSeP 风格 patch |

---

## 目录结构

```
MCSpatNet/
├── 01_train_mcspat.py          # MCSpatNet 训练
├── 02_test_vis_mcspat.py       # MCSpatNet 测试与可视化
├── 03_eval_localization_fscore.py  # MCSpatNet 点定位 F-score 评估
├── model_arch.py               # U-Net + VGG 多分支网络
├── my_dataloader_w_kfunc.py    # 训练用 DataLoader（含 K-map）
├── my_dataloader.py            # 测试/简单 DataLoader
├── cluster_helper.py           # 子类聚类辅助
├── spatial_analysis_utils_v2_sh.py  # K-function 计算（R 接口）
├── data/                       # MCSpatNet 主实验数据
├── data_splits/                # MCSpatNet 数据划分
├── data_prepare/               # 数据预处理脚本
├── exp/                        # MCSpatNet 训练/测试输出
├── src/                        # Web 交互平台
│   └── src/
│       ├── app.py              # Flask 后端（推理 API）
│       ├── model_arch.py       # 模型定义（与根目录同步副本）
│       ├── requirements.txt    # Web 端依赖
│       ├── templates/          # 前端页面
│       ├── static/             # CSS / JS
│       ├── uploads/            # 上传图像（运行时生成）
│       ├── outputs/            # 推理结果（运行时生成）
│       └── models/             # 可选：放置 .pth 权重
└── contrast/                   # 对比实验
    ├── 01_train.py             # 统一训练入口（E2ECR / Faster R-CNN）
    ├── 02_test.py              # 统一测试入口
    ├── e2ecr_train.py / e2ecr_test.py
    ├── faster_rcnn_train.py / faster_rcnn_test.py
    ├── e2ecr_dataset.py / faster_rcnn_dataset.py
    ├── data/                   # 对比实验数据
    ├── exp/                    # 对比实验输出
    └── HoverNet/               # HoVer-Net 训练/推理/评估
        ├── run_train.py
        ├── run_infer.py
        └── evaluate.py         # 与 MCSpatNet 点标注对齐的评估
```

---

## 环境配置

### 推荐环境

- Python 3.10+
- PyTorch 2.x + CUDA（按本机 GPU 安装）
- 建议使用虚拟环境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy scipy scikit-image opencv-python pillow tqdm matplotlib pandas
pip install albumentations   # E2ECR 训练增强需要
```

### MCSpatNet 额外依赖

- **K-function 预计算**：`data_prepare/2_calc_kmaps.py` 依赖 R 及 `spatial_analysis_utils_v2_sh.py` 中的 R 脚本接口。若关闭 K-function 损失（`use_k_function_loss=False`），可跳过 K-map 生成。

### HoVer-Net 额外依赖

HoVer-Net 有独立依赖，见 `contrast/HoverNet/requirements.txt` 与 `contrast/HoverNet/README.md`。建议在 `contrast/HoverNet/` 下单独配置环境。

---

## 数据准备

### MCSpatNet 数据格式

每个数据集目录通常包含：

```
<dataset_root>/
├── images/           # RGB patch，*.png
├── gt_custom/        # 标注
│   ├── *.npy         # 膨胀点图（多通道）
│   └── *_gt_dots.npy # 细胞中心点图（多通道）
└── k_func_maps/      # K-function 图（训练时需要）
    └── *_gt_kmap.npy
```

**CoNSeP 类别映射（3 类）**：1=lymphocyte，2=tumor，3=stromal。

**预处理脚本**（在 `data_prepare/` 下执行，需先修改脚本内路径）：

| 脚本 | 作用 |
|------|------|
| `1_generate_dot_maps_consep.py` | 从 CoNSeP 原始 mat 标注生成点图/膨胀图 |
| `1_generate_dot_maps_monusac.py` | MoNuSAC 点图生成 |
| `2_calc_kmaps.py` | 由点标注计算 cross K-function 图 |

**数据划分**：`data_splits/<dataset>/train_split.txt`、`val_split.txt`（每行一个图像文件名）。

### 对比实验数据格式

位于 `contrast/data/`，常见子目录：

| 目录 | 用途 | 方法 |
|------|------|------|
| `CoNSeP_MCSpatNet_point` | 点标注 | E2ECR |
| `MoNuSAC_point` | 点标注（CSV） | E2ECR |
| `CoNSeP_box` / `CoNSeP_box_patch` | 边界框 CSV | Faster R-CNN |
| `MoNuSAC_box` | 边界框 CSV | Faster R-CNN |
| `MoNuSAC_HoverNet` | HoVer-Net patch（`.npy`） | HoVer-Net |

**Faster R-CNN 数据目录结构**：

```
<box_dataset>/
├── images/{train,val,test}/*.png
├── annotations/boxes.csv
└── metadata/classes.csv
```

**E2ECR 点数据**：`images/` + `gt_custom/*_gt_dots.npy` + `data_splits/`（CoNSeP/BRCA-M2C），或 `MoNuSAC_point` 的 `annotations/points.csv` 格式。

---

## MCSpatNet 实验流程

> 所有 MCSpatNet 脚本通过**文件顶部全局变量**配置路径与超参，修改后直接运行（无 argparse）。

### 1. 训练

```bash
python 01_train_mcspat.py
```

**主要配置项**（脚本顶部）：

| 变量 | 说明 |
|------|------|
| `checkpoints_folder_name` | 实验名，输出到 `exp/<name>/` |
| `train_data_root` / `test_data_root` | 训练/验证数据根目录 |
| `train_split_filepath` / `test_split_filepath` | 划分文件 |
| `epochs` | 训练轮数（CoNSeP 建议 300） |
| `use_k_function_loss` | 是否启用 K-function L1 损失 |
| `use_dice_loss` / `use_focal_loss` | 主分支 Dice / Focal（互斥） |
| `use_uce_*` | U-CE 损失（与 Focal 互斥） |
| `use_point_heatmap_loss` | 点热图高斯监督 |
| `use_det_cls_heatmap_consistency_loss` | 检测-分类热图一致性 |
| `epoch_start_classification` | 分阶段训练：前 N epoch 仅检测，之后加入分类 |

**输出**：

- `exp/<checkpoints_folder_name>/mcspat_epoch_*.pth` — 验证 F1 提升时保存的最优权重
- `exp/<checkpoints_folder_name>/train_log.txt` — 训练日志
- `exp/<checkpoints_folder_name>/code/` — 训练时源码备份

**网络输出头**：检测（1 通道）、分类（n_classes）、子类聚类（可选）、K-function（r_classes × n_classes 通道）。

### 2. 测试与可视化

```bash
python 02_test_vis_mcspat.py
```

修改 `checkpoints_folder_name`、`epoch`、`test_data_root` 等，加载对应 checkpoint，在测试集上生成预测点图与可视化。

**输出目录**：`exp/<checkpoints_folder_name>_e<epoch>/`

- `*_pred_dots_class*.npy` — 各类预测点图
- `*_gt_dots_class*.npy` — GT 点图（供评估）
- 可视化 PNG（`visualize=True` 时）

### 3. F-score 评估

```bash
python 03_eval_localization_fscore.py
```

将 `data_dir` 指向 `02_test_vis_mcspat.py` 的输出目录（如 `exp/exp18_consep_e298/`）。

**评估方式**：在 1~`max_dist_thresh` 像素距离内做点匹配，统计 TP/FP/FN，计算各类及整体检测的 Precision / Recall / F1。

**输出**：`fscore_eval.txt`（写在 `data_dir` 下）。

---

## Web 交互平台

基于 **Flask + PyTorch** 的浏览器端细胞检测演示系统，推理逻辑与 `02_test_vis_mcspat.py` 保持一致（滞后阈值检测、连通域质心、分类 argmax 着色）。

### 功能

- 拖拽/点击上传组织病理 patch（PNG、JPG、TIF，最大 16MB）
- 在线推理：细胞检测、3 类分类、检测热力图
- 展示细胞总数、各类别计数、原图/检测图/分类图/热力图对比
- 一键下载检测图、分类图、热力图及 JSON 统计

### 环境

Web 端有独立依赖，见 `src/src/requirements.txt`（Flask、torch、opencv、scikit-image 等）。可与主项目共用同一虚拟环境，或单独安装：

```bash
cd src/src
pip install -r requirements.txt
```

### 准备模型权重

启动前需准备训练好的 MCSpatNet 权重（与 `MODEL_CONFIG` 结构一致，`n_classes=3`）：

```bash
# 方式 1：复制到 Web 目录 models/
cp ../../exp/<your_exp>/mcspat_epoch_*.pth src/src/models/

# 方式 2：保留在 exp/ 下，由 app.py 自动搜索
# 默认会查找 ../exp/exp12_consep_no_cluster_e263/ 或 ../exp/exp12_consep_no_cluster/
```

也可在 `app.py` 的 `load_model(model_path=...)` 中显式指定 checkpoint 路径。

### 启动服务

```bash
cd src/src
python app.py
```

浏览器访问：**http://localhost:5000**

默认监听 `0.0.0.0:5000`；修改端口或关闭 debug 模式请编辑 `app.py` 末尾 `app.run(...)`。

### 页面与 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面（上传与结果展示） |
| `/api/detect` | POST | 上传图像，返回检测/分类结果（`multipart/form-data`，字段 `image`） |
| `/api/model_status` | GET | 模型是否已加载、当前 device |
| `/api/result/<filename>` | GET | 获取输出图像（检测/分类/热力图） |
| `/api/upload/<filename>` | GET | 获取上传的原图 |

**`/api/detect` 响应示例**：

```json
{
  "success": true,
  "total_count": 150,
  "class_details": [
    {"class_id": 0, "class_name": "炎症细胞", "count": 45, "color": [0, 162, 232]},
    {"class_id": 1, "class_name": "上皮细胞", "count": 80, "color": [255, 0, 0]},
    {"class_id": 2, "class_name": "间质细胞", "count": 25, "color": [0, 255, 0]}
  ],
  "result_image": "/api/result/xxx_result.png",
  "class_image": "/api/result/xxx_class.png",
  "heatmap_image": "/api/result/xxx_heatmap.png",
  "original_image": "/api/upload/xxx.png"
}
```

### 类别与可视化

Web 端默认 **3 类**（与 CoNSeP 训练配置一致），颜色定义见 `app.py` 中 `COLOR_SET`：

| class_id | 名称 | 颜色 |
|----------|------|------|
| 0 | 炎症细胞 | 蓝 |
| 1 | 上皮细胞 | 红 |
| 2 | 间质细胞 | 绿 |

检测图：所有细胞中心以**黑色**方块标记；分类图：按类别着色。

### 注意事项

- 推理为**单张 patch**，不做 WSIs 分块；大图请先裁剪为与训练相近尺寸的 patch。
- 后处理阈值与 `02_test_vis_mcspat.py` 相同（`THRESH_LOW/HIGH=0.5`，`SIZE_THRESH=5`），可在 `app.py` 顶部修改。
- GPU 不可用时自动回退 CPU；显存不足可在 `app.py` 中设 `DEVICE = torch.device('cpu')`。
- 更详细的 Web 端说明见 [`src/src/README.md`](src/src/README.md)。

---

## 对比实验流程

> 在 `contrast/` 目录下运行，或确保 Python 路径包含 `contrast/`。

### E2ECR / Faster R-CNN（统一入口）

#### 训练

编辑 `contrast/01_train.py`：

```python
CHECKPOINTS_FOLDER_NAME = 'exp32_faster_rcnn_consep'
TYPE = 'faster_rcnn'   # 或 'e2ecr'
SEED = 42
```

```bash
cd contrast
python 01_train.py
```

各方法的具体超参在对应模块内修改：

- **E2ECR**：`e2ecr_train.py` — `DATA_ROOT`、`DATASET_TYPE`（`consep` / `monusac` / `brca-m2c`）、学习率、Focal 检测损失等
- **Faster R-CNN**：`faster_rcnn_train.py` — `DATA_ROOT`、`DATASET_TYPE`（`consep` / `monusac`）、输入尺寸、epoch 等

**验证选模**：两种方法均按 validation 上的 classification-aware F1 保存 checkpoint（文件名含 `epoch{N}_f1{score}.pth`）。

#### 测试

编辑 `contrast/02_test.py`：

```python
CHECKPOINTS_FOLDER_NAME = 'exp32_faster_rcnn_consep'
EPOCH = 81
TYPE = 'faster_rcnn'   # 或 'e2ecr'
```

```bash
cd contrast
python 02_test.py
```

**输出**：`contrast/exp/<name>_e<epoch>/`

- `test_results.txt` — 汇总指标
- `visualizations/` — GT/预测点可视化（`*_gt.png`、`*_pred.png`）

也可直接调用底层函数：

```python
from e2ecr_test import e2ecr_test
from faster_rcnn_test import faster_rcnn_test
```

### HoVer-Net

详见 `contrast/HoverNet/README.md`。本仓库中的典型流程：

#### 1. 配置

编辑 `contrast/HoverNet/config.py`：`dataset_name`、`train_dir_list`、`valid_dir_list`、`log_dir`、`nr_type` 等。

#### 2. 训练

```bash
cd contrast/HoverNet
python run_train.py --gpu=0
```

#### 3. 推理

```bash
python run_infer.py tile \
  --model_path=<checkpoint.tar> \
  --nr_types=3 \
  --type_info_path=type_info_consep.json \
  --input_dir=<images> \
  --output_dir=<output>
```

#### 4. 与 MCSpatNet 点标注对齐评估

```bash
cd contrast/HoverNet
python evaluate.py \
  --exp_dir ../exp/exp28_HoverNet_monusac_e100 \
  --gt_root ../data/MoNuSAC_MCSpatNet_point/test/gt_custom \
  --img_root ../data/MoNuSAC_MCSpatNet_point/test/images \
  --type_info type_info_monusac.json
```

评估规则：预测实例 mask 覆盖 GT 点且类别一致 → TP；无匹配 GT 的预测 → FP；未被任何预测覆盖的 GT 点 → FN。

---

## 损失函数（MCSpatNet 训练）

`01_train_mcspat.py` 支持多种可组合损失，通过全局开关控制：

| 损失 | 开关 | 说明 |
|------|------|------|
| Dice | `use_dice_loss` | 检测 + 分类分支（与 Focal 互斥） |
| Focal | `use_focal_loss` | 检测二值 Focal + 分类多类 Focal CE |
| U-CE | `use_uce_detection_loss` / `use_uce_classification_loss` | MC-Dropout 不确定性加权 CE（与 Focal 互斥） |
| K-function L1 | `use_k_function_loss` | 空间统计分支回归 |
| 点热图 | `use_point_heatmap_loss` | GT 点高斯热图 MSE 监督 |
| 热图一致性 | `use_det_cls_heatmap_consistency_loss` | `sigmoid(det)` 与 `max(softmax(cls))` 的 L1 |

分阶段训练：`epoch < epoch_start_classification` 时仅优化检测相关损失；达到该 epoch 后加入分类损失。

---

## 数据集与类别说明

### CoNSeP（MCSpatNet / 多数对比实验）

- 3 类：lymphocyte、tumor、stromal
- 训练/验证：`data/CoNSeP_train` + `data_splits/consep/`
- 测试：`data/CoNSeP_test/`

### MoNuSAC

- MCSpatNet：`data/MoNuSAC_MCSpatNet_point/`
- E2ECR / Faster R-CNN：常用 2 类（epithelial + lymphocyte），见各 `*_dataset.py`
- HoVer-Net：`contrast/data/MoNuSAC_HoverNet/`

### BRCA-M2C

- 对比实验：`contrast/data/BRCA-M2C/`
- E2ECR 2 类设置

---

## 常见问题

**Q: 训练报 K-map 通道数不匹配？**  
A: 确保 `n_classes`、`class_indx` 与 `2_calc_kmaps.py` 中 `n_classes` 一致，且每张图都有对应 `k_func_maps/*_gt_kmap.npy`。或暂时设 `use_k_function_loss=False`。

**Q: 相对路径找不到数据？**  
A: 请在**项目根目录**运行 MCSpatNet 脚本；对比实验脚本在 `contrast/` 下运行，或检查各脚本内 `Path("data")` 是否指向 `contrast/data/`。

**Q: Windows 上 DataLoader 报错？**  
A: 对比实验脚本已按平台将 `NUM_WORKERS` 设为 0（Windows）或更大值（Linux）。HoVer-Net 在 Windows 上建议 `debug=True` 关闭多进程。

**Q: 如何复现实验？**  
A: 各脚本默认 `seed=42`；训练开始时会将当前 `.py` 备份到实验目录 `code/`，以该备份为准核对超参。

**Q: Web 端提示模型未加载？**  
A: 确认 `src/src/models/` 或 `exp/` 下存在与 `MODEL_CONFIG` 匹配的 `.pth` 文件，或在 `load_model()` 中传入正确路径。

**Q: Web 端上传后报错？**  
A: 检查图像格式与大小（≤16MB）；查看终端 traceback；确保在 `src/src/` 目录下启动 `app.py`。

---

## 引用

若使用本仓库中的第三方方法，请引用对应论文：

- **HoVer-Net**: Graham et al., Medical Image Analysis, 2019
- **E2ECR / Faster R-CNN**: 见 `contrast/` 内各实现及实验配置
- **CoNSeP / MoNuSAC**: 见各数据集原始论文

---

## 快速命令参考

```bash
# MCSpatNet：训练 → 测试 → 评估
python 01_train_mcspat.py
python 02_test_vis_mcspat.py
python 03_eval_localization_fscore.py

# Web 交互平台
cd src/src && pip install -r requirements.txt && python app.py
# 浏览器打开 http://localhost:5000

# 对比实验：E2ECR / Faster R-CNN
cd contrast && python 01_train.py && python 02_test.py

# HoVer-Net
cd contrast/HoverNet && python run_train.py --gpu=0
python run_infer.py tile --model_path=... --input_dir=... --output_dir=...
python evaluate.py --exp_dir=... --gt_root=... --img_root=...
```
