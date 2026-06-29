# MCSpatNet 细胞检测前端系统

基于 MCSpatNet 深度学习模型的细胞检测 Web 前端应用。支持上传组织病理图像，自动检测并分类细胞，返回检测结果和统计信息。

## 功能特性

- **图像上传**: 支持拖拽或点击上传 PNG、JPG、TIF 格式图像
- **细胞检测**: 使用 MCSpatNet 模型自动检测图像中的细胞
- **细胞分类**: 支持 4 类细胞分类：
  - 淋巴细胞 (蓝色)
  - 肿瘤细胞 (红色)
  - 其他细胞 (绿色)
  - 中性粒细胞 (黄色)
- **结果可视化**: 显示检测框、热力图和统计信息
- **结果下载**: 支持下载检测结果图像和统计数据

## 项目结构

```
src/
├── app.py                 # Flask 后端主程序
├── model_arch.py          # MCSpatNet 模型架构
├── requirements.txt       # Python 依赖
├── README.md             # 本文件
├── uploads/              # 上传图像存储目录
├── outputs/              # 检测结果输出目录
├── models/               # 模型权重文件目录
├── static/
│   ├── css/
│   │   └── style.css     # 前端样式
│   └── js/
│       └── main.js       # 前端交互逻辑
└── templates/
    └── index.html        # 主页面
```

## 环境要求

- Python 3.8 或更高版本
- CUDA (可选，用于 GPU 加速)
- 足够的磁盘空间存储上传的图像和模型

## 安装步骤

### 1. 安装依赖

```bash
cd src
pip install -r requirements.txt
```

### 2. 准备模型权重

将训练好的模型权重文件复制到 `models/` 目录下：

```bash
# 从 exp12_consep_no_cluster 复制模型权重
cp ../exp/exp12_consep_no_cluster/mcspat_epoch_263_*.pth models/
```

如果没有指定模型文件，程序会自动在以下位置查找：
- `models/` 目录
- `../exp/exp12_consep_no_cluster_e263/`
- `../exp/exp12_consep_no_cluster/`

### 3. 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用方法

1. 打开浏览器访问 `http://localhost:5000`
2. 点击上传区域或拖拽图像到上传区域
3. 等待检测完成（进度条显示处理状态）
4. 查看检测结果：
   - 细胞总数统计
   - 各类别细胞数量
   - 原始图像、检测结果图、热力图
5. 点击"下载结果"保存检测图像和统计数据

## API 接口

### POST /api/detect
上传图像进行检测

**请求**: `multipart/form-data`
- `image`: 图像文件

**响应**:
```json
{
  "success": true,
  "total_count": 150,
  "class_details": [
    {"class_id": 0, "class_name": "淋巴细胞", "count": 45, "color": [0, 162, 232]},
    {"class_id": 1, "class_name": "肿瘤细胞", "count": 80, "color": [255, 0, 0]},
    {"class_id": 2, "class_name": "其他细胞", "count": 25, "color": [0, 255, 0]}
  ],
  "result_image": "/api/result/xxx_result.png",
  "heatmap_image": "/api/result/xxx_heatmap.png",
  "original_image": "/api/upload/xxx.png"
}
```

### GET /api/model_status
检查模型加载状态

**响应**:
```json
{
  "model_loaded": true,
  "device": "cuda",
  "cuda_available": true
}
```

## 模型配置

模型配置位于 `app.py` 中的 `MODEL_CONFIG`：

```python
MODEL_CONFIG = {
    'dropout_prob': 0,
    'initial_pad': 126,
    'interpolate': 'False',
    'conv_init': 'he',
    'n_classes': 3,        # 主分类类别数
    'n_channels': 3,       # 输入通道数
    'n_heads': 4,          # 输出头数量
    'head_classes': [1, 3, 15, 21],  # 各输出头类别数
}
```

## 常见问题

### 1. CUDA 内存不足
如果 GPU 内存不足，可以修改 `app.py` 中的设备配置：
```python
DEVICE = torch.device('cpu')  # 使用 CPU
```

### 2. 模型文件找不到
确保模型权重文件已复制到正确位置，或在 `load_model()` 函数中指定正确的模型路径。

### 3. 端口被占用
修改 `app.py` 最后一行的端口号：
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

## 技术栈

- **后端**: Flask, PyTorch
- **前端**: HTML5, CSS3, JavaScript (原生)
- **深度学习**: MCSpatNet (U-Net + VGG 多任务网络)

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题，请参考原始 MCSpatNet 项目文档。
