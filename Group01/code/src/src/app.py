"""
MCSpatNet 细胞检测前端应用
基于 Flask 的后端 API，提供细胞检测服务
"""

import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from skimage import io
import cv2
import torch
import torch.nn as nn
from skimage import filters
from skimage.measure import label
import glob
import json
from datetime import datetime

# 添加父目录到路径以导入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_arch import UnetVggMultihead

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 最大文件大小

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# 模型配置 - 与 02_test_vis_mcspat.py 保持一致
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_CLASSES = 3  # 主分类头中的细胞类别数
N_CLUSTERS = 5  # 每个主类别进一步细分的聚类数
N_CLASSES2 = N_CLUSTERS * N_CLASSES  # 子类别总数
R_STEP = 15
R_RANGE = range(0, 100, R_STEP)
R_CLASSES = len(R_RANGE)
R_CLASSES_ALL = R_CLASSES * N_CLASSES

MODEL_CONFIG = {
    'dropout_prob': 0,
    'initial_pad': 126,
    'interpolate': 'False',
    'conv_init': 'he',
    'n_classes': N_CLASSES,
    'n_channels': 3,
    'n_heads': 4,
    'head_classes': [1, N_CLASSES, N_CLASSES2, R_CLASSES_ALL],
}

# 类别颜色定义 - 与 02_test_vis_mcspat.py 保持一致
COLOR_SET = {
    0: (0, 162, 232),    # 炎症细胞 - 蓝色
    1: (255, 0, 0),      # 上皮细胞 - 红色
    2: (0, 255, 0),      # 间质细胞 - 绿色
    3: (255, 255, 0),    # 中性粒细胞 - 黄色
}

CLASS_NAMES = {
    0: '炎症细胞',
    1: '上皮细胞',
    2: '间质细胞',
    3: '中性粒细胞',
}

# 检测后处理阈值 - 与 02_test_vis_mcspat.py 保持一致
THRESH_LOW = 0.5
THRESH_HIGH = 0.5
SIZE_THRESH = 5

# 全局模型变量
model = None
criterion_sig = None
criterion_softmax = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model(model_path=None):
    """加载 MCSpatNet 模型"""
    global model, criterion_sig, criterion_softmax
    
    # 创建模型
    model = UnetVggMultihead(kwargs=MODEL_CONFIG)
    model.to(DEVICE)
    
    # 激活函数
    criterion_sig = nn.Sigmoid()
    criterion_softmax = nn.Softmax(dim=1)
    
    # 加载权重
    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
    else:
        # 自动查找模型文件
        possible_paths = [
            os.path.join('..', 'exp', 'exp12_consep_no_cluster_e263'),
            os.path.join('..', 'exp', 'exp12_consep_no_cluster'),
            os.path.join(MODEL_FOLDER),
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                # 查找 epoch 263 的模型
                pattern = os.path.join(path, '*epoch_263*.pth')
                model_files = glob.glob(pattern)
                if model_files:
                    model_path = model_files[0]
                    print(f"Found model: {model_path}")
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
                    found = True
                    break
                # 查找任何 mcspat_epoch 文件
                pattern = os.path.join(path, 'mcspat_epoch_*.pth')
                model_files = glob.glob(pattern)
                if model_files:
                    model_path = model_files[0]
                    print(f"Found model: {model_path}")
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
                    found = True
                    break
        
        if not found:
            print("Warning: No pretrained model found. Using random initialization.")
    
    model.eval()
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path):
    """预处理输入图像 - 与 02_test_vis_mcspat.py 保持一致"""
    img = io.imread(image_path) / 255.0
    
    # 如果是灰度图，转换为三通道
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
    elif img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]
    
    # 转换为 tensor [C, H, W]
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度
    
    return img_tensor, img


def detect_cells(et_all_sig, thresh_low=THRESH_LOW, thresh_high=THRESH_HIGH, size_thresh=SIZE_THRESH):
    """
    细胞检测 - 与 02_test_vis_mcspat.py 中的逻辑完全一致
    返回检测到的细胞中心点
    """
    # 对检测头概率图做滞后阈值分割，得到二值预测图
    e_hard = filters.apply_hysteresis_threshold(et_all_sig, thresh_low, thresh_high)
    e_hard2 = (e_hard > 0).astype(np.uint8)
    comp_mask = label(e_hard2)
    
    # 过滤掉面积过小的噪声区域
    e_count = comp_mask.max()
    if size_thresh > 0:
        for c in range(1, comp_mask.max() + 1):
            s = (comp_mask == c).sum()
            if s < size_thresh:
                e_count -= 1
                e_hard2[comp_mask == c] = 0
    
    # 从每个预测连通域中提取质心，作为最终的检测点
    e_dot = np.zeros(et_all_sig.shape)
    contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    centers = []
    for idx in range(len(contours)):
        contour_i = contours[idx]
        M = cv2.moments(contour_i)
        if M["m00"] == 0:
            continue
        cx = round(M["m10"] / M["m00"])
        cy = round(M["m01"] / M["m00"])
        # e_dot 只保留单个中心点
        e_dot[min(int(cy), e_dot.shape[0] - 1), min(int(cx), e_dot.shape[1] - 1)] = 1
        centers.append((int(cx), int(cy)))
    
    return centers, e_dot


def classify_cells(centers, et_class_argmax, e_dot_all):
    """
    细胞分类 - 与 02_test_vis_mcspat.py 中的逻辑一致
    根据分类结果将检测点分配到各类别
    """
    class_centers = {i: [] for i in range(N_CLASSES)}
    
    for cx, cy in centers:
        if 0 <= cy < et_class_argmax.shape[0] and 0 <= cx < et_class_argmax.shape[1]:
            cls = int(et_class_argmax[cy, cx])
            if cls < N_CLASSES:
                class_centers[cls].append((cx, cy))
    
    return class_centers


def create_detection_overlay(img, centers, color=(0, 0, 0), marker_size=3):
    """
    创建检测叠加图 - 与 02_test_vis_mcspat.py 中的可视化逻辑一致
    在图像上用黑色方块标记检测到的细胞中心
    """
    img_overlay = img.copy()
    for cx, cy in centers:
        # 在可视化图像上把预测中心附近涂黑，方便肉眼观察
        y_start = max(0, cy - marker_size)
        y_end = min(img.shape[0], cy + marker_size + 1)
        x_start = max(0, cx - marker_size)
        x_end = min(img.shape[1], cx + marker_size + 1)
        img_overlay[y_start:y_end, x_start:x_end, :] = color
    return img_overlay


def create_class_overlay(img, class_centers, color_set, marker_size=3):
    """
    创建分类叠加图 - 与 02_test_vis_mcspat.py 中的可视化逻辑一致
    按类别颜色绘制预测中心
    """
    img_overlay = img.copy()
    for cls, centers in class_centers.items():
        color = color_set.get(cls, (128, 128, 128))
        for cx, cy in centers:
            y_start = max(0, cy - marker_size)
            y_end = min(img.shape[0], cy + marker_size + 1)
            x_start = max(0, cx - marker_size)
            x_end = min(img.shape[1], cx + marker_size + 1)
            img_overlay[y_start:y_end, x_start:x_end, :] = color
    return img_overlay


def create_heatmap(et_all_sig):
    """创建检测热力图 - 使用 JET 颜色映射"""
    heatmap = (et_all_sig * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap_color


def process_image(img_tensor, img_original):
    """
    处理图像 - 完整的检测和分类流程
    与 02_test_vis_mcspat.py 的逻辑一致
    """
    # 推理
    with torch.no_grad():
        et_dmap_lst = model(img_tensor)
    
    # 4 个输出头分别对应：
    # 0: 全部细胞的检测图
    # 1: 主类别分类图
    # 2: 子类别分类图
    # 3: K-function 回归图
    # [2:-2, 2:-2] 用于去掉边界 padding 对输出的影响
    et_dmap_all = et_dmap_lst[0][:, :, 2:-2, 2:-2]
    et_dmap_class = et_dmap_lst[1][:, :, 2:-2, 2:-2]
    
    # 应用激活函数
    et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy().squeeze()
    et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()
    
    # 将输入图像恢复成 HWC 格式并放缩回 0~255，便于直接可视化保存
    img_display = img_original.squeeze().transpose(1, 2, 0) * 255 if img_original.ndim == 4 else (img_original * 255).astype(np.uint8)
    
    # 检测所有细胞
    centers, e_dot_all = detect_cells(et_all_sig, THRESH_LOW, THRESH_HIGH, SIZE_THRESH)
    
    # 创建检测叠加图（所有细胞）- 黑色标记
    img_det_overlay = create_detection_overlay(img_display, centers, color=(0, 0, 0), marker_size=3)
    
    # 分类
    et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
    class_centers = classify_cells(centers, et_class_argmax, e_dot_all)
    
    # 创建分类叠加图 - 按类别颜色标记
    img_class_overlay = create_class_overlay(img_display, class_centers, COLOR_SET, marker_size=3)
    
    # 统计各类别数量
    class_counts = {cls: len(centers_list) for cls, centers_list in class_centers.items()}
    
    return {
        'total_count': len(centers),
        'class_counts': class_counts,
        'centers': centers,
        'class_centers': class_centers,
        'img_det_overlay': img_det_overlay,
        'img_class_overlay': img_class_overlay,
        'et_all_sig': et_all_sig,
    }


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """检测 API"""
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图像'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预处理
        img_tensor, img_original = preprocess_image(filepath)
        img_tensor = img_tensor.to(DEVICE)
        
        # 处理图像
        result = process_image(img_tensor, img_original)
        
        # 创建热力图
        heatmap = create_heatmap(result['et_all_sig'])
        
        # 保存结果
        result_filename = filename.rsplit('.', 1)[0] + '_result.png'
        class_filename = filename.rsplit('.', 1)[0] + '_class.png'
        heatmap_filename = filename.rsplit('.', 1)[0] + '_heatmap.png'
        
        result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)
        class_path = os.path.join(app.config['OUTPUT_FOLDER'], class_filename)
        heatmap_path = os.path.join(app.config['OUTPUT_FOLDER'], heatmap_filename)
        
        # 保存图像 - 与 02_test_vis_mcspat.py 使用相同的格式
        io.imsave(result_path, result['img_det_overlay'].astype(np.uint8))
        io.imsave(class_path, result['img_class_overlay'].astype(np.uint8))
        cv2.imwrite(heatmap_path, heatmap)
        
        # 准备响应
        class_details = []
        for cls in range(N_CLASSES):
            count = result['class_counts'].get(cls, 0)
            class_details.append({
                'class_id': cls,
                'class_name': CLASS_NAMES.get(cls, f'类别{cls}'),
                'count': count,
                'color': COLOR_SET.get(cls, (128, 128, 128))
            })
        
        response = {
            'success': True,
            'total_count': result['total_count'],
            'class_details': class_details,
            'result_image': f'/api/result/{result_filename}',
            'class_image': f'/api/result/{class_filename}',
            'heatmap_image': f'/api/result/{heatmap_filename}',
            'original_image': f'/api/upload/{filename}',
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/result/<filename>')
def get_result(filename):
    """获取结果图像"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))


@app.route('/api/upload/<filename>')
def get_upload(filename):
    """获取上传的原始图像"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/api/model_status')
def model_status():
    """检查模型状态"""
    return jsonify({
        'model_loaded': model is not None,
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
    })


if __name__ == '__main__':
    # 加载模型
    load_model()
    
    # 运行 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=True)
