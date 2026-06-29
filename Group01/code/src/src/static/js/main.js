/**
 * MCSpatNet 前端 JavaScript
 * 处理文件上传、图像检测和结果展示
 */

// DOM 元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// 结果元素
const totalCountEl = document.getElementById('totalCount');
const classStatsEl = document.getElementById('classStats');
const originalImageEl = document.getElementById('originalImage');
const resultImageEl = document.getElementById('resultImage');
const classImageEl = document.getElementById('classImage');
const heatmapImageEl = document.getElementById('heatmapImage');
const legendItemsEl = document.getElementById('legendItems');

// 按钮
const downloadBtn = document.getElementById('downloadBtn');
const newUploadBtn = document.getElementById('newUploadBtn');

// 当前结果数据
let currentResult = null;

// 类别配置
const CLASS_CONFIG = {
    0: { name: '炎症细胞', color: 'rgb(0, 162, 232)' },
    1: { name: '上皮细胞', color: 'rgb(255, 0, 0)' },
    2: { name: '间质细胞', color: 'rgb(0, 255, 0)' },
    3: { name: '中性粒细胞', color: 'rgb(255, 255, 0)' },
};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkModelStatus();
});

// 初始化事件监听
function initEventListeners() {
    // 点击上传
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // 下载按钮
    downloadBtn.addEventListener('click', downloadResults);

    // 重新上传按钮
    newUploadBtn.addEventListener('click', resetUpload);
}

// 检查模型状态
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model_status');
        const data = await response.json();
        
        if (!data.model_loaded) {
            showError('模型未加载，请检查模型文件是否存在');
        }
        
        console.log('Model status:', data);
    } catch (error) {
        console.error('Failed to check model status:', error);
    }
}

// 处理文件
function handleFile(file) {
    // 验证文件类型
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif'];
    const allowedExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
    
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions.includes(fileExtension)) {
        showError('不支持的文件格式，请上传 PNG, JPG 或 TIF 格式的图像');
        return;
    }

    // 验证文件大小 (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('文件过大，请上传小于 16MB 的图像');
        return;
    }

    // 上传并检测
    uploadAndDetect(file);
}

// 上传并检测
async function uploadAndDetect(file) {
    // 显示进度条
    showProgress();
    
    const formData = new FormData();
    formData.append('image', file);

    try {
        progressText.textContent = '正在上传图像...';
        
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '检测失败');
        }

        const result = await response.json();
        
        if (result.success) {
            currentResult = result;
            displayResults(result);
        } else {
            throw new Error(result.error || '检测失败');
        }
    } catch (error) {
        hideProgress();
        showError(error.message);
    }
}

// 显示进度条
function showProgress() {
    uploadArea.style.display = 'none';
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

// 隐藏进度条
function hideProgress() {
    progressSection.style.display = 'none';
}

// 显示结果
function displayResults(result) {
    hideProgress();
    resultsSection.style.display = 'block';

    // 更新总数
    totalCountEl.textContent = result.total_count;

    // 更新各类别统计
    classStatsEl.innerHTML = '';
    result.class_details.forEach((cls, index) => {
        const classCard = document.createElement('div');
        classCard.className = `class-card class-${cls.class_id}`;
        classCard.innerHTML = `
            <div class="class-name">${cls.class_name}</div>
            <div class="class-count">${cls.count}</div>
        `;
        classStatsEl.appendChild(classCard);
    });

    // 更新图像
        originalImageEl.src = result.original_image;
        resultImageEl.src = result.result_image;
        classImageEl.src = result.class_image;
        heatmapImageEl.src = result.heatmap_image;

    // 更新图例
    legendItemsEl.innerHTML = '';
    result.class_details.forEach(cls => {
        const config = CLASS_CONFIG[cls.class_id];
        if (config) {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <div class="legend-color" style="background-color: ${config.color};"></div>
                <span>${config.name}</span>
            `;
            legendItemsEl.appendChild(legendItem);
        }
    });

    // 滚动到结果区域
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 下载结果
function downloadResults() {
    if (!currentResult) return;

    // 下载细胞检测图像（黑色标记）
    const resultLink = document.createElement('a');
    resultLink.href = currentResult.result_image;
    resultLink.download = 'detection_result.png';
    resultLink.click();

    // 下载细胞分类图像（彩色标记）
    setTimeout(() => {
        const classLink = document.createElement('a');
        classLink.href = currentResult.class_image;
        classLink.download = 'classification_result.png';
        classLink.click();
    }, 500);

    // 下载热力图
    setTimeout(() => {
        const heatmapLink = document.createElement('a');
        heatmapLink.href = currentResult.heatmap_image;
        heatmapLink.download = 'detection_heatmap.png';
        heatmapLink.click();
    }, 1000);

    // 下载统计信息为 JSON
    const statsData = {
        total_count: currentResult.total_count,
        class_details: currentResult.class_details,
        timestamp: new Date().toISOString(),
    };
    
    const statsBlob = new Blob([JSON.stringify(statsData, null, 2)], { type: 'application/json' });
    const statsUrl = URL.createObjectURL(statsBlob);
    
    setTimeout(() => {
        const statsLink = document.createElement('a');
        statsLink.href = statsUrl;
        statsLink.download = 'detection_stats.json';
        statsLink.click();
        URL.revokeObjectURL(statsUrl);
    }, 1500);
}

// 重置上传
function resetUpload() {
    currentResult = null;
    fileInput.value = '';
    
    uploadArea.style.display = 'block';
    progressSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // 滚动到顶部
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// 显示错误
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'flex';
}

// 隐藏错误
function hideError() {
    errorSection.style.display = 'none';
}

// 键盘快捷键
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hideError();
    }
});
