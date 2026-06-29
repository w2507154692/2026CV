"""MoNuSAC 切片读取器。

MoNuSAC 提供的 .tif 在部分 Windows + Pillow 组合下会在 load/crop 时触发
原生层崩溃（exit code 0xC0000409）。TIFF 路径改用 OpenCV 读取。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OBJECTIVE_PROPERTY_KEYS = (
    "openslide.objective-power",
    "aperio.AppMag",
    "hamamatsu.SourceLens",
)


def choose_best_level(slide, target_magnification: float) -> int:
    objective = None
    for key in OBJECTIVE_PROPERTY_KEYS:
        value = slide.properties.get(key)
        if value is None:
            continue
        try:
            objective = float(value)
            break
        except ValueError:
            continue

    if objective is None or target_magnification <= 0:
        return 0

    target_downsample = max(objective / target_magnification, 1.0)
    return min(
        range(slide.level_count),
        key=lambda idx: abs(float(slide.level_downsamples[idx]) - target_downsample),
    )


class SlideReader:
    def __init__(
        self,
        image_path: Path,
        target_magnification: float,
        tif_reference_magnification: float | None = None,
    ):
        self.image_path = Path(image_path)
        self._cv_image: np.ndarray | None = None
        self._openslide = None
        self._level = 0
        self._downsample = 1.0
        self._tif_scale = 1.0

        if self.image_path.suffix.lower() in {".tif", ".tiff"}:
            bgr = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"OpenCV 无法读取 TIFF: {self.image_path}")
            self._cv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            native_height, native_width = self._cv_image.shape[:2]
            if (
                tif_reference_magnification is not None
                and tif_reference_magnification > 0
                and target_magnification > 0
            ):
                self._tif_scale = target_magnification / tif_reference_magnification
            self.width = max(1, int(round(native_width * self._tif_scale)))
            self.height = max(1, int(round(native_height * self._tif_scale)))
            return

        try:
            import openslide
        except ImportError as exc:
            raise RuntimeError(
                "读取 .svs 文件需要安装 openslide-python，且系统中可用 OpenSlide 库。"
                "若同目录存在 .tif，请优先使用 .tif。"
            ) from exc

        self._openslide = openslide.OpenSlide(str(self.image_path))
        self._level = choose_best_level(self._openslide, target_magnification)
        self._downsample = float(self._openslide.level_downsamples[self._level])
        self.width, self.height = self._openslide.level_dimensions[self._level]

    def _resize_to_patch(self, patch: Image.Image, patch_size: int) -> Image.Image:
        if patch.size != (patch_size, patch_size):
            patch = patch.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        return patch.convert("RGB")

    def crop(self, x: int, y: int, patch_size: int) -> tuple[Image.Image, float, float]:
        x = max(0, min(x, max(self.width - 1, 0)))
        y = max(0, min(y, max(self.height - 1, 0)))
        actual_w = min(patch_size, self.width - x)
        actual_h = min(patch_size, self.height - y)
        if actual_w <= 0 or actual_h <= 0:
            raise ValueError(
                f"无法裁剪 patch: 起点=({x}, {y}), 逻辑尺寸=({self.width}, {self.height})"
            )

        scale_x = patch_size / actual_w
        scale_y = patch_size / actual_h

        if self._cv_image is not None:
            native_height, native_width = self._cv_image.shape[:2]
            if self._tif_scale != 1.0:
                inv_scale = 1.0 / self._tif_scale
                x0 = int(round(x * inv_scale))
                y0 = int(round(y * inv_scale))
                x1 = min(int(round((x + actual_w) * inv_scale)), native_width)
                y1 = min(int(round((y + actual_h) * inv_scale)), native_height)
            else:
                x0, y0 = x, y
                x1 = min(x + actual_w, native_width)
                y1 = min(y + actual_h, native_height)
            patch_arr = self._cv_image[y0:y1, x0:x1]
            patch = Image.fromarray(patch_arr)
            return self._resize_to_patch(patch, patch_size), scale_x, scale_y

        location = (
            int(round(x * self._downsample)),
            int(round(y * self._downsample)),
        )
        level_w = max(1, actual_w)
        level_h = max(1, actual_h)
        patch = self._openslide.read_region(location, self._level, (level_w, level_h))
        return self._resize_to_patch(patch, patch_size), scale_x, scale_y

    def close(self) -> None:
        self._cv_image = None
        if self._openslide is not None:
            self._openslide.close()
            self._openslide = None
