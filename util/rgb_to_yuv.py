import cv2
import numpy as np
import torch
from PIL import Image


def rgb_to_yuv_tensor(image):
    """
        将 RGB 图像转换为 YUV 空间并返回 PyTorch Tensor。

        Args:
        image (Union[Image.Image, np.ndarray]): 输入的 RGB 图像，可以是 PIL 图像或 NumPy 数组。

        Returns:
        torch.Tensor: 转换后的 YUV 图像，形状为 [C, H, W]，值范围在 [0, 1]。
    """

    # 如果输入是 PIL 图像，转换为 NumPy 数组
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))  # 确保图像是 RGB 格式

    # 确保图像是 NumPy 数组并且有 3 个通道
    if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
        raise ValueError("Input image must be an RGB image.")

    # 将 RGB 图像转换为 YUV 空间
    yuv_image_np = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    # 重新排列通道顺序，使其符合 [C, H, W] 格式
    yuv_image_np = yuv_image_np.transpose((2, 0, 1))

    # 将 NumPy 数组转换为 PyTorch Tensor
    yuv_image_tensor = torch.tensor(yuv_image_np, dtype=torch.float32) / 255.0

    return yuv_image_tensor


