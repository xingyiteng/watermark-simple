import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model.decoder import Decoder
from model.encoder import Encoder
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.noiser import Noiser
from util.rgb_to_yuv import rgb_to_yuv_tensor


def main():
    # 编码图片
    encoded_and_cover = encoder()

    # 添加噪音
    noise_encoded = noise(encoded_and_cover, 'cuda' if torch.cuda.is_available() else 'cpu')

    # 解码图片
    watermark = decoder(noise_encoded)
    print('extract message: ', watermark)


def encoder():
    # 1. 加载图片
    image = Image.open('test.png')

    # 转为Tensor类型
    transform = transforms.ToTensor()
    rgb_tensor_image = transform(image).unsqueeze(0)

    # 调用方法将 RGB 图像转换为 YUV Tensor
    # yuv_tensor_image = rgb_to_yuv_tensor(image).unsqueeze(0)

    # 2. 生成随机水印
    message = torch.Tensor(np.random.choice([0, 1], (rgb_tensor_image.shape[0], 30)))

    print('original message: ', message.numpy())

    # 3. 嵌入水印
    encoder_model = Encoder()
    res = encoder_model(rgb_tensor_image, message)

    # 4. 保存编码图片
    save_image(res, 'test_emb.png')

    print('generate encoder image successful')

    # 5. 返回编码图片 原始图片
    return [res, rgb_tensor_image]


def noise(encoded_and_cover: list, device):
    # 引入噪声层模块，并指定对应的噪声列表
    height_ratio_range = (0.2, 0.3)  # 随机裁剪高度比例范围
    width_ratio_range = (0.4, 0.5)  # 随机裁剪宽度比例范围

    noiser = Noiser([Crop(height_ratio_range, width_ratio_range), Cropout(height_ratio_range, width_ratio_range)], device)

    # [编码图像，原始图像]作为噪声层的输入
    noised_and_cover = noiser(encoded_and_cover)

    # 获得加了噪声后的图像
    noised_image = noised_and_cover[0]

    # 保存噪声图片
    save_image(noised_image, 'test_emb_noise.png')

    print('generate noise image successful')

    return noised_image


def decoder(noise_encoded):
    # 解析水印
    decoder_model = Decoder()
    message = decoder_model(noise_encoded)

    message = message.detach().cpu().numpy().round().clip(0, 1)

    print('extract message successful')
    return message


def decoder_v1():
    # 1. 加载图片
    image = Image.open('test_emb_noise.png')

    # 2. 调用方法将 RGB 图像转换为 YUV Tensor
    # yuv_tensor_image = rgb_to_yuv_tensor(image).unsqueeze(0)

    transform = transforms.ToTensor()
    rgb_tensor_image = transform(image).unsqueeze(0)

    # 3. 解析水印
    decoder_model = Decoder()
    res = decoder_model(rgb_tensor_image)
    # detach(): 从计算图中分离张量, （在推理或处理输出时, 不需要反向传播）
    # cpu(): 将方法转到CPU运算，因为将张量转换为 NumPy 数组需要张量在 CPU 上，因为 NumPy 不支持 GPU 张量。
    # numpy(): 将张量转换为 NumPy 数组
    # round(): 四舍五入到最近的整数
    # clip(0, 1): 每个元素限制在指定范围 [0, 1] 之间。任何小于 0 的值都会被设置为 0，任何大于 1 的值都会被设置为 1。
    res = res.detach().cpu().numpy().round().clip(0, 1)
    print('extract message: ', res)


if __name__ == '__main__':
    main()
