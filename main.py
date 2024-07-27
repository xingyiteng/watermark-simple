import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model.decoder import Decoder
from model.encoder import Encoder
from util.rgb_to_yuv import rgb_to_yuv_tensor


def main():
    encoder()
    decoder()


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
    res = encoder_model(rgb_tensor_image, message).squeeze(0)

    # 4. 保存生成的图片
    save_image(res, 'test_emb.png')


def decoder():
    # 1. 加载图片
    image = Image.open('test_emb.png')

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
