import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model.decoder import Decoder
from model.encoder import Encoder
from util.rgb_to_yuv import rgb_to_yuv_tensor


def main():
    # encoder()
    decoder()


def encoder():
    # 1. 加载图片
    image = Image.open('test.png')

    # 转为Tensor类型
    transform = transforms.ToTensor()
    # rgb_tensor_image = transform(image).unsqueeze(0)

    # 调用方法将 RGB 图像转换为 YUV Tensor
    yuv_tensor_image = rgb_to_yuv_tensor(image).unsqueeze(0)

    # 2. 生成随机水印
    message = torch.Tensor(np.random.choice([0, 1], (yuv_tensor_image.shape[0], 30)))

    print('original message: ', message.numpy())

    # 3. 嵌入水印
    encoder_model = Encoder()
    res = encoder_model(yuv_tensor_image, message).squeeze(0)

    # 4. 保存生成的图片
    save_image(res, 'test_emb.png')


def decoder():
    # 1. 加载图片
    image = Image.open('test_emb.png')

    # 2. 调用方法将 RGB 图像转换为 YUV Tensor
    yuv_tensor_image = rgb_to_yuv_tensor(image).unsqueeze(0)

    # 3. 解析水印
    decoder_model = Decoder()
    res = decoder_model(yuv_tensor_image)
    res = res.detach().cpu().numpy().round().clip(0, 1)
    print(res)


if __name__ == '__main__':
    main()
