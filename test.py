import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model.encoder import Encoder


def main():
    # 加载图片
    image = Image.open('test.png')

    # 转为Tensor类型
    transform = transforms.ToTensor()
    tensor_image = transform(image).unsqueeze(0)

    # 随机水印
    message = torch.Tensor(np.random.choice([0, 1], (tensor_image.shape[0], 30)))

    # 二值水印
    # message = Image.open('D:\\workspace\\watermark\\watermark-project\\watermark-simple\\message_binary.png')
    # message = transform(message)

    print('original message: ', message.numpy())

    # 嵌入水印
    encoder = Encoder()
    res = encoder(tensor_image, message).squeeze(0)

    # 保存到本地
    save_image(res, 'test_emb.png')


if __name__ == '__main__':
    main()
