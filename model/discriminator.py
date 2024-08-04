import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    接收输入图像并判断图像中是否包含水印，输出一个标量表示图像是否包含水印的可信度
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # 卷积层的通道数
        discriminator_channels = 64
        # 卷积块数
        discriminator_blocks = 3

        layers = [ConvBNRelu(3, discriminator_channels)]
        for _ in range(discriminator_blocks - 1):
            layers.append(ConvBNRelu(discriminator_channels, discriminator_channels))

        # 将输出的空间维度变为 1x1。
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        # 将卷积层的输出通道数映射到单个标量输出，该值表示网络判断图像是否包含水印的置信度
        self.linear = nn.Linear(discriminator_channels, 1)

    def forward(self, image):
        X = self.before_linear(image)

        # 压缩掉最后两个维度，得到形状为 (batch_size, channels) 的张量。
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        return X


class ConvBNRelu(nn.Module):

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
