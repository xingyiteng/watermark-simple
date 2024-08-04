import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.channels = 64  # 卷积通道数
        self.decoder_blocks = 7  # 卷积块块数
        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(self.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        layers.append(ConvBNRelu(self.channels, 30))

        # 将输入特征图池化为一个指定大小的输出。将输入特征图的空间维度（高度和宽度）池化为 1x1（例如池化为：1 * 30 * 1 * 1）
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        # 全连接层（线性层）。 输入特征的维度，输出特征维度
        self.linear = nn.Linear(30, 30)

    def forward(self, image_with_wm):
        # 1 * 30 * 1 * 1
        x = self.layers(image_with_wm)
        # squeeze_不指定维度时，会移除所有大小为 1 的维度。squeeze_(dim)会移除特定维度 dim，且该维度大小为1时。
        x.squeeze_(3).squeeze_(2)
        # 全连接层（线性层）一般用于处理二维张量，形状为 (batch_size, num_features)
        x = self.linear(x)
        return x


class ConvBNRelu(nn.Module):

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            # 卷积 提取特征
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            # 批量归一化 加速收敛
            nn.BatchNorm2d(channels_out),
            # 激活函数 引入非线性
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)



