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

        # 将输入特征图池化为一个指定大小的输出。
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        # 恒等映射，不改变输入的大小或形状。
        self.linear = nn.Linear(30, 30)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # 去除张量x中第三个和第二个维度（从0开始）
        x.squeeze_(3).squeeze_(2)
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



