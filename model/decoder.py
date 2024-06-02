import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        layers = [ConvBNRelu(3, 64)]
        for _ in range(6):
            layers.append(ConvBNRelu(64, 64))

        layers.append(ConvBNRelu(64, 30))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        # 恒等映射，不改变输入的大小或形状。
        self.linear = nn.Linear(30, 30)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
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



