import torch
from torch import nn
from torch.nn import init


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        layers = [ConvBNRelu(33, 64)]
        for _ in range(3):
            layer = ConvBNRelu(64, 64)
            layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(64 + 3 + 30, 64)
        self.final_layer = nn.Conv2d(64, 3, kernel_size=1)
        # self.se = SEAttention(channel=97)

    def forward(self, image, message):
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        # 1 * 30 * 128 * 128
        expanded_message = expanded_message.expand(-1, -1, 128, 128)

        # 分离出YUV空间的各通道
        Y_channel = image[:, 0, :, :].unsqueeze(1)
        U_channel = image[:, 1, :, :].unsqueeze(1)
        V_channel = image[:, 2, :, :].unsqueeze(1)

        # 将水印concat到U通道
        U_channel_concat = torch.cat((U_channel, expanded_message), dim=1)

        # 再将Y, 新的U和V通道concat成最终的图像, 1 * 64 * 128 * 128
        final_image = torch.cat((Y_channel, U_channel_concat, V_channel), dim=1)

        # 1 * 64 * 128 * 128
        encoded_image = self.conv_layers(final_image)

        # skip connection  1 * 97 * 128 * 128
        combined = torch.cat([expanded_message, encoded_image, image], dim=1)

        # SEAttention
        # combined = self.se(combined)

        # 1 * 64 * 128 * 128
        im_w = self.after_concat_layer(combined)

        # 1 * 3 * 128 * 128
        im_w = self.final_layer(im_w)
        return im_w


# 1. 主干网络 + 模块
# 2. 主干网络 + (模块 + 模块 + ...)。模块与模块串行、并行、包含...


class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

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
