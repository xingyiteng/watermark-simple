import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        layers = [ConvBNRelu(3, 64)]
        for _ in range(3):
            layer = ConvBNRelu(64, 64)
            layers.append(layer)
        self.conv_layers = nn.Sequential(*layers)
        # 3通道 + 64通道 + 30长度  => 64通道
        self.after_concat_layer = ConvBNRelu(64 + 3 + 30, 64)
        # 64通道 => 3通道
        self.final_layer = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, image, message):
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1, -1, 128, 128)
        encoded_image = self.conv_layers(image)
        combined = torch.cat([expanded_message, encoded_image, image], dim=1)
        im_w = self.after_concat_layer(combined)
        im_w = self.final_layer(im_w)
        return im_w


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)
