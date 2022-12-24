import numpy as np

import mindspore.nn as nn
import mindspore.ops as P
#from mindspore.common.initializer import Normal(sigma=0.01, mean=0.0)


class ConvBNRelu(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, pad_mode='pad', padding=padding, dilation=dilation,
                              has_bias=bias, weight_init='HeUniform')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ParsingNet(nn.Cell):
    def __init__(self, backbone_type, backbone, cls_dim=(37, 10, 4), use_aux=False):
        super(ParsingNet, self).__init__()

        self.cls_dim = cls_dim
        self.use_aux = use_aux
        self.total_dim = int(np.prod(cls_dim))

        self.backbone = backbone

        if self.use_aux:
            self.aux_header2 = nn.SequentialCell([
                ConvBNRelu(128, 128, kernel_size=3, stride=1, padding=1) if backbone_type in [
                    '34', '18'] else ConvBNRelu(512, 128, kernel_size=3, stride=1, padding=1),
                ConvBNRelu(128, 128, 3, padding=1),
                ConvBNRelu(128, 128, 3, padding=1),
                ConvBNRelu(128, 128, 3, padding=1)
            ])
            self.aux_header3 = nn.SequentialCell([
                ConvBNRelu(256, 128, kernel_size=3, stride=1, padding=1) if backbone_type in [
                    '34', '18'] else ConvBNRelu(1024, 128, kernel_size=3, stride=1, padding=1),
                ConvBNRelu(128, 128, 3, padding=1),
                ConvBNRelu(128, 128, 3, padding=1)
            ])
            self.aux_header4 = nn.SequentialCell([
                ConvBNRelu(512, 128, kernel_size=3, stride=1, padding=1) if backbone_type in [
                    '34', '18'] else ConvBNRelu(2048, 128, kernel_size=3, stride=1, padding=1),
                ConvBNRelu(128, 128, 3, padding=1)
            ])
            self.aux_combine = nn.SequentialCell([
                ConvBNRelu(384, 256, 3, padding=2, dilation=2),
                ConvBNRelu(256, 128, 3, padding=2, dilation=2),
                ConvBNRelu(128, 128, 3, padding=2, dilation=2),
                ConvBNRelu(128, 128, 3, padding=4, dilation=4),
                nn.Conv2d(128, cls_dim[-1] + 1, 1,
                          pad_mode="pad", padding=0, has_bias=True, weight_init='HeUniform')
            ])

        self.classier = nn.SequentialCell([
            nn.Dense(1800, 2048),
            nn.ReLU(),
            nn.Dense(2048, self.total_dim)
        ])

        self.concat = P.Concat(axis=1)

        self.pool = nn.Conv2d(512, 8, 1, pad_mode="pad", padding=0, has_bias=True, weight_init='HeUniform') if backbone_type in [
            '34', '18'] else nn.Conv2d(2048, 8, 1, pad_mode="pad", padding=0, has_bias=True, weight_init='HeUniform')

    def construct(self, x):
        x2, x3, fea = self.backbone(x)
        if self.use_aux and self.training:
            x2 = self.aux_header2(x2)
            x3 = self.aux_header3(x3)
            x3 = P.ResizeBilinear((x3.shape[2] * 2, x3.shape[3] * 2))(x3)
            x4 = self.aux_header4(fea)
            x4 = P.ResizeBilinear((x4.shape[2] * 4, x4.shape[3] * 4))(x4)
            aux_seg = self.concat((x2, x3, x4))
            aux_seg = self.aux_combine(aux_seg)
        else:
            aux_seg = None

        fea = self.pool(fea).view((-1, 1800))

        group_cls = self.classier(fea).view((-1, self.cls_dim[0], self.cls_dim[1], self.cls_dim[2]))

        if self.use_aux and self.training:
            return group_cls, aux_seg

        return group_cls
