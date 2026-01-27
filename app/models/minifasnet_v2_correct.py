"""
MiniFASNetV2 Model Architecture from Silent-Face-Anti-Spoofing repository.
Compatible with original checkpoint weights.
https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    BatchNorm1d,
    PReLU,
    Sequential,
    Module,
    Flatten,
    Linear,
)


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Conv_block(Module):
    """Conv -> BatchNorm -> PReLU"""

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    """Conv -> BatchNorm (no PReLU)"""

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    """Depth-wise separable convolution block"""

    def __init__(
        self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1
    ):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3

        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    """Residual block with multiple Depth_Wise layers"""

    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(
                Depth_Wise(
                    c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNetV2(Module):
    """
    MiniFASNetV2 Architecture for Anti-Spoofing.
    Exact replica of the architecture from Silent-Face-Anti-Spoofing.
    Compatible with '1.8M_' checkpoint.
    """

    MODEL_CONFIG = {
        # keep: [input, keep[0], ..., output]
        "1.8M_": [
            32,
            32,
            103,
            103,
            64,
            13,
            13,
            64,
            13,
            13,
            64,
            13,
            13,
            64,
            13,
            13,
            64,
            231,
            231,
            128,
            231,
            231,
            128,
            52,
            52,
            128,
            26,
            26,
            128,
            77,
            77,
            128,
            26,
            26,
            128,
            26,
            26,
            128,
            308,
            308,
            128,
            26,
            26,
            128,
            26,
            26,
            128,
            512,
            512,
        ]
    }

    def __init__(
        self,
        embedding_size: int = 128,
        conv6_kernel=(7, 7),
        drop_p: float = 0.0,
        num_classes: int = 1,  # Binary classification
        img_channel: int = 3,
    ):
        super(MiniFASNetV2, self).__init__()
        self.embedding_size = embedding_size

        keep = self.MODEL_CONFIG["1.8M_"]

        # Initial convolutions
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(
            keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1]
        )

        # conv_23: Depth_Wise block
        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(
            c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3]
        )

        # conv_3: Residual block (4 layers)
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(
            c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # conv_34: Depth_Wise block
        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(
            c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19]
        )

        # conv_4: Residual block (6 layers)
        c1 = [
            (keep[19], keep[20]),
            (keep[22], keep[23]),
            (keep[25], keep[26]),
            (keep[28], keep[29]),
            (keep[31], keep[32]),
            (keep[34], keep[35]),
        ]
        c2 = [
            (keep[20], keep[21]),
            (keep[23], keep[24]),
            (keep[26], keep[27]),
            (keep[29], keep[30]),
            (keep[32], keep[33]),
            (keep[35], keep[36]),
        ]
        c3 = [
            (keep[21], keep[22]),
            (keep[24], keep[25]),
            (keep[27], keep[28]),
            (keep[30], keep[31]),
            (keep[33], keep[34]),
            (keep[36], keep[37]),
        ]
        self.conv_4 = Residual(
            c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # conv_45: Depth_Wise block
        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(
            c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40]
        )

        # conv_5: Residual block (2 layers)
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(
            c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # Final layers
        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(
            keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_flatten = Flatten()

        # Classifier
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p) if drop_p > 0 else torch.nn.Identity()
        self.prob = Linear(embedding_size, num_classes, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (BatchNorm2d, BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)

        if self.embedding_size != 512:
            out = self.linear(out)

        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)

        return out

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding features before classification."""
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)

        if self.embedding_size != 512:
            out = self.linear(out)

        out = self.bn(out)
        return out

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding features before classification."""
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)

        if self.embedding_size != 512:
            out = self.linear(out)

        out = self.bn(out)
        return out