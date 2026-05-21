import torch
import torch.nn as nn
import timm


class MIMobileNetV3(nn.Module):
    """
    MobileNetV3 Large backbone với adapter layer chuẩn hóa output về 512 channels.
    Thiết kế đơn giản (không FSM) để dùng cho Knowledge Distillation + Quantization.
    Input: (B, 3, 112, 112)  →  Output: (B, 512, H, W)
    """

    def __init__(self, model_name: str = 'mobilenetv3_large_100', pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        # Lấy số channels của feature map cuối cùng bằng dummy forward
        dummy = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            out_channels = self.backbone(dummy)[-1].shape[1]

        self.target_channels = 512
        # Adapter: align channels về 512, giữ nguyên spatial size
        self.adapter_conv = nn.Conv2d(out_channels, self.target_channels, kernel_size=1, bias=False)
        self.adapter_bn   = nn.BatchNorm2d(self.target_channels)
        self.adapter_act  = nn.PReLU(self.target_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)[-1]                          # (B, C_last, H, W)
        return self.adapter_act(self.adapter_bn(self.adapter_conv(feat)))  # (B, 512, H, W)


def create_mimobilenetv3(model_name: str = 'mobilenetv3_large_100', **kwargs) -> MIMobileNetV3:
    return MIMobileNetV3(model_name, **kwargs)
