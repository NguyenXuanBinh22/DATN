import torch
import torch.nn as nn

from .backbone.mobilenetv3_backbone import create_mimobilenetv3
from .head.id import MagLinear

seed = 42
torch.manual_seed(seed)


class FaceRecognitionMobileNetV3(nn.Module):
    """
    Student model: MobileNetV3 Large thuần cho face recognition (single-task).

    Không có FSM, không có auxiliary tasks, không có GRL.
    Thiết kế tối giản để:
      1. Train với MagFace loss (standalone hoặc Knowledge Distillation từ ConvNeXt V2)
      2. Quantize INT8 cho edge device

    Luồng dữ liệu:
        Input (B, 3, 112, 112)
            ↓  MobileNetV3 Large + Adapter Conv1×1
        (B, 512, H, W)
            ↓  BN → AdaptiveAvgPool → Flatten
        (B, 512)  ← embedding, dùng cho inference
            ↓  MagLinear  [chỉ dùng khi train]
        ([cos_theta, cos_theta_m], x_norm)
    """

    def __init__(self, num_classes: int, backbone: str = 'mobilenetv3_large_100'):
        super().__init__()

        self.backbone = create_mimobilenetv3(backbone)

        # Embedding: spatial features → 512-D vector
        self.embedding = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Classification head — chỉ dùng khi training, bỏ khi deploy
        self.maglinear = MagLinear(512, num_classes)

    def forward(self, x: torch.Tensor):
        """Training forward — trả về logits và feature norm cho MagFace loss."""
        feat = self.backbone(x)          # (B, 512, H, W)
        emb  = self.embedding(feat)      # (B, 512)
        logits, x_norm = self.maglinear(emb)
        return logits, x_norm

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference / Evaluation forward — chỉ trả về 512-D embedding.
        Dùng cho: AUC tính cosine similarity, KD loss, export ONNX.
        """
        feat = self.backbone(x)
        return self.embedding(feat)      # (B, 512)
