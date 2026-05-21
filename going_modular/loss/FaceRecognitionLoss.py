import torch.nn as nn
from .WeightClassMagLoss import WeightClassMagLoss


class FaceRecognitionLoss(nn.Module):
    """
    Loss đơn giản cho single-task face recognition.

    Chỉ dùng WeightClassMagLoss (MagFace + class-balanced weighting).
    Không auto-weighting, không auxiliary tasks.

    forward() trả về: scalar loss
    """

    def __init__(self, metadata_path: str):
        super().__init__()
        self.id_loss = WeightClassMagLoss(metadata_path)

    def forward(self, logits, y):
        x_id_logits, x_id_norm = logits
        id_label = y[:, 0]
        return self.id_loss(x_id_logits, id_label, x_id_norm)
