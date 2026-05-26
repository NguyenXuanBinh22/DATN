import torch
import torch.nn as nn
import torch.nn.functional as F


def fsp_matrix(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """
    Compute FSP (Flow of Solution Procedure) matrix.

    G_{i,j} = (1 / H*W) * sum_{s,t} F1_{s,t,i} * F2_{s,t,j}

    f1 : (B, C1, H, W)
    f2 : (B, C2, H, W)  — must have same H, W as f1
    Returns G : (B, C1, C2)
    """
    B, C1, H, W = f1.shape
    C2 = f2.shape[1]
    f1_flat = f1.view(B, C1, H * W)                               # (B, C1, HW)
    f2_flat = f2.view(B, C2, H * W)                               # (B, C2, HW)
    return torch.bmm(f1_flat, f2_flat.transpose(1, 2)) / (H * W)  # (B, C1, C2)


class FSPDistillLoss(nn.Module):
    """
    FSP Knowledge Distillation Loss (Yim et al., CVPR 2017).

    Adapted for cross-architecture distillation where teacher and student
    have different channel dimensions. Trainable 1x1 Conv adapters project
    student features into teacher's channel space before FSP computation.

    These adapters are auxiliary — they exist only for Stage 1 FSP pre-training
    and are discarded before Stage 2 task training.

    Args:
        teacher_ch_pairs : [(C_T1, C_T2), ...]  — teacher channel dims per pair
        student_ch_pairs : [(C_S1, C_S2), ...]  — student channel dims per pair

    Usage:
        fsp_loss = FSPDistillLoss(
            teacher_ch_pairs=[(192, 384), (384, 768)],
            student_ch_pairs=[(40, 96),   (96, 160)],
        )
        loss = fsp_loss(teacher_pairs, student_pairs)

        where teacher_pairs = [(f_T1_pair0, f_T2_pair0), (f_T1_pair1, f_T2_pair1)]
              student_pairs = [(f_S1_pair0, f_S2_pair0), ...]
    """

    def __init__(
        self,
        teacher_ch_pairs: list,
        student_ch_pairs: list,
    ):
        super().__init__()
        assert len(teacher_ch_pairs) == len(student_ch_pairs)

        self.adapters_f1 = nn.ModuleList()
        self.adapters_f2 = nn.ModuleList()

        for (C_T1, C_T2), (C_S1, C_S2) in zip(teacher_ch_pairs, student_ch_pairs):
            self.adapters_f1.append(
                nn.Conv2d(C_S1, C_T1, kernel_size=1, bias=False)
                if C_S1 != C_T1 else nn.Identity()
            )
            self.adapters_f2.append(
                nn.Conv2d(C_S2, C_T2, kernel_size=1, bias=False)
                if C_S2 != C_T2 else nn.Identity()
            )

    @staticmethod
    def _align(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if t.shape[2] != h or t.shape[3] != w:
            return F.adaptive_max_pool2d(t, (h, w))
        return t

    def forward(
        self,
        teacher_pairs: list,   # [(f_T1, f_T2), ...]  — teacher features (detached)
        student_pairs: list,   # [(f_S1, f_S2), ...]  — student features
    ) -> torch.Tensor:
        """
        Returns mean FSP L2 loss over all pairs.

        Both f1 and f2 within a pair are pooled to the smaller spatial
        resolution (as in the original paper) before FSP matrix computation.
        """
        total = torch.tensor(0.0, device=student_pairs[0][0].device)

        for i, ((f_T1, f_T2), (f_S1, f_S2)) in enumerate(
            zip(teacher_pairs, student_pairs)
        ):
            # Project student channels to match teacher
            f_S1_p = self.adapters_f1[i](f_S1)
            f_S2_p = self.adapters_f2[i](f_S2)

            # Determine target spatial: smallest among the four tensors
            h = min(f_T1.shape[2], f_T2.shape[2], f_S1_p.shape[2], f_S2_p.shape[2])
            w = min(f_T1.shape[3], f_T2.shape[3], f_S1_p.shape[3], f_S2_p.shape[3])

            f_T1_a = self._align(f_T1, h, w)
            f_T2_a = self._align(f_T2, h, w)
            f_S1_a = self._align(f_S1_p, h, w)
            f_S2_a = self._align(f_S2_p, h, w)

            # Teacher FSP — no gradient needed
            G_T = fsp_matrix(f_T1_a.detach(), f_T2_a.detach())
            G_S = fsp_matrix(f_S1_a, f_S2_a)

            total = total + F.mse_loss(G_S, G_T)

        return total / len(teacher_pairs)