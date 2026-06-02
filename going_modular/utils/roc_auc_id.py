import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def _collect_embeddings(dataloader, model, device, modal_idx=None):
    """Thu thập toàn bộ embedding và identity label từ dataloader.

    modal_idx: truyền 0 hoặc 1 khi batch X có shape [B, 2, 3, H, W]
               (concat loader) để slice đúng modal trước khi feed model.
               None = không slice (single-modal loader, hành vi mặc định).
    """
    model.eval()
    id_labels_list  = []
    embeddings_list = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            if modal_idx is not None and X.dim() == 5:
                X = X[:, modal_idx]  # [B, 2, 3, H, W] → [B, 3, H, W]
            emb = model.get_embedding(X)
            id_labels_list.append(y[:, 0])
            embeddings_list.append(emb.cpu())
    all_ids = torch.cat(id_labels_list, dim=0)           # (N,)
    all_emb = F.normalize(torch.cat(embeddings_list, dim=0), p=2, dim=1)  # (N, 512)
    return all_ids, all_emb


def compute_id_auc(dataloader, model, device, modal_idx=None) -> dict:
    """
    Tính AUC nhận dạng khuôn mặt.
      - Cosine similarity AUC
      - Euclidean distance AUC (negated)

    Returns:
        {'id_cosine': float, 'id_euclidean': float}
    """
    all_ids, all_emb = _collect_embeddings(dataloader, model, device, modal_idx)

    cosine_sim     = torch.mm(all_emb, all_emb.t())
    euclidean_dist = torch.cdist(all_emb, all_emb, p=2)

    same_id_matrix = (all_ids.unsqueeze(1) == all_ids.unsqueeze(0)).int()
    triu_mask      = torch.triu(torch.ones_like(same_id_matrix), diagonal=1).bool()

    labels           = same_id_matrix[triu_mask].numpy()
    scores_cosine    = cosine_sim[triu_mask].numpy()
    scores_euclidean = -euclidean_dist[triu_mask].numpy()

    auc_scores = {}
    try:
        auc_scores['id_cosine']    = roc_auc_score(labels, scores_cosine)
        auc_scores['id_euclidean'] = roc_auc_score(labels, scores_euclidean)
    except Exception as e:
        print(f" Lỗi tính AUC ID: {e}")
        auc_scores['id_cosine']    = 0.0
        auc_scores['id_euclidean'] = 0.0

    return auc_scores


def compute_id_auc_gallery_probe(gallery_dl, probe_dl, model, device) -> dict:
    """
    Verification AUC theo chuẩn gallery-probe.

    Với mỗi cặp (probe_i, gallery_j):
      label = 1 nếu cùng identity, 0 nếu khác
      score  = cosine_sim(probe_emb_i, gallery_emb_j)

    Returns {'id_cosine': float, 'id_euclidean': float}
    """
    g_ids, g_emb = _collect_embeddings(gallery_dl, model, device)
    p_ids, p_emb = _collect_embeddings(probe_dl,   model, device)

    cos_sim  = torch.mm(p_emb, g_emb.t())          # [P, G]
    euc_dist = torch.cdist(p_emb, g_emb, p=2)      # [P, G]

    labels = (p_ids.unsqueeze(1) == g_ids.unsqueeze(0)).int().flatten().numpy()

    try:
        auc_cos = roc_auc_score(labels, cos_sim.flatten().numpy())
        auc_euc = roc_auc_score(labels, -euc_dist.flatten().numpy())
    except Exception as e:
        print(f'Lỗi tính gallery-probe AUC: {e}')
        auc_cos = auc_euc = 0.0

    return {'id_cosine': auc_cos, 'id_euclidean': auc_euc}


def compute_rank1_gallery_probe(gallery_dl, probe_dl, model, device) -> float:
    """
    Rank-1 Identification Accuracy theo chuẩn gallery-probe.

    Với mỗi probe i: tìm gallery j có cosine similarity cao nhất.
    Đúng nếu identity[gallery_j] == identity[probe_i].

    Returns rank1_acc: float trong [0, 1]
    """
    g_ids, g_emb = _collect_embeddings(gallery_dl, model, device)
    p_ids, p_emb = _collect_embeddings(probe_dl,   model, device)

    cos_sim          = torch.mm(p_emb, g_emb.t())  # [P, G]
    top1_gallery_idx = cos_sim.argmax(dim=1)        # [P]
    predicted_ids    = g_ids[top1_gallery_idx]      # [P]

    return (predicted_ids == p_ids).float().mean().item()


def compute_rank1(dataloader, model, device) -> float:
    """
    Tính Rank-1 Identification Accuracy.

    Với mỗi ảnh query i, tìm ảnh j có cosine similarity cao nhất
    trong toàn bộ tập (j != i). Nếu id[j] == id[i] thì đúng.

    Returns:
        rank1_acc: float trong [0, 1]
    """
    all_ids, all_emb = _collect_embeddings(dataloader, model, device)

    cosine_sim = torch.mm(all_emb, all_emb.t())   # (N, N)

    # Loại self-match bằng cách đặt đường chéo = -inf
    cosine_sim.fill_diagonal_(float('-inf'))

    # Top-1 match cho mỗi query
    top1_indices = cosine_sim.argmax(dim=1)        # (N,)
    top1_ids     = all_ids[top1_indices]           # (N,)

    rank1_acc = (top1_ids == all_ids).float().mean().item()
    return rank1_acc
