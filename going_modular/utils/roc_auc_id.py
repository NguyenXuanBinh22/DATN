import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def _collect_embeddings(dataloader, model, device):
    """Thu thập toàn bộ embedding và identity label từ dataloader."""
    model.eval()
    id_labels_list  = []
    embeddings_list = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            emb = model.get_embedding(X)
            id_labels_list.append(y[:, 0])
            embeddings_list.append(emb.cpu())
    all_ids = torch.cat(id_labels_list, dim=0)           # (N,)
    all_emb = F.normalize(torch.cat(embeddings_list, dim=0), p=2, dim=1)  # (N, 512)
    return all_ids, all_emb


def compute_id_auc(dataloader, model, device) -> dict:
    """
    Tính AUC nhận dạng khuôn mặt.
      - Cosine similarity AUC
      - Euclidean distance AUC (negated)

    Returns:
        {'id_cosine': float, 'id_euclidean': float}
    """
    all_ids, all_emb = _collect_embeddings(dataloader, model, device)

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
