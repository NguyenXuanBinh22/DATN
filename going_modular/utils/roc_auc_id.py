import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_id_auc(dataloader, model, device) -> dict:
    """
    Tính AUC nhận dạng khuôn mặt cho single-task model.

    Dùng model.get_embedding() để lấy 512-D embedding, sau đó tính:
      - Cosine similarity AUC
      - Euclidean distance AUC (negated)

    Returns:
        {'id_cosine': float, 'id_euclidean': float}
    """
    model.eval()

    id_labels_list  = []
    embeddings_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            emb = model.get_embedding(X)          # (B, 512)

            id_labels_list.append(y[:, 0])
            embeddings_list.append(emb.cpu())

    all_ids = torch.cat(id_labels_list, dim=0)    # (N,)
    all_emb = torch.cat(embeddings_list, dim=0)   # (N, 512)
    all_emb = F.normalize(all_emb, p=2, dim=1)

    # Pairwise similarity / distance — chỉ lấy upper triangle để tránh duplicate
    cosine_sim     = torch.mm(all_emb, all_emb.t())
    euclidean_dist = torch.cdist(all_emb, all_emb, p=2)

    same_id_matrix = (all_ids.unsqueeze(1) == all_ids.unsqueeze(0)).int()
    triu_mask      = torch.triu(torch.ones_like(same_id_matrix), diagonal=1).bool()

    labels          = same_id_matrix[triu_mask].numpy()
    scores_cosine   = cosine_sim[triu_mask].numpy()
    scores_euclidean = -euclidean_dist[triu_mask].numpy()  # negated: cao hơn = gần hơn

    auc_scores = {}
    try:
        auc_scores['id_cosine']    = roc_auc_score(labels, scores_cosine)
        auc_scores['id_euclidean'] = roc_auc_score(labels, scores_euclidean)
    except Exception as e:
        print(f" Lỗi tính AUC ID: {e}")
        auc_scores['id_cosine']    = 0.0
        auc_scores['id_euclidean'] = 0.0

    return auc_scores
