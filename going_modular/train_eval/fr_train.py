import os
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

from ..utils.roc_auc_id import compute_id_auc
from ..utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ..utils.ModelCheckPoint import ModelCheckpoint

seed = 42
torch.manual_seed(seed)


def fit(
    conf: dict,
    start_epoch: int,
    model: Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler,
    early_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint,
    existing_manager=None,
):
    if existing_manager:
        log_dir = existing_manager.log_dir
        manager = existing_manager
    else:
        log_dir = os.path.join(conf['checkpoint_dir'], conf['type'], 'logs')

        class _Dummy:
            def log_text(self, m): print(m)
            def log_metrics(self, e, m): pass

        manager = _Dummy()

    writer  = SummaryWriter(log_dir=log_dir)
    device  = conf['device']

    manager.log_text(f"BAT DAU TRAINING: {conf['note']}")

    for epoch in range(start_epoch, conf['epochs']):
        manager.log_text(f"\n--- Epoch {epoch+1}/{conf['epochs']} ---")

        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, device)

        # AUC tính trên cả train lẫn test để theo dõi overfitting
        train_auc = compute_id_auc(train_dataloader, model, device)
        test_auc  = compute_id_auc(test_dataloader,  model, device)

        train_metrics = {
            "loss":             train_loss,
            "auc_id_cosine":    train_auc['id_cosine'],
            "auc_id_euclidean": train_auc['id_euclidean'],
        }
        test_metrics = {
            "auc_id_cosine":    test_auc['id_cosine'],
            "auc_id_euclidean": test_auc['id_euclidean'],
        }

        # TensorBoard
        writer.add_scalar('Loss/train',                  train_loss,                       epoch + 1)
        writer.add_scalars('AUC/id_cosine',    {'train': train_auc['id_cosine'],    'test': test_auc['id_cosine']},    epoch + 1)
        writer.add_scalars('AUC/id_euclidean', {'train': train_auc['id_euclidean'], 'test': test_auc['id_euclidean']}, epoch + 1)

        # Console
        _display(epoch + 1, train_metrics, test_metrics)

        # File log
        manager.log_metrics(epoch + 1, {**train_metrics, **test_metrics})

        # Checkpoint + Scheduler
        model_checkpoint(model, optimizer, epoch + 1, test_metrics, scheduler)
        scheduler.step(epoch)

    writer.close()
    manager.log_text("TRAINING HOAN TAT.")


# ---------------------------------------------------------------------------

def train_epoch(dataloader, model, criterion, optimizer, device) -> float:
    model.to(device)
    model.train()

    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)          # (logits, x_norm)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def _display(epoch: int, train_metrics: dict, test_metrics: dict):
    """Hiển thị bảng metrics ra console."""
    rows = []
    for key, val in train_metrics.items():
        test_val = test_metrics.get(key, "-")
        rows.append([key, f"{val:.4f}" if isinstance(val, float) else val,
                     f"{test_val:.4f}" if isinstance(test_val, float) else test_val])
    print(f"\nEp {epoch}:")
    print(tabulate(rows, headers=["Metric", "Train", "Test"], tablefmt="fancy_grid"))
