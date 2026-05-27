
import torch
import os
import shutil

class ModelCheckpoint:
    def __init__(self, output_dir, mode='min', best_metric_name='loss'):
        self.output_dir = output_dir
        self.mode = mode
        self.best_metric_name = best_metric_name
        self.best_metric_val = float('inf') if mode == 'min' else -float('inf')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, model, optimizer, epoch, metrics=None, scheduler=None, extra_state=None):
        # 1. Luôn lưu 'last_model.pth' (Model mới nhất)
        last_path = os.path.join(self.output_dir, 'last_model.pth')

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric_val': self.best_metric_val
        }
        if scheduler:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if extra_state:
            state.update(extra_state)

        torch.save(state, last_path)

        # 2. Kiểm tra lưu 'best_model.pth'
        if metrics is not None and self.best_metric_name in metrics:
            current_val = metrics[self.best_metric_name]
            is_best = False

            if self.mode == 'min':
                if current_val < self.best_metric_val: is_best = True
            else: # mode max
                if current_val > self.best_metric_val: is_best = True

            if is_best:
                self.best_metric_val = current_val
                best_path = os.path.join(self.output_dir, 'best_model.pth')
                shutil.copyfile(last_path, best_path)
                print(f"--> SAVE BEST MODEL ({self.best_metric_name}: {current_val:.4f})")