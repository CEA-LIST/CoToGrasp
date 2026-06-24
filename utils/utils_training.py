from datetime import timedelta, datetime
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_only
import os
import torch
import lightning as L
import statistics



class SaveBestStateDictCallback(L.Callback):
    def __init__(self, save_dir, train_name, save_best_epoch=False):
        super().__init__()
        self.save_path = os.path.join(save_dir, f'{train_name}_state_dict.pth')
        self.best_loss = float('inf')
        self.save_best_epoch = save_best_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if self.save_best_epoch:
            current_loss = trainer.callback_metrics.get("loss")
            if (current_loss is not None) and (current_loss < self.best_loss):
                self.best_loss = current_loss
                torch.save(pl_module.state_dict(), self.save_path)
        else:
            torch.save(pl_module.state_dict(), self.save_path)



class CustomProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_epoch = -1
        self.best_loss = float('inf')
        self.estimated_time_per_epoch = None
        self.start_time = None

        self.BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.start_time = datetime.now()

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        # Reset estimated time per epoch at the start of each epoch
        self.estimated_time_per_epoch = None

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            if not self.train_progress_bar.disable:
                self.train_progress_bar.n = n
                self.train_progress_bar.refresh()

        self.update_progress_bar(trainer, pl_module)
            
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        loss = trainer.callback_metrics.get("loss")
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = trainer.current_epoch

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

    def update_progress_bar(self, trainer, pl_module):
        # Get default metrics
        # metrics = self.get_metrics(trainer, pl_module)
        metrics = {}

        # Add custom information to metrics
        if self.start_time is not None:
            # Track per-epoch durations for a more robust average
            if not hasattr(self, "epoch_durations"):
                self.epoch_durations = []
                self.last_epoch_time = self.start_time

            # Only update durations if a new epoch has started
            if getattr(self, "last_recorded_epoch", -1) != trainer.current_epoch:
                now = datetime.now()
                if trainer.current_epoch > 0:
                    duration = (now - self.last_epoch_time).total_seconds()
                    self.epoch_durations.append(duration)
                self.last_epoch_time = now
                self.last_recorded_epoch = trainer.current_epoch

            # Use median of last N epochs for robustness
            N = min(5, len(self.epoch_durations))
            if N > 0:
                median_epoch_time = statistics.median(self.epoch_durations[-N:])
                remaining_epochs = max(trainer.max_epochs - trainer.current_epoch - 1, 0)
                remaining_total = median_epoch_time * remaining_epochs
                remaining_time_str = str(timedelta(seconds=int(remaining_total)))

                # Add custom information to metrics
                metrics["remaining"] = remaining_time_str
                metrics["best_epoch"] = f"{self.best_epoch}" if self.best_epoch != -1 else "n/a"
                metrics["loss"] = f"{self.best_loss:.4f}" if self.best_epoch != -1 else "n/a"

        # Update bar description and postfix
        description = f"Epoch {trainer.current_epoch}/{trainer.max_epochs - 1} | "
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info() 
            total_gb = total / (1024 ** 3)
            mem_used_gb = (total - free) / (1024 ** 3)
            gpu_util = torch.cuda.utilization()
            description += f"GPU ({gpu_util:.0f} %) : {mem_used_gb:.2f}G/{total_gb:.2f}G "
        
        self.train_progress_bar.set_description(description)
        self.train_progress_bar.set_postfix(metrics)