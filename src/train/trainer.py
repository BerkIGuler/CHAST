from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import complex_grid_to_2ch

from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int = 50
    min_delta: float = 1e-5


@dataclass(frozen=True)
class CheckpointConfig:
    out_dir: Path
    filename: str = "best.pt"

    @property
    def path(self) -> Path:
        return self.out_dir / self.filename


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint: CheckpointConfig,
        early_stopping: EarlyStoppingConfig = EarlyStoppingConfig(),
        run_config: Optional[Dict[str, Any]] = None,
        tb_writer: Optional["SummaryWriter"] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping
        self.run_config = run_config
        self.tb_writer = tb_writer

        self.criterion = nn.MSELoss()

        self.best_val_nmse_db = float("inf")
        self.best_epoch = -1
        self._no_improve = 0

        self.checkpoint.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _batch_to_xy(
        batch: Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ls_sparse, h_true, _stats = batch
        x = complex_grid_to_2ch(ls_sparse).to(device)
        y = complex_grid_to_2ch(h_true).to(device)
        return x, y

    @staticmethod
    def _nmse_sums(
        pred_2ch: torch.Tensor, target_2ch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (sum_num, sum_den) across batch for NMSE:
            sum_num = Σ ||pred - target||^2
            sum_den = Σ ||target||^2
        """
        err = pred_2ch - target_2ch
        dims = tuple(range(1, pred_2ch.ndim))
        num = (err * err).sum(dim=dims)  # (B,)
        den = (target_2ch * target_2ch).sum(dim=dims)  # (B,)
        return num.sum(), den.sum()

    def train(self, *, epochs: int, epoch_offset: int = 0) -> Dict[str, Any]:
        """
        Run `epochs` training steps with a fresh optimizer/scheduler (caller responsibility).

        `epoch_offset` is added so TensorBoard and logs use **global** 1-based epoch indices:
        first step uses epoch == epoch_offset + 1, last == epoch_offset + epochs
        (unless early stopping cuts short).
        """
        if epoch_offset < 0:
            raise ValueError(f"epoch_offset must be >= 0, got {epoch_offset}")
        for epoch in range(epoch_offset + 1, epoch_offset + epochs + 1):
            train_loss = self._train_one_epoch()
            val_nmse_db = self._validate()

            if self.scheduler is not None:
                self.scheduler.step()

            improved = (self.best_val_nmse_db - val_nmse_db) > self.early_stopping.min_delta
            if improved:
                self.best_val_nmse_db = val_nmse_db
                self.best_epoch = epoch
                self._no_improve = 0
                self._save_checkpoint(epoch=epoch, val_nmse_db=val_nmse_db)
            else:
                self._no_improve += 1

            lr = self.optimizer.param_groups[0].get("lr", None)
            lr_str = f"{lr:.3e}" if isinstance(lr, float) else str(lr)
            print(
                f"epoch {epoch:03d} | train_mse={train_loss:.6e} | val_nmse_db={val_nmse_db:.3f} | best={self.best_val_nmse_db:.3f} @ {self.best_epoch:03d} | lr={lr_str}"
            )

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("loss/train_mse", train_loss, epoch)
                self.tb_writer.add_scalar("nmse/val_db", val_nmse_db, epoch)
                if isinstance(lr, float):
                    self.tb_writer.add_scalar("optim/lr", lr, epoch)

            if self._no_improve >= self.early_stopping.patience:
                print(
                    f"early stopping: no improvement for {self.early_stopping.patience} epochs (min_delta={self.early_stopping.min_delta})"
                )
                break

        return {
            "best_val_nmse_db": self.best_val_nmse_db,
            "best_epoch": self.best_epoch,
            "best_checkpoint_path": str(self.checkpoint.path),
        }

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_weighted_loss = 0.0
        total_samples = 0

        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for batch in pbar:
            x, y = self._batch_to_xy(batch, self.device)
            batch_size = x.size(0)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(x, sparse_input=x)
            loss = self.criterion(pred, y)

            loss.backward()
            self.optimizer.step()

            total_weighted_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            pbar.set_postfix(mse=float(loss.detach().cpu()))

        return total_weighted_loss / max(total_samples, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()

        num_sum = torch.tensor(0.0, device=self.device)
        den_sum = torch.tensor(0.0, device=self.device)

        pbar = tqdm(self.val_loader, desc="val", leave=False)
        for batch in pbar:
            x, y = self._batch_to_xy(batch, self.device)
            pred = self.model(x, sparse_input=x)
            bnum, bden = self._nmse_sums(pred, y)
            num_sum += bnum
            den_sum += bden
            nmse = num_sum / den_sum
            pbar.set_postfix(nmse_db=float((10.0 * torch.log10(nmse)).detach().cpu()))

        nmse = num_sum / den_sum
        nmse_db = 10.0 * torch.log10(nmse)
        return float(nmse_db.detach().cpu())

    def _save_checkpoint(self, *, epoch: int, val_nmse_db: float) -> None:
        payload = {
            "epoch": epoch,
            "val_nmse_db": val_nmse_db,
            "config": self.run_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(payload, self.checkpoint.path)

    @staticmethod
    def load_model_weights(
        path: Union[str, Path],
        *,
        model: nn.Module,
        map_location: Union[str, torch.device] = "cpu",
    ) -> None:
        """Load only `model_state_dict` (no optimizer/scheduler)."""
        ckpt = torch.load(str(path), map_location=map_location)
        model.load_state_dict(ckpt["model_state_dict"])

    @staticmethod
    def load_checkpoint(
        path: Union[str, Path],
        *,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        ckpt = torch.load(str(path), map_location=map_location)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt

