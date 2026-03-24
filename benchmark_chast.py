"""
Benchmark CHAST forward latency on a test split.

Timing methodology:
- We report mean/std of *batch* forward times (seconds). Per-sample throughput is
  batch_time / batch_size (divide the reported batch stats by batch_size if you
  want per-sample averages).
- DataLoader work is outside the timed region. Each batch is moved to the device
  untimed; we sync, then time only model(x, sparse_input=x) with eval + no_grad.
- Use a moderate batch_size (e.g. 64): std is over batches; dividing by batch_size
  gives an amortized per-sample time comparable across batch sizes for this model.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from src.data import get_in_distribution_test_datasets
from src.model import CHAST
from src.utils import complex_grid_to_2ch

DEFAULT_PILOT_CONFIGS: Tuple[Sequence[int], ...] = (
    (2,),
    (2, 3),
    (2, 7, 11),
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark CHAST forward pass (mean/std batch latency).")
    p.add_argument("--data_path", type=str, required=True, help="Test root, e.g. .../test/TDLA/")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 keeps preload simple).")
    p.add_argument("--snr", type=int, default=20, help="Fixed SNR for TDLDataset (single value => deterministic).")
    p.add_argument("--num_subcarriers", type=int, default=120)
    p.add_argument("--num_symbols", type=int, default=14)
    p.add_argument("--warmup_batches", type=int, default=5)
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda | cpu. Default: cuda if available else cpu.",
    )
    return p


def _select_device(name: str | None) -> torch.device:
    if name is not None:
        dev = str(name).lower()
        if dev.startswith("cuda") and not torch.cuda.is_available():
            print(f"Requested '{name}' but CUDA not available; using cpu.")
            return torch.device("cpu")
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward_batches(
    model: torch.nn.Module,
    device: torch.device,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    warmup_batches: int,
) -> List[float]:
    """Returns wall-clock seconds per batch for model forward only (inputs already sized)."""
    with torch.no_grad():
        for i in range(min(warmup_batches, len(batches))):
            x, _y = batches[i]
            x = x.to(device, non_blocking=(device.type == "cuda"))
            _sync_if_cuda(device)
            _ = model(x, sparse_input=x)

        times: List[float] = []
        for x, _y in batches:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = model(x, sparse_input=x)
            _sync_if_cuda(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return times


def _preload_batches(
    loader: DataLoader,
    max_batches: int | None = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Materialize batches on CPU to separate loading from timed forwards."""
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, (ls_sparse, h_true, _stats) in enumerate(loader):
        x = complex_grid_to_2ch(ls_sparse)
        y = complex_grid_to_2ch(h_true)
        out.append((x, y))
        if max_batches is not None and i + 1 >= max_batches:
            break
    return out


def _model_stats(model: torch.nn.Module) -> Tuple[int, int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes
    return total_params, trainable_params, total_bytes


def main() -> None:
    args = _build_argparser().parse_args()
    data_root = Path(args.data_path)
    if not data_root.is_dir():
        raise ValueError(f"--data_path is not a directory: {data_root}")

    device = _select_device(args.device)
    # Fresh (untrained) model for pure forward-pass latency benchmarking.
    model = CHAST(num_subcarriers=args.num_subcarriers, num_symbols=args.num_symbols).to(device)
    model.eval()

    print(f"device={device}, batch_size={args.batch_size}, warmup_batches={args.warmup_batches}")
    total_params, trainable_params, total_bytes = _model_stats(model)
    print(
        "model_stats: "
        f"total_params={total_params:,}, "
        f"trainable_params={trainable_params:,}, "
        f"size_mib={total_bytes / (1024 ** 2):.4f}"
    )
    print("Pilot configs:", [list(c) for c in DEFAULT_PILOT_CONFIGS])

    for pilot_symbols in DEFAULT_PILOT_CONFIGS:
        pilot_key = ",".join(str(p) for p in pilot_symbols)
        print(f"\n=== pilot_symbols=[{pilot_key}] ===")
        all_times_s: List[float] = []
        scenario_count = 0

        for _folder_name, dataset in get_in_distribution_test_datasets(
            data_root,
            return_pilots_only=False,
            SNRs=[args.snr],
            pilot_symbols=list(pilot_symbols),
        ):
            scenario_count += 1
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            batches = _preload_batches(loader)
            if not batches:
                continue

            times_s = _time_forward_batches(model, device, batches, args.warmup_batches)
            all_times_s.extend(times_s)

        if not all_times_s:
            print(f"  no valid batches across {scenario_count} scenarios")
            continue

        mean_s = statistics.mean(all_times_s)
        std_s = statistics.stdev(all_times_s) if len(all_times_s) > 1 else 0.0
        mean_ms = mean_s * 1e3
        std_ms = std_s * 1e3
        per_sample_ms = mean_ms / args.batch_size

        print(
            f"  all_scenarios: n_scenarios={scenario_count}, n_batches={len(all_times_s)}, "
            f"batch_time_ms mean={mean_ms:.4f} std={std_ms:.4f}, "
            f"amortized_per_sample_ms={per_sample_ms:.6f}"
        )


if __name__ == "__main__":
    main()
