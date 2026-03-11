import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import get_in_distribution_test_datasets
from src.model import CHAST
from src.utils import complex_grid_to_2ch


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained CHAST checkpoint (NMSE dB).")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30])
    p.add_argument("--pilot_symbols", type=int, nargs="+", default=[2, 7, 11])
    p.add_argument("--pilot_every_n", type=int, default=2)
    p.add_argument("--num_subcarriers", type=int, default=120)
    p.add_argument("--num_symbols", type=int, default=14)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation (e.g. 'cuda' or 'cpu'). Defaults to CUDA if available, else CPU.",
    )
    return p


@torch.no_grad()
def main() -> None:
    args = _build_argparser().parse_args()

    if args.device is not None:
        dev_name = args.device.lower()
        if dev_name == "cuda" and not torch.cuda.is_available():
            print("Requested device 'cuda' but CUDA is not available; falling back to 'cpu'.")
            dev_name = "cpu"
        if dev_name not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported device '{args.device}'; use 'cpu' or 'cuda'.")
        device = torch.device(dev_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CHAST(num_subcarriers=args.num_subcarriers, num_symbols=args.num_symbols).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for snr in args.snrs:
        print(f"=== SNR = {snr} dB ===")
        for folder_name, dataset in get_in_distribution_test_datasets(
            Path(args.data_path),
            return_pilots_only=False,
            SNRs=[snr],
            pilot_symbols=args.pilot_symbols,
        ):
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            num_sum = torch.tensor(0.0, device=device)
            den_sum = torch.tensor(0.0, device=device)

            for ls_sparse, h_true, _stats in loader:
                x = complex_grid_to_2ch(ls_sparse).to(device)
                y = complex_grid_to_2ch(h_true).to(device)
                pred = model(x, sparse_input=x)

                err = pred - y
                num_sum += (err * err).sum()
                den_sum += (y * y).sum()

            nmse = num_sum / den_sum
            nmse_db = 10.0 * torch.log10(nmse)
            print(f"  folder={folder_name}: NMSE (dB) = {float(nmse_db.detach().cpu()):.3f}")


if __name__ == "__main__":
    main()

