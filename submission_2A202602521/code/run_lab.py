"""Run reproducible validation-only experiments for the Day 1 neural-network lab.

Run from the repository root, for example:
    .venv\\Scripts\\python.exe code\\run_lab.py --out submission_MSSV
The script deliberately does not evaluate on the held-out eval set.  Call
scripts/evaluate.py only after selecting the final configuration from val scores.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data import prepare_data
from plots import plot_compare, plot_run
from results_table import save_result
from train import DEFAULT_CFG, run_experiment


def configs() -> list[dict]:
    """A fair, fixed 20-epoch validation experiment plan covering all lab topics."""
    base = dict(epochs=20, batch=2048, hidden=(256, 128), seed=1, precision="fp32")
    return [
        {**base, "exp_id": "base-s1", "group": "baseline", "description": "M-base, AdamW lr=1e-3, He", "optimizer": "adamw", "lr": 1e-3, "init": "he"},
        {**base, "exp_id": "base-s2", "group": "baseline", "description": "Baseline repeat, seed 2", "optimizer": "adamw", "lr": 1e-3, "init": "he", "seed": 2},
        {**base, "exp_id": "base-s3", "group": "baseline", "description": "Baseline repeat, seed 3", "optimizer": "adamw", "lr": 1e-3, "init": "he", "seed": 3},
        {**base, "exp_id": "loss-mse", "group": "loss", "description": "MSE on one-hot labels", "optimizer": "adamw", "lr": 1e-3, "init": "he", "loss": "mse"},
        {**base, "exp_id": "opt-sgdm-lr01", "group": "optimizer", "description": "SGD momentum, lr=0.01", "optimizer": "sgd_momentum", "lr": 1e-2, "init": "he"},
        {**base, "exp_id": "opt-sgdm-lr05", "group": "optimizer", "description": "SGD momentum, lr=0.05", "optimizer": "sgd_momentum", "lr": 5e-2, "init": "he"},
        {**base, "exp_id": "opt-adam-lr3e4", "group": "optimizer", "description": "Adam, lr=3e-4", "optimizer": "adam", "lr": 3e-4, "init": "he"},
        {**base, "exp_id": "opt-adam-lr1e3", "group": "optimizer", "description": "Adam, lr=1e-3", "optimizer": "adam", "lr": 1e-3, "init": "he"},
        {**base, "exp_id": "lr-adamw-3e4", "group": "hyperparameter", "description": "AdamW, lower learning rate", "optimizer": "adamw", "lr": 3e-4, "init": "he"},
        {**base, "exp_id": "dropout-p03", "group": "dropout", "description": "Dropout p=0.3", "optimizer": "adamw", "lr": 1e-3, "init": "he", "dropout": 0.3},
        {**base, "exp_id": "clip-highlr-none", "group": "clipping", "description": "AdamW lr=0.01, no clipping", "optimizer": "adamw", "lr": 1e-2, "init": "he", "clip_norm": None},
        {**base, "exp_id": "clip-highlr-1", "group": "clipping", "description": "AdamW lr=0.01, clip norm 1", "optimizer": "adamw", "lr": 1e-2, "init": "he", "clip_norm": 1.0},
        {**base, "exp_id": "precision-bf16", "group": "mixed_precision", "description": "BF16 requested; CPU falls back to FP32", "optimizer": "adamw", "lr": 1e-3, "init": "he", "precision": "bf16"},
        {**base, "exp_id": "precision-fp16", "group": "mixed_precision", "description": "FP16 with GradScaler on GPU", "optimizer": "adamw", "lr": 1e-3, "init": "he", "precision": "fp16"},
        {**base, "exp_id": "init-xavier", "group": "initialization", "description": "Xavier normal initialization", "optimizer": "adamw", "lr": 1e-3, "init": "xavier"},
        {**base, "exp_id": "init-normal", "group": "initialization", "description": "Normal N(0, 0.01) initialization", "optimizer": "adamw", "lr": 1e-3, "init": "normal"},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission_2A202602521")
    parser.add_argument("--only", nargs="*", help="run only these exp_id values")
    args = parser.parse_args()
    out = Path(args.out)
    figures, results = out / "figures", out / "results"
    figures.mkdir(parents=True, exist_ok=True); results.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    data = prepare_data("cuda" if torch.cuda.is_available() else "cpu")
    completed = []
    for cfg in configs():
        if args.only and cfg["exp_id"] not in args.only:
            continue
        print(f"\n=== {cfg['exp_id']} ===", flush=True)
        result = run_experiment(deepcopy({**DEFAULT_CFG, **cfg}), data)
        save_result(result, str(results))
        plot_run(result, str(figures / f"{cfg['exp_id']}.png"))
        print(result["summary"], flush=True)
        completed.append(result)
    for group in sorted({r["cfg"]["group"] for r in completed}):
        group_results = [r for r in completed if r["cfg"]["group"] == group]
        if len(group_results) > 1:
            plot_compare(group_results, "val_macro_f1", str(figures / f"compare_{group}.png"), f"Validation macro-F1: {group}")
    if completed:
        ranked = sorted(completed, key=lambda r: r["summary"]["val_macro_f1"], reverse=True)
        print("\nTop validation runs:")
        for r in ranked[:5]:
            print(r["cfg"]["exp_id"], f"F1={r['summary']['val_macro_f1']:.4f}", f"loss={r['summary']['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
