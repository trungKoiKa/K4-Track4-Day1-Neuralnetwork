"""Vẽ biểu đồ từng thí nghiệm và so sánh các cấu hình.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg, hist, summary = result["cfg"], result["history"], result["summary"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    epoch = hist["epoch"]
    axes[0].plot(epoch, hist["train_loss"], label="train loss")
    axes[0].plot(epoch, hist["val_loss"], label="val loss")
    axes[0].axvline(summary["best_epoch"], color="black", ls=":", label="best epoch")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epoch, hist["val_acc"], label="val accuracy")
    axes[1].plot(epoch, hist["val_macro_f1"], label="val macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[2].plot(epoch, hist["grad_norm"], label="mean gradient norm")
    if cfg["clip_norm"] is not None:
        axes[2].axhline(cfg["clip_norm"], color="crimson", ls="--", label="clip threshold")
    axes[2].set(title="Gradient norm before clipping", xlabel="Epoch", ylabel="L2 norm")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.suptitle(f"{cfg['exp_id']} | {cfg['optimizer']} lr={cfg['lr']} batch={cfg['batch']} | "
                 f"init={cfg['init']} dropout={cfg['dropout']} clip={cfg['clip_norm']} {cfg['precision']}")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in results:
        h, cfg = result["history"], result["cfg"]
        ax.plot(h["epoch"], h[metric], label=cfg["exp_id"])
    ax.set(title=title or metric, xlabel="Epoch", ylabel=metric)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
