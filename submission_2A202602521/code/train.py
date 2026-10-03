"""Huấn luyện, đánh giá và tạo file dự đoán cho bài lab.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import copy
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=1e-3,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros(7), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(7), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=(precision + recall) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    device = next(model.parameters()).device
    parts = []
    for start in range(0, len(X), batch_size):
        parts.append(model(X[start:start + batch_size].to(device)).argmax(dim=1))
    return torch.cat(parts)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_correct = 0.0, 0
    cm = np.zeros((7, 7), dtype=np.int64)
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size].to(device), y[start:start + batch_size].to(device)
        logits = model(xb)
        if loss_name == "ce":
            total_loss += float(F.cross_entropy(logits, yb, reduction="sum").item())
        elif loss_name == "mse":
            total_loss += float(F.mse_loss(logits, F.one_hot(yb, 7).float(), reduction="sum").item())
        else:
            raise ValueError(f"unknown loss {loss_name!r}")
        pred = logits.argmax(1)
        total_correct += int((pred == yb).sum().item())
        np.add.at(cm, (yb.cpu().numpy(), pred.cpu().numpy()), 1)
    divisor = len(X) if loss_name == "ce" else len(X) * 7
    return {"loss": total_loss / divisor, "acc": total_correct / len(X), "macro_f1": macro_f1_from_confusion(cm)}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        return F.mse_loss(logits, F.one_hot(y, num_classes=7).float())
    raise ValueError(f"unknown loss {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] must be specified")
    set_seed(cfg["seed"])
    device = data["X_tr"].device
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])]
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"], cfg["momentum"])
    use_amp = cfg["precision"] in {"fp16", "bf16"} and device.type == "cuda"
    amp_dtype = torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and cfg["precision"] == "fp16")
    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    step0 = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])["loss"]
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s")}
    best_loss, best_epoch, best_state, diverged = math.inf, 0, None, False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg["epochs"] + 1):
        begin = time.perf_counter()
        model.train()
        norms = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], generator):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = compute_loss(model(xb), yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            scaler.scale(loss).backward()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            norms.append(clip_gradients(model.parameters(), cfg["clip_norm"]))
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        train_scores = evaluate(model, data["X_tr"], data["y_tr"], cfg["loss"])
        val_scores = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        history["epoch"].append(epoch)
        history["train_loss"].append(train_scores["loss"])
        history["val_loss"].append(val_scores["loss"])
        history["val_acc"].append(val_scores["acc"])
        history["val_macro_f1"].append(val_scores["macro_f1"])
        history["grad_norm"].append(float(np.mean(norms)) if norms else float("nan"))
        history["epoch_time_s"].append(time.perf_counter() - begin)
        if val_scores["loss"] < best_loss:
            best_loss, best_epoch = val_scores["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if diverged: break
    pick = best_epoch - 1
    summary = {
        "step0_loss": step0, "best_val_loss": best_loss, "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1], "final_val_loss": history["val_loss"][-1],
        "val_acc": history["val_acc"][pick], "val_macro_f1": history["val_macro_f1"][pick],
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])),
        "peak_mem_MB": (torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0),
        "diverged": diverged,
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["row_id", "pred"])
        writer.writerows(zip(np.asarray(row_id).astype(int), np.asarray(preds).astype(int)))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    device = data["X_eval"].device
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    write_predictions(data["eval_row_id"], predict(model, data["X_eval"]).cpu().numpy(), pred_path)
