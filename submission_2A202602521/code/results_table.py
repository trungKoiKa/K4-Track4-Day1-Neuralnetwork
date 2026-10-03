"""Lưu lịch sử thí nghiệm và điền bảng kết quả theo mẫu của bài lab.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    root = Path(results_dir); root.mkdir(parents=True, exist_ok=True)
    payload = {key: result[key] for key in ("cfg", "history", "summary")}
    path = root / f"{result['cfg']['exp_id']}.json"
    nonfinite = {}
    for key, values in payload['history'].items():
        indices = [i + 1 for i, value in enumerate(values) if isinstance(value, float) and not math.isfinite(value)]
        if indices: nonfinite[key] = indices
    if nonfinite:
        payload['summary'] = {**payload['summary'], 'nonfinite_history_epochs': nonfinite}
    def finite_json(value):
        if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    path.write_text(json.dumps(finite_json(payload), indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(results_dir).glob("*.json"))]


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg, summary = result["cfg"], result["summary"]
    fields = ("exp_id", "group", "description", "loss", "optimizer", "lr", "weight_decay", "batch", "epochs", "dropout", "clip_norm", "precision", "init", "seed")
    row = {key: cfg.get(key) for key in fields}
    row["hidden"] = "-".join(map(str, cfg["hidden"]))
    row.update(summary)
    row.update({"eval_acc": None, "eval_macro_f1": None, "figure_file": f"figures/{cfg['exp_id']}.png", "notes": notes})
    if eval_scores:
        row["eval_acc"] = eval_scores.get("accuracy", eval_scores.get("eval_acc"))
        row["eval_macro_f1"] = eval_scores.get("macro_f1", eval_scores.get("eval_macro_f1"))
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    # The template is intentionally copied first so its required sheets, styles and formulas survive.
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("Install openpyxl to populate the required template workbook.") from exc
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, out)
    wb = openpyxl.load_workbook(out)
    ws = wb["Experiments"]
    headers = {cell.value: cell.column for cell in ws[1] if cell.value}
    formula_columns = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"}
    for row_index, row in enumerate(rows, start=2):
        for key, value in row.items():
            if key in headers and key not in formula_columns:
                ws.cell(row_index, headers[key], value)
    for extra_row in ws.iter_rows(min_row=len(rows) + 2):
        for cell in extra_row[:29]:
            cell.value = None
    for index in range(2, 7):
        wb['Seeds'].cell(index, 1).value = f'base-s{index-1}' if index < 5 else None
    for address, value in {'A5': 'hyperparameter', 'A8': 'mixed_precision', 'A9': 'initialization'}.items():
        wb['Summary'][address] = value
    wb['Summary']['A14'] = '7 chủ đề: loss, optimizer, hyperparameter, dropout, clipping, mixed_precision, initialization.'
    wb['Summary'].column_dimensions['A'].width = 22
    wb.save(out)
