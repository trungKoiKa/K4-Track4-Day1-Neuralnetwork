"""Finalize Colab validation runs; evaluate only baseline and the selected model."""
from pathlib import Path
import json
import sys
import shutil
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import prepare_data
from model import MLP, count_params, activation_stats
from train import run_experiment, final_eval, set_seed
from plots import plot_run, plot_compare
from results_table import load_results, save_result, to_row, write_xlsx


def main(out_dir='submission_2A202602521'):
    root = Path.cwd()
    out = root / out_dir
    results = load_results(out / 'results')
    selected = max(results, key=lambda r: r['summary']['val_macro_f1'])['cfg']['exp_id']
    print('Selected by validation:', selected, flush=True)
    data = prepare_data('cuda' if torch.cuda.is_available() else 'cpu')
    scores = {}
    for exp_id in dict.fromkeys(['base-s1', selected]):
        old = next(r for r in results if r['cfg']['exp_id'] == exp_id)
        result = run_experiment(old['cfg'], data)
        # Repeat with the same seed to recover weights; record the actual repeated run.
        save_result(result, str(out / 'results'))
        plot_run(result, str(out / 'figures' / f'{exp_id}.png'))
        pred = out / ('predictions_eval.csv' if exp_id == selected else 'predictions_baseline.csv')
        target = out / ('eval_result.json' if exp_id == selected else 'eval_baseline.json')
        final_eval(result['cfg'], result, data, str(pred))
        subprocess.run([sys.executable, 'scripts/evaluate.py', '--pred', str(pred), '--out', str(target)], check=True)
        scores[exp_id] = json.loads(target.read_text())
    results = load_results(out / 'results')
    baseline = next(r for r in results if r['cfg']['exp_id'] == 'base-s1')
    final = next(r for r in results if r['cfg']['exp_id'] == selected)
    seeds = [r for r in results if r['cfg']['group'] == 'baseline']
    noise = 2 * np.std([r['summary']['val_macro_f1'] for r in seeds], ddof=1)
    for group in sorted({r['cfg']['group'] for r in results}):
        members = [r for r in results if r['cfg']['group'] == group]
        if group != 'baseline':
            members = [baseline] + members
        plot_compare(members, 'val_macro_f1', str(out / 'figures' / f'compare_{group}.png'), f'Validation macro-F1: {group}')

    set_seed(1)
    model = MLP().to(data['X_tr'].device)
    xb, yb = data['X_tr'][:20], data['y_tr'][:20]
    shape = list(model(xb).shape)
    F.cross_entropy(model(xb), yb).backward()
    gradients = all(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(500):
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
    health = dict(parameters=count_params(model), logits_shape=shape, all_parameter_gradients_nonzero=gradients,
                  overfit20_loss=F.cross_entropy(model(xb), yb).item(), torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    health['activation_stats'] = {}
    for init in ['he', 'xavier', 'normal', 'zeros']:
        set_seed(1)
        health['activation_stats'][init] = activation_stats(MLP(init=init).to(xb.device), data['X_tr'][:2048])
    (out / 'health_checks.json').write_text(json.dumps(health, indent=2), encoding='utf-8')

    rows = [to_row(r, scores.get(r['cfg']['exp_id']),
                   (f"Rerun same seed for eval. Health: parameters={health['parameters']}, logits=(20,7), gradients={gradients}, overfit20_loss={health['overfit20_loss']:.8f}." if r['cfg']['exp_id'] == 'base-s1'
                    else 'Rerun same seed to recover weights for eval.' if r['cfg']['exp_id'] in scores else '')) for r in results]
    write_xlsx(rows, str(root / 'templates/experiment_table_template.xlsx'), str(out / 'experiments.xlsx'))
    import openpyxl
    wb = openpyxl.load_workbook(out / 'experiments.xlsx')
    ws = wb['Experiments']
    for row in ws.iter_rows(min_row=len(rows)+2):
        for cell in row[:29]:
            cell.value = None
    for i in range(2, 7):
        wb['Seeds'].cell(i, 1).value = f'base-s{i-1}' if i < 5 else None
    wb.save(out / 'experiments.xlsx')

    cm = np.asarray(scores[selected]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Fraction within true class')
    ax.set(xlabel='Predicted label', ylabel='True label', title=f'Eval confusion: {selected}', xticks=range(7), yticks=range(7))
    fig.tight_layout(); fig.savefig(out / 'figures/confusion_eval.png', dpi=150); plt.close(fig)

    lines = ['# Báo cáo Lab Day 1 — Hoàng Trung Anh — 2A202602521', '', '## 1. Thiết lập',
             f"Google Colab, {health['gpu']}, PyTorch {health['torch_version']}. M-base 54→256→128→7, 47 879 tham số. Chuẩn hoá 10 cột số chỉ bằng train; 44 cột nhị phân giữ nguyên.",
             'Train/eval theo metadata: 464 809/116 203. Validation phân tầng 20%, seed 42: 371 847 train, 92 962 val. Accuracy đoán lớp đa số: 0.4876.',
             'Baseline: CE, AdamW, lr=0.001, batch=2048, 20 epoch, He, FP32, không dropout/weight decay/clipping. Batch cuối nhỏ hơn được giữ lại. Mọi thí nghiệm cùng số bước mỗi epoch.',
             '', '## 2. Kiểm tra ban đầu và độ nhiễu',
             f"Logits: {shape}; gradient khác 0 ở mọi tensor tham số: {gradients}. Overfit 20 mẫu, 500 bước Adam lr=0.01: loss={health['overfit20_loss']:.6f}.",
             f"Step-0 CE baseline: {baseline['summary']['step0_loss']:.6f}; ln(7)=1.945910. He có logits ngẫu nhiên với phương sai khác 0 nên CE có thể cao hơn ln(7).",
             f"Baseline 3 seed: val macro-F1={np.mean([r['summary']['val_macro_f1'] for r in seeds]):.6f} ± {noise/2:.6f}; ngưỡng 2σ={noise:.6f} (độ lệch chuẩn mẫu).",
             '', '## 3. Kết quả theo chủ đề', '| exp_id | lr | val acc | val macro-F1 | epoch tốt nhất | giây/epoch |', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        c, s = r['cfg'], r['summary']
        lines.append(f"| {c['exp_id']} | {c['lr']} | {s['val_acc']:.6f} | {s['val_macro_f1']:.6f} | {s['best_epoch']} | {s['time_per_epoch_s']:.4f} |")
    explanations = {
        'loss': 'CE tối ưu xác suất lớp qua softmax; MSE ở đây lấy trung bình bình phương lỗi giữa logits thô và one-hot, không phải giữa xác suất và one-hot. Hai thang loss khác nhau; chỉ so accuracy/F1.',
        'optimizer': 'SGD momentum và Adam được thử hai learning rate. AdamW với weight_decay=0 tương đương Adam; đây không phải bằng chứng cho lợi ích của decoupled weight decay.',
        'hyperparameter': 'Learning rate nhỏ có thể hội tụ chậm trong ngân sách 20 epoch. Kết luận chỉ đúng trong lưới lr đã thử.',
        'dropout': 'Dropout p=0.3 thêm nhiễu và giảm năng lực học. Nếu baseline chưa quá khớp mạnh, dropout có thể làm giảm F1 trong cùng ngân sách.',
        'clipping': 'So sánh lr=0.01 có/không clip norm=1. Grad norm ghi trước clip. Không kết luận clipping cứu phân kỳ nếu cả hai đều hội tụ; trung bình grad norm không cho biết chính xác tỷ lệ batch kích hoạt clip.',
        'mixed_precision': 'T4 không hỗ trợ BF16 native; PyTorch có thể thực thi BF16 qua fallback/emulation. FP16 dùng GradScaler, unscale trước khi đo/clip gradient. Thời gian còn gồm đánh giá FP32 toàn train/val; tốc độ mạng nhỏ có thể bị giới hạn bởi overhead.',
        'initialization': 'He giữ phương sai phù hợp ReLU, Xavier cân bằng fan-in/fan-out; normal 0.01 có thể giảm tín hiệu. Zeros phá đối xứng và chặn gradient qua các hidden layer, không phải khởi tạo phù hợp.'}
    for group, explanation in explanations.items():
        group_runs = [r for r in results if r['cfg']['group'] == group]
        best = max(group_runs, key=lambda r: r['summary']['val_macro_f1'])
        delta = best['summary']['val_macro_f1'] - baseline['summary']['val_macro_f1']
        lines += ['', f'### {group}', explanation,
                  f"Tốt nhất nhóm: `{best['cfg']['exp_id']}`, ΔF1 so base-s1={delta:+.6f}; |Δ| {'vượt' if abs(delta)>noise else 'chưa vượt'} ngưỡng 2σ baseline. Chỉ baseline được lặp seed nên đây là so sánh thăm dò.",
                  f'![{group}](figures/compare_{group}.png)']
    lines += ['', '## 4. Đánh giá cuối trên eval',
              f"Chọn `{selected}` bằng macro-F1 validation trước khi xem eval; checkpoint chọn bằng val loss thấp nhất. Sau đó chạy scripts/evaluate.py cho baseline và cấu hình cuối.",
              '| Cấu hình | eval accuracy | eval macro-F1 |', '|---|---:|---:|']
    for exp_id, score in scores.items():
        lines.append(f"| {exp_id} | {score['accuracy']:.6f} | {score['macro_f1']:.6f} |")
    lines += ['', '| Lớp | Support | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for c in scores[selected]['per_class']:
        lines.append(f"| {c['cls']} | {c['support']} | {c['precision']:.6f} | {c['recall']:.6f} | {c['f1']:.6f} |")
    hardest = min(scores[selected]['per_class'], key=lambda c: c['f1'])
    errors = cm[hardest['cls']].copy(); errors[hardest['cls']] = 0
    lines += [f"Lớp khó nhất: {hardest['cls']} (F1={hardest['f1']:.6f}), nhầm nhiều nhất sang lớp {errors.argmax()}. Mất cân bằng và vùng đặc trưng chồng lấn là giả thuyết, chưa kiểm chứng; có thể thử weighted CE bằng validation.",
              '![Confusion](figures/confusion_eval.png)', '', '## 5. Câu hỏi dẫn dắt',
              '1. Chọn bộ tối ưu qua bảng ở lr tốt nhất trong lưới đã thử; lr chưa chỉnh có thể đảo thứ hạng. Lưới nhỏ và AdamW wd=0 giới hạn kết luận.',
              '2. Dropout chỉ nên thêm khi train–val gap và độ nhiễu cho thấy overfit; chưa overfit thì regularization có thể làm underfit.',
              '3. Clipping giới hạn chuẩn gradient để giảm bước cập nhật quá lớn; phải đối chiếu grad norm và đường loss, không suy luận từ F1 một mình.',
              '4. So precision-fp16 với base-s1 trong bảng: thời gian gồm overhead và đánh giá FP32, nên mixed precision không đảm bảo nhanh hơn cho MLP nhỏ.',
              '5. Zeros làm neuron đối xứng và gradient hidden không học được; He dùng phương sai 2/fan_in cho ReLU, Xavier dùng 2/(fan_in+fan_out).',
              '6. Nếu loss không giảm sau 2 000 bước: kiểm tra nhãn/dtype/shape và CE dùng logits; kiểm tra gradient, zero_grad/backward/step và scale đầu vào; thử overfit 20 mẫu rồi quét lr. Ba bước tách lỗi dữ liệu, lỗi cập nhật và lỗi cấu hình.',
              '', '## 6. Hạn chế',
              'Ba seed chỉ đo nhiễu baseline, chưa đo nhiễu từng cấu hình. Lặp baseline/final cùng seed để lấy trọng số cho eval và ghi lại kết quả thực tế; GPU có thể không bitwise deterministic. Không có weight decay khác 0, lưới lr nhỏ, không khảo sát độ rộng/độ sâu. Các giải thích lý thuyết được bổ sung sau lượt chạy đầu, không phải bản dự đoán được ghi trước thí nghiệm.',
              'Chỉ có một seed eval mỗi cấu hình nên không khẳng định cải thiện eval có ý nghĩa thống kê. Ngưỡng nhiễu validation chỉ là tham chiếu.',
              'peak_mem_MB đo tổng bộ nhớ PyTorch đang cấp phát, gồm dữ liệu và các đối tượng còn giữ trong kernel. Khi lặp baseline/final để khôi phục trọng số, còn bộ dữ liệu của lượt trước nên peak cao hơn. Không dùng các số peak này để kết luận mixed precision tiết kiệm bộ nhớ; cần đo lại ở các kernel tách biệt.',
              '', '## 7. Phụ lục',
              'Code, notebook, experiments.xlsx, predictions_eval.csv, eval_result.json, figures và results. predictions_baseline.csv/eval_baseline.json giữ để đối chiếu. Không nộp dữ liệu hoặc trọng số.']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    code_out = out / 'code'; code_out.mkdir(exist_ok=True)
    for file in Path(__file__).resolve().parent.iterdir():
        if file.suffix not in {'.py', '.ipynb', '.txt'}:
            continue
        target = code_out / file.name
        if file.resolve() != target.resolve():
            shutil.copy2(file, target)
    (out / 'selection.json').write_text(json.dumps({'selected_exp_id': selected, 'criterion': 'highest validation macro-F1 before eval', 'eval_scores': scores}, indent=2), encoding='utf-8')
    print('FINAL_EVAL_COMPLETE', json.dumps(scores), flush=True)


if __name__ == '__main__':
    main()
