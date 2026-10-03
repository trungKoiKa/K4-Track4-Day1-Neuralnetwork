# Đối chiếu bài nộp — 03/10/2026

Bài nộp của **Hoàng Trung Anh — 2A202602521** đã được hoàn thiện bằng lượt chạy mới trên Google Colab Tesla T4. Các thiếu sót kỹ thuật của bản trước đã được sửa; lịch sử đã xem điểm eval vẫn được công khai.

## Nguồn yêu cầu

Đã đối chiếu toàn bộ Markdown của bài: [README.md](README.md), [GUIDE.md](GUIDE.md), [RUBRIC.md](RUBRIC.md), [data/README.md](data/README.md), [REPORT_TEMPLATE.md](templates/REPORT_TEMPLATE.md) và [báo cáo thực tế](submission_2A202602521/REPORT.md). Giữ nguyên đề, rubric, dữ liệu và template. Ghi chú đầu README/GUIDE phân biệt khung ban đầu với phần triển khai hiện tại.

## Kết quả kiểm tra

| Yêu cầu | Bằng chứng trong submission_2A202602521 |
|---|---|
| Split metadata cố định, val 20% stratify seed 42, chuẩn hoá chỉ từ train | code/data.py; health_checks.json; 371 847 train, 92 962 val, 116 203 eval |
| M-base tự định nghĩa, logits thô, dropout sau ReLU, 47 879 tham số | code/model.py; assert và output notebook |
| Kiểm tra sức khoẻ trước chạy dài | Overfit 20 mẫu accuracy 100%, loss 0.00000439, health_overfit20.png; gradient mọi tham số; uniform CE = ln 7; activation std trên val |
| Baseline đúng GUIDE | SGD momentum 0,9; batch 512; 20 epoch; He/CE/FP32; lr 0.1 chọn qua lưới val 0.01/0.05/0.1 |
| Nhiễu baseline | base-s1..3; val F1 0.853510 ± 0.012555, std mẫu; Seeds có công thức và giá trị tính lại |
| Đủ 7 chủ đề, giả thuyết trước run mới | 25 dòng Experiments, 25 JSON và 25 PNG riêng; protocol.json và planned/ ghi kế hoạch trước worker |
| So sánh công bằng | Adam/AdamW mỗi bộ 3 lr; cùng split/seed/ngân sách; thay batch riêng; cặp clipping cùng lr; khác biệt số update và optimizer/lr được giải thích |
| Clipping có căn cứ | c = grad_p90 baseline; clip_fraction; cặp highlr-none/highlr-clip; không tuyên bố cứu phân kỳ khi cả hai đều không diverged |
| Precision đo được | Mỗi run process GPU riêng; FP32/FP16/BF16 có thời gian/bộ nhớ/F1; nêu BF16 không native và overhead |
| Log hợp lệ | Train loss ở eval mode, val metric, grad trước clip, step0 loss, epoch time, peak memory, diverged; JSON chuẩn, giá trị không hữu hạn ghi null kèm epoch |
| Chọn final bằng val | adam-lr0.003, checkpoint min val loss; protocol khoá trước scoring mới; final seed 2/3 không retune |
| Eval chính thức và phân tích lớp | predictions_eval.csv đủ 116 203 ID, nhãn 0..6; eval_result.json khớp báo cáo/Excel; bảng từng lớp và confusion_eval.png |
| Excel và ảnh | 4 sheet giữ cột mẫu, 25 dòng; không lỗi công thức; Summary đủ 7 chủ đề, số health/seed/từng lớp; ảnh riêng và ảnh nhóm |
| Notebook và đóng gói | 8 code cell có execution_count/output, không lỗi, clean-kernel nbconvert thành công trên T4; Part 0–4; toàn bộ code trong code/, không data/cache/weights |
| Báo cáo theo template | 7 phần, cơ chế và nhiễu seed, câu hỏi dẫn dắt; render QA 3 trang, đọc được |

## Kết quả và giới hạn

Final eval **accuracy 0.914030, macro-F1 0.869999**; baseline seed 1 F1 0.840992, chênh lệch +0.029007. Final eval seed 1–3 F1 0.868871 ± 0.003280. Baseline eval chỉ một seed; nhiễu val không thay thế kiểm định hai nhóm trên eval.

Lượt mới không dùng điểm eval mới để chọn cấu hình. Tuy nhiên eval của bản cũ đã được xem trước lượt sửa: báo cáo và protocol ghi rõ điều này. Không thể khẳng định lịch sử toàn bài đáp ứng tuyệt đối điều kiện eval chưa từng được xem, hoặc bảo đảm điểm chấm của giảng viên.

FP16 có gradient mean không hữu hạn tại epoch 12, 15, 19 trong khi loss hữu hạn; GradScaler xử lý overflow. JSON ghi null và metadata, đồ thị để khoảng trống. Các chủ đề ngoài baseline/final chỉ một seed, các chênh lệch nhỏ chưa đủ kết luận.

Kiểm tra tự động đã đối chiếu cấu hình kế hoạch/kết quả, chọn epoch/config, Excel/JSON/eval, notebook và cấu trúc bài nộp. PDF render chỉ dùng QA, không thêm vào bài nộp.
