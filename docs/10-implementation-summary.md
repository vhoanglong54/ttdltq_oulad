# 10 — Tổng kết những gì đã triển khai

Ngày cập nhật: 07/10/2026. Trạng thái: **bản nền đã có trên repo `ttdltq_oulad`; thay đổi hiện tại đang ở local, chưa commit**.

## 1. Quyết định cuối

- Giữ nguyên tên đề tài: **Nghiên cứu và phân tích các yếu tố ảnh hưởng đến kết quả học tập của sinh viên đại học**.
- Dữ liệu: OULAD.
- Công nghệ: Python, Pandas, scikit-learn, Streamlit, Plotly.
- Kết quả mô tả: điểm assessment và Distinction/Pass/Fail/Withdrawn.
- Model: Fail so với Pass/Distinction; Withdrawn tách riêng.
- Ngôn ngữ kết luận: “liên quan/đi cùng/khác biệt”, không tuyên bố nhân quả.
- Rubric không bị sửa.

Plan nguồn: [`plan_approved.md`](plan_approved.md).

## 2. Pipeline dữ liệu

Đã tổ chức tách biệt raw → interim → processed → model/dashboard artifacts. `studentVle` được aggregate và bảo toàn click; assessment/VLE được gom về một dòng cho mỗi `module × presentation × student` trước join. Pipeline có guard khóa trùng, unmatched, missing và feature leakage.

Nhãn mới:

- `Academic_Fail=1` khi `final_result=Fail`.
- `Academic_Fail=0` khi `Pass/Distinction`.
- `Withdrawn_Flag=1` khi Withdrawn.
- `At_Risk` chỉ còn là alias tương thích của `Academic_Fail`.

Dashboard mart đã được tái tạo với target Fail. Scatter nộp bài trên Trang 2 chỉ dùng bài nộp `<=105`, phù hợp phạm vi checkpoint.

## 3. EDA và insight

`src/eda_analysis.py` sinh 15 bảng và 6 hình. `insight_evidence.csv` hiện có đủ 8 insight:

| ID | Kết luận chính |
|---|---|
| INS-01 | Tỷ lệ Fail khác nhau theo học vấn đầu vào. |
| INS-02 | Quartile VLE click thấp có Fail cao hơn rõ. |
| INS-03 | Không hoàn thành bài đến hạn là tín hiệu mạnh nhất. |
| INS-04 | VLE thấp + điểm thấp tạo nhóm Fail nổi bật. |
| INS-05 | Lịch sử học lại đi cùng tỷ lệ Fail cao hơn. |
| INS-06 | Có chênh lệch vùng; chỉ dùng như bối cảnh. |
| INS-07 | Độ trễ nộp bài cao đi cùng tỷ lệ Fail cao hơn. |
| INS-08 | Ít ngày hoạt động trong 28 ngày gần nhất đi cùng tỷ lệ Fail rất cao. |

Chi tiết số liệu, `N`, phạm vi và giới hạn nằm ở [`insight-log.md`](insight-log.md).

## 4. Model đã sửa đúng trọng tâm

Model cũ gộp Fail+Withdrawn đã được thay bằng model `Academic_Fail`.

- Audit checkpoint: 30/60/90/105.
- Chọn ngày 105 vì là mốc sớm nhất vượt toàn bộ cổng chất lượng; giao diện gọi **cảnh báo giữa khóa**.
- Group split theo `id_student`.
- Logistic Regression L1, `C=0,3`, threshold `0,335` chọn trên validation với Recall floor 0,75.
- Version: `lr-oulad-academic-fail-c105-s42-v5`.
- Artifact chuẩn: `data/processed/model_academic_fail/c105_final/`.

Kết quả test:

| Metric | Kết quả |
|---|---:|
| Accuracy | 83,68% |
| Accuracy KTC 95% | 82,42%–85,04% |
| Balanced Accuracy | 81,48% |
| Precision Fail | 73,29% |
| Recall Fail | 75,55% |
| F1 Fail | 74,40% |
| ROC-AUC | 0,902 |
| PR-AUC | 0,852 |
| Brier | 0,107 |

11/11 verification checks PASS. Model cao hơn dummy baseline 15,07 điểm phần trăm. Có 246 lượt Fail bị bỏ sót và 277 cảnh báo nhầm; dashboard hiển thị trực tiếp các lỗi này.

Giải thích đầy đủ và cách đối chiếu: [`04-model.md`](04-model.md).

## 5. Dashboard bốn trang

### Trang 1 — Bức tranh kết quả

- 4 KPI rõ ràng.
- Geographic Filled Map theo tỷ lệ Fail, cross-filter vùng.
- 100% stacked bar tổng thể Distinction/Pass/Fail/Withdrawn, drill xuống học vấn đầu vào.
- Violin + box so sánh phân bố điểm quá trình theo kết quả cuối.
- Story tách Fail và Withdrawn, nêu chênh lệch theo học vấn/vùng kèm `N`.

### Trang 2 — Các yếu tố học tập

- VLE line Fail so với Pass/Distinction.
- Completion bar.
- Submission delay scatter + trendline, chỉ dùng dữ liệu đến ngày 105.
- Grouped bar so sánh cơ cấu sử dụng tài nguyên VLE giữa hai nhóm kết quả.
- Story trả lời trực tiếp yếu tố nào liên quan rõ nhất.

### Trang 3 — Kết hợp nhiều yếu tố

- VLE × điểm assessment heatmap.
- Học vấn × IMD heatmap.
- Điểm theo số lần từng học box plot.
- Story nêu nhóm bất lợi cộng dồn và chênh lệch lịch sử học lại.

### Trang 4 — Mô hình và yếu tố dự báo

- Định nghĩa rõ model dự báo Fail, không dự báo điểm.
- KPI động từ dữ liệu đang lọc; caption metric chuẩn từ test artifact.
- Bar + line đối chiếu xác suất dự báo với tỷ lệ `Fail` thật theo 10 nhóm.
- Confusion matrix heatmap cho TP/TN/FP/FN.
- Bar hệ số giải thích tín hiệu toàn mô hình.
- Story đối chiếu profile High/Low và kết luận tín hiệu liên quan đến Fail.
- Không hiển thị danh sách, mã sinh viên, học phần hoặc lớp học; dự báo chỉ được diễn giải ở cấp nhóm.

Tất cả thuật ngữ VLE, IMD, module ẩn danh, checkpoint, Recall và bỏ sót đều có giải thích tiếng Việt trong giao diện.

## 6. Quy tắc trực quan đã áp dụng

- Palette nhất quán; đỏ = Fail/cảnh báo, xanh = Pass/Distinction.
- Biểu đồ có title, trục, đơn vị, hover, legend, caption và cỡ mẫu khi cần.
- Không gộp Fail với Withdrawn.
- Không dùng Geographic Map chỉ để so sánh tùy ý hai vùng; bản đồ cho bức tranh phân bố và hỗ trợ cross-filter.
- Insight ngắn, có số liệu so sánh và đưa về hành động.
- Mọi biểu đồ đều trả lời cấu trúc `yếu tố → kết quả` hoặc `dự báo → kết quả thật`; đã loại module ẩn danh, treemap tổng click, gauge, donut và bảng cá nhân.

## 7. Kiểm thử đã chạy

- `python -m compileall dashboard src tests`: PASS.
- `python -m unittest discover -s tests -v`: **19/19 PASS**, gồm kiểm tra mart và hợp đồng “mỗi visual phải trả lời một câu hỏi kết quả”.
- `streamlit.testing.v1.AppTest`: 4/4 trang không exception sau sửa model/dashboard.
- Validation model: 11/11 PASS.

## 8. Những gì giữ nguyên và những gì thay đổi

Giữ nguyên: OULAD, tên đề tài, dữ liệu raw, pipeline join/aggregate, Logistic Regression, Python/Streamlit/Plotly, Geographic Map và rubric.

Thay đổi: target model, cutoff audit, artifact/version/threshold, EDA target, insight, dashboard wording, risk bands, metric hiển thị, giải thích model và tài liệu.

Model/artifact cũ tại `data/processed/model/` được coi là legacy và không còn là nguồn dashboard. Không xóa vội để tránh mất bằng chứng; nghiệm thu cuối có thể archive sau khi leader duyệt.

## 9. Việc còn lại cho thay đổi hiện tại

1. Leader duyệt nội dung và giao diện Trang 4 sau khi bỏ bảng cá nhân.
2. Chỉ commit/push lên repo hiện hành khi leader cho phép.
