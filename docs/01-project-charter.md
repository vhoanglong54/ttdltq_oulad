# 01 — Project charter

## Đề tài

**Nghiên cứu và phân tích các yếu tố ảnh hưởng đến kết quả học tập của sinh viên đại học**.

Trong báo cáo, “ảnh hưởng” được kiểm tra dưới dạng **liên hệ/khác biệt thống kê**. OULAD là dữ liệu quan sát nên cơ chế giải thích chỉ được nêu như giả thuyết hợp lý, không phải quan hệ nhân quả đã chứng minh.

## Phạm vi đo lường

- Đơn vị phân tích: một sinh viên trong một `code_module × code_presentation`.
- Kết quả mô tả: điểm assessment và `Distinction`, `Pass`, `Fail`, `Withdrawn`.
- Nhãn model: `Academic_Fail=1` cho `Fail`; `0` cho `Pass/Distinction`.
- `Withdrawn` được phân tích riêng về duy trì học tập và loại khỏi model học thuật.
- OULAD không có GPA toàn khóa, loại tốt nghiệp, thời gian tự học, điểm danh hoặc sức khỏe tinh thần; không tạo biến giả để lấp khoảng trống.

## Câu hỏi nghiên cứu

1. Kết quả học tập khác nhau thế nào theo module, presentation và vùng?
2. Tiến độ hoàn thành assessment, điểm sớm và thời điểm nộp liên quan thế nào đến `Fail`?
3. Nhóm `Fail` và `Pass/Distinction` có nhịp hoạt động VLE khác nhau thế nào?
4. Lịch sử học lại, học vấn đầu vào và hoàn cảnh khu vực liên quan ra sao?
5. Khi nhiều tín hiệu bất lợi cùng xuất hiện, nhóm nào có tỷ lệ `Fail` cao nhất?
6. Logistic Regression có thể cảnh báo `Fail` với Accuracy trên 80% và Recall đủ dùng hay không?

## Công nghệ và sản phẩm

| Phần | Lựa chọn |
|---|---|
| Dữ liệu | 7 bảng OULAD, pipeline Python tái tạo được |
| Phân tích | Pandas, NumPy, Matplotlib/Seaborn |
| Model | scikit-learn Logistic Regression |
| Trực quan | Streamlit + Plotly, bốn trang tương tác |
| Bằng chứng | CSV, hình EDA, metrics, bootstrap CI, verification và kiểm thử |

## Tiêu chí hoàn thành

- Không sửa rubric.
- Geographic Map hoạt động và có kiểm tra mapping.
- Insight có kết luận, số liệu, cỡ mẫu, phạm vi lọc và giới hạn.
- Model không leakage; split theo sinh viên; test chỉ dùng một lần cho đánh giá cuối.
- Accuracy và cận dưới KTC 95% ≥ 80%; các cổng Recall, Balanced Accuracy, F1, ROC-AUC, PR-AUC và Brier đều đạt.
- Dashboard nói rõ model dự báo `Fail`, tách `Withdrawn`, đưa ra nhóm ưu tiên và hành động hỗ trợ.
