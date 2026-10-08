# QA dashboard — kiểm định Logistic Regression v9

Ngày kiểm tra: 08/10/2026. Trạng thái local, chưa commit.

## Baseline

| Kiểm tra | Kết quả |
|---|---:|
| Lượt học toàn bảng mô tả | 32.593 |
| Sinh viên duy nhất | 28.785 |
| Tỷ lệ Pass/Distinction | 47,2% |
| Tỷ lệ Fail | 21,6% |
| Tỷ lệ Withdrawn | 31,2% |
| Cohort model ngày 105 | 22.427 |
| Test rows | 3.205 |
| Test Accuracy | 83,68% |
| Test Recall Fail | 75,55% |
| Threshold | 0,335 |

## Kiểm tra tự động

- `python -m unittest discover -s tests -v`: **20/20 PASS**.
- `at_risk_model.py validate`: **11/11 PASS**.
- Streamlit AppTest: **4/4 trang, 0 exception**.
- Số Plotly chart theo trang: **3 / 5 / 3 / 5**.
- Năm biểu đồ kiểm định Logistic Regression hiển thị trực tiếp, không đặt trong expander: PASS.
- Tương tác threshold Trang 4: đổi `33,5% → 50,0%` làm Recall/Precision/F1 đổi từ `75,5% / 73,3% / 74,4%` sang `63,6% / 83,4% / 72,2%`; nút reset đưa về `33,5%`, 0 exception.
- Browser QA mới cho v9: **chưa chạy**; ảnh v7 chỉ là baseline lịch sử.

## Kiểm tra nghiệp vụ

- Fail và Withdrawn tách riêng: PASS.
- Day 105 gọi “cảnh báo giữa khóa”: PASS.
- Trang 4 nói rõ dự báo Fail, không dự báo điểm/GPA: PASS.
- Khối mở đầu nói ngắn gọn dự báo Fail tại ngày 105, cách đọc ngưỡng và chuỗi thao tác; không liệt kê công thức dài: PASS.
- Biểu đồ 12 có đường Sigmoid, threshold và xác suất từng quan sát theo lớp thực tế: PASS.
- Sau confusion matrix có kết luận quyết định; KPI chuẩn và ROC/PR nằm cuối trang: PASS.
- Đã loại biểu đồ so sánh A/B vì không phải đầu ra cốt lõi của Logistic Regression: PASS.
- ROC/PR đọc artifact test; confusion matrix cập nhật theo threshold đang thử: PASS.
- Threshold và metric đọc từ artifact: PASS.
- Geographic Map dùng tỷ lệ Fail và cross-filter: PASS.
- Submission scatter giới hạn `date_submitted <=105`: PASS.
- 8 insight có evidence sinh tự động: PASS.
- Mỗi chart nối yếu tố với kết quả hoặc dự báo với kết quả thật: PASS.
- Rubric không bị chỉnh sửa: PASS.

## Ảnh baseline trước thay đổi v9

- [Trang 1](evidence/screenshots/overview-page-v7.png)
- [Trang 2](evidence/screenshots/behavior-page-v7.png)
- [Trang 3](evidence/screenshots/interaction-page-v7.png)
- [Trang 4 — model và yếu tố dự báo](evidence/screenshots/prediction-page-v7.png)

Ảnh v2–v7 là mốc lịch sử. Trang 4 v9 cần chụp lại sau khi leader duyệt bố cục kiểm định rút gọn.
