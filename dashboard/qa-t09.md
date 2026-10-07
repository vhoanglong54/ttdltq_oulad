# QA dashboard — target Academic Fail v5

Ngày kiểm tra: 07/10/2026. Trạng thái local, chưa commit.

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
| High trên test | 1.037 |

## Kiểm tra tự động

- `python -m unittest discover -s tests -v`: **17/17 PASS**.
- `at_risk_model.py validate`: **11/11 PASS**.
- Streamlit AppTest: **4/4 trang, 0 exception**.
- Số Plotly chart theo trang: **3 / 4 / 3 / 3**.
- Browser QA Chrome headless: **4/4 trang, 0 exception**, đúng số chart.

## Kiểm tra nghiệp vụ

- Fail và Withdrawn tách riêng: PASS.
- Day 105 gọi “cảnh báo giữa khóa”: PASS.
- Trang 4 nói rõ dự báo Fail, không dự báo điểm/GPA: PASS.
- Threshold và metric đọc từ artifact: PASS.
- Geographic Map dùng tỷ lệ Fail và cross-filter: PASS.
- Submission scatter giới hạn `date_submitted <=105`: PASS.
- 8 insight có evidence sinh tự động: PASS.
- Rubric không bị chỉnh sửa: PASS.

## Ảnh bằng chứng hiện tại

- [Trang 1](evidence/screenshots/overview-page-v5.png)
- [Trang 2](evidence/screenshots/behavior-page-v5.png)
- [Trang 3](evidence/screenshots/interaction-page-v5.png)
- [Trang 4](evidence/screenshots/prediction-page-v5.png)

Ảnh v2/v3/v4 là mốc lịch sử, không dùng để nghiệm thu target hiện tại.
