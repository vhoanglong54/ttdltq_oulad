# Bằng chứng dashboard

`screenshots/` lưu ảnh QA trực quan qua các phiên bản. Tên có hậu tố `v2`–`v7` là mốc lịch sử, không phải nguồn xác nhận hành vi hiện tại của ứng dụng.

Nguồn kiểm tra hiện hành:

- `dashboard/app.py` và `dashboard/dashboard_data.py`: hành vi giao diện.
- `tests/test_dashboard_contract.py`: schema, KPI, filter và artifact contract.
- `tests/test_at_risk_model.py`: target, leakage, split và model contract.
- `data/processed/model_academic_fail/c105_final/model_verification.csv`: cổng nghiệm thu model.

Trước khi nộp, chạy lại test và chụp đủ bốn trang ở cùng một phiên bản; không dùng ảnh lịch sử để chứng minh giao diện mới.
