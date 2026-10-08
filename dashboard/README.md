# Dashboard Streamlit

Dashboard gồm bốn trang theo đúng mạch phân tích:

1. **Bức tranh kết quả học tập:** 4 KPI, cơ cấu tổng thể, bản đồ và phân bố điểm theo kết quả.
2. **Các yếu tố học tập:** mức tham gia trực tuyến, hoàn thành bài, thời điểm nộp và cách sử dụng tài nguyên giữa các nhóm kết quả.
3. **Kết hợp nhiều yếu tố:** mức tham gia × điểm, học vấn × hoàn cảnh khu vực và lịch sử học lại.
4. **Mô hình và yếu tố dự báo:** nói rõ dự báo Fail tại ngày 105; minh họa Sigmoid và xác suất từng quan sát; kéo ngưỡng để xem Precision/Recall/F1 và lỗi thay đổi; kết luận ngưỡng rồi kiểm định tổng thể bằng ROC/PR.

Mỗi trang có khối **Story** riêng, trả lời trực tiếp yếu tố nào liên quan đến kết quả. Mọi chart đều nối một yếu tố với điểm, kết quả cuối hoặc chất lượng dự báo; không có chart liệt kê mã ẩn danh hay độ phổ biến đơn thuần. Mô hình dự đoán khả năng `Fail` ở ngày 105; `Withdrawn` tách riêng và model không dự đoán điểm số. Giao diện không công bố danh sách, mã sinh viên hay lớp học.

## Chạy local

```powershell
python src/dashboard_features.py
streamlit run dashboard/app.py
```

App cần:

- `data/processed/clean_dataset.csv`
- `data/processed/dashboard/*`
- `data/processed/model/*`
- `dashboard/assets/oulad_regions.geojson`

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py` | Giao diện bốn trang, state, filter, chart và Story |
| `dashboard_data.py` | Loader, schema guard, filter và KPI |
| `chart-inventory.md` | Inventory visual và mục đích phân tích |
| `wireframe.md` | Bố cục bốn trang |
| `qa-t09.md` | Baseline và cổng QA |
| `assets/` | GeoJSON, mapping audit, nguồn/giấy phép |
| `evidence/` | Automated QA và screenshot browser thật |

Chi tiết logic ở [đặc tả dashboard](../docs/05-dashboard-spec.md), model ở [04-model.md](../docs/04-model.md), tổng kết ở [10-implementation-summary.md](../docs/10-implementation-summary.md).
