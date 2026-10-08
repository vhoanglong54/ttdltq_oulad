# Wireframe dashboard bốn trang

## Trang 1 — Bức tranh kết quả

```text
FILTER: giới tính │ tuổi │ học vấn đầu vào │ IMD
KPI: sinh viên │ điểm TB │ Pass/Distinction │ Fail
STORY: cơ cấu kết quả tổng thể, chênh lệch theo học vấn và vùng
1. 100% STACKED BAR — tổng thể; drill có chủ đích xuống học vấn đầu vào
2. FILLED MAP — vùng cư trú × tỷ lệ Fail, click vùng để cross-filter
3. VIOLIN + BOX — điểm quá trình × kết quả cuối
```

## Trang 2 — Các yếu tố học tập

```text
STORY: completion và tính liên tục của VLE là hai tín hiệu rõ nhất
4. BAR AUC — mức phân biệt kết quả của từng yếu tố đơn lẻ
5. MULTI-LINE — nhịp VLE của Fail so với Pass/Distinction
6. BAR — completion đến ngày 105 × tỷ lệ Fail
7. BAR CHUẨN HÓA — submission delay × chênh lệch điểm
8. DOT PLOT TRỤC LOG — mức dùng tài nguyên VLE giữa hai nhóm kết quả
```

## Trang 3 — Kết hợp nhiều yếu tố

```text
STORY: VLE thấp + điểm thấp; học vấn + IMD; lịch sử học lại
9. HEATMAP — VLE quartile × assessment-score quartile → Fail
10. HEATMAP — education × IMD → Fail
11. BOX PLOT — previous attempts × điểm ngày 105
```

## Trang 4 — Mô hình và yếu tố dự báo

```text
FILTER: giới tính │ tuổi │ học vấn đầu vào │ IMD
KPI: Accuracy │ Recall Fail │ Precision Fail
MODEL INPUT: nền tảng │ tiến độ assessment │ hoạt động VLE đến ngày 105
MODEL SUMMARY: dự báo Fail tại ngày 105 │ cách đọc p(Fail) │ chuỗi thao tác
12. SIGMOID + PROBABILITY STRIP — đường xác suất và từng quan sát theo lớp thật
13. DIVERGING BAR — tín hiệu global của Logistic Regression
14. THRESHOLD LINE — ngưỡng × Recall/Precision/F1; nút trở về 33,5%
15. CONFUSION MATRIX — kết quả thật × kết quả dự báo tại ngưỡng đang chọn
DECISION: phát hiện đúng, bỏ sót, cảnh báo nhầm và khuyến nghị ngưỡng
16. ROC + PRECISION–RECALL — kiểm định tổng thể trên mọi ngưỡng
STORY: dự báo gì → chọn dữ liệu → xem xác suất → thử threshold → kiểm tra lỗi → ra quyết định → kiểm định tổng thể
```

## Quy tắc UX

- Mỗi visual phải có cấu trúc `yếu tố → kết quả` hoặc `dự báo → kết quả thật`.
- Story ngắn, có kết luận định lượng; không lặp mô tả trục.
- Không dùng mã học phần ẩn danh làm insight chính.
- Không dùng biểu đồ chỉ liệt kê mức sử dụng hoặc số lượng.
- Không hiển thị danh sách, mã sinh viên hoặc lớp học.
