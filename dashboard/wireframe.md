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
4. MULTI-LINE — nhịp VLE của Fail so với Pass/Distinction
5. BAR — completion đến ngày 105 × tỷ lệ Fail
6. SCATTER + TRENDLINE — submission delay × score
7. GROUPED BAR — cơ cấu tài nguyên VLE × nhóm kết quả
```

## Trang 3 — Kết hợp nhiều yếu tố

```text
STORY: VLE thấp + điểm thấp; học vấn + IMD; lịch sử học lại
8. HEATMAP — VLE quartile × assessment-score quartile → Fail
9. HEATMAP — education × IMD → Fail
10. BOX PLOT — previous attempts × điểm ngày 105
```

## Trang 4 — Mô hình và yếu tố dự báo

```text
FILTER: giới tính │ tuổi │ học vấn đầu vào │ IMD
KPI: Accuracy │ Recall Fail │ Precision Fail
11. BAR + LINE — decile xác suất dự báo × tỷ lệ Fail thật
12. CONFUSION MATRIX — kết quả thật × kết quả dự báo
13. DIVERGING BAR — tín hiệu global của Logistic Regression
STORY: model phân tầng được nguy cơ không, sai ở đâu, dựa vào yếu tố gì
```

## Quy tắc UX

- Mỗi visual phải có cấu trúc `yếu tố → kết quả` hoặc `dự báo → kết quả thật`.
- Story ngắn, có kết luận định lượng; không lặp mô tả trục.
- Không dùng mã học phần ẩn danh làm insight chính.
- Không dùng biểu đồ chỉ liệt kê mức sử dụng hoặc số lượng.
- Không hiển thị danh sách, mã sinh viên hoặc lớp học.
