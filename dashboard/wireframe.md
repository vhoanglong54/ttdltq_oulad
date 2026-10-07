# Wireframe dashboard bốn trang

## Trang 1 — Bức tranh kết quả

```text
FILTER: module │ presentation │ giới tính
KPI: sinh viên │ điểm TB │ Pass/Distinction │ Fail
STORY: Fail và Withdrawn tách riêng; khác biệt module/vùng có N
1. FILLED MAP — tỷ lệ Fail, click region để cross-filter
2. 100% STACKED BAR — Distinction/Pass/Fail/Withdrawn, drill module → presentation
3. HISTOGRAM — phân bố điểm quá trình theo kết quả cuối
```

## Trang 2 — Các yếu tố học tập

```text
STORY: completion và VLE là hai tín hiệu rõ nhất
4. MULTI-LINE — Fail so với Pass/Distinction theo thời gian
5. BAR — completion đến ngày 105 × Fail
6. SCATTER + TRENDLINE — submission delay × score, chỉ bài nộp <=105
7. TREEMAP — activity_type × clicks
```

## Trang 3 — Kết hợp nhiều yếu tố

```text
STORY: thấp–thấp/cao–cao + lịch sử học lại
8. HEATMAP — VLE quartile × assessment-score quartile
9. HEATMAP — education × IMD
10. BOX PLOT — score ngày 105 × previous attempts
```

## Trang 4 — Cảnh báo giữa khóa

```text
FILTER: Low/Medium/High │ IMD
KPI: Accuracy │ Recall Fail │ High count
11. GAUGE — xác suất Fail, threshold đọc từ artifact
12. DONUT — TP/TN/FP/FN
13. BAR — tín hiệu global của Logistic Regression
ACTION LIST — probability Fail + dự báo + mức cảnh báo
STORY: dự báo cái gì, đúng/bỏ sót bao nhiêu, nhóm High có gì khác, hỗ trợ ra sao
```

## Quy tắc UX

- Giải thích VLE, IMD, AAA–GGG, B/J và metric trước khi dùng.
- Story ngắn, có số liệu và hành động; không lặp mô tả trục.
- Drill chỉ một cấp; không dùng sunburst dày đặc.
- Action List để ưu tiên hỗ trợ, không phải quyết định tự động.
