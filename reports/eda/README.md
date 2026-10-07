# EDA evidence package

Đây là bộ bằng chứng tái tạo được cho RQ1–RQ5, H01–H10 và sáu insight phân tích. Bộ này **không chứa kết luận đánh giá model**.

## Chạy lại

Từ thư mục gốc repository:

```powershell
python src/eda_analysis.py
```

Có thể chạy cùng luồng qua `notebooks/03_eda.ipynb`. Script dừng nếu thiếu input hoặc khóa một lượt học bị trùng.

## Hai phạm vi phân tích

| Phạm vi | Nguồn | N | Dùng cho |
|---|---|---:|---|
| Toàn khóa | `data/processed/clean_dataset.csv` | 32.593 attempts | Phân bố kết quả, module/presentation, region |
| Cảnh báo sớm ngày 105 | `data/processed/model/feature_snapshot.csv` | 25.132 eligible attempts | VLE, assessment, prior attempts, tương tác đa yếu tố |

Không trộn hai mẫu số. Dữ liệu toàn khóa mô tả kết quả; snapshot ngày 105 chỉ giữ lượt học đủ điều kiện để tạo feature tại cutoff. Một dòng là một learning attempt `(code_module, code_presentation, id_student)`, không phải một người duy nhất.

## Phương pháp

- `At_Risk` trong các bảng hiện tại là alias của `Academic_Fail`: `Fail=1`, `Pass/Distinction=0`; `Withdrawn` được loại khỏi snapshot model và mô tả riêng ở bảng toàn khóa.
- Các rate đi kèm `at_risk_count`, `attempts` và khoảng tin cậy Wilson 95%.
- Quartile dùng `rank(method="first")` rồi `qcut` để tạo bốn nhóm có kích thước gần bằng nhau.
- Điểm assessment bị thiếu được giữ thành nhóm `No scored assessment by cutoff`, không bị gán vào quartile.
- Xu hướng VLE dùng 15 tuần trọn vẹn, ngày 0–104. Mẫu số mỗi tuần gồm toàn bộ eligible attempts của từng outcome, kể cả attempt có 0 click trong tuần.
- Week chứa riêng ngày 105 không được vẽ để tránh tạo sụt giảm giả do tuần chưa đủ bảy ngày.
- Tất cả so sánh là mô tả liên hệ; không suy diễn nguyên nhân.

## Output

| File | Ý nghĩa |
|---|---|
| `insight_evidence.csv` | Sáu so sánh chính, N, rate, chênh lệch, risk ratio, phạm vi và giới hạn |
| `hypothesis_results.csv` | Kết quả H01–H10 theo ID trong `docs/plan_approved.md` |
| `outcome_by_module_presentation.csv` | Bốn kết quả theo module/presentation |
| `module_presentation_risk.csv` | At-Risk rate và CI theo module/presentation |
| `vle_weekly_trend.csv` | Click trung bình và tỷ lệ active theo tuần |
| `engagement_quartiles.csv` | At-Risk rate theo quartile VLE clicks |
| `active_days_quartiles.csv` | At-Risk rate theo quartile active days |
| `assessment_completion.csv` | At-Risk rate theo mức hoàn thành assessment |
| `assessment_score_quartiles.csv` | At-Risk rate theo quartile điểm và nhóm chưa có điểm |
| `engagement_assessment_matrix.csv` | Ma trận tương tác VLE × assessment |
| `previous_attempts.csv` | So sánh 0 và 1+ lần học trước |
| `education_risk.csv` | At-Risk rate theo education |
| `region_risk.csv` | At-Risk rate theo 13 region OULAD |

Sáu hình tĩnh nằm tại `reports/figures/eda/`. Diễn giải nghiệm thu nằm tại `docs/insight-log.md`.

## Cổng kiểm tra

1. Tổng `attempts` của bảng nhóm phải khớp N của phạm vi tương ứng.
2. `at_risk_count / attempts = at_risk_rate` trong sai số số thực.
3. CI nằm trong `[0, 1]` và bao quanh rate.
4. H01–H10 phải có đúng một dòng mỗi giả thuyết.
5. Insight phải có evidence table/figure, N và giới hạn; metric model không được đưa vào đây.
6. Geographic Map chưa được coi là hoàn thành chỉ vì có `region_risk.csv`; geometry vẫn phải qua cổng nguồn, giấy phép và mapping.
