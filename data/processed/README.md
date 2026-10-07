# Hợp đồng bảng processed — T07

`clean_dataset.csv` là **một tập processed chuẩn duy nhất** được build tái lập từ raw; không sửa thủ công bằng spreadsheet hoặc trong dashboard. File được Git theo dõi làm nguồn dùng chung giữa các bước và phải tái tạo được bằng các lệnh:

```powershell
python src/oulad_pipeline.py audit data/raw
python src/oulad_pipeline.py clean data/raw
python src/oulad_pipeline.py build data/raw
python src/dashboard_features.py
```

## Hạt, kích thước và khóa

- Hạt: một lượt học `(code_module, code_presentation, id_student)`.
- Kích thước lần chạy hiệu chỉnh 06/10/2026: 32.593 dòng, 35 cột, 0 duplicate attempt key; `clean_dataset.csv` SHA-256 `70db1ff19a1dc8553005b0e3801786d20ab1488cc3420a2fe15c5a2254d80f9a`.
- Base là `studentInfo`; `studentRegistration` và `courses` left join one-to-one/many-to-one. `studentAssessment` và `studentVle` được aggregate trước left join nên không nhân dòng.
- `At_Risk`: `Fail`/`Withdrawn` = 1, `Pass`/`Distinction` = 0. Đây là nhãn, không phải feature model.

## Schema

| Nhóm | Cột |
|---|---|
| Khóa và raw student | `code_module`, `code_presentation`, `id_student`, `gender`, `region`, `highest_education`, `imd_band`, `imd_band_display`, `age_band`, `num_of_prev_attempts`, `studied_credits`, `disability`, `final_result` |
| Registration/course | `date_registration`, `date_unregistration`, `module_presentation_length` |
| Assessment aggregate | `assessment_event_count`, `assessment_scored_count`, `assessment_score_missing_count`, `assessment_score_sum_all_time`, `assessment_score_mean_all_time`, `assessment_score_min_all_time`, `assessment_score_max_all_time`, `assessment_banked_count`, `assessment_late_submission_count_all_time`, `assessment_type_nunique` |
| VLE aggregate | `vle_event_count`, `vle_total_clicks_all_time`, `vle_active_days_all_time`, `vle_resource_count_all_time`, `vle_activity_type_count_all_time`, `vle_first_event_day`, `vle_last_event_day` |
| Calculated result | `At_Risk`, `Performance_Level` |

## Handoff guardrails

- Tỷ lệ/KPI dùng mẫu số **lượt học**, không suy ra số sinh viên unique nếu chưa deduplicate theo `id_student` theo định nghĩa riêng.
- `*_all_time` chỉ là aggregate mô tả cho EDA/dashboard và không dùng làm feature model. Model v5 dùng cutoff ngày 105, threshold 0,335 và dựng snapshot riêng từ các bảng interim; giá trị cuối phải khớp artifact `model_academic_fail/c105_final`.
- Với Average Assessment Score theo filter, dùng `SUM(assessment_score_sum_all_time) / SUM(assessment_scored_count)` khi mẫu số lớn hơn 0; không dùng trung bình trực tiếp của `assessment_score_mean_all_time` vì sẽ sai trọng số.
- Tuyệt đối loại khỏi model feature: `final_result`, `At_Risk`, `date_unregistration` và bất cứ assessment/VLE nào sau cutoff đã chốt.
- `imd_band` được chuẩn hóa `10-20` → `10-20%`; raw không thay đổi. `imd_band` missing vẫn nullable, còn `imd_band_display` dùng `Unknown` cho dashboard; không tự diễn giải là thu nhập cá nhân.
- Count/tổng assessment và VLE được điền `0` khi lượt học không có event; score mean/min/max và ngày event đầu/cuối vẫn nullable. `has_registration_record` bị loại vì zero variance.
- `studentVle` raw có 10.655.280 dòng. T06 gom về 8.459.320 khóa student–resource–day và cộng `sum_click`; tổng click được bảo toàn ở 39.605.099. `vle_event_count` vì vậy đếm dòng logic theo ngày/tài nguyên, không đếm số dòng raw.
- Trong dashboard Python: categorical fields phải giữ kiểu phân loại/string; `date_registration`, `date_unregistration`, `vle_first_event_day`, `vle_last_event_day` là số ngày tương đối, không phải calendar date. VLE click là proxy tương tác, không phải attendance/study hours.
- Thư mục `dashboard/` chứa năm data mart phục vụ timeline, scatter và so sánh tài nguyên theo kết quả. Hai mart VLE toàn kỳ phải cùng bảo toàn tổng 39.605.099 clicks; `vle_activity_summary_day105.csv.gz` là mart riêng chỉ dùng dữ liệu đến checkpoint; `assessment_submissions.csv.gz` phải giữ công thức `submission_delay = date_submitted - due_date`.
- Chi tiết số dòng, missing, duplicate, join và test: [Data Quality Report](../../reports/data-quality-report.md).
