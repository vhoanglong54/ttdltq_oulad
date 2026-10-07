# Data Quality Report — OULAD (T05–T07)

**Ngày chạy pipeline:** 07/10/2026.
**Task:** T05–T07.
**Hạt đầu ra:** một lượt học `(code_module, code_presentation, id_student)`.  
**Công nghệ:** Python, pandas 2.3.3, NumPy 2.3.5.  
**Input:** bảy CSV raw cục bộ tại `data/raw/`, đã xác minh T01/T02; raw không bị sửa.

## T05 — Audit raw

Lệnh tái tạo: `python src/oulad_pipeline.py audit data/raw`.

| Bảng | Dòng | Cột | Duplicate toàn dòng | Key / duplicate key | Blank / `?` |
|---|---:|---:|---:|---|---|
| `courses.csv` | 22 | 3 | 0 | code_module, code_presentation; duplicate 0 | 0 |
| `studentInfo.csv` | 32,593 | 12 | 0 | code_module, code_presentation, id_student; duplicate 0 | 1,111 |
| `studentRegistration.csv` | 32,593 | 5 | 0 | code_module, code_presentation, id_student; duplicate 0 | 22,566 |
| `assessments.csv` | 206 | 6 | 0 | id_assessment; duplicate 0 | 11 |
| `studentAssessment.csv` | 173,912 | 5 | 0 | id_assessment, id_student; duplicate 0 | 173 |
| `vle.csv` | 6,364 | 6 | 0 | id_site; duplicate 0 | 10,486 |
| `studentVle.csv` | 10,655,280 | 6 | 787,154 | code_module, code_presentation, id_student, id_site, date; duplicate 2,195,960 | 0 |

### Schema quan sát

| Bảng | Cột theo thứ tự raw | Dtype đọc bởi pandas |
|---|---|---|
| `courses.csv` | `code_module`, `code_presentation`, `module_presentation_length` | `code_module`: object, `code_presentation`: object, `module_presentation_length`: int64 |
| `studentInfo.csv` | `code_module`, `code_presentation`, `id_student`, `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `num_of_prev_attempts`, `studied_credits`, `disability`, `final_result` | `code_module`: object, `code_presentation`: object, `id_student`: int64, `gender`: object, `region`: object, `highest_education`: object, `imd_band`: object, `age_band`: object, `num_of_prev_attempts`: int64, `studied_credits`: int64, `disability`: object, `final_result`: object |
| `studentRegistration.csv` | `code_module`, `code_presentation`, `id_student`, `date_registration`, `date_unregistration` | `code_module`: object, `code_presentation`: object, `id_student`: int64, `date_registration`: object, `date_unregistration`: object |
| `assessments.csv` | `code_module`, `code_presentation`, `id_assessment`, `assessment_type`, `date`, `weight` | `code_module`: object, `code_presentation`: object, `id_assessment`: int64, `assessment_type`: object, `date`: object, `weight`: float64 |
| `studentAssessment.csv` | `id_assessment`, `id_student`, `date_submitted`, `is_banked`, `score` | `id_assessment`: int64, `id_student`: int64, `date_submitted`: int64, `is_banked`: int64, `score`: object |
| `vle.csv` | `id_site`, `code_module`, `code_presentation`, `activity_type`, `week_from`, `week_to` | `id_site`: int64, `code_module`: object, `code_presentation`: object, `activity_type`: object, `week_from`: object, `week_to`: object |
| `studentVle.csv` | `code_module`, `code_presentation`, `id_student`, `id_site`, `date`, `sum_click` | `code_module`: string, `code_presentation`: string, `id_student`: int64, `id_site`: int64, `date`: int64, `sum_click`: int64 |

### Missing, invalid và outlier

`?` được phát hiện trên raw: `imd_band` 1.111; `date_registration` 45; `date_unregistration` 22.521; `assessments.date` 11; `studentAssessment.score` 173; `vle.week_from`/`week_to` mỗi cột 5.243. Đây là mã missing, không phải blank.

| Bảng / kiểm tra | Kết quả thực | Diễn giải T05 |
|---|---:|---|
| `courses.csv.module_presentation_length` | range 234.0–269.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentInfo.csv.id_student` | range 3733.0–2716795.0; IQR diagnostic 6,460 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentInfo.csv.num_of_prev_attempts` | range 0.0–6.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentInfo.csv.studied_credits` | range 30.0–655.0; IQR diagnostic 350 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentInfo.csv.final_result` invalid category | 0 | So với domain OULAD đã ghi. |
| `studentInfo.csv.gender` invalid category | 0 | So với domain OULAD đã ghi. |
| `studentInfo.csv.disability` invalid category | 0 | So với domain OULAD đã ghi. |
| `studentRegistration.csv.id_student` | range 3733.0–2716795.0; IQR diagnostic 6,460 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentRegistration.csv.date_registration` | range -322.0–167.0; IQR diagnostic 339 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentRegistration.csv.date_unregistration` | range -365.0–444.0; IQR diagnostic 41 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `assessments.csv.id_assessment` | range 1752.0–40088.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `assessments.csv.date` | range 12.0–261.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `assessments.csv.weight` | range 0.0–100.0; IQR diagnostic 24 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `assessments.csv.assessment_type` invalid category | 0 | So với domain OULAD đã ghi. |
| `studentAssessment.csv.id_assessment` | range 1752.0–37443.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentAssessment.csv.id_student` | range 6516.0–2698588.0; IQR diagnostic 35,636 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentAssessment.csv.date_submitted` | range -11.0–608.0; IQR diagnostic 59 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentAssessment.csv.is_banked` | range 0.0–1.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentAssessment.csv.score` | range 0.0–100.0; IQR diagnostic 3,813 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentAssessment.csv.is_banked` invalid category | 0 | So với domain OULAD đã ghi. |
| `vle.csv.id_site` | range 526721.0–1077905.0; IQR diagnostic 17 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `vle.csv.week_from` | range 0.0–29.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `vle.csv.week_to` | range 0.0–29.0; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentVle.csv.id_student` | range 6,516–2,698,588; IQR diagnostic — | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentVle.csv.id_site` | range 526,721–1,049,562; IQR diagnostic — | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentVle.csv.date` | range -25–269; IQR diagnostic 0 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |
| `studentVle.csv.sum_click` | range 1–6,977; IQR diagnostic 1,314,646 | Cờ IQR chỉ để xem xét; T06 không xóa máy móc. |

### Khóa, cardinality và kiểm tra nghiệp vụ

| Kiểm tra | Kết quả |
|---|---:|
| `registration_outside_studentInfo` | 0 |
| `studentInfo_without_registration` | 0 |
| `assessments_without_courses` | 0 |
| `studentAssessment_missing_assessment` | 0 |
| `studentAssessment_inferred_attempt_outside_studentInfo` | 0 |
| `studentVle_studentInfo_unmatched` | 0 |
| `studentVle_vle_unmatched` | 0 |
| `late_submissions` | 49,318 |
| `banked_distribution` | 0=172003, 1=1909 |

`date_unregistration` missing không đồng nghĩa chắc chắn với không rút: trong raw có 93 lượt `Withdrawn` vẫn missing, trong khi 10.063 lượt `Withdrawn` có ngày rút. Do đó T06 giữ missing và mọi bước model phải cấm cột này.

`studentVle` là bảng đóng góp click. Nhiều dòng cùng learner–resource–day được gom theo khóa `(code_module, code_presentation, id_student, id_site, date)` và cộng `sum_click`; không xóa chỉ vì toàn dòng giống nhau. T06 bắt buộc bảo toàn tổng click trước/sau chuẩn hóa.

## T06 — Cleaning

Lệnh tái tạo: `python src/oulad_pipeline.py clean data/raw`.

- Raw CSV are read only; '?' is normalized to nullable missing in interim outputs, never imputed.
- String fields are trimmed; numeric fields are cast to nullable Int64 or Float64 according to observed values.
- Exact full-row duplicates are dropped only in non-event tables; outliers are retained for later interpretation.
- studentVle rows are consolidated by code_module, code_presentation, id_student, id_site and date; sum_click is summed and its total must remain unchanged.
- date_unregistration missing is retained as unknown/not-recorded and is prohibited from prediction features; it is not assumed equivalent to not Withdrawn.
- studentAssessment.score missing and assessment/vle unknown dates/weeks remain missing; no score/date imputation is performed.

| Bảng | Dòng trước | Dòng sau | Exact duplicate quan sát | Dòng gom theo event key | Duplicate bị xóa | Missing sau |
|---|---:|---:|---:|---:|---:|---:|
| `courses.csv` | 22 | 22 | 0 | 0 | 0 | 0 |
| `studentInfo.csv` | 32,593 | 32,593 | 0 | 0 | 0 | 1,111 |
| `studentRegistration.csv` | 32,593 | 32,593 | 0 | 0 | 0 | 22,566 |
| `assessments.csv` | 206 | 206 | 0 | 0 | 0 | 11 |
| `studentAssessment.csv` | 173,912 | 173,912 | 0 | 0 | 0 | 173 |
| `vle.csv` | 6,364 | 6,364 | 0 | 0 | 0 | 10,486 |
| `studentVle.csv` | 10,655,280 | 8,459,320 | 787,170 | 2,195,960 | 0 | 0 |

`studentVle.sum_click` được bảo toàn: 39,605,099 trước và 39,605,099 sau khi gom event key.

Quyết định sử dụng: `imd_band` giữ missing nullable (không thay bằng median); EDA/dashboard có thể hiển thị category `Unknown` nhưng phải ghi rõ mẫu số. Cleaning không đặt ngưỡng model; model v5 dùng cutoff ngày 105 và threshold 0,335 từ artifact đã kiểm định.

## T07 — Join, aggregate và calculated fields

Lệnh tái tạo: `python src/oulad_pipeline.py build data/raw`.

| Chỉ số | Kết quả |
|---|---:|
| `studentInfo` | 32,593 |
| `studentAssessment_events` | 173,912 |
| `assessment_aggregated_attempts` | 25,843 |
| `studentVle_events` | 8,459,320 |
| `vle_aggregated_attempts` | 29,228 |
| `joined_output` | 32,593 |
| unmatched `assessment_dimension` | 0 |
| unmatched `vle_dimension` | 0 |
| unmatched `registration` | 0 |
| unmatched `courses` | 0 |
| `imd_band` `10-20` normalized to `10-20%` | 3,516 |
| `imd_band_display = Unknown` | 1,111 |
| nulls after selected aggregate zero-fill | 0 |
| excluded zero-variance QA field | `has_registration_record` |
| duplicate attempt key sau join | 0 |
| `final_result` | Distinction=3,024, Fail=7,052, Pass=12,361, Withdrawn=10,156 |
| `At_Risk` | 0=25,541, 1=7,052 |

`Academic_Fail = 1` chỉ cho `Fail`; `0` cho `Pass`/`Distinction`. `Withdrawn_Flag` được mô tả riêng và `At_Risk` là alias tương thích của `Academic_Fail`. Các aggregate `*_all_time` chỉ dành cho mô tả; model phải dựng snapshot cutoff riêng. Không tạo attendance, study hours, sleep hoặc previous grade giả.

## Đánh giá điều kiện nghiệm thu T05–T07

| Điều kiện | Trạng thái | Bằng chứng / giới hạn |
|---|---|---|
| Pipeline tái tạo từ 7 CSV | Đạt về chạy cục bộ | Script và các lệnh trên; `clean_dataset.csv` được theo dõi và bản hiệu chỉnh chưa commit trong đợt tái cấu trúc. |
| Missing/outlier/duplicate có quyết định | Đạt về pipeline cục bộ | Báo cáo T05/T06; `studentVle` được gom theo learner–resource–day và bảo toàn tổng `sum_click`; bảng khác chỉ loại exact duplicate khi có. |
| Join không nhân dòng | Đạt theo test T07 | Output cùng số dòng `studentInfo`, duplicate attempt key 0; event được aggregate trước join. |
| Dùng được cho EDA/dashboard Python và làm nền model | Đạt local có giới hạn | Bảng processed dành cho mô tả; model dùng bảng interim và feature theo cutoff, không dùng aggregate `*_all_time`. |
| Hiệu chỉnh event VLE | Đã kiểm tra local | Logic bảo toàn click, report và output local đã tái tạo; chưa commit/push. |

## Cách sử dụng và giới hạn

- EDA dùng `clean_dataset.csv` tái tạo cục bộ cùng Data Quality Report; các tỷ lệ dùng mẫu số là lượt học, không phải sinh viên unique.
- Model/dashboard dùng target `Academic_Fail`, guard leakage và split theo `id_student`. Model v5 dùng cutoff ngày 105, threshold 0,335; phải đối chiếu artifact trước khi công bố.
- Không có thao tác dashboard trong T05–T07. Không có insight hay kết quả model được công bố ở đây.
