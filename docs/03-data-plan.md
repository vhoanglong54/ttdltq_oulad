# 03 — Kế hoạch và hợp đồng dữ liệu OULAD

## Nguồn và phạm vi

OULAD gồm bảy bảng gốc: `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, `studentAssessment`, `studentVle`. Raw giữ nguyên tại `data/raw/`; không chỉnh tay và không commit dữ liệu raw.

Khóa chính của một lượt học:

```text
code_module + code_presentation + id_student
```

Các event assessment và VLE phải aggregate về đúng khóa này trước khi nối với `studentInfo`. Pipeline dừng nếu output có khóa trùng.

## Kết quả và nhãn

| Trường | Ý nghĩa | Dùng trong model? |
|---|---|---|
| `final_result` | Distinction/Pass/Fail/Withdrawn | Chỉ tạo nhãn/đánh giá |
| `Academic_Fail` | 1 nếu Fail; 0 nếu Pass/Distinction | Target, không feature |
| `Withdrawn_Flag` | 1 nếu Withdrawn | Mô tả riêng, loại khỏi model |
| `At_Risk` | Alias tương thích của `Academic_Fail` | Không feature |
| `assessment_score_mean_all_time` | Điểm quá trình trung bình toàn khóa | Chỉ EDA mô tả |

## Nhóm yếu tố thực sự đo được

- Assessment: số bài đến hạn, số bài nộp, tỷ lệ hoàn thành, điểm sớm, nộp trễ, điểm có trọng số.
- VLE: tổng click, ngày hoạt động, độ gần của lần hoạt động cuối, click 7/28 ngày, số loại/tài nguyên sử dụng.
- Lịch sử/đăng ký: số lần từng học, số tín chỉ, ngày đăng ký.
- Bối cảnh: module, presentation, vùng, IMD, học vấn đầu vào, tuổi, giới tính, khuyết tật.

Không suy diễn những biến OULAD không đo: thời gian tự học, điểm danh, động lực, sức khỏe tinh thần, việc làm thêm, GPA hoặc loại tốt nghiệp.

## Hai bảng phục vụ hai mục đích

1. `data/processed/clean_dataset.csv`: một dòng/lượt học, dùng EDA toàn khóa và dashboard mô tả. Các cột `*_all_time` không được dùng cho model checkpoint.
2. `data/processed/model_academic_fail/c105_final/feature_snapshot.csv`: snapshot chỉ dùng event `<=105`, dùng model và phân tích hành vi tại checkpoint.

Không nối trực tiếp `studentVle` hàng triệu dòng khi render dashboard.

## Cohort model

- Chỉ giữ `Fail`, `Pass`, `Distinction`.
- Loại `Withdrawn` để không biến rút học thành thất bại học thuật.
- Loại lượt chưa đăng ký hoặc đã rút trước/tại checkpoint.
- Split train/validation/test theo `id_student`, tránh cùng sinh viên xuất hiện ở nhiều tập.
- Cấm feature: target, kết quả cuối, mã sinh viên, `date_unregistration`, mọi aggregate toàn khóa và event sau cutoff.

## Checkpoint

Audit các ngày 30, 60, 90 và 105. Việc chọn cutoff chỉ dùng train/validation. Day 105 là mốc duy nhất vượt toàn bộ cổng chất lượng, nên sản phẩm gọi là **cảnh báo giữa khóa**, không gọi là dự báo sớm.

## Kiểm tra chất lượng bắt buộc

- Bảo toàn tổng `sum_click` khi gom event VLE.
- Missing không bị điền tùy tiện; category không xác định ghi rõ `Unknown/Không xác định`.
- Không nhân dòng sau join.
- Không target leakage hoặc post-cutoff leakage.
- Mapping Geographic Map phủ đủ 13 nhãn region và không trùng vùng hình học.
- Mọi tỷ lệ phải nêu mẫu số là lượt học hay sinh viên duy nhất.

## Lệnh tái tạo

```powershell
python src/oulad_pipeline.py clean data/raw
python src/oulad_pipeline.py build data/raw
python src/dashboard_features.py
python src/eda_analysis.py
python -m unittest discover -s tests -v
```
