# 04 — Logistic Regression dự báo nguy cơ trượt học phần

Đây là tài liệu chuẩn duy nhất cho toàn bộ phần model.

## 1. Mục đích dự báo

Tại một checkpoint trong khóa học, model trả lời:

> Với thông tin đã có đến thời điểm này, xác suất một lượt học kết thúc bằng `Fail`, thay vì `Pass/Distinction`, là bao nhiêu?

Đơn vị dự báo là `(code_module, code_presentation, id_student)`. Model **không dự đoán GPA, điểm số chính xác, loại tốt nghiệp hoặc danh tính “sinh viên yếu”**.

Nhãn:

- `Academic_Fail=1`: `final_result == Fail`.
- `Academic_Fail=0`: `Pass` hoặc `Distinction`.
- `Withdrawn`: loại khỏi cohort model; được phân tích riêng.

Output giữ cả tên tường minh `failure_probability`, `predicted_fail`, `actual_fail` và alias cũ để tương thích dashboard.

## 2. Vì sao dùng Logistic Regression

- Đúng thuật toán rubric yêu cầu.
- Phù hợp bài toán nhị phân Fail/không Fail.
- Xuất được xác suất, threshold và hệ số có thể giải thích.
- Cho phép kiểm định rõ bằng Accuracy, Recall, F1, ROC-AUC, PR-AUC, Brier và confusion matrix.

`DummyClassifier` chỉ là baseline kỹ thuật để chứng minh model học được tín hiệu; không phải model chính thứ hai.

## 3. Dữ liệu model được phép nhìn

Chỉ dùng dữ liệu có trước hoặc tại cutoff:

- Tiến độ assessment: bài đến hạn, đã nộp, bỏ lỡ, điểm sớm, điểm có trọng số, độ trễ.
- Hoạt động VLE: tổng click, ngày hoạt động, lần hoạt động gần nhất, cửa sổ 7/28 ngày, số tài nguyên/loại hoạt động.
- Lịch sử trước khóa: số lần từng học, số tín chỉ, ngày đăng ký.
- Module/presentation và học vấn đầu vào.

Cấm: `final_result`, `Academic_Fail`, `At_Risk`, `Withdrawn_Flag`, `date_unregistration`, `id_student`, aggregate `*_all_time` và event sau cutoff. Gender, region, IMD, tuổi, disability dùng cho QA/bối cảnh chứ không làm feature quyết định chính.

## 4. Chọn checkpoint mà không nhìn test

Audit độ phủ:

| Cutoff | Lượt đủ điều kiện | Có VLE | Có assessment nộp | Số ngày còn lại trung bình |
|---:|---:|---:|---:|---:|
| 30 | 22.416 | 97,00% | 76,70% | 226 |
| 60 | 22.422 | 97,93% | 91,91% | 196 |
| 90 | 22.426 | 98,09% | 93,88% | 166 |
| 105 | 22.427 | 98,15% | 93,99% | 151 |

Screening cùng cấu hình trên validation:

| Cutoff | Accuracy | Balanced Accuracy | Recall Fail | ROC-AUC |
|---:|---:|---:|---:|---:|
| 30 | 74,27% | 73,21% | 70,35% | 0,810 |
| 60 | 79,12% | 77,21% | 72,07% | 0,857 |
| 90 | 83,17% | 79,62% | 70,08% | 0,882 |
| 105 | 84,52% | 80,87% | 71,07% | 0,896 |

Ngày 90 không vượt cổng hơn baseline 15 điểm phần trăm trên đánh giá cuối. Ngày 105 là mốc sớm nhất vượt **toàn bộ** cổng đã duyệt. Vì vậy, phải gọi đây là **cảnh báo giữa khóa ngày 105**.

## 5. Split, preprocessing và tuning

- Group split theo `id_student`: 70% train, 15% validation, 15% test; một sinh viên không đi qua nhiều tập.
- Numeric: median imputation, biến count lệch dùng `log1p`, sau đó standardize.
- Categorical: điền `Unknown`, one-hot encode.
- Tuning hyperparameter chỉ trên train-CV.
- Threshold chọn trên validation: Accuracy cao nhất với Recall Fail tối thiểu 75%.
- Test bị khóa đến đánh giá cuối.

Cấu hình cuối: Logistic Regression L1/liblinear, `C=0,3`, `class_weight=None`; threshold **0,335**; model version `lr-oulad-academic-fail-c105-s42-v5`.

Risk band:

- `Low`: probability < 0,1675.
- `Medium`: 0,1675 đến dưới 0,335.
- `High`: probability ≥ 0,335, tức model dự báo Fail.

Risk band là mức ưu tiên vận hành, không phải điểm đậu/rớt chính thức.

## 6. Kết quả trên tập kiểm tra độc lập

Tập test có 3.205 lượt học; tỷ lệ Fail nền 31,39%.

| Chỉ số | Kết quả | Ý nghĩa |
|---|---:|---|
| Accuracy | **83,68%** | đúng khoảng 84/100 lượt |
| KTC 95% Accuracy | **82,42%–85,04%** | cận dưới vẫn trên 80% |
| Balanced Accuracy | **81,48%** | trung bình khả năng nhận diện hai lớp |
| Precision Fail | **73,29%** | trong các cảnh báo Fail, khoảng 73% đúng |
| Recall Fail | **75,55%** | phát hiện khoảng 76/100 lượt thực sự Fail |
| F1 Fail | **74,40%** | cân bằng Precision và Recall |
| ROC-AUC | **0,902** | khả năng xếp hạng Fail cao hơn non-Fail |
| PR-AUC | **0,852** | chất lượng lớp Fail, cao hơn prevalence 0,314 |
| Brier | **0,107** | sai số xác suất; thấp hơn là tốt |

Confusion matrix: 246 False Negative (bỏ sót Fail) và 277 False Positive (cảnh báo nhầm). Accuracy dummy baseline là 68,61%; model cao hơn **15,07 điểm phần trăm**.

## 7. Cổng nghiệm thu

`model_verification.csv` có 11/11 kiểm tra `PASS`:

- Artifact tồn tại và SHA-256 khớp metadata.
- Accuracy ≥ 0,80 và cận dưới KTC 95% ≥ 0,80.
- Recall Fail ≥ 0,70; Balanced Accuracy ≥ 0,78; F1 ≥ 0,70.
- ROC-AUC ≥ 0,85; PR-AUC cao hơn prevalence ít nhất 0,20.
- Brier ≤ 0,15.
- Accuracy hơn dummy baseline ít nhất 0,15.
- Accuracy thấp nhất theo presentation ≥ 0,80.

KTC bootstrap 2.000 lần, resample theo `id_student` để tôn trọng cấu trúc lặp.

## 8. Model dựa vào tín hiệu gì

Hệ số là liên hệ toàn mô hình sau preprocessing, không phải nguyên nhân và không phải giải thích cá nhân.

| Tín hiệu | Hệ số | Diễn giải |
|---|---:|---|
| Lâu không hoạt động trên VLE | +0,698 | đi cùng nguy cơ cao hơn |
| Bỏ lỡ nhiều bài đến hạn | +0,249 | đi cùng nguy cơ cao hơn |
| Từng học học phần này | +0,162 | đi cùng nguy cơ cao hơn |
| Nhiều ngày hoạt động VLE trong 28 ngày | -0,577 | đi cùng nguy cơ thấp hơn |
| Dùng đa dạng tài nguyên VLE | -0,498 | đi cùng nguy cơ thấp hơn |
| Tích lũy điểm ở bài đến hạn | -0,410 | đi cùng nguy cơ thấp hơn |
| Điểm bài tập có trọng số cao | -0,337 | đi cùng nguy cơ thấp hơn |

Một số interaction theo module giúp Logistic Regression cho phép độ mạnh của tín hiệu khác nhau giữa học phần; không dùng chúng để kết luận module gây Fail.

## 9. Model xuất gì cho dashboard

`model_predictions.csv` lưu dự báo ở cấp lượt học để kiểm định kỹ thuật:

- `failure_probability`, `predicted_fail`, `actual_fail`.
- `risk_band`, `error_type`, threshold, cutoff, version.
- Thông tin nhóm để lọc; không chứa lý do nhân quả.

Dashboard trình bày:

1. Một khối mở đầu ngắn nói model dự báo Fail tại ngày 105, cách đọc ngưỡng 0,335 và chuỗi thao tác; không liệt kê công thức dài.
2. Đường Sigmoid và xác suất của từng lượt học trên tập test theo lớp thực tế.
3. Hệ số global giải thích tín hiệu tăng/giảm rủi ro.
4. Thanh trượt ngưỡng để thấy Recall, Precision, F1, tỷ lệ cảnh báo, bỏ sót và cảnh báo nhầm thay đổi thế nào; có nút trở về ngưỡng nghiệm thu 0,335.
5. Confusion matrix cập nhật theo ngưỡng đang thử để biết model đúng và sai ở đâu.
6. Một khối kết luận giúp quyết định giữ, tăng hay giảm ngưỡng dựa trên bỏ sót và cảnh báo nhầm.
7. Cuối trang mới đặt Accuracy, Recall Fail, Precision Fail, F1 Fail và ROC/PR trên toàn bộ test.

Dashboard không công bố mã sinh viên, học phần, lớp học hoặc danh sách người được dự báo. Không dùng model để tự động xử phạt hoặc quyết định kết quả.

## 10. Artifact chuẩn

```text
data/processed/model_academic_fail/c105_final/
  feature_snapshot.csv
  model_predictions.csv
  model_metrics.csv
  model_confidence_intervals.csv
  model_coefficients.csv
  model_subgroup_metrics.csv
  model_verification.csv
  model_metadata.json
models/logistic_academic_fail_c105_final.joblib
```

## 11. Cách tái tạo và đối chiếu

```powershell
python src/at_risk_model.py audit-cutoffs --cutoffs 30 60 90 105 --output data/processed/model_academic_fail/cutoff_audit.csv
python src/screen_academic_fail_cutoffs.py --cutoffs 30 60 90 105 --output data/processed/model_academic_fail/cutoff_screening.csv
python src/at_risk_model.py train --cutoff 105 --minimum-recall 0.75 --output-dir data/processed/model_academic_fail/c105_final --model-path models/logistic_academic_fail_c105_final.joblib --n-jobs -1
python src/at_risk_model.py validate --output-dir data/processed/model_academic_fail/c105_final --model-path models/logistic_academic_fail_c105_final.joblib
python -m unittest discover -s tests -v
```

Đối chiếu thủ công: lấy hàng `logistic_regression,test` trong `model_metrics.csv`; so với KTC trong `model_confidence_intervals.csv`; đếm TP/TN/FP/FN từ `model_predictions.csv`; kiểm tra tất cả dòng của `model_verification.csv` là `PASS`; kiểm tra metadata đúng target, cutoff, threshold và model version.
