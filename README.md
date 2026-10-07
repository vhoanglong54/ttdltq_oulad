# Nghiên cứu và phân tích các yếu tố ảnh hưởng đến kết quả học tập của sinh viên đại học

Đồ án dùng **OULAD**, Python, Streamlit, Plotly và Logistic Regression. Mục tiêu là làm rõ những yếu tố **liên quan** đến kết quả học tập, không tuyên bố quan hệ nhân quả từ dữ liệu quan sát.

Kết quả được xem theo hai lớp:

- Phân tích mô tả: điểm assessment và `Distinction / Pass / Fail / Withdrawn`.
- Model học thuật: dự báo `Fail` so với `Pass/Distinction`; `Withdrawn` được tách riêng.

Model dùng dữ liệu có đến ngày 105 để đưa ra **cảnh báo giữa khóa**. Trên tập kiểm tra độc lập, Accuracy đạt **83,68%**, Recall Fail **75,55%**, Balanced Accuracy **81,48%**, ROC-AUC **0,902**. Model không dự đoán GPA, điểm cuối chính xác hay nguyên nhân nhân quả.

## Tài liệu chính

1. [Plan đã duyệt](docs/plan_approved.md)
2. [Mục tiêu và phạm vi](docs/01-project-charter.md)
3. [Rubric traceability — không sửa tiêu chí](docs/02-rubric-traceability.md)
4. [Kế hoạch dữ liệu](docs/03-data-plan.md)
5. [Model Logistic Regression](docs/04-model.md)
6. [Đặc tả dashboard](docs/05-dashboard-spec.md)
7. [Data dictionary](docs/09-data-dictionary.md)
8. [Nhật ký insight](docs/insight-log.md)
9. [Tổng kết triển khai](docs/10-implementation-summary.md)

## Cấu trúc

```text
data/raw/                         7 CSV OULAD gốc, không commit
data/interim/                     dữ liệu đã kiểm tra và chuẩn hóa
data/processed/clean_dataset.csv  bảng mô tả một dòng/lượt học
data/processed/model_academic_fail/c105_final/
                                  artifact model đã kiểm định
src/                              pipeline dữ liệu, EDA và model
dashboard/                        ứng dụng Streamlit bốn trang
reports/                          EDA, hình và báo cáo chất lượng
models/                           bundle model
tests/                            kiểm thử dữ liệu, leakage và dashboard
docs/                             tài liệu nghiệp vụ/kỹ thuật
```

## Chạy cục bộ

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/at_risk_model.py validate --output-dir data/processed/model_academic_fail/c105_final --model-path models/logistic_academic_fail_c105_final.joblib
streamlit run dashboard/app.py
```

Dashboard gồm bốn trang: bức tranh kết quả; yếu tố học tập; kết hợp nhiều yếu tố; cảnh báo giữa khóa. Bản đồ là biểu đồ Geographic Map bắt buộc, có cross-filter theo vùng.

## Trạng thái Git

Mọi thay đổi hiện chỉ ở local; các remote cũ đã được gỡ khỏi cấu hình. Không fetch, pull, commit hoặc push cho đến khi leader cung cấp repository mới và duyệt kết quả.
