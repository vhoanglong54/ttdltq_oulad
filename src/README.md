# Python pipeline

## OULAD data pipeline

`oulad_pipeline.py` đọc 7 CSV cục bộ và không sửa raw. Môi trường QA hiện tại dùng Python 3.12.10; phiên bản thư viện nằm trong `requirements.txt`.

```powershell
python src/oulad_pipeline.py audit data/raw
python src/oulad_pipeline.py clean data/raw
python src/oulad_pipeline.py build data/raw
python src/oulad_pipeline.py report
```

- `audit` ghi metrics cục bộ vào `data/interim/t05_audit_metrics.json` (tên file được giữ để tương thích lịch sử).
- `clean` tạo 7 CSV interim ở `data/interim/`; raw luôn bất biến.
- `build` aggregate hai event table trước khi left join về hạt `(code_module, code_presentation, id_student)`, rồi tạo `data/processed/clean_dataset.csv` dùng chung theo D16.
- `report` tạo hiện vật tracked [Data Quality Report](../reports/data-quality-report.md).

Các aggregate có hậu tố `*_all_time` là dữ liệu mô tả dùng cho EDA/dashboard và không được dùng làm feature model. `studentVle` được gom theo learner–resource–day bằng phép cộng `sum_click`, không xóa exact duplicate riêng lẻ. `final_result`, `Academic_Fail`, `At_Risk`, `Withdrawn_Flag` và `date_unregistration` không phải feature.

## EDA, map, model và dashboard marts

- `eda_analysis.py`: tạo 15 bảng bằng chứng và sáu hình tĩnh cho H01–H10/tám insight.
- `build_oulad_regions_geojson.py`: tải ONS geometry, tạo map 13 vùng và mapping audit.
- `at_risk_model.py`: tạo snapshot ngày 105, Logistic Regression dự báo Fail và model artifacts đã kiểm định.
- `dashboard_features.py`: tạo bốn data mart nén cho deadline, submission scatter, VLE timeline và activity treemap; app không phải đọc 8,4 triệu VLE event khi render.

Toàn bộ mục tiêu, feature contract, thí nghiệm, metric, artifact và cách dùng model trong dashboard được quản lý tại [tài liệu model](../docs/04-model.md).
