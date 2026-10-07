# Hướng dẫn dựng và chạy dashboard

## Nguồn bắt buộc

```text
data/processed/clean_dataset.csv
data/processed/dashboard/*.csv[.gz]
data/processed/model_academic_fail/c105_final/*
dashboard/assets/oulad_regions.geojson
```

Không train model và không đọc hàng triệu event VLE khi render.

## Dựng lại dữ liệu

```powershell
python src/oulad_pipeline.py build data/raw
python src/oulad_pipeline.py report
python src/dashboard_features.py
python src/eda_analysis.py
python src/at_risk_model.py validate --output-dir data/processed/model_academic_fail/c105_final --model-path models/logistic_academic_fail_c105_final.joblib
```

## Chạy app

```powershell
streamlit run dashboard/app.py
```

## Contract quan trọng

- Target dashboard/model là Fail; Withdrawn tách riêng.
- Cutoff = 105; gọi cảnh báo giữa khóa.
- Threshold = 0,335 từ artifact.
- Low <0,1675; Medium 0,1675–<0,335; High ≥0,335.
- Một dòng model = một `module × presentation × student` đủ điều kiện.
- KPI mô tả và KPI model không được trộn mẫu số.

## QA

```powershell
python -m compileall dashboard src tests
python -m unittest discover -s tests -v
```

Sau khi đổi data/model/UI phải chụp lại 4 trang và cập nhật `qa-t09.md`. Không dùng ảnh v2/v3/v4 để nghiệm thu phiên bản v5.
