# Automated QA — cập nhật 07/10/2026

## Lệnh

```powershell
python -m py_compile dashboard/app.py dashboard/dashboard_data.py src/dashboard_features.py src/eda_analysis.py src/build_oulad_regions_geojson.py src/at_risk_model.py
python -m unittest discover -s tests -v
```

## Kết quả

- Compile: PASS.
- Unit tests: **19/19 PASS**.
- Dashboard mart click preservation: PASS ở cả daily và activity mart.
- Assessment delay contract: PASS.
- AppTest Trang 1: 3 Plotly charts, 4 metrics, 0 exception.
- AppTest Trang 2: 4 Plotly charts, 0 exception.
- AppTest Trang 3: 3 Plotly charts, 0 exception.
- AppTest Trang 4: 3 Plotly charts, 3 metrics, 0 dataframe, 0 exception.
- Markdown relative links: 0 missing.
- `git diff --check`: PASS.
- Protected rubric files: không có diff.

## Model/Geo gates kế thừa

- Logistic Regression test: Accuracy 0,8368; Recall 0,7555; verification 11/11 PASS.
- GeoJSON: 13 region, mapping 218 ONS areas duy nhất, geometry hợp lệ.
