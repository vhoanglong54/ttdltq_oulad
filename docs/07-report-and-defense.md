# 07 — Báo cáo, demo và vấn đáp

Một người thực hiện chính chịu trách nhiệm hiểu và trình bày toàn bộ pipeline.

## Cấu trúc báo cáo

| Phần | Nội dung cần chứng minh |
|---|---|
| Introduction & Related Work | Bối cảnh, mục tiêu, RQ, phạm vi và tài liệu tham khảo IEEE |
| Data & Method | 7 bảng OULAD, nguồn, khóa/hạt, audit, cleaning, calculated fields và giới hạn |
| EDA & Insights | Sáu chart tĩnh, H01–H10, tương tác đa biến, sáu insight và story |
| Prediction | Target/cutoff, leakage guard, split, Logistic Regression, metric/CI, lỗi và giới hạn |
| Interactive Dashboard | Streamlit + Plotly, 8 loại chart thường, Geographic Map riêng, tương tác và QA |
| Conclusion | Trả lời RQ, khuyến nghị thận trọng, hạn chế và hướng tiếp theo |
| Installation & Demo | Cách tái tạo data/model, chạy app, kịch bản demo và video backup |

Báo cáo phải đạt **ít nhất 40 trang** và dùng trích dẫn IEEE. Sơ đồ pipeline: **bài toán → OULAD → audit/cleaning → join/feature → EDA/hypothesis → insight/risk profile → Logistic Regression → dashboard Python → báo cáo/demo**.

## Kịch bản Story khi demo

1. Nêu câu hỏi trung tâm, hạt lượt học và giới hạn dữ liệu quan sát.
2. Trang Bức tranh kết quả nêu tỷ lệ Pass/Distinction, Fail và Withdrawn tách riêng; học phần và vùng là bối cảnh so sánh.
3. Demo map cross-filter và drill từ học phần xuống đợt mở.
4. Trang Các yếu tố học tập kết luận hoàn thành bài là yếu tố liên quan rõ nhất, sau đó đến mức tham gia học trực tuyến.
5. Trang Kết hợp nhiều yếu tố so sánh nhóm thấp ở cả mức tham gia và điểm với nhóm cao ở cả hai; nêu thêm lịch sử học lại.
6. Trang Dự đoán nói rõ mô hình dự báo `Fail` so với `Pass/Distinction`, loại `Withdrawn`, không dự đoán điểm; trình bày Sigmoid và xác suất từng quan sát, kéo threshold để giải thích đánh đổi Precision–Recall–F1, rồi dùng confusion matrix và ROC/PR để trình bày model đúng/sai đến đâu.
7. Kết luận Story, hành động thận trọng và giới hạn.

## Nội dung phải tự giải thích được

- Nguồn, giấy phép, 7 bảng, khóa và grain.
- Quy tắc cleaning, missing, outlier và aggregation.
- RQ/H, cách tính KPI, lý do chọn từng loại chart.
- Geometry/mapping của Geographic Map.
- Filter, drill-down, tooltip và cross-filtering.
- Target, cutoff, feature, split, threshold và các metric model.
- Vì sao insight chỉ là mối liên hệ, không phải kết luận nhân quả.

## Bằng chứng nộp

- PDF báo cáo, slide, link/code demo và video backup.
- Inventory 8 chart thường và Geographic Map được kiểm tra riêng.
- Insight Log 5–7 mục.
- Model verification và dashboard QA.
- Lệnh tái tạo, requirements, checksum input và ảnh/video interaction.
