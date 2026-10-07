# 08 — Quyết định và điểm cần xác nhận

| ID | Trạng thái | Quyết định | Ảnh hưởng |
|---|---|---|---|
| D01 | Chốt | Giữ OULAD và tên đề tài. | Toàn dự án |
| D02 | Chốt | Python + Streamlit + Plotly; không Tableau/Power BI. | Dashboard |
| D03 | Chốt | `Academic_Fail`: Fail=1, Pass/Distinction=0; Withdrawn tách riêng. | Data/model/insight |
| D04 | Chốt sau audit | Checkpoint ngày 105; gọi “cảnh báo giữa khóa”. | Model/UI |
| D05 | Chốt trên validation | Threshold 0,335; Low <0,1675; Medium <0,335; High ≥0,335. | Prediction |
| D06 | Chốt | Split theo `id_student`; test chỉ đánh giá cuối. | QA model |
| D07 | Chốt | Accuracy và cận dưới CI ≥80%; thêm Recall/Balanced/F1/AUC/Brier/baseline gates. | Nghiệm thu |
| D08 | Chốt | Geographic Map theo region, màu = tỷ lệ Fail, có cross-filter. | Rubric/dashboard |
| D09 | Chốt | Insight phân tích tách khỏi metric model. | Story/report |
| D10 | Chốt | Dùng “liên quan/đi cùng”; cơ chế giải thích là giả thuyết, không kết luận nhân quả. | Ngôn ngữ |
| D11 | Chốt | Không tạo GPA, attendance, study hours hoặc biến OULAD không đo. | Data integrity |
| D12 | Chốt | Không dùng `date_unregistration`, target, `*_all_time` hay post-cutoff event làm feature. | Leakage guard |
| D13 | Chốt | Dashboard chỉ đọc artifact đã verification; không train khi render. | Kiến trúc |
| D14 | Chờ leader | Remote Git mới, commit và push. | Git |

## Điểm mở trước nghiệm thu cuối

- Ảnh QA cũ phải thay bằng ảnh sau khi target/UI đã sửa.
- DOCX/báo cáo cần kiểm tra khi file không bị khóa.
- Artifact legacy Fail+Withdrawn giữ tạm để truy vết; chỉ archive/xóa sau khi leader duyệt.
