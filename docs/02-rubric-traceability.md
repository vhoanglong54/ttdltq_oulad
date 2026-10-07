# 02 — Ma trận rubric và bằng chứng

Trạng thái lúc khởi tạo: **chưa nghiệm thu**. Tích `[x]` chỉ khi có hiện vật, bằng chứng kiểm tra và leader xác nhận; cột đường dẫn là nơi sẽ đặt kết quả. Các điểm số dưới đây giữ nguyên barem trong DOCX, tổng 10 điểm.

| Mục | Điểm | Bằng chứng cần nộp | Thực hiện | Đạt |
|---|---:|---|---|:---:|
| Bài toán và dataset hợp lệ | 0,5 | [Charter](01-project-charter.md), [nguồn/checksum](../data/README.md), [dictionary 7 bảng](09-data-dictionary.md) | Leader | [x] |
| Làm sạch dữ liệu | 0,5 | [Pipeline](../src/oulad_pipeline.py), [Data Quality Report](../reports/data-quality-report.md), notebook T05–T06 | Leader | [x] |
| Biến đổi dữ liệu | 0,75 | [Pipeline](../src/oulad_pipeline.py), [processed contract](../data/processed/README.md): aggregate trước join, 0 unmatched/duplicate key | Leader | [x] |
| EDA | 0,75 | `03_eda.ipynb`, ít nhất 3–5 biểu đồ tĩnh Matplotlib/Seaborn và diễn giải | Leader | [x] |
| UI/UX Dashboard | 0,5 | Streamlit có layout rõ, màu nhất quán, tiêu đề, chú thích, định nghĩa KPI | Leader | [x] |
| Đa dạng biểu đồ | 1,0 | Ít nhất 8 **loại** biểu đồ khác nhau, phù hợp kiểu dữ liệu; inventory và ảnh | Leader | [x] |
| Bản đồ địa lý | 0,5 | Ít nhất 1 geographic map thực sự theo `region`, tỷ lệ/số đo và mapping vùng được kiểm tra | Leader | [x] |
| Tương tác | 1,5 | Filter nhiều cấp, drill-down, tooltip hover, cross-filtering; checklist kiểm thử | Leader | [x] |
| Insight/storytelling | 1,0 | 5–7 insight có bằng chứng, xu hướng/nhóm/tương tác, giới hạn diễn giải | Leader | [x] |
| Mô hình dự báo | 0,5 | Logistic Regression đúng target, split/encoding hợp lệ, chỉ số đánh giá | Leader | [x] |
| Trực quan dự báo | 0,5 | Xác suất/phân lớp và Actual vs Predicted trên Streamlit, đối chiếu với artifact Python | Leader | [x] |
| Báo cáo khoa học | 1,0 | ≥40 trang, pipeline và sơ đồ hệ thống, logic chọn chart, code/pseudocode, trích dẫn IEEE | Leader | [ ] |
| Demo ứng dụng | 1,0 | Dashboard chạy trực tiếp, kịch bản Data Analyst, video backup, link trong báo cáo | Leader | [ ] |

Điều kiện bắt buộc có thể làm đồ án không hợp lệ: dataset dưới 5.000 dòng hoặc chỉ một bảng đơn giản. Yêu cầu đầy đủ theo DOCX là **từ 3 bảng trở lên**, có khóa nối tự nhiên. Dashboard hiện được triển khai bằng **Python Streamlit + Plotly**.

## Cổng kiểm tra chất lượng

- **Dữ liệu:** nguồn và số dòng xác minh trên file đã tải; 7 bảng/khóa; data dictionary; data audit; cleaning; bảng phân tích với calculated fields; kiểm tra không nhân dòng ngoài dự kiến.
- **Insight:** 3–5 biểu đồ tĩnh tối thiểu, kiểm tra 8–10 giả thuyết theo khả năng dữ liệu, 5–7 insight chính có số liệu và leader xác nhận.
- **Dashboard:** 4 trang, đủ 8 loại chart và map, kiểm thử tương tác, đối chiếu KPI/forecast, nội dung demo ổn định.
- **Nộp cuối:** báo cáo ≥40 trang, tài liệu tham khảo IEEE, slide, video backup, link demo và khả năng giải thích data/analysis/model/dashboard. DOCX cảnh báo vấn đáp yếu hoặc không hiểu code có thể bị trừ tới 4 điểm hay bị hủy kết quả nếu vi phạm liêm chính.

Xem [backlog](06-tasks-and-dependencies.md) để biết task tạo từng bằng chứng.
