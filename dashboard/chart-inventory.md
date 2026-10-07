# Inventory trực quan và câu hỏi nghiên cứu

Quy tắc nghiệm thu: một biểu đồ chỉ được giữ khi nối trực tiếp ít nhất một yếu tố với điểm số, kết quả cuối hoặc chất lượng dự báo. Không dùng biểu đồ chỉ để liệt kê danh mục, mã ẩn danh hay số lượng thuần túy.

| # | Trực quan | Yếu tố → kết quả | Câu hỏi/insight phải trả lời |
|---:|---|---|---|
| 1 | 100% Stacked Bar | Tổng thể hoặc học vấn đầu vào → bốn kết quả | Cơ cấu Xuất sắc/Qua môn/Trượt/Rút học ra sao; cơ cấu thay đổi thế nào theo học vấn? |
| 2 | Geographic Filled Map | Vùng cư trú → tỷ lệ `Fail` | Khu vực nào có tỷ lệ trượt cao/thấp và chênh lệch bao nhiêu? |
| 3 | Violin + Box | Điểm quá trình → kết quả cuối | Phân bố và trung vị điểm khác nhau thế nào giữa bốn kết quả? |
| 4 | Multi-Line | Nhịp hoạt động VLE → `Fail`/qua môn | Hoạt động trực tuyến của hai nhóm tách nhau như thế nào theo thời gian? |
| 5 | Bar | Hoàn thành bài đến hạn → tỷ lệ `Fail` | Tỷ lệ trượt giảm ra sao khi mức hoàn thành tăng? |
| 6 | Scatter + Trendline | Độ trễ nộp bài → điểm | Nộp trễ có đi cùng điểm thấp hơn không và mức liên hệ ra sao? |
| 7 | Grouped Bar | Loại tài nguyên VLE → nhóm kết quả | Cơ cấu sử dụng tài nguyên khác nhau thế nào giữa nhóm Trượt và Qua môn/Xuất sắc? |
| 8 | Heatmap | Mức hoạt động × điểm sớm → tỷ lệ `Fail` | Hai bất lợi cùng xuất hiện làm tỷ lệ trượt thay đổi thế nào? |
| 9 | Heatmap | Học vấn × IMD → tỷ lệ `Fail` | Tổ hợp nền tảng đầu vào và hoàn cảnh khu vực nào có tỷ lệ trượt cao? |
| 10 | Box Plot | Lịch sử học lại → điểm sớm | Điểm số phân bố khác nhau thế nào theo số lần từng học? |
| 11 | Bar + Line | Nhóm xác suất dự báo → tỷ lệ `Fail` thật | Xác suất cao hơn có thực sự đi cùng tỷ lệ trượt cao hơn không? |
| 12 | Confusion Matrix Heatmap | Dự báo ↔ kết quả thật | Model phát hiện, bỏ sót và cảnh báo nhầm bao nhiêu? |
| 13 | Diverging Bar | Tín hiệu đầu vào → nguy cơ model | Tín hiệu nào đi cùng nguy cơ `Fail` cao hơn hoặc thấp hơn? |

Biểu đồ đã loại:

- Module/presentation ẩn danh: khó diễn giải về yếu tố ảnh hưởng.
- Treemap tổng click tài nguyên: chỉ cho biết loại nào phổ biến, không nối với kết quả.
- Gauge xác suất trung bình: một con số trang trí, không kiểm tra chất lượng dự báo.
- Donut TP/TN/FP/FN: khó so sánh lỗi hơn confusion matrix.
- Danh sách dự báo cá nhân: lệch khỏi mục tiêu nghiên cứu yếu tố.

Màu dùng nhất quán: xanh cho Qua môn/Xuất sắc hoặc tín hiệu giảm nguy cơ; đỏ cho `Fail`/nguy cơ cao; cam cho cần chú ý; xám cho thiếu dữ liệu.
