# Inventory trực quan và câu hỏi nghiên cứu

Quy tắc nghiệm thu: một biểu đồ chỉ được giữ khi nối trực tiếp ít nhất một yếu tố với điểm số, kết quả cuối hoặc chất lượng dự báo. Không dùng biểu đồ chỉ để liệt kê danh mục, mã ẩn danh hay số lượng thuần túy.

| # | Trực quan | Yếu tố → kết quả | Câu hỏi/insight phải trả lời |
|---:|---|---|---|
| 1 | 100% Stacked Bar | Tổng thể hoặc học vấn đầu vào → bốn kết quả | Cơ cấu Xuất sắc/Qua môn/Trượt/Rút học ra sao; cơ cấu thay đổi thế nào theo học vấn? |
| 2 | Geographic Filled Map | Vùng cư trú → tỷ lệ `Fail` | Khu vực nào có tỷ lệ trượt cao/thấp và chênh lệch bao nhiêu? |
| 3 | Violin + Box | Điểm quá trình → kết quả cuối | Phân bố và trung vị điểm khác nhau thế nào giữa bốn kết quả? |
| 4 | Bar xếp hạng AUC | Từng yếu tố đơn lẻ ↔ khả năng phân biệt kết quả | Yếu tố nào tự nó phân biệt nhóm kết quả rõ hơn trước khi kết hợp trong model? |
| 5 | Multi-Line | Nhịp hoạt động VLE → `Fail`/qua môn | Hoạt động trực tuyến của hai nhóm tách nhau như thế nào theo thời gian? |
| 6 | Bar | Hoàn thành bài đến hạn → tỷ lệ `Fail` | Tỷ lệ trượt giảm ra sao khi mức hoàn thành tăng? |
| 7 | Bar chuẩn hóa | Độ trễ nộp bài → chênh lệch điểm | Nộp trễ có đi cùng điểm thấp hơn sau khi trừ độ khó bài không? |
| 8 | Dot Plot trục log | Loại tài nguyên VLE ↔ nhóm kết quả | Nhóm qua môn dùng từng tài nguyên nhiều gấp bao nhiêu lần nhóm kết quả thấp? |
| 9 | Heatmap | Mức hoạt động × điểm sớm → tỷ lệ `Fail` | Hai bất lợi cùng xuất hiện làm tỷ lệ trượt thay đổi thế nào? |
| 10 | Heatmap | Học vấn × IMD → tỷ lệ `Fail` | Tổ hợp nền tảng đầu vào và hoàn cảnh khu vực nào có tỷ lệ trượt cao? |
| 11 | Box Plot | Lịch sử học lại → điểm sớm | Điểm số phân bố khác nhau thế nào theo số lần từng học? |
| 12 | Sigmoid + Probability Strip | Điểm `z` → `p(Fail)`; xác suất từng quan sát ↔ lớp thật | Logistic biến điểm tuyến tính thành xác suất thế nào và hai lớp nằm ở đâu so với ngưỡng 33,5%? |
| 13 | Diverging Bar | Tín hiệu đầu vào → nguy cơ model | Tín hiệu nào đi cùng nguy cơ `Fail` cao hơn hoặc thấp hơn? |
| 14 | Threshold Trade-off Line | Ngưỡng phân loại ↔ Recall/Precision/F1 | Hạ hoặc tăng ngưỡng làm khả năng phát hiện và độ chính xác cảnh báo thay đổi ra sao? |
| 15 | Confusion Matrix Heatmap | Dự báo ↔ kết quả thật | Model phát hiện, bỏ sót và cảnh báo nhầm bao nhiêu tại ngưỡng đang chọn? |
| 16 | ROC + Precision–Recall | Khả năng phân biệt ↔ kết quả thật | Logistic Regression xếp hạng hai lớp tốt đến đâu trên mọi ngưỡng? |

Biểu đồ đã loại:

- Module/presentation ẩn danh: khó diễn giải về yếu tố ảnh hưởng.
- Treemap tổng click tài nguyên: chỉ cho biết loại nào phổ biến, không nối với kết quả.
- Gauge xác suất trung bình: một con số trang trí, không kiểm tra chất lượng dự báo.
- Donut TP/TN/FP/FN: khó so sánh lỗi hơn confusion matrix.
- Danh sách dự báo cá nhân: lệch khỏi mục tiêu nghiên cứu yếu tố.
- Biểu đồ so sánh hai mức A/B: trùng với phân tích yếu tố ở các trang trước và không phải đầu ra cốt lõi của Logistic Regression.

Màu dùng nhất quán: xanh cho Qua môn/Xuất sắc hoặc tín hiệu giảm nguy cơ; đỏ cho `Fail`/nguy cơ cao; cam cho cần chú ý; xám cho thiếu dữ liệu.
