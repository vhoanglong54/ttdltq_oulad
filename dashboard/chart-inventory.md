# Inventory trực quan

Dashboard có 13 visual, gồm Geographic Map bắt buộc và nhiều loại biểu đồ hỗ trợ đúng câu hỏi.

| # | Loại | Câu hỏi trả lời |
|---:|---|---|
| 1 | Filled Geographic Map | Tỷ lệ Fail phân bố ra sao giữa các vùng? |
| 2 | 100% Stacked Bar | Cơ cấu bốn kết quả khác nhau theo module/presentation thế nào? |
| 3 | Histogram | Phân bố điểm quá trình khác nhau theo kết quả cuối ra sao? |
| 4 | Multi-Line | Nhịp hoạt động VLE của Fail và Pass/Distinction khác nhau thế nào? |
| 5 | Bar | Completion đến hạn liên quan thế nào đến Fail? |
| 6 | Scatter + Trendline | Nộp trễ liên hệ với điểm thế nào? |
| 7 | Treemap | Tài nguyên VLE nào được sử dụng nhiều? |
| 8 | Heatmap | VLE × điểm sớm tạo nhóm Fail nào? |
| 9 | Heatmap | Học vấn × IMD có tổ hợp nào đáng chú ý? |
| 10 | Box Plot | Điểm phân tán theo lịch sử học lại thế nào? |
| 11 | Gauge | Xác suất Fail trung bình trong bộ lọc là bao nhiêu? |
| 12 | Donut | Model cảnh báo đúng, nhầm và bỏ sót bao nhiêu? |
| 13 | Diverging Bar | Tín hiệu nào đi cùng nguy cơ cao/thấp trong model? |

Action List là bảng hỗ trợ hành động, không tính như biểu đồ.

Màu dùng nhất quán: xanh cho Pass/Distinction hoặc tín hiệu giảm nguy cơ; đỏ cho Fail/nguy cơ cao; cam cho cần chú ý. Threshold 0,335 và risk band đều đọc từ artifact model, không ghi cứng trong logic giao diện.
