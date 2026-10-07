# 05 — Đặc tả dashboard Streamlit

## Story tổng thể

```text
Kết quả hiện tại
→ yếu tố học tập nào đi cùng kết quả
→ nhiều yếu tố bất lợi cùng lúc tạo nhóm nào
→ model cảnh báo lượt có nguy cơ Fail
→ giảng viên nên kiểm tra và hỗ trợ điều gì
```

Insight dữ liệu nằm ở ba trang đầu; model là phần riêng ở Trang 4. Không lấy metric model thay cho insight phân tích.

## Quy tắc trình bày

- Tiêu đề biểu đồ là câu hỏi hoặc kết luận dễ hiểu.
- Trục, đơn vị, legend, hover, màu và cỡ mẫu `N` đầy đủ.
- Xanh: kết quả tốt/an toàn; cam: cần chú ý; đỏ: Fail/cảnh báo; xám: thiếu dữ liệu.
- Không gộp `Fail` và `Withdrawn`.
- “VLE” luôn giải thích là hệ thống học trực tuyến; click không phải thời gian tự học hay điểm danh.
- `AAA–GGG` là mã ẩn danh; `B/J` là đợt bắt đầu tháng 2/tháng 10.
- Day 105 phải ghi “cảnh báo giữa khóa”.
- Mỗi story nêu kết luận, hai số so sánh, cỡ mẫu/phạm vi và hành động hoặc giới hạn.

## Trang 1 — Bức tranh kết quả học tập

Mục tiêu: cho biết kết quả nào phổ biến và khác biệt tập trung ở đâu.

- Filter: module, presentation, giới tính; click map tạo cross-filter vùng.
- KPI: tổng sinh viên, điểm assessment trung bình, tỷ lệ Pass/Distinction, tỷ lệ Fail.
- Geographic Filled Map: tỷ lệ Fail theo 13 vùng OULAD; click vùng lọc cả trang.
- 100% stacked bar: Distinction/Pass/Fail/Withdrawn; drill module → presentation.
- Histogram: phân bố điểm quá trình theo kết quả cuối.
- Story: Fail và Withdrawn tách riêng; nêu module và vùng có tỷ lệ Fail cao/thấp cùng `N`.

Địa lý là bối cảnh tổng hợp: khác biệt vùng có thể đi cùng cơ cấu người học, điều kiện kinh tế-xã hội hoặc module; dashboard không suy diễn vùng cư trú tự nó gây Fail.

## Trang 2 — Các yếu tố học tập

Mục tiêu: chỉ ra yếu tố đo được nào liên quan rõ nhất đến Fail.

- Multi-line: hoạt động VLE theo thời gian của `Fail` và `Pass/Distinction`, kèm deadline.
- Bar: tỷ lệ Fail theo mức hoàn thành assessment đã đến hạn.
- Scatter + trendline: ngày nộp trễ và điểm; kích thước theo số lần từng học.
- Treemap: cơ cấu tài nguyên VLE theo click.
- Story: so sánh quartile VLE thấp/cao và completion 0%/100%; ưu tiên nhóm vừa ít hoạt động vừa chưa hoàn thành bài.

## Trang 3 — Kết hợp nhiều yếu tố

Mục tiêu: không nhìn từng yếu tố rời rạc.

- Heatmap 1: quartile VLE × quartile điểm assessment, màu = tỷ lệ Fail, mỗi ô ghi `N`.
- Heatmap 2: học vấn đầu vào × IMD, màu = tỷ lệ Fail, mỗi ô ghi `N`.
- Box plot: điểm assessment theo số lần từng học học phần.
- Story: đối chiếu nhóm thấp-thấp với cao-cao và nhóm học lần đầu với từng học lại.

## Trang 4 — Cảnh báo giữa khóa nguy cơ Fail

Mục tiêu: trả lời rõ “dự báo cái gì, đúng đến đâu, ai cần xem xét và vì sao”.

- Filter: risk band và IMD.
- KPI: Accuracy, Recall Fail, số lượt High.
- Gauge: xác suất Fail trung bình; Low `<16,75%`, Medium `16,75–<33,5%`, High `≥33,5%`; vạch tím là threshold model.
- Donut: TP/TN/FP/FN giữa dự báo và kết quả thật.
- Bar hệ số: những tín hiệu global đi cùng nguy cơ cao/thấp.
- Action list: student, module, presentation, probability, predicted label; sắp giảm dần.
- Story: model dự báo Fail chứ không dự báo điểm; nêu Accuracy/Recall, số bỏ sót, profile High so với Low và bước hỗ trợ.

## Nguồn dữ liệu giao diện

- Trang 1: `clean_dataset.csv`.
- Trang 2: dashboard marts + snapshot cutoff.
- Trang 3: `feature_snapshot.csv`.
- Trang 4: `model_predictions.csv`, metrics, CI, coefficients và verification.

Ứng dụng không train model và không join event thô khi render.

## QA bắt buộc

- 4/4 trang chạy không exception bằng `streamlit.testing.v1.AppTest`.
- Mapping map khớp đủ region; cross-filter không nhân KPI.
- Bộ lọc rỗng có thông báo, không crash.
- Metric Trang 4 khớp artifact test; threshold không ghi cứng.
- Không còn nội dung Power BI/Tableau hoặc nhãn Fail+Withdrawn cũ ngoài rubric được bảo vệ.
