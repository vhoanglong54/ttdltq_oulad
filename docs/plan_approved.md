# PLAN_APPROVED — Phương án triển khai OULAD đã chốt

**Tên đề tài:** Nghiên cứu và phân tích các yếu tố ảnh hưởng đến kết quả học tập của sinh viên đại học

**Ngày duyệt:** 07/10/2026

**Công nghệ duy nhất:** Python, Pandas, scikit-learn, Streamlit và Plotly

**Quy tắc Git:** chỉ commit/push thay đổi mới sau khi leader duyệt kết quả
**Rubric:** giữ nguyên nội dung và tiêu chí; không sửa file rubric nguồn

> **Kết quả triển khai:** đã audit 30/60/90/105; chỉ ngày 105 vượt toàn bộ cổng chất lượng nên sản phẩm gọi là **cảnh báo giữa khóa**. Model v5 dự báo Fail đạt Accuracy 83,68%, Recall Fail 75,55%, ROC-AUC 0,902 và 11/11 verification PASS. Dashboard bốn trang và 8 insight đã chuyển sang target Fail, tách Withdrawn.

## 1. Phạm vi nghiên cứu

Dự án dùng 7 bảng của Open University Learning Analytics Dataset (OULAD) để phân tích các yếu tố **liên quan** đến kết quả học tập trong môi trường đại học trực tuyến. Tên đề tài được giữ nguyên; báo cáo phải nói rõ OULAD là dữ liệu quan sát của một trường đại học tại Vương quốc Anh nên không tự động đại diện cho toàn bộ sinh viên đại học.

Kết quả học tập được đo bằng hai tầng:

- Điểm assessment từ 0 đến 100, gồm điểm trung bình và điểm có trọng số.
- Kết quả cuối của một module-presentation: `Distinction`, `Pass`, `Fail`.

`Withdrawn` là kết quả về duy trì học tập và được phân tích riêng, không gộp thành `Fail` trong model học thuật chính. OULAD không có GPA toàn khóa hoặc loại tốt nghiệp; giới hạn này phải xuất hiện trong dashboard và báo cáo.

Đơn vị phân tích là một lượt học theo khóa:

```text
(code_module, code_presentation, id_student)
```

Không đồng nhất số lượt học với số sinh viên duy nhất.

## 2. Nhóm yếu tố được phân tích

1. Cá nhân: giới tính, nhóm tuổi, tình trạng khuyết tật.
2. Nền tảng học tập: học vấn trước đó, số lần từng học module, số tín chỉ đăng ký.
3. Kinh tế–xã hội và địa lý: `imd_band`, `region`.
4. Mức tham gia học trực tuyến: tổng click, ngày/tuần hoạt động, độ gián đoạn, xu hướng hoạt động và loại tài nguyên VLE.
5. Hành vi làm bài: số bài đến hạn, tỷ lệ hoàn thành, nộp đúng/trễ hạn và điểm đầu khóa.

VLE click chỉ là dấu vết sử dụng nền tảng, không phải giờ tự học hoặc điểm danh. IMD là chỉ số thiếu thốn của khu vực cư trú, không phải thu nhập cá nhân.

## 3. Câu hỏi nghiên cứu

1. Cơ cấu kết quả tổng thể ra sao và khác nhau thế nào theo vùng, học vấn đầu vào và các nhóm sinh viên?
2. Học vấn trước đó, số lần học lại, tín chỉ và điều kiện kinh tế–xã hội liên quan thế nào đến kết quả?
3. Nhóm `Pass/Distinction` và `Fail` có nhịp tương tác VLE khác nhau từ thời điểm nào?
4. Hoàn thành assessment và nộp đúng hạn liên quan thế nào đến điểm và kết quả cuối?
5. Khi tương tác thấp, điểm sớm thấp và hoàn thành bài thấp cùng xuất hiện, nguy cơ `Fail` thay đổi ra sao?
6. Logistic Regression có thể cảnh báo sớm nguy cơ `Fail` với accuracy trên 80% hay không?

## 4. Insight phải khai thác

| ID | Nội dung phải kết luận | Vai trò |
|---|---|---|
| INS-01 | Cơ cấu Distinction/Pass/Fail/Withdrawn tổng thể và khác biệt theo học vấn đầu vào | Bức tranh kết quả |
| INS-02 | Chênh lệch `Fail` theo mức hoàn thành assessment đến cutoff | Tiến độ học tập |
| INS-03 | Liên hệ giữa điểm đầu khóa, nộp đúng/trễ và kết quả cuối | Hành vi làm bài |
| INS-04 | Chênh lệch kết quả theo mức độ, tính liên tục và độ gián đoạn VLE | Mức tham gia học |
| INS-05 | Rủi ro khi tương tác thấp, điểm sớm thấp và hoàn thành bài thấp cùng xuất hiện | Bất lợi cộng dồn |
| INS-06 | Chênh lệch theo học vấn trước đó, số lần học lại và tải tín chỉ | Nền tảng học tập |
| INS-07 | Chênh lệch theo IMD/region sau khi đặt trong cùng module/presentation | Bối cảnh kinh tế–địa lý |
| INS-08 | Khác biệt về tỷ trọng sử dụng từng loại tài nguyên VLE giữa nhóm `Fail` và `Pass/Distinction` | Chiến lược học tập |

Mỗi insight phải có câu kết luận dễ hiểu, nhóm so sánh, tử số/mẫu số, `N`, chênh lệch điểm phần trăm hoặc effect size, filter context, đường dẫn bằng chứng, giới hạn và đề xuất hành động. Không hard-code số liệu chưa được sinh từ pipeline.

## 5. Story dashboard bốn trang

### Trang 1 — Kết quả học tập hiện tại

- KPI: tổng lượt học, điểm trung bình, tỷ lệ Pass/Distinction và tỷ lệ Fail.
- Geographic Map theo 13 vùng, có `N`, bộ lọc và cross-filter.
- 100% stacked bar: `Distinction/Pass/Fail/Withdrawn` tổng thể, drill xuống học vấn đầu vào.
- Violin + box phân phối điểm assessment theo kết quả cuối.
- Story: kết quả đang ra sao và khác biệt xuất hiện theo học vấn/vùng ở đâu.

### Trang 2 — Hành vi học tập liên quan trực tiếp

- Multi-line: tương tác VLE theo tuần giữa các nhóm kết quả.
- Bar: tỷ lệ Fail theo mức hoàn thành assessment.
- Scatter + trendline: thời điểm nộp bài và điểm.
- Box plot: điểm theo mức tương tác hoặc lịch sử học lại.
- Grouped bar: tỷ trọng loại tài nguyên VLE theo nhóm kết quả.
- Story: tiến độ làm bài, điểm đầu khóa và tính liên tục của hoạt động học là các tín hiệu gần kết quả nhất.

### Trang 3 — Nhiều yếu tố xuất hiện cùng lúc

- Heatmap: tương tác × điểm đầu khóa.
- Heatmap: hoàn thành assessment × mức tương tác.
- Heatmap: học vấn × IMD.
- Bar/box plot: số lần học lại và tải tín chỉ.
- Story: xác định nhóm chịu nhiều yếu tố bất lợi đồng thời, không gán định kiến cá nhân.

### Trang 4 — Cảnh báo sớm nguy cơ Fail

- Một ô mở đầu nói model dự báo Fail tại ngày 105, cách đọc ngưỡng 33,5% và chuỗi thao tác; không liệt kê công thức dài.
- Sigmoid + probability strip: đường xác suất và từng quan sát theo lớp thật, có threshold 33,5%.
- Thanh trượt threshold: cho thấy đánh đổi Recall/Precision/F1, tỷ lệ cảnh báo, bỏ sót và cảnh báo nhầm; có nút trở về ngưỡng artifact đã nghiệm thu.
- Confusion matrix heatmap cho đúng, cảnh báo nhầm và bỏ sót.
- Khối quyết định giải thích nên giữ, tăng hay giảm ngưỡng dựa trên bỏ sót và cảnh báo nhầm.
- Cuối trang: KPI Accuracy/Recall/Precision/F1 và ROC + Precision–Recall trên toàn bộ test.
- ROC/PR curve trong phần kiểm định.
- Bar hệ số cho các tín hiệu làm xác suất tăng/giảm.
- Story: model ước lượng khả năng `Fail`, không dự đoán GPA hoặc điểm chính xác.

Mạch Story chung:

```text
Kết quả hiện tại → yếu tố đơn lẻ → bất lợi cộng dồn → cảnh báo sớm → hỗ trợ phù hợp
```

## 6. Quy định trực quan

- Tối thiểu 8 loại biểu đồ thường và một Geographic Map riêng.
- Không dùng sunburst/drill-down hình tròn khó đọc.
- Một biểu đồ chỉ trả lời một câu hỏi chính.
- Mỗi biểu đồ phải nối một yếu tố với điểm/kết quả hoặc nối dự báo với kết quả thật; không dùng visual để liệt kê mã/danh mục/độ phổ biến đơn thuần.
- Tiêu đề nêu nội dung; trục, đơn vị, legend và tooltip dùng tiếng Việt dễ hiểu.
- Mọi tỷ lệ có `N` trong chart, caption hoặc tooltip.
- Màu nhất quán: xanh cho kết quả tốt/an toàn, cam cho cần chú ý, đỏ cho Fail/nguy cơ cao, xám cho thiếu dữ liệu.
- Không để title, legend, nhãn hoặc biểu đồ chèn lên nhau ở viewport trình chiếu.
- Mỗi trang có một khối **Story** gồm 2–3 câu: kết luận, bằng chứng chính và hành động.
- Chi tiết kỹ thuật/giới hạn nằm trong tooltip hoặc expander, không làm rối luồng chính.
- Filter chung: region, giới tính, tuổi, học vấn và IMD.
- Drill-down có ý nghĩa: cơ cấu tổng thể → học vấn đầu vào.
- Map click phải cập nhật KPI và các visual liên quan.

Inventory hiện hành: Geographic Map, 100% stacked bar, violin, multi-line, bar, scatter, grouped bar, box plot, heatmap, probability validation, confusion matrix và diverging bar.

## 7. Mục đích và target dự báo

Model chính trả lời câu hỏi:

> Tại một checkpoint đầu khóa, xác suất một lượt học sẽ kết thúc bằng `Fail`, thay vì `Pass` hoặc `Distinction`, là bao nhiêu?

Quy ước:

- `Academic_Fail = 1` nếu `final_result == Fail`.
- `Academic_Fail = 0` nếu `final_result` là `Pass` hoặc `Distinction`.
- Loại `Withdrawn` khỏi cohort model học thuật chính; phân tích duy trì học tập riêng.
- Không dự báo GPA, điểm chính xác hoặc loại tốt nghiệp.

Mỗi output dự báo phải có xác suất Fail, mức Low/Medium/High, nhãn dự báo, actual label trên tập kiểm định, loại lỗi, các tín hiệu chính làm xác suất tăng/giảm và đề xuất hỗ trợ phù hợp.

## 8. Feature và chống data leakage

Feature được phép dùng nếu tồn tại tại checkpoint:

- Module/presentation, học vấn trước đó, số lần học module, tín chỉ và thời điểm đăng ký.
- Click, ngày hoạt động, khoảng gián đoạn, xu hướng gần đây và loại tài nguyên VLE.
- Assessment đã đến hạn, tỷ lệ hoàn thành, điểm sớm và thời điểm nộp trước cutoff.

Cấm tuyệt đối:

- `final_result`, target và identifier.
- `date_unregistration` trong model học thuật.
- Điểm, submission hoặc VLE event sau cutoff.
- Aggregate `*_all_time`.
- Preprocessing fit trên validation/test.

Giới tính, tuổi, region, IMD và disability ưu tiên làm cột audit/subgroup; không tự động dùng để quyết định hỗ trợ cá nhân nếu chưa có lý do và kiểm tra chênh lệch nhóm.

## 9. Chọn checkpoint

Không mặc định giữ ngày 105. Phải kiểm tra các checkpoint 30, 60, 90 và 105 ngày bằng cùng protocol. Chọn **mốc sớm nhất** đạt toàn bộ cổng chất lượng. Nếu chỉ ngày 105 đạt, giao diện phải gọi là “cảnh báo giữa khóa”, không gọi là “cảnh báo sớm”.

Lịch sử so sánh cutoff phải được công bố để tránh cherry-pick.

## 10. Cổng độ chính xác model

Model chỉ được nghiệm thu khi đồng thời đạt:

- Test Accuracy ≥ 0,80.
- Cận dưới bootstrap 95% của Accuracy ≥ 0,80.
- Recall lớp Fail ≥ 0,70.
- Balanced Accuracy ≥ 0,78.
- F1 lớp Fail ≥ 0,70.
- ROC-AUC ≥ 0,85.
- PR-AUC cao hơn rõ ràng tỷ lệ Fail nền.
- Brier Score ≤ 0,15.
- Accuracy cao hơn Dummy baseline ít nhất 0,15.
- Không trùng `id_student` giữa train, validation và test.
- Threshold chỉ chọn trên validation; test chỉ dùng để báo cáo cuối.
- Có temporal/presentation stress test để đánh giá độ ổn định.

Không được đổi test, chọn cutoff theo test hoặc đưa feature sau cutoff vào chỉ để vượt 80%.

## 11. Nhận xét model phải đưa ra

Dashboard phải trả lời rõ:

1. Nhóm nào có xác suất Fail cao nhất?
2. Tín hiệu nào liên quan mạnh nhất đến dự báo Fail?
3. Trong 100 lượt thực sự Fail, model phát hiện và bỏ sót bao nhiêu?
4. Các tín hiệu của nhóm cảnh báo cao gợi ý cần cải thiện điều gì?

Ví dụ diễn giải đúng:

> Tại checkpoint, mức hoàn thành assessment thấp, điểm sớm thấp và hoạt động gần đây giảm cùng xuất hiện trong nhóm nguy cơ Fail cao. Đây là bằng chứng về mối liên hệ ở cấp nhóm, không phải kết luận cho từng cá nhân hoặc bằng chứng nhân quả.

Đề xuất hành động:

- Chưa hoàn thành bài: nhắc hạn và hỗ trợ hoàn thành assessment.
- Điểm đầu khóa thấp: bổ sung học thuật.
- Hoạt động giảm/ngắt quãng: liên hệ để tìm trở ngại.
- Lịch sử học lại hoặc tải tín chỉ cao: tư vấn kế hoạch học.

## 12. Phần giữ lại và phần phải làm lại

Giữ lại: 7 bảng raw, pipeline audit/clean/join, processed base, dashboard marts, geometry/mapping 13 vùng, hạ tầng bootstrap/leakage guard/tests và kiến trúc bốn trang.

Làm lại: target, cutoff, model artifacts, metric, insight theo outcome tách biệt, Story, phân tích yếu tố tổng hợp, tài liệu còn nhắc Tableau, ảnh QA và báo cáo model.

Model cũ và ảnh QA cũ chỉ được archive/xóa sau khi bản mới vượt kiểm định và leader duyệt.

## 13. Trình tự thực hiện

1. Khóa data contract và outcome.
2. Tính lại EDA/insight với `Fail` và `Withdrawn` tách riêng.
3. Huấn luyện Logistic Regression theo các checkpoint.
4. Chọn checkpoint sớm nhất vượt cổng chất lượng.
5. Cập nhật dashboard bốn trang và Story.
6. Kiểm tra map, filter, drill-down, cross-filter và bố cục.
7. Đồng bộ README, model, insight log, dashboard spec và báo cáo.
8. Chạy automated tests, visual QA, leakage audit và đối chiếu rubric.
9. Báo thay đổi để leader duyệt.
10. Chỉ kết nối và commit lên repo Git mới sau khi được cho phép.
