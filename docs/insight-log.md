# Insight log — kết quả học tập OULAD

Đây là nơi nghiệm thu insight phân tích dữ liệu. Model ở `04-model.md` là phần riêng. Tất cả kết luận dưới đây nói **liên quan/đi cùng**, không chứng minh nhân quả.

## Tóm tắt dễ trình bày

1. **Nền tảng học vấn đi cùng khác biệt rõ:** nhóm không có bằng cấp chính quy có tỷ lệ Fail 27,38%, còn nhóm sau đại học là 10,86%.
2. **Mức tham gia VLE phân biệt kết quả rõ:** 25% ít click nhất có tỷ lệ Fail 58,25%; 25% nhiều click nhất 12,40%.
3. **Hoàn thành bài đến hạn là tín hiệu mạnh nhất:** nhóm chưa hoàn thành bài nào có tỷ lệ Fail 95,88%; nhóm hoàn thành đủ 18,75%.
4. **Hai bất lợi cộng dồn làm nhóm rủi ro nổi bật:** VLE thấp + điểm thấp có tỷ lệ Fail 65,56%; cao ở cả hai chỉ 4,58%.
5. **Lịch sử học lại cần được chú ý:** nhóm từng học học phần có tỷ lệ Fail 48,83%; học lần đầu 29,02%.
6. **Có chênh lệch vùng nhưng địa lý là bối cảnh:** Wales 29,72%; South Region 17,59%.
7. **Nộp muộn ảnh hưởng điểm số:** nộp trễ quá 7 ngày làm điểm trung bình (đã trừ độ khó bài) giảm mạnh (-6.77), nhưng nộp sớm chưa chắc điểm cao hơn đúng hạn.
8. **Hoạt động gần đây đặc biệt hữu ích để can thiệp:** 25% ít ngày hoạt động nhất trong 28 ngày có tỷ lệ Fail 69,02%; 25% nhiều ngày nhất 9,65%.

## Bảng bằng chứng

| ID | Claim có số liệu | N / phạm vi | Bằng chứng | Ý nghĩa hành động và giới hạn |
|---|---|---|---|---|
| INS-01 | Không có bằng cấp chính quy Fail **27,38%** (N=347), sau đại học **10,86%** (N=313), chênh **16,51 điểm %**. | Toàn khóa, N=32.593 lượt học. | `insight_evidence.csv`, cơ cấu kết quả theo học vấn | Học vấn đi cùng nhiều khác biệt nền tảng khác; không phải tác động nhân quả độc lập. |
| INS-02 | Quartile VLE click thấp nhất Fail **58,25%** (N=5.607), cao nhất **12,40%** (N=5.607), chênh **45,85 điểm %**. | Cohort model ngày 105, N=22.427. | `engagement_quartiles.csv`, VLE line | Ít dùng hệ thống là tín hiệu theo dõi; click không đo thời gian hay chất lượng học. |
| INS-03 | Completion 0% Fail **95,88%** (N=1.382), completion 100% **18,75%** (N=17.569), chênh **77,12 điểm %**. | Assessment đã đến hạn tại ngày 105. | `assessment_completion.csv` | Ưu tiên nhắc bài đến hạn; lịch assessment khác nhau giữa module. |
| INS-04 | VLE thấp + điểm thấp Fail **65,56%** (N=1.507), cao + cao **4,58%** (N=2.098), chênh **60,98 điểm %**. | Snapshot ngày 105, nhóm có điểm để xếp quartile. | `engagement_assessment_matrix.csv` | Dùng hai tín hiệu cùng lúc tốt hơn nhìn một chỉ số; không phải công thức nhân quả. |
| INS-05 | Có ≥1 lần học trước Fail **48,83%** (N=2.697), lần đầu **29,02%** (N=19.730), chênh **19,82 điểm %**. | Cohort model ngày 105. | `previous_attempts.csv` | Cho phép ưu tiên hỗ trợ bổ sung, không gắn nhãn năng lực. |
| INS-06 | Wales Fail **29,72%** (N=2.086), South Region **17,59%** (N=3.092), chênh **12,13 điểm %**. | Toàn khóa; vùng cư trú OULAD. | `region_risk.csv`, Geographic Map | Xem thêm cơ cấu module/IMD/cohort; bản đồ không chứng minh vùng gây Fail. |
| INS-07 | Nộp trễ >7 ngày có median điểm chênh lệch **-6.77**, trong khi đúng hạn là **+0.60** và sớm 1-7 ngày là **+5.18**. Spearman = **-0.09**. | Các bài tập đã nộp. | `delay_score_buckets.csv`, bar chart | Nộp trễ >7 ngày kéo điểm xuống rõ rệt. Xu hướng không tuyến tính hoàn toàn. |
| INS-08 | Quartile ít ngày hoạt động nhất trong 28 ngày Fail **69,02%** (N=5.607), nhiều nhất **9,65%** (N=5.607), chênh **59,37 điểm %**. | Cohort model ngày 105. | `feature_snapshot.csv`, coefficient model | Theo dõi sự gián đoạn gần đây để liên hệ hỗ trợ; VLE không bao quát học offline. |

## Story dùng khi thuyết trình

1. Bắt đầu từ cơ cấu kết quả tổng thể và cho thấy học vấn đầu vào/vùng có chênh lệch.
2. Đi vào yếu tố học tập: hoàn thành bài và tham gia VLE phân biệt nhóm Fail rõ nhất.
3. Làm rõ rằng nộp muộn, gián đoạn hoạt động và lịch sử học lại là tín hiệu bổ sung.
4. Kết hợp VLE thấp với điểm thấp để xác định nhóm bất lợi cộng dồn.
5. Sau insight dữ liệu mới chuyển sang Logistic Regression: dự báo Fail ở ngày 105, kiểm tra sai số và giải thích các tín hiệu liên quan ở cấp nhóm.

Insight bổ sung từ visual tài nguyên: nhóm Qua môn/Xuất sắc có tỷ lệ tương tác các tài nguyên như `forumng`, `quiz` cao gấp **~2,3 đến 2,8 lần** nhóm Trượt. Mức độ cảnh báo của loại tài nguyên vẫn thấp hơn so với việc hoàn thành bài và cường độ tương tác tổng thể, nên đây là tín hiệu phụ hỗ trợ cho việc thiết kế can thiệp.

## Cách kiểm tra lại

```powershell
python src/eda_analysis.py
python src/dashboard_features.py
python -m unittest discover -s tests -v
```

Cả tám insight được sinh tự động trong `reports/eda/insight_evidence.csv`; các bảng thành phần nằm cùng thư mục.
