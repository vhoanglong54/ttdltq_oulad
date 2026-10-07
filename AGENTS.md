# Quy tắc làm việc trong repo

- Đọc `README.md`, `docs/02-rubric-traceability.md`, `docs/03-data-plan.md` và tài liệu của phần đang sửa trước khi làm việc.
- Ghi đúng trạng thái thực tế; không viết rằng insight, mô hình hay dashboard đã hoàn thành khi chưa có hiện vật và bằng chứng kiểm tra.
- DOCX gốc là yêu cầu; OULAD là nguồn quyết định các cột thực tế. Không tạo dữ liệu hoặc cột giả để làm đủ rubric. Ghi chênh lệch tại `docs/08-decisions-and-open-questions.md`.
- Không sửa barem, điểm số hoặc tiêu chí chấm trong `docs/02-rubric-traceability.md` và **PHẦN II. RUBRIC ĐỒ ÁN CUỐI KỲ** của DOCX.
- Công nghệ dashboard duy nhất là **Python với Streamlit + Plotly**. Pandas/NumPy xử lý dữ liệu; Matplotlib/Seaborn phục vụ EDA tĩnh; scikit-learn huấn luyện Logistic Regression; Streamlit + Plotly trình bày dashboard tương tác và output model đã kiểm tra.
- Không commit hoặc push nếu chủ dự án chưa cho phép rõ ràng. Mọi thay đổi phải được báo cáo để duyệt trước.
- Không commit 7 CSV gốc, dữ liệu trung gian, secrets, thông tin định danh ngoài OULAD hoặc notebook có output nặng. Chỉ `data/processed/clean_dataset.csv` được theo dõi làm nguồn processed chuẩn; phải giữ script tái tạo và checksum.
- Nhãn học thuật chính là `Academic_Fail`: `Fail = 1`, `Pass/Distinction = 0`; loại `Withdrawn` khỏi cohort model và mô tả riêng. `At_Risk` chỉ là alias tương thích của `Academic_Fail`. Không dùng nhãn hoặc thông tin xảy ra sau mốc dự báo làm feature.
- Hạt dữ liệu là **một lượt học theo `(code_module, code_presentation, id_student)`**; không đồng nhất số lượt học với số sinh viên duy nhất.
- Insight là mối liên hệ trong dữ liệu quan sát; không viết quan hệ nhân quả nếu chưa có thiết kế chứng minh. Mỗi insight phải có RQ/H liên quan, số liệu, mẫu số, filter context, cỡ mẫu, giới hạn và đề xuất hành động.
- Dashboard chỉ đọc bảng đã aggregate/output model đã kiểm tra. KPI phải có công thức và baseline Python để đối chiếu. Map cần geometry/mapping có nguồn, không tự chế tọa độ hoặc polygon.
- Inventory sau review: 8 loại biểu đồ không phải map và một Geographic Map bắt buộc riêng. Không đổi chart chỉ để đủ số lượng; mọi thay đổi phải giữ liên kết RQ/insight và filter, drill-down, tooltip, cross-filtering.
- Chỉ đánh dấu checklist hoàn thành khi có đường dẫn hiện vật, bằng chứng kiểm tra và chủ dự án xác nhận. Không sửa file rubric.
