# Quy tắc làm việc

Project do **một người thực hiện chính** trên toàn bộ chuỗi dữ liệu, phân tích, model, dashboard, báo cáo và demo. Không duy trì phân công theo thành viên hoặc thủ tục bàn giao nội bộ.

## Quy trình gọn

1. Chọn một cụm việc trong [thứ tự thực hiện](docs/06-tasks-and-dependencies.md).
2. Xác định đầu vào, đầu ra, phụ thuộc và tiêu chí kiểm tra trước khi sửa.
3. Thực hiện trên nhánh làm việc; không push trực tiếp vào `main`.
4. Chạy kiểm tra tương ứng và ghi bằng chứng cạnh hiện vật.
5. Báo danh sách file, kết quả đạt/chưa đạt và phần còn thiếu.
6. Chỉ commit, push hoặc chuyển sang repo mới sau khi chủ dự án xác nhận rõ ràng.

## Nhánh

`main` là nhánh tích hợp ổn định. Dùng tiền tố theo loại việc: `data/`, `analysis/`, `model/`, `dashboard/`, `docs/` hoặc `fix/`. Không cần tách nhánh theo người thực hiện.

## Quy tắc dữ liệu và phân tích

- `data/raw/` giữ nguyên bảy CSV nguồn; ghi nguồn, phiên bản và checksum. Không sửa raw.
- `data/processed/clean_dataset.csv` phải tái tạo được bằng script. Không sửa CSV thủ công.
- Join theo đúng khóa; bảng sự kiện nhiều dòng phải aggregate về hạt lượt học trước khi ghép.
- Hạt phân tích là `(code_module, code_presentation, id_student)`; không nhầm learning attempts với distinct learners.
- Ghi rõ cửa sổ thời gian, tử số/mẫu số, missing và outlier.
- EDA phải có ít nhất 3–5 biểu đồ tĩnh và kiểm tra H01–H10.
- Chỉ chọn 5–7 insight có bằng chứng mạnh nhất; mỗi insight có RQ/H, số liệu, `N`, giới hạn và ý nghĩa hành động.
- Không suy diễn nhân quả từ dữ liệu quan sát.

## Quy tắc model và dashboard

- `Academic_Fail = 1` cho `Fail`, `0` cho `Pass/Distinction`; `Withdrawn` tách riêng. `At_Risk` chỉ là alias tương thích của `Academic_Fail`.
- Không dùng target hoặc thông tin sau cutoff làm feature dự báo sớm.
- App Streamlit chỉ đọc bảng đã aggregate và output model đã kiểm tra; không train model khi render.
- Inventory sau review là **8 loại biểu đồ không phải map + 1 Geographic Map bắt buộc riêng**, đúng mức tối thiểu rubric. Map không được dùng để bù vào nhóm biểu đồ thường.
- Geographic Map phải dùng geometry có nguồn/giấy phép, mapping đủ 13 region, tooltip và kiểm tra tổng `N`.
- Filter nhiều cấp, drill-down, tooltip và cross-filtering phải có bằng chứng hoạt động.

## Điều kiện hoàn tất một phần

Hiện vật đúng đường dẫn; số liệu tái tạo được; kiểm tra đạt; tài liệu đồng bộ với code; không có dữ liệu giả hoặc bằng chứng suy diễn. File tồn tại hoặc app khởi động được chưa tự động đồng nghĩa tiêu chí rubric đã hoàn thành.
