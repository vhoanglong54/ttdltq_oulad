# 06 — Thứ tự thực hiện và quan hệ phụ thuộc

Project do leader thực hiện chính. Tài liệu này quản lý cổng kỹ thuật, không chia việc theo thành viên hoặc tuần.

| Bước | Đầu ra | Phụ thuộc | Trạng thái local |
|---:|---|---|---|
| 1 | Nguồn, raw checksum, data dictionary | — | Đã có |
| 2 | Audit/clean/join, `clean_dataset.csv` | 1 | Đã cập nhật target mới; cần chạy lại pipeline cuối |
| 3 | RQ/Hypothesis/plan | 1 | Đã chốt trong `plan_approved.md` |
| 4 | EDA, 8 insight, 10 hypothesis | 2–3 | Đã sinh 15 bảng + 6 hình |
| 5 | Logistic Regression checkpoint-safe | 2 | Day 105 v5, 11/11 verification PASS |
| 6 | Dashboard bốn trang + map | 4–5 | Đã sửa theo target Fail; AppTest PASS |
| 7 | QA tích hợp | 4–6 | Unit test cần chạy lại sau thay đổi cuối |
| 8 | DOCX/báo cáo/slide/demo | 7 | Chưa hoàn tất |
| 9 | Leader duyệt rồi commit repo mới | 7–8 | Đã hoàn thành bản nền; thay đổi tiếp theo vẫn chờ duyệt |

## Điểm chặn bắt buộc

- Không sửa rubric.
- Không gọi day 105 là cảnh báo sớm.
- Không gộp Fail với Withdrawn.
- Không nghiệm thu model nếu verification không PASS hoặc dashboard dùng sai artifact/version/threshold.
- Không nghiệm thu insight nếu thiếu claim, số liệu, `N`, phạm vi và giới hạn.
- Không nghiệm thu map nếu mapping không đủ 13 region.
- Không commit/push thay đổi mới trước khi leader duyệt.
