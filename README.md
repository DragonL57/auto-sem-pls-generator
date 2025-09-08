# Tự động hóa sinh dữ liệu và kiểm định mô hình SEM/PLS (Python)

## Mục đích

- Tự động sinh dữ liệu Likert cho các mô hình SEM/PLS/phân tích nhân tố dựa trên cấu hình linh hoạt.
- Tự động kiểm định các tiêu chí: Cronbach's Alpha, EFA, hồi quy, checklist xuất báo cáo.
- Chỉ cần chỉnh file `config.py` cho mọi thay đổi mô hình, không cần sửa code.
- Hỗ trợ tự động cập nhật ma trận tương quan tiềm ẩn tốt nhất sau mỗi lần chạy.

## Cấu trúc thư mục

```
auto/
├── main.py                # Chạy toàn bộ pipeline tối ưu hóa, sinh dữ liệu, kiểm định, xuất báo cáo
├── config.py              # File cấu hình duy nhất cần chỉnh cho mọi mô hình
├── evaluation.py          # Hàm đánh giá, sinh dữ liệu, kiểm định các tiêu chí
├── diagnostics.py         # Xuất báo cáo kiểm định, checklist, EFA, hồi quy
├── data_generation.py     # Hàm sinh dữ liệu Likert từ latent factors
├── metrics.py             # Các hàm tính toán chỉ số (Cronbach, ...)
├── utils.py               # Hàm phụ trợ (ma trận xác định dương, ...)
├── genetic_algorithm.py   # GA tối ưu hóa tham số sinh dữ liệu
├── output/
│   └── output.xlsx        # File dữ liệu giả mạo cuối cùng
│   └── terminal.log       # Log chi tiết quá trình chạy
```

## Hướng dẫn sử dụng

1. **Chỉnh mô hình trong `config.py`**
   - Khai báo các latent factors và số lượng biến quan sát trong `factors_config`.
   - Khai báo các mô hình hồi quy, thứ tự ảnh hưởng trong `regression_models`.
   - (Tùy chọn) Đặt ma trận tương quan tiềm ẩn khởi đầu ở `latent_correlation_matrix`.
2. **Chạy pipeline**
   - Kích hoạt môi trường ảo (nếu có):
     ```
     .\venv\Scripts\activate
     ```
   - Chạy script:
     ```
     python auto/main.py
     ```
3. **Xem kết quả**
   - Dữ liệu giả mạo và các chỉ số kiểm định sẽ được xuất ra `auto/output/output.xlsx` và log chi tiết ở `auto/output/terminal.log`.
   - Ma trận tương quan tiềm ẩn tốt nhất sẽ được tự động cập nhật vào `config.py` cho lần chạy sau.

## Tùy biến mô hình
- **Thay đổi số lượng latent factor, số biến quan sát, hoặc cấu trúc mô hình:**
  - Chỉ cần sửa `factors_config` và `regression_models` trong `config.py`.
- **Thêm biến interaction:**
  - Chỉ cần thêm tên biến dạng "A x B" vào regression_models, hệ thống sẽ tự động tạo biến tương tác nếu đủ thành phần.
- **Kiểm soát ma trận tương quan tiềm ẩn:**
  - Đặt biến `latent_correlation_matrix` bằng ma trận mong muốn, hoặc để None để hệ thống tự tối ưu.

## Các kiểm định tự động
- Cronbach's Alpha cho từng nhân tố
- EFA (Promax), tự động phát hiện item xoay sai nhóm
- Hồi quy tuyến tính, kiểm tra thứ tự ảnh hưởng, standardized beta
- Checklist kiểm định xuất báo cáo tự động

## Yêu cầu thư viện
- numpy, pandas, scikit-learn, statsmodels, factor_analyzer

Cài đặt nhanh:
```
pip install -r requirements.txt
```

## Ghi chú
- Mọi thay đổi mô hình chỉ cần chỉnh `config.py`, không cần sửa code.
- Nếu gặp lỗi về ma trận tương quan tiềm ẩn, hãy kiểm tra lại số dòng/cột và giá trị trong `latent_correlation_matrix`.
- Log chi tiết quá trình chạy luôn được lưu ở `auto/output/terminal.log`.

---

**Tác giả:** Võ Mai Thế Long

Mọi thắc mắc, góp ý vui lòng liên hệ tác giả hoặc để lại issue trên repo.
