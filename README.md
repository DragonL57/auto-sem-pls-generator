# Máy Tạo Dữ Liệu Tổng Hợp SEM/PLS

Công cụ tạo dữ liệu tổng hợp cho nghiên cứu Structural Equation Modeling (SEM) và Partial Least Squares (PLS) sử dụng Bayesian Optimization để tối ưu hóa các tham số.

## 🌟 Tính Năng

- **Tự động tối ưu hóa** tham số mô hình sử dụng Bayesian Optimization
- **Tạo dữ liệu Likert-scale** chất lượng cao với cấu trúc nhân tố tiềm ẩn
- **Xác thực thống kê** đầy đủ (Cronbach's Alpha, EFA, KMO-Bartlett, hồi quy)
- **Xuất kết quả** sang file Excel với nhiều sheet phân tích
- **Hỗ trợ biến tương tác** (interaction variables) tự động
- **Cập nhật tự động** ma trận tương quan tiềm ẩn tối ưu

## 📋 Yêu Cầu Hệ Thống

- Python 3.8+
- Các thư viện Python (xem `requirements.txt`)

## 🚀 Cài Đặt

### 1. Clone repository
```bash
git clone https://github.com/DragonL57/auto-sem-pls-generator.git
cd auto-sem-pls-generator
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 🔧 Cấu Hình

Chỉnh sửa file `config.py` để thiết lập mô hình nghiên cứu:

### 1. Cấu hình ma trận tương quan tiềm ẩn
```python
latent_correlation_matrix = [
    [1.000, 0.253, 0.629, 0.572, 0.625, 0.567],
    [0.253, 1.000, 0.436, 0.435, 0.313, 0.302],
    [0.629, 0.436, 1.000, 0.529, 0.526, 0.526],
    [0.572, 0.435, 0.529, 1.000, 0.727, 0.626],
    [0.625, 0.313, 0.526, 0.727, 1.000, 0.535],
    [0.567, 0.302, 0.526, 0.626, 0.535, 1.000]
]
```

### 2. Cấu hình nhân tố và biến quan sát
```python
factors_config = {
    "PI":   {"original_items": ["PI1", "PI2", "PI3", "PI4", "PI5"]},
    "PA":   {"original_items": ["PA1", "PA2", "PA3", "PA4", "PA5"]},
    "CONF": {"original_items": ["CONF1", "CONF2", "CONF3", "CONF4"]},
    "PU":   {"original_items": ["PU1", "PU2", "PU3", "PU4"]},
    "SAT":  {"original_items": ["SAT1", "SAT2", "SAT3", "SAT4"]},
    "CI":   {"original_items": ["CI1", "CI2"]}
}
```

### 3. Cấu hình mô hình hồi quy
```python
regression_models = [
    {"dependent": "PA_composite", "independent": ["PI_composite"]},
    {"dependent": "CONF_composite", "independent": ["PI_composite", "PA_composite"]},
    {"dependent": "PU_composite", "independent": ["PI_composite", "PA_composite", "CONF_composite"]},
    {"dependent": "SAT_composite", "independent": ["PU_composite", "CONF_composite"]},
    {"dependent": "CI_composite", "independent": ["PU_composite", "SAT_composite"]}
]
```

### 4. Tham số Bayesian Optimization
```python
num_observations = 367     # Số quan sát
# Tham số tối ưu được tự động điều chỉnh
```

## 🏃 Chạy Chương Trình

### Chạy chương trình
```bash
python main.py
```

**Lưu ý**: Chương trình tự động sử dụng (số lõi CPU - 1) processes để tối ưu hóa hiệu suất:
- CPU 8 lõi → 7 processes
- CPU 4 lõi → 3 processes
- Luôn giữ lại 1 lõi cho hệ thống

Nếu gặp lỗi multiprocessing, chương trình sẽ tự động giảm số processes.

## 📊 Kết Quả Đầu Ra

Sau khi chạy xong, chương trình sẽ tạo:

1. **File Excel**: `output/output.xlsx`
   - Generated Data: Dữ liệu tổng hợp thô
   - Statistical Analysis: Thống kê mô tả, ma trận tương quan
   - Factor Analysis: Kết quả EFA, tải nhân tố
   - Regression Results: Kết quả hồi quy
   - Diagnostics: Cronbach's Alpha, KMO-Bartlett

2. **Console Output**: Hiển thị tiến trình tối ưu hóa
   - Điểm số từng iteration
   - Thông số tốt nhất tìm được
   - Kết quả xác thực thống kê

3. **Log File**: `output/terminal.log`
   - Ghi lại toàn bộ output của chương trình

## 🧠 Bayesian Optimization

Hệ thống sử dụng Bayesian Optimization để tối ưu hóa:

- **Parameters**: 
  - Loading strength (0.45-0.65)
  - Error strength (0.35-0.55)
  - Latent correlations (0.01-0.7)

- **Fitness Function**: 
  - Cronbach's Alpha (mục tiêu: 0.7-0.9)
  - Factor structure quality
  - Correlation matrix validity
  - Regression model fit

- **Optimization Features**:
  - Expected Improvement (EI) acquisition function
  - Early stopping để tối ưu hiệu suất
  - Tự động điều chỉnh không gian tìm kiếm

## 📈 Xác Thực Thống Kê

Chương trình tự động thực hiện:

### 1. Độ tin cậy (Reliability)
- **Cronbach's Alpha** cho từng nhân tố
- Mục tiêu: α ≥ 0.7

### 2. Tính hiệu lực (Validity)
- **Exploratory Factor Analysis (EFA)**
- **KMO Test** (mục tiêu: ≥ 0.6)
- **Bartlett's Test** (p < 0.05)

### 3. Mô hình hồi quy
- **R-squared** và **Adjusted R-squared**
- **p-values** cho các hệ số hồi quy
- **VIF** kiểm tra đa cộng tuyến

## 🔍 Khắc Phục Sự Cố

### Lỗi Multiprocessing
```
Error: Can't get local object
```
**Giải pháp**: Sử dụng ít processes hơn
```bash
python main.py --processes 1
```

### Lỗi Heywood Cases
```
Error: Heywood (Latent Diag > 1)
```
**Giải pháp**: Tăng số iterations hoặc điều chỉnh bounds

### Lỗi Encoding
```
UnicodeEncodeError: 'charmap' codec
```
**Giải pháp**: Chạy trong terminal hỗ trợ UTF-8

## 📝 Cấu Trúc File

```
auto-sem-pls-generator/
├── main.py                 # File chạy chính
├── config.py              # Cấu hình mô hình
├── bayesian_optimizer.py   # Bayesian optimization
├── evaluation.py          # Đánh giá fitness
├── data_generation.py     # Tạo dữ liệu
├── diagnostics.py         # Xác thực thống kê
├── utils.py              # Hàm tiện ích
├── metrics.py            # Tính toán metrics
├── latent_utils.py       # Xử lý biến tiềm ẩn
├── requirements.txt       # Dependencies
├── README.md             # Hướng dẫn sử dụng
└── output/               # Thư mục kết quả
    ├── output.xlsx       # File Excel kết quả
    └── terminal.log      # Log file
```

## 🎯 Mô Hình Mẫu

Dựa trên nghiên cứu về **ý định tiếp tục sử dụng ứng dụng ngân hàng AI**:

- **PI** (Perceived Intelligence): Nhận thức về trí tuệ
- **PA** (Perceived Anthropomorphism): Nhân cách hóa
- **CONF** (Confirmation): Xác nhận kỳ vọng
- **PU** (Perceived Usefulness): Tính hữu ích perceived
- **SAT** (Satisfaction): Sự hài lòng
- **CI** (Continuance Intention): Ý định tiếp tục sử dụng

## 🤝 Đóng Góp

Mọi đóng góp và cải tiến đều được chào đón!

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 Giấy Phép

This project is for educational and research purposes.

## 🔗 Liên Kết

- **GitHub**: https://github.com/DragonL57/auto-sem-pls-generator
- **Issues**: https://github.com/DragonL57/auto-sem-pls-generator/issues

---

**Note**: Công cụ này dành cho mục đích học thuật và nghiên cứu. Vui lòng tham khảo tài liệu SEM/PLS phù hợp khi sử dụng kết quả.