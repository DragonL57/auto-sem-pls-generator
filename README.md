# Máy Tạo Dữ Liệu Tổng Hợp SEM/PLS với Bayesian Optimization

Công cụ toàn diện tạo dữ liệu tổng hợp chất lượng cao cho nghiên cứu Structural Equation Modeling (SEM) và Partial Least Squares (PLS), sử dụng thuật toán Bayesian Optimization hiện đại để tự động tối ưu hóa các tham số mô hình.

## 🌟 Tính Năng Nổi Bật

### 🤖 Tối Ưu Hóa Thông Minh
- **Bayesian Optimization** tự động tìm kiếm tham số tối ưu
- **Expected Improvement (EI)** acquisition function hiệu quả
- **Early stopping** thông minh để tiết kiệm thời gian
- **Adaptive search space** tự điều chỉnh không gian tìm kiếm

### 📊 Tạo Dữ Liệu Chất Lượng Cao
- **Likert-scale data** với phân phối chuẩn xác
- **Latent factor structure** tuân thủ ma trận tương quan mục tiêu
- **Controlled error variance** với độ lỗi tùy chỉnh
- **Realistic factor loadings** mô phỏng dữ liệu thực

### 🔍 Xác Thực Thống Kê Toàn Diện
- **Cronbach's Alpha** kiểm tra độ tin cậy
- **Exploratory Factor Analysis (EFA)** với promax rotation
- **KMO-Bartlett tests** kiểm tra tính phù hợp phân tích nhân tố
- **Regression analysis** với đầy đủ thống kê
- **Heywood cases detection** và tự động sửa chữa

### 📈 Xuất Kết Quả Chuyên Nghiệp
- **Multi-sheet Excel** với dữ liệu và phân tích
- **Real-time console output** hiển thị tiến trình
- **Comprehensive logging** cho debugging và kiểm tra
- **Automatic model validation** với chi tiết từng bước

## 📋 Yêu Cầu Hệ Thống

### Phần Cứng Tối Thiểu
- **CPU**: 4 cores trở lên (khuyến nghị 8+ cores)
- **RAM**: 8GB RAM (khuyến nghị 16GB+)
- **Storage**: 1GB không gian trống

### Phần Mềm
- **Python**: 3.8 trở lên (khuyến nghị 3.9+)
- **OS**: Windows 10+, macOS 10.15+, Linux Ubuntu 18.04+

### Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
statsmodels>=0.12.0
factor_analyzer>=0.4.0
openpyxl>=3.0.0
scikit-optimize>=0.9.0
```

## 🚀 Hướng Dẫn Cài Đặt Chi Tiết

### 1. Clone Repository
```bash
git clone https://github.com/DragonL57/auto-sem-pls-generator.git
cd auto-sem-pls-generator
```

### 2. Tạo Môi Trường Ảo (Bắt Buộc)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Cập nhật pip
pip install --upgrade pip
```

### 3. Cài Đặt Dependencies
```bash
# Cài đặt từ requirements.txt
pip install -r requirements.txt

# Xác nhận cài đặt thành công
python -c "import numpy, pandas, sklearn, statsmodels; print('All dependencies installed successfully')"
```

### 4. Cấu Hình VSCode (Khuyến Nghị)
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python",
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

## 🔧 Cấu Hình Mô Hình Chi Tiết

### 1. Cấu Hình Ma Trận Tương Quan Tiềm Ẩn

File `config.py` cho phép bạn kiểm soát mối quan hệ giữa các nhân tố tiềm ẩn:

```python
# ================== MA TRẬN TƯƠNG QUAN TIỀM ẨN ==================
latent_correlation_matrix = None  # Tự động tối ưu hóa
# HOẶC đặt ma trận cụ thể:
latent_correlation_matrix = [
    [1.000, 0.300, 0.250, 0.400, 0.350, 0.200, 0.150, 0.100],
    [0.300, 1.000, 0.350, 0.300, 0.250, 0.400, 0.300, 0.200],
    [0.250, 0.350, 1.000, 0.450, 0.300, 0.250, 0.350, 0.250],
    [0.400, 0.300, 0.450, 1.000, 0.500, 0.300, 0.200, 0.150],
    [0.350, 0.250, 0.300, 0.500, 1.000, 0.350, 0.250, 0.200],
    [0.200, 0.400, 0.250, 0.300, 0.350, 1.000, 0.600, 0.400],
    [0.150, 0.300, 0.350, 0.200, 0.250, 0.600, 1.000, 0.500],
    [0.100, 0.200, 0.250, 0.150, 0.200, 0.400, 0.500, 1.000]
]
```

**Lưu ý quan trọng:**
- Ma trận phải vuông và đối xứng
- Đường chéo luôn = 1.0 (tự tương quan)
- Giá trị ngoài đường chéo: -1 đến 1 (thường 0.1-0.8)
- Thứ tự nhân tố phải khớp với `factors_config`

### 2. Cấu Hình Nhân Tố và Biến Quan Sát

Định nghĩa các nhân tố tiềm ẩn và biến quan sát tương ứng:

```python
factors_config = {
    "SI":  {"original_items": ["SI1", "SI2", "SI3"]},                    # Ảnh hưởng xã hội
    "GOV": {"original_items": ["GOV1", "GOV2", "GOV3", "GOV4", "GOV5", "GOV6"]},  # Chính phủ
    "LCI": {"original_items": ["LCI1", "LCI2", "LCI3"]},                  # Cơ sở hạ tầng sạc
    "PU":  {"original_items": ["PU1", "PU2", "PU3"]},                     # Nhận thức hữu ích
    "PE":  {"original_items": ["PE1", "PE2", "PE3"]},                     # Nhận thức dễ sử dụng
    "EA":  {"original_items": ["EA1", "EA2", "EA3", "EA4", "EA5"]},        # Môi trường
    "PN":  {"original_items": ["PN1", "PN2", "PN3", "PN4"]},              # Chuẩn mực cá nhân
    "BI":  {"original_items": ["BI1", "BI2", "BI3", "BI4"]}               # Ý định sử dụng xe điện
}
```

### 3. Cấu Hình Mô Hình Hồi Quy

Xác định các mối quan hệ nhân quả và thứ tự ảnh hưởng mong đợi:

```python
regression_models = [
    # Mô hình 1: Môi trường tác động đến Chuẩn mực cá nhân
    {"dependent": "PN_composite", 
     "independent": ["EA_composite"], 
     "order": ["EA_composite"]},
    
    # Mô hình 2: Các yếu tố tác động đến Ý định sử dụng xe điện
    # Thứ tự: PE > PU > GOV > LCI > SI > PN (từ mạnh nhất đến yếu nhất)
    {"dependent": "BI_composite",
     "independent": ["PE_composite", "PU_composite", "GOV_composite", "LCI_composite", "SI_composite", "PN_composite"],
     "order": ["PE_composite", "PU_composite", "GOV_composite", "LCI_composite", "SI_composite", "PN_composite"]}
]
```

**Giải thích cấu trúc:**
- `dependent`: Biến kết quả (phải có `_composite` suffix)
- `independent`: Danh sách các biến độc lập ảnh hưởng
- `order`: Thứ tự mong đợi độ mạnh ảnh hưởng (từ mạnh → yếu)

### 4. Cấu Hình Tham Số Bayesian Optimization

```python
# ================== THAM SỐ BAYESIAN OPTIMIZATION ==================
num_observations = 367                    # Số mẫu cần tạo

# Thông số tối ưu hóa
bo_n_calls = 120                         # Số lần đánh giá tối đa
bo_n_initial_points = 15                 # Số điểm khám phá ban đầu
bo_acq_func = 'EI'                       # Acquisition function
bo_n_jobs = -1                          # Số processes (-1 = tất cả cores)
bo_early_stopping = True                 # Bật early stopping
bo_patience = 12                        # Số iteration chờ trước khi dừng

# Không gian tìm kiếm
bo_latent_cor_min = 0.01                 # Tương quan tiềm ẩn tối thiểu
bo_latent_cor_max = 0.5                  # Tương quan tiềm ẩn tối đa
bo_error_strength_min = 0.25             # Độ lỗi tối thiểu
bo_error_strength_max = 0.45              # Độ lỗi tối đa
bo_loading_strength_min = 0.55            # Tải nhân tố tối thiểu
bo_loading_strength_max = 0.75            # Tải nhân tố tối đa
```

## 🏃 Hướng Dẫn Sử Dụng

### 1. Chạy Chương Trình Cơ Bản
```bash
# Kích hoạt môi trường ảo
venv\Scripts\activate

# Chạy chương trình
python main.py
```

### 2. Tùy Chọn Số Processes
```bash
# Sử dụng 4 processes (khuyến nghị cho CPU 8 cores)
python main.py --processes 4

# Sử dụng 1 process (nếu gặp lỗi multiprocessing)
python main.py --processes 1
```

### 3. Giám Sát Tiến Trình
Chương trình sẽ hiển thị tiến trình real-time:
```
==================================================
BẮT ĐẦU QUÁ TRÌNH TỐI ƯU HÓA (BAYESIAN OPTIMIZATION)
Số evaluations: 120
Số điểm khởi tạo: 15
Acquisition function: EI
==================================================
Evaluation 5/120: Best score = 1850.42, Current = 1420.15
Evaluation 10/120: Best score = 1920.78, Current = 1680.45
...
```

## 📊 Kết Quả Đầu Ra Chi Tiết

### 1. File Excel (`output/output.xlsx`)

**Sheet 1: Generated Data**
- Dữ liệu tổng hợp thô cho tất cả biến quan sát
- Composite scores cho từng nhân tố
- Interaction variables (nếu có)

**Sheet 2: Statistical Analysis**
- Thống kê mô tả (mean, SD, min, max, skewness, kurtosis)
- Ma trận tương quan Pearson
- Histogram và Q-Q plots

**Sheet 3: Factor Analysis**
- Kết quả EFA với promax rotation
- Factor loadings matrix
- Communalities và uniqueness
- Factor correlation matrix

**Sheet 4: Regression Results**
- Regression coefficients và standard errors
- t-values và p-values
- R-squared, Adjusted R-squared
- VIF (Variance Inflation Factors)
- Residual analysis

**Sheet 5: Diagnostics**
- Cronbach's Alpha cho từng nhân tố
- KMO và Bartlett's test results
- Reliability statistics
- Validity measures

### 2. Console Output Real-time
```
==================================================
QUÁ TRÌNH TỐI ƯU HÓA (BAYESIAN OPTIMIZATION) HOÀN TẤT
Tổng thời gian chạy: 45.32 giây
Số evaluations thực tế: 98
==================================================
Điểm số tốt nhất tìm được: 2150.75
Lý do: Valid model with good fit indices

Bộ tham số tốt nhất:
  Độ mạnh tải nhân tố (Loading Strength): 0.685
  Độ mạnh sai số (Error Strength): 0.342
  Các giá trị tương quan tiềm ẩn: [0.245, 0.189, 0.321, ...]
```

### 3. Log File (`output/terminal.log`)
- Ghi lại toàn bộ output console
- Dùng cho debugging và kiểm tra
- UTF-8 encoding hỗ trợ tiếng Việt

## 🧠 Bayesian Optimization Chi Tiết

### Thuật To toán Tối Ưu Hóa
Hệ thống sử dụng **Gaussian Process-based Bayesian Optimization** với:

- **Surrogate Model**: Gaussian Process Regression với RBF kernel
- **Acquisition Function**: Expected Improvement (EI)
- **Search Strategy**: Tree-structured Parzen Estimator (TPE)
- **Convergence Criteria**: Early stopping với patience

### Không Gian Tìm Kiếm
| Parameter | Min | Max | Mục Đích |
|-----------|-----|-----|----------|
| Latent Correlations | 0.01 | 0.5 | Tránh Heywood cases |
| Error Strength | 0.25 | 0.45 | Tăng reliability |
| Loading Strength | 0.55 | 0.75 | Tăng convergent validity |

### Fitness Function
Đánh giá chất lượng mô hình dựa trên:

1. **Reliability Scores** (40%)
   - Cronbach's Alpha ≥ 0.7
   - Composite reliability ≥ 0.7

2. **Validity Scores** (30%)
   - Convergent validity (AVE ≥ 0.5)
   - Discriminant validity
   - Cross-loadings < 0.4

3. **Model Fit** (20%)
   - CFI ≥ 0.90
   - TLI ≥ 0.90
   - RMSEA ≤ 0.08

4. **Regression Quality** (10%)
   - R-squared ≥ 0.3
   - Significant coefficients (p < 0.05)
   - Expected beta coefficient order

## 📈 Xác Thực Thống Kê Toàn Diện

### 1. Reliability Analysis
- **Cronbach's Alpha**: Đo lường tính nhất quán nội bộ
- **Composite Reliability**: Đo lường độ tin cậy tổng hợp
- **Average Variance Extracted (AVE)**: Đo lường convergent validity

### 2. Validity Analysis
- **Exploratory Factor Analysis (EFA)**: Khám phá cấu trúc nhân tố
- **Confirmatory Factor Analysis (CFA)**: Xác nhận cấu trúc giả thuyết
- **Discriminant Validity**: Phân biệt giữa các nhân tố
- **Convergent Validity**: Tính hội tụ của các biến

### 3. Regression Analysis
- **Multiple Linear Regression**: Phân tích mối quan hệ nhân quả
- **Hierarchical Regression**: Phân tích theo cấp bậc
- **Moderation Analysis**: Phân tích hiệu quả điều tiết

### 4. Advanced Diagnostics
- **Heywood Cases Detection**: Phát hiện và sửa lỗi ma trận
- **Multicollinearity Check**: Kiểm tra đa cộng tuyến
- **Normality Tests**: Kiểm tra phân phối chuẩn
- **Outlier Detection**: Phát hiện giá trị ngoại lai

## 🔍 Khắc Phục Sự Cố Chi Tiết

### 1. Lỗi Multiprocessing
```
Error: Can't get local object 'function_name'
```
**Nguyên nhân**: Python serialization issues with multiprocessing
**Giải pháp**:
```bash
# Sử dụng single process
python main.py --processes 1

# Hoặc giảm số processes
python main.py --processes 2
```

### 2. Heywood Cases
```
Error: Heywood (Latent Diag > 1)
```
**Nguyên nhân**: Correlation values quá cao tạo ma trận không xác định dương
**Giải pháp**:
- Giảm `bo_latent_cor_max` (ví dụ: 0.5 → 0.4)
- Tăng số iterations (`bo_n_calls`)
- Kiểm tra lại factor structure

### 3. Convergence Issues
```
Error: No convergence in maximum iterations
```
**Giải pháp**:
- Tăng `bo_n_initial_points` (ví dụ: 15 → 20)
- Thay đổi `bo_acq_func` ('EI' → 'LCB')
- Tăng `bo_patience` (ví dụ: 12 → 15)

### 4. Low Fitness Scores
```
Best score: < 1000
```
**Giải pháp**:
- Mở rộng search space bounds
- Kiểm tra lại model specification
- Tăng số observations
- Điều chỉnh regression model

### 5. Encoding Issues
```
UnicodeEncodeError: 'charmap' codec
```
**Giải pháp**:
```bash
# Sử dụng terminal hỗ trợ UTF-8
chcp 65001
python main.py
```

## 📝 Cấu Trúc File Chi Tiết

```
auto-sem-pls-generator/
├── main.py                    # Entry point chính
├── config.py                 # Cấu hình toàn bộ mô hình
├── bayesian_optimizer.py     # Bayesian optimization engine
├── evaluation.py            # Fitness function và model evaluation
├── data_generation.py       # Tạo dữ liệu tổng hợp
├── diagnostics.py           # Statistical validation
├── utils.py                # Utility functions
├── metrics.py              # Statistical calculations
├── latent_utils.py         # Latent variable processing
├── requirements.txt         # Python dependencies
├── .vscode/
│   └── settings.json        # VSCode configuration
├── README.md               # Documentation
├── .gitignore              # Git ignore rules
├── venv/                   # Virtual environment
└── output/                 # Results directory
    ├── output.xlsx         # Excel results
    ├── terminal.log        # Execution log
    └── temp/               # Temporary files
```

### Mô Tả Các File Chính

**main.py**: Entry point của chương trình
- Khởi tạo Bayesian Optimization
- Quản lý tiến trình thực thi
- Xử lý output và logging

**config.py**: File cấu hình trung tâm
- Định nghĩa factor structure
- Cấu hình regression models
- Thiết lập optimization parameters

**bayesian_optimizer.py**: Core optimization engine
- Gaussian Process implementation
- Acquisition function calculations
- Search space management

**evaluation.py**: Model evaluation system
- Fitness function implementation
- Statistical validation
- Penalty score calculations

## 🎯 Mô Hình Hiện Tại: Ý Định Sử Dụng Xe Điện

### Các Nhân Tố Nghiên Cứu
1. **SI** (Social Influence): Ảnh hưởng xã hội (3 items)
2. **GOV** (Government): Chính phủ (6 items)
3. **LCI** (Charging Infrastructure): Cơ sở hạ tầng sạc (3 items)
4. **PU** (Perceived Usefulness): Nhận thức hữu ích (3 items)
5. **PE** (Perceived Ease of Use): Nhận thức dễ sử dụng (3 items)
6. **EA** (Environmental Awareness): Nhận thức môi trường (5 items)
7. **PN** (Personal Norms): Chuẩn mực cá nhân (4 items)
8. **BI** (Behavioral Intention): Ý định sử dụng xe điện (4 items)

### Mô Hình Hồi Quy
- **Model 1**: EA → PN (Môi trường → Chuẩn mực cá nhân)
- **Model 2**: [PE, PU, GOV, LCI, SI, PN] → BI (Thứ tự strength: PE > PU > GOV > LCI > SI > PN)

### Thứ Tự Ảnh Hưởng Mong Đợi
1. **EA → PN** (Mạnh nhất)
2. **PE → BI** (Mạnh thứ 2)
3. **PU → BI** (Mạnh thứ 3)
4. **GOV → BI** (Mạnh thứ 4)
5. **LCI → BI** (Mạnh thứ 5)
6. **SI → BI** (Mạnh thứ 6)
7. **PN → BI** (Yếu nhất)

## 🚀 Tối Ưu Hóa Hiệu Suất

### Tối Ưu Cho Máy Nhiều Cores
```bash
# CPU 16 cores → sử dụng 15 processes
python main.py --processes 15

# CPU 8 cores → sử dụng 7 processes
python main.py --processes 7

# CPU 4 cores → sử dụng 3 processes
python main.py --processes 3
```

### Tối Ưu Thời Gian Chạy
- **Fast mode**: `bo_n_calls = 60`, `bo_n_initial_points = 10`
- **Normal mode**: `bo_n_calls = 120`, `bo_n_initial_points = 15`
- **Thorough mode**: `bo_n_calls = 200`, `bo_n_initial_points = 25`

### Tối Ưu Chất Lượng
- **Quality mode**: Giảm search space, tăng iterations
- **Exploration mode**: Mở rộng search space, tăng initial points
- **Balanced mode**: Cân bằng giữa exploration và exploitation

## 📊 Diễn Giải Kết Quả

### 1. Đọc Kết Quả Regression
- **R-squared**: > 0.3 (acceptable), > 0.5 (good), > 0.7 (excellent)
- **Beta coefficients**: Giá trị dương/negative phù hợp giả thuyết
- **p-values**: < 0.05 (significant), < 0.01 (highly significant)
- **VIF**: < 5 (acceptable), < 3 (good)

### 2. Đánh Giá Model Fit
- **Cronbach's Alpha**: > 0.7 (acceptable), > 0.8 (good)
- **KMO**: > 0.6 (acceptable), > 0.8 (good)
- **Factor Loadings**: > 0.5 (acceptable), > 0.7 (good)
- **AVE**: > 0.5 (acceptable)

### 3. Xác Thực Hypotheses
- **Hypothesis supported**: p < 0.05 và beta đúng dấu
- **Hypothesis rejected**: p ≥ 0.05 hoặc beta sai dấu
- **Effect size**: Small (0.1), Medium (0.3), Large (0.5)

## 🤝 Đóng Góp và Phát Triển

### Cách Đóng Góp
1. **Fork** repository
2. Tạo **feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push**: `git push origin feature/amazing-feature`
5. **Pull Request**: Mở PR trên GitHub

### Quy Trình Phát Triển
- Tuân thủ PEP 8 cho coding style
- Thêm tests cho các functions mới
- Cập nhật documentation khi thay đổi
- Review code trước khi merge

### Areas for Improvement
- Thêm acquisition functions mới
- Hỗ trợ các loại dữ liệu khác (ordinal, nominal)
- Thêm visualization tools
- Tăng tốc độ convergence

## 📄 Giấy Phép và Sử Dụng

### Giấy Phép
This project is licensed under the MIT License - see the LICENSE file for details.

### Mục Đích Sử Dụng
- ✅ Academic research
- ✅ Educational purposes
- ✅ Methodological development
- ❌ Commercial use without permission
- ❌ Medical/clinical applications

### Citation
Nếu sử dụng công cụ này trong nghiên cứu, vui lòng citation:
```
Auto SEM/PLS Data Generator (Version 1.0)
https://github.com/DragonL57/auto-sem-pls-generator
```

## 🔗 Liên Kết Hữu Ích

### Documentation
- **Official Documentation**: [Link]
- **API Reference**: [Link]
- **Tutorial Videos**: [Link]

### Community
- **GitHub Issues**: https://github.com/DragonL57/auto-sem-pls-generator/issues
- **Discussions**: https://github.com/DragonL57/auto-sem-pls-generator/discussions
- **Email Support**: [Contact]

### Related Tools
- **R semTools Package**: https://cran.r-project.org/package=semTools
- **Python semopy Package**: https://github.com/georgy-seledtskov/semopy
- **Lavaan (R)**: https://lavaan.ugent.be/

---

**Note**: Công cụ này được phát triển cho mục đích học thuật và nghiên cứu. Vui lòng tham khảo tài liệu SEM/PLS phù hợp khi sử dụng kết quả trong các công bố khoa học.

**Last Updated**: September 2025
**Version**: 1.0.0
**Maintainer**: DragonL57