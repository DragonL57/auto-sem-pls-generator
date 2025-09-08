# Công Cụ Tạo Dữ Liệu Giả Mạo SEM/PLS

Hệ thống toàn diện để tạo dữ liệu khảo sát giả mạo cho Phân tích Mô hình Cấu trúc (SEM) và Bình phương Cực tiểu (PLS), tích hợp tối ưu hóa thuật toán di truyền và xác thực thống kê mạnh mẽ.

## 🚀 Tính Năng Mới Trong Phiên Bản Cải Tiến

### ✅ Cải Tiến Chính

1. **Kiến trúc Module**: Tách biệt rõ ràng các chức năng
2. **Xử lý lỗi mạnh mẽ**: Xử lý ngoại lệ toàn diện với các loại lỗi tùy chỉnh
3. **Xác thực nâng cao**: Xác thực thống kê đa cấp độ với báo cáo chi tiết
4. **Xuất báo cáo chuyên nghiệp**: Báo cáo Excel định dạng đẹp với nhiều sheet
5. **An toàn kiểu dữ liệu**: Gợi ý kiểu đầy đủ trong toàn bộ mã nguồn
6. **Giao diện CLI**: Giao diện dòng lệnh với nhiều tùy chọn
7. **Quản lý cấu hình**: Xác thực cấu hình dựa trên Pydantic
8. **Tối ưu hóa hiệu suất**: Thuật toán hiệu quả và xử lý song song

## 📁 Cấu Trúc Dự Án

```
auto/
├── src/
│   ├── core/                    # Module chính
│   │   ├── data_generator.py    # Bộ điều phối chính
│   │   ├── config_manager.py    # Quản lý cấu hình
│   │   └── exceptions.py        # Ngoại lệ tùy chỉnh
│   ├── optimization/            # Module tối ưu hóa
│   │   ├── genetic_optimizer.py # Tối ưu hóa GA
│   │   └── genetic_algorithm.py # Cài đặt GA
│   ├── validation/              # Module xác thực
│   │   ├── data_validator.py    # Trình xác thực chính
│   │   └── statistical_validator.py # Xác thực thống kê
│   ├── export/                  # Module xuất
│   │   └── results_exporter.py  # Xuất Excel/JSON
│   └── utils/                   # Module tiện ích
│       ├── data_generator_utils.py # Tiện ích tạo dữ liệu
│       └── math_utils.py        # Tiện ích toán học
├── main_new.py                  # Điểm vào CLI mới
├── main.py                      # Phiên bản gốc (tương thích ngược)
├── config.py                    # File cấu hình
├── requirements.txt             # Dependencies
└── README.md                    # Tài liệu này
```

## 🛠️ Cài Đặt

1. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

2. **Chạy phiên bản cải tiến**:
```bash
python main_new.py --help
```

## 🎯 Sử Dụng

### Cơ Bản

```bash
# Chạy pipeline đầy đủ với cài đặt mặc định
python main_new.py

# Sử dụng file cấu hình tùy chỉnh
python main_new.py --config config_cua_ban.py

# Chỉ định thư mục đầu ra
python main_new.py --output ./ket_qua

# Sử dụng nhiều tiến trình để tối ưu hóa
python main_new.py --processes 4

# Chỉ chạy xác thực
python main_new.py --validation-only

# In tóm tắt và thoát
python main_new.py --summary
```

### Tùy Chọn CLI

| Tùy Chọn | Mô Tả | Mặc Định |
|----------|--------|----------|
| `--config, -c` | Đường dẫn file cấu hình | `config.py` |
| `--output, -o` | Thư mục đầu ra | `auto/output` |
| `--processes, -p` | Số tiến trình | Số CPU - 1 |
| `--log-level, -l` | Mức độ logging | `INFO` |
| `--validation-only, -v` | Chỉ chạy xác thực | False |
| `--summary, -s` | In tóm tắt và thoát | False |

### Sử Dụng Chương Trình

```python
from src.core.data_generator import SEMDataGenerator

# Tải cấu hình
config_dict = load_config_from_old_format('config.py')

# Khởi tạo generator
generator = SEMDataGenerator(config_dict, './output')

# Chạy pipeline đầy đủ
results = generator.run_full_pipeline()

# Truy cập kết quả
print(f"Điểm tốt nhất: {results['optimization']['best_score']}")
print(f"Kích thước dữ liệu: {results['generated_data'].shape}")
print(f"Đường dẫn xuất: {results['export_path']}")
```

## 🔧 Cấu Hình

Phiên bản cải tiến duy trì tương thích ngược với định dạng `config.py` cũ trong khi thêm xác thực và tính năng mới:

### Định Dạng Cũ (Vẫn Hỗ Trợ)

```python
factors_config = {
    "PI": {"original_items": ["PI1", "PI2", "PI3", "PI4", "PI5"]},
    "PA": {"original_items": ["PA1", "PA2", "PA3", "PA4", "PA5"]},
    # ... thêm factors
}

regression_models = [
    {"dependent": "PA_composite", "independent": ["PI_composite"], "order": ["PI_composite"]},
    # ... thêm models
]
```

### Tính Năng Mới

- **Xác thực Cấu hình**: Xác thực tự động các tham số cấu hình
- **Giới hạn Tham Số**: Định nghĩa giới hạn tham số linh hoạt
- **Cấu hình GA**: Quản lý tham số thuật toán di truyền riêng biệt
- **Xử lý Lỗi**: Xử lý lỗi cấu hình một cách duyên dáng

## 📊 Tính Năng Xác Thực

### Xác Thực Thống Kê

- **Cronbach's Alpha**: Phân tích độ tin cậy cho mỗi factor
- **Phân tích Factor**: EFA với xoay Promax
- **Kiểm tra KMO & Bartlett**: Đánh giá factorability
- **Phân tích Cross-loading**: Xác thực cấu trúc factor
- **Xác thực Hồi quy**: Kiểm tra fit model và ý nghĩa thống kê

### Kiểm Tra Chất Lượng Dữ Liệu

- **Độ Lớp Mẫu Đủ**: Yêu cầu kích thước mẫu tối thiểu
- **Phân tích Dữ Liệu Thiếu**: Đánh giá giá trị thiếu
- **Phát hiện Outlier**: Xác định outlier dựa trên IQR
- **Xác thực Likert Scale**: Kiểm tra khoảng và phân phối
- **Phân tích Phương Sai**: Phát hiện phương sai đủ

## 📈 Tính Năng Xuất

### Xuất Excel

Phiên bản cải tiến tạo báo cáo Excel toàn diện với:

- **Sheet Dữ Liệu**: Dữ liệu giả mạo thô với định dạng phù hợp
- **Sheet Kết Quả Xác Thực**: Kết quả xác thực thống kê
- **Sheet Cấu Hình**: Tóm tắt cấu hình model
- **Sheet Tham Số Tối Ưu**: Kết quả tối ưu hóa GA
- **Sheet Tóm Tắt**: Tổng quan phân tích và thống kê

### Định Dạng Xuất Khác

- **Xuất JSON**: Định dạng kết quả có thể đọc được bằng máy
- **Báo Cáo Xác Thực**: Báo cáo văn bản chi tiết

## 🎮 Tính Năng Tối Ưu Hóa

### Cải Tiện Thuật Toán Di Truyền

- **Tỷ lệ Mutation Thích ứng**: Điều chỉnh mutation động
- **Lựa chọn Tournament**: Lựa chọn cha mẹ mạnh mẽ
- **Bảo toàn Elitism**: Các cá thể tốt nhất được giữ lại
- **Xử lý Song Song**: Tối ưu hóa đa tiến trình
- **Phát hiện Hội Tụ**: Giám sát sự trì trệ

### Tối Ưu Hóa Hiệu Suất

- **Hoạt động Vector hóa**: Tính toán số hiệu quả
- **Quản lý Bộ Nhớ**: Cấu trúc dữ liệu được tối ưu hóa
- **Bộ nhớ đệm**: Lưu trữ chiến lược các hoạt động tốn kém
- **Xử lý Song Song**: Sử dụng đa lõi

## 🔍 Xử Lý Lỗi

### Ngoại Lệ Tùy Chỉnh

- `SEMDataGenerationError`: Lớp ngoại lệ cơ sở
- `ConfigurationError`: Lỗi liên quan đến cấu hình
- `OptimizationError`: Lỗi tối ưu hóa
- `ValidationError`: Vấn đề xác thực
- `DataGenerationError`: Vấn đề tạo dữ liệu
- `ExportError`: Lỗi xuất

### Suy thoái Duyên Dáng

- **Lỗi không nghiêm trọng**: Tiếp tục xử lý với cảnh báo
- **Phục hồi Lỗi**: Phục hồi tự động từ các vấn đề tạm thời
- **Logging Chi Tiết**: Ghi nhật ký lỗi toàn diện
- **Thông báo Thân thiện với Người dùng**: Mô tả lỗi rõ ràng

## 🧪 Kiểm Thử

### Ví Dụ Xác Thực

```python
# Kiểm tra tạo dữ liệu
data = generator.generate_data(parameters)

# Xác thực dữ liệu
validation_results = generator.validate_data(data)

# Kiểm tra tính hợp lệ tổng thể
if validation_results['overall_validity']:
    print("Xác thực dữ liệu thành công!")
else:
    print("Xác thực dữ liệu thất bại")
    print("Vấn đề:", validation_results['errors'])
```

## 📚 Tham Khảo API

### Các Lớp Chính

- `SEMDataGenerator`: Lớp điều phối chính
- `ConfigManager`: Quản lý và xác thực cấu hình
- `GeneticOptimizer`: Tối ưu hóa thuật toán di truyền
- `DataValidator`: Xác thực dữ liệu toàn diện
- `ResultsExporter`: Chức năng xuất đa định dạng

### Hàm Tiện Ích

- `generate_items_from_latent()`: Tạo mục Likert từ factor tiềm ẩn
- `nearest_positive_definite()`: Tính xác định dương của ma trận
- `create_latent_correlation_matrix()`: Tạo ma trận tương quan

## 🔄 Di Chuyển Từ Phiên Bản Gốc

### Đối với Người Dùng

1. **Không cần thay đổi**: File `config.py` hiện có của bạn vẫn hoạt động
2. **CLI mới**: Sử dụng `main_new.py` thay vì `main.py`
3. **Tính năng nâng cao**: Truy cập tính năng xác thực và xuất mới
4. **Xử lý lỗi tốt hơn**: Thông báo lỗi mô tả hơn

### Đối với Nhà Phát Triển

1. **Cấu trúc Module**: Dễ mở rộng và bảo trì
2. **Gợi ý Kiểu**: Hỗ trợ IDE tốt hơn và rõ ràng mã hơn
3. **Tài liệu Toàn diện**: Docstring chi tiết và ví dụ
4. **Kiến trúc Kiểm thử**: Thiết kế module tạo điều kiện kiểm thử

## 🎯 So Sánh Hiệu Suất

| Tính Năng | Phiên Bản Gốc | Phiên Bản Cải Tiến | Cải Tiện |
|-----------|---------------|-------------------|----------|
| Cấu trúc Mã | Đơn tầng | Module | ✅ Khả năng Bảo trì |
| Xử lý Lỗi | Cơ bản | Toàn diện | ✅ Độ tin cậy |
| Xác thực | Hạn chế | Nâng cao | ✅ Độ chính xác |
| Xuất | Cơ bản | Chuyên nghiệp | ✅ Khả năng Sử dụng |
| Hiệu Suất | Đơn luồng | Đa tiến trình | ✅ Tốc độ |
| Tài liệu | Tối thiểu | Toàn diện | ✅ Trải nghiệm Nhà phát Triển |

## 🚀 Cải Tiện Tương Lai

Kế hoạch cải tiến cho các phiên bản tương lai:

- **Giao diện GUI**: Giao diện người dùng dựa trên web
- **Model Nâng cao**: Hỗ trợ các model SEM phức tạp
- **Xác thực Thời gian Thực**: Phản hồi xác thực trực tiếp
- **Tích hợp Đám mây**: Xử lý dựa trên đám mây
- **Phân tích Nâng cao**: Phân tích thống kê tinh vi hơn
- **Tích hợp Cơ sở dữ liệu**: Kết nối cơ sở dữ liệu trực tiếp

## 🤝 Đóng Góp

Phiên bản cải tiến được thiết kế để có thể mở rộng và dễ bảo trì. Các lĩnh vực đóng góp chính:

- **Phương pháp Xác thực Mới**: Các kiểm tra thống kê bổ sung
- **Định dạng Xuất**: Hỗ trợ nhiều định dạng file hơn
- **Thuật toán Tối ưu hóa**: Phương pháp tối ưu hóa thay thế
- **Cải tiến Giao diện Người dùng**: Cải tiến GUI hoặc giao diện web
- **Tài liệu**: Tài liệu và ví dụ được nâng cao
- **Tính năng Mới**: Tích hợp mô hình nâng cao, phân tích thời gian thực

## 📄 Giấy Phép

Dự án này tiếp tục dưới cùng giấy phép với phiên bản gốc.

## 🙏 Lời Tri Ân

- **Tác giả gốc**: Đóng góp nền tảng cho việc tạo dữ liệu SEM/PLS
- **Cộng đồng Thống kê**: Cho các phương pháp xác thực và thông lệ tốt nhất
- **Người đóng góp Mã nguồn Mở**: Cho các thư viện và công cụ làm nên dự án này

---

**Lưu ý**: Phiên bản cải tiến này duy trì tương thích ngược đầy đủ trong khi cung cấp cải tiến đáng kể về chất lượng mã, tính năng và trải nghiệm người dùng.

**Tác giả**: Võ Mai Thế Long  
**Email**: vo.maithelong@gmail.com  
**GitHub**: DragonL57