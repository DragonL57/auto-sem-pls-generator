# ================== CẤU HÌNH MA TRẬN TƯƠNG QUAN TIỀM ẨN ==================
#
# 1. Đây là nơi bạn kiểm soát mức độ liên hệ giữa các latent factors (nhân tố tiềm ẩn) trong mô hình.
# 2. Thứ tự các latent factors PHẢI đúng với thứ tự trong factors_config bên dưới (ví dụ: ["PI", "PA", "CONF", "PU", "SAT", "CI"])
# 3. Bạn chỉ cần nhập một ma trận vuông (dạng list lồng list), mỗi phần tử là hệ số tương quan giữa hai nhân tố (giá trị từ -1 đến 1, thường là 0.1–0.8 với dữ liệu xã hội học).
# 4. Đường chéo LUÔN là 1.0 (tự động), các giá trị ngoài đường chéo là tương quan bạn muốn đặt.
# 5. Nếu không chắc chắn, chỉ cần để latent_correlation_matrix = None, hệ thống sẽ tự tối ưu các giá trị này.
# 6. Nếu muốn dùng ma trận cụ thể, hãy thay None bằng ma trận của bạn, ví dụ:
#
# latent_correlation_matrix = [
#     [0.5, 1.0, 0.3, 0.4, 0.2, 0.1],
#     [0.4, 0.3, 1.0, 0.5, 0.3, 0.2],
#     [0.6, 0.4, 0.5, 1.0, 0.4, 0.3],
#     [0.3, 0.2, 0.3, 0.4, 1.0, 0.5],
#     [0.2, 0.1, 0.2, 0.3, 0.5, 1.0],
# ]
#
# Lưu ý: Nếu bạn thêm/bớt latent factor, phải cập nhật lại số dòng/cột cho đúng!
#
latent_correlation_matrix = [
    [1.000, 0.247, 0.315, 0.043, 0.331, 0.250, 0.152, 0.314],
    [0.247, 1.000, 0.219, 0.295, 0.350, 0.134, 0.258, 0.291],
    [0.315, 0.219, 1.000, 0.125, 0.166, 0.352, 0.166, 0.120],
    [0.043, 0.295, 0.125, 1.000, 0.248, 0.129, 0.332, 0.389],
    [0.331, 0.350, 0.166, 0.248, 1.000, 0.288, 0.221, 0.328],
    [0.250, 0.134, 0.352, 0.129, 0.288, 1.000, 0.399, 0.365],
    [0.152, 0.258, 0.166, 0.332, 0.221, 0.399, 1.000, 0.345],
    [0.314, 0.291, 0.120, 0.389, 0.328, 0.365, 0.345, 1.000]
]

# ================= HƯỚNG DẪN CHỈNH MÔ HÌNH HỒI QUY =================
# - Chỉ cần chỉnh sửa danh sách factors_config và regression_models bên dưới để thay đổi mô hình, biến độc lập/phụ thuộc, hoặc thêm biến interaction.
# - Để tạo biến interaction (tương tác), chỉ cần đặt tên biến theo dạng "A x B" (ví dụ: "PUxSAT"), hệ thống sẽ tự động tạo nếu đủ biến thành phần.
# - KHÔNG cần sửa bất kỳ file nào khác ngoài config.py, mọi thứ sẽ tự động cập nhật.
# - File kết quả sẽ xuất ra là output.xlsx trong thư mục output.
#
# ================== MÔ HÌNH: Ý ĐỊNH SỬ DỤNG XE ĐIỆN ==================
#
# 1. Các latent factors và số lượng biến quan sát:
#   - SI: 3 biến (SI1, SI2, SI3) - Ảnh hưởng xã hội
#   - GOV: 6 biến (GOV1, GOV2, GOV3, GOV4, GOV5, GOV6) - Chính phủ
#   - LCI: 3 biến (LCI1, LCI2, LCI3) - Cơ sở hạ tầng sạc
#   - PU: 3 biến (PU1, PU2, PU3) - Nhận thức hữu ích
#   - PE: 3 biến (PE1, PE2, PE3) - Nhận thức dễ sử dụng
#   - EA: 5 biến (EA1, EA2, EA3, EA4, EA5) - Môi trường
#   - PN: 4 biến (PN1, PN2, PN3, PN4) - Chuẩn mực cá nhân
#   - BI: 4 biến (BI1, BI2, BI3, BI4) - Ý định sử dụng xe điện
#
# 2. Các mối quan hệ hồi quy:
#    - 'dependent': Biến phụ thuộc (kết quả)
#    - 'independent': Danh sách các biến độc lập (nhân tố ảnh hưởng)
#    - 'order': Thứ tự mong đợi của độ mạnh ảnh hưởng từ mạnh đến yếu
#
#    Ví dụ:
#    regression_models = [
#        {"dependent": "Y", "independent": ["A", "B", "C"], "order": ["A", "B", "C"]},
#        # Trong mô hình này, A được kỳ vọng sẽ có ảnh hưởng mạnh nhất, tiếp theo là B, yếu nhất là C
#    ]

factors_config = {
    "SI":  {"original_items": ["SI1", "SI2", "SI3"]},
    "GOV": {"original_items": ["GOV1", "GOV2", "GOV3", "GOV4", "GOV5", "GOV6"]},
    "LCI": {"original_items": ["LCI1", "LCI2", "LCI3"]},
    "PU":  {"original_items": ["PU1", "PU2", "PU3"]},
    "PE":  {"original_items": ["PE1", "PE2", "PE3"]},
    "EA":  {"original_items": ["EA1", "EA2", "EA3", "EA4", "EA5"]},
    "PN":  {"original_items": ["PN1", "PN2", "PN3", "PN4"]},
    "BI":  {"original_items": ["BI1", "BI2", "BI3", "BI4"]}
}

regression_models = [
    # Mô hình 1: Môi trường (EA) tác động đến Chuẩn mực cá nhân (PN)
    # "Môi trường" là yếu tố mạnh nhất và nó tác động đến PN
    {"dependent": "PN_composite", "independent": ["EA_composite"], "order": ["EA_composite"]},

    # Mô hình 2: Các yếu tố tác động đến Ý định sử dụng xe điện (BI)
    # Thứ tự các biến độc lập được sắp xếp theo độ mạnh ảnh hưởng từ mạnh nhất đến yếu nhất:
    # "dễ sử dụng- sự hữu ích- chính phủ- cơ sở hạ tầng- ảnh hưởng xã hội- chuẩn mực cá nhân"
    {"dependent": "BI_composite",
     "independent": ["PE_composite", "PU_composite", "GOV_composite", "LCI_composite", "SI_composite", "PN_composite"],
     "order": ["PE_composite", "PU_composite", "GOV_composite", "LCI_composite", "SI_composite", "PN_composite"]}
]




n_latent_factors = len(factors_config)
latent_factor_names = list(factors_config.keys())
n_latent_cor_values = n_latent_factors * (n_latent_factors - 1) // 2



# --- Bayesian Optimization Config ---
num_observations = 367

# Bayesian Optimization parameters - Tối ưu hóa để giảm Heywood cases và tăng hiệu suất
bo_n_calls = 50                    # Số lần đánh giá (giảm để tập trung vào không gian tìm kiếm hẹp hơn)
bo_n_initial_points = 15           # Số điểm khởi tạo ngẫu nhiên (tăng để khám phá không gian tốt hơn)
bo_acq_func = 'EI'                 # Acquisition function (Expected Improvement)
bo_n_jobs = -1                     # Số processes (-1 = tất cả cores)
bo_early_stopping = True           # Bật early stopping
bo_patience = 12                   # Số iteration chờ trước khi dừng (tăng để cho phép hội tụ tốt hơn)

# Search space bounds for Bayesian Optimization - Tối ưu hóa để giảm Heywood cases
bo_latent_cor_min = 0.01           # Giá trị tối thiểu cho tương quan tiềm ẩn
bo_latent_cor_max = 0.5            # Giá trị tối đa cho tương quan tiềm ẩn (giảm mạnh để tránh Heywood)
bo_error_strength_min = 0.25       # Giá trị tối thiểu cho error strength (giảm để tăng reliability)
bo_error_strength_max = 0.45       # Giá trị tối đa cho error strength
bo_loading_strength_min = 0.55     # Giá trị tối thiểu cho loading strength (tăng để tăng factor loadings)
bo_loading_strength_max = 0.75     # Giá trị tối đa cho loading strength