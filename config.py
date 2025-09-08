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
    [1.000, 0.230, 0.204, 0.464, 0.366, 0.170],
    [0.230, 1.000, 0.290, 0.265, 0.481, 0.424],
    [0.204, 0.290, 1.000, 0.376, 0.274, 0.298],
    [0.464, 0.265, 0.376, 1.000, 0.483, 0.307],
    [0.366, 0.481, 0.274, 0.483, 1.000, 0.145],
    [0.170, 0.424, 0.298, 0.307, 0.145, 1.000]
]

# ================= HƯỚNG DẪN CHỈNH MÔ HÌNH HỒI QUY =================
# - Chỉ cần chỉnh sửa danh sách factors_config và regression_models bên dưới để thay đổi mô hình, biến độc lập/phụ thuộc, hoặc thêm biến interaction.
# - Để tạo biến interaction (tương tác), chỉ cần đặt tên biến theo dạng "A x B" (ví dụ: "PUxSAT"), hệ thống sẽ tự động tạo nếu đủ biến thành phần.
# - KHÔNG cần sửa bất kỳ file nào khác ngoài config.py, mọi thứ sẽ tự động cập nhật.
# - File kết quả sẽ xuất ra là output.xlsx trong thư mục output.
#
# ================== MÔ HÌNH: Ý ĐỊNH TIẾP TỤC SỬ DỤNG APP NGÂN HÀNG AI ==================
#
# 1. Các latent factors và số lượng biến quan sát:
#   - PI: 5 biến (PI1–PI5)
#   - PA: 5 biến (PA1–PA5)
#   - CONF: 4 biến (CONF1–CONF4)
#   - PU: 4 biến (PU1–PU4)
#   - SAT: 4 biến (SAT1–SAT4)
#   - CI: 2 biến (CI1–CI2)
#
# 2. Các mối quan hệ hồi quy (thứ tự ảnh hưởng mạnh yếu thể hiện qua thứ tự trong 'order')
#
# regression_models = [
#     {"dependent": "Y", "independent": ["A", "B", "AxB"], "order": ["A", "B", "AxB"]},
#     ...
# ]

factors_config = {
    "PI":   {"original_items": ["PI1", "PI2", "PI3", "PI4", "PI5"]},
    "PA":   {"original_items": ["PA1", "PA2", "PA3", "PA4", "PA5"]},
    "CONF": {"original_items": ["CONF1", "CONF2", "CONF3", "CONF4"]},
    "PU":   {"original_items": ["PU1", "PU2", "PU3", "PU4"]},
    "SAT":  {"original_items": ["SAT1", "SAT2", "SAT3", "SAT4"]},
    "CI":   {"original_items": ["CI1", "CI2"]}
}

regression_models = [
    # Perceived Anthropomorphism (PA) ~ Perceived Intelligence (PI)
    {"dependent": "PA_composite", "independent": ["PI_composite"], "order": ["PI_composite"]},
    # Confirmation (CONF) ~ PI + PA
    {"dependent": "CONF_composite", "independent": ["PI_composite", "PA_composite"], "order": ["PI_composite", "PA_composite"]},
    # Perceived Usefulness (PU) ~ PI + PA + CONF
    {"dependent": "PU_composite", "independent": ["PI_composite", "PA_composite", "CONF_composite"], "order": ["PI_composite", "CONF_composite", "PA_composite"]},
    # Satisfaction (SAT) ~ PU + CONF
    {"dependent": "SAT_composite", "independent": ["PU_composite", "CONF_composite"], "order": ["PU_composite", "CONF_composite"]},
    # Continuance Intention (CI) ~ PU + SAT
    {"dependent": "CI_composite", "independent": ["PU_composite", "SAT_composite"], "order": ["PU_composite", "SAT_composite"]}
]




n_latent_factors = len(factors_config)
latent_factor_names = list(factors_config.keys())
n_latent_cor_values = n_latent_factors * (n_latent_factors - 1) // 2



# --- Bayesian Optimization Config ---
num_observations = 367

# Bayesian Optimization parameters - Tối ưu hóa để giảm Heywood cases và tăng hiệu suất
bo_n_calls = 120                    # Số lần đánh giá (giảm để tập trung vào không gian tìm kiếm hẹp hơn)
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