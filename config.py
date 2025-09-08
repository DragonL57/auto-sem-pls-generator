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
    [1.000, 0.253, 0.629, 0.572, 0.625, 0.567],
    [0.253, 1.000, 0.436, 0.435, 0.313, 0.302],
    [0.629, 0.436, 1.000, 0.529, 0.526, 0.526],
    [0.572, 0.435, 0.529, 1.000, 0.727, 0.626],
    [0.625, 0.313, 0.526, 0.727, 1.000, 0.535],
    [0.567, 0.302, 0.526, 0.626, 0.535, 1.000]
],

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



# --- Genetic Algorithm (GA) Config ---
num_observations = 367

# GA parameter bounds
param_bounds = {
    'latent_cor_values': [0.01, 0.95],
    'loading_strength': [0.45, 0.65],
    'error_strength': [0.35, 0.55]
}


bounds_list = []
for _ in range(n_latent_cor_values):
    bounds_list.append(param_bounds['latent_cor_values'])
bounds_list.append(param_bounds['error_strength'])
bounds_list.append(param_bounds['loading_strength'])

# GA hyperparameters
population_size = 100
num_generations = 200
crossover_rate = 0.8
base_mutation_rate = 0.15
mutation_scale = 0.08
elitism_count = 5
stagnation_threshold = 7
mutation_increase_factor = 1.3
mutation_decrease_factor = 0.8
max_mutation_rate = 0.3
min_mutation_rate = 0.05