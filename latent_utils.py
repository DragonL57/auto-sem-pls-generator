import numpy as np

def generate_latent_factors(n_samples, n_factors, random_state=None):
    # Khởi tạo ma trận tương quan latent với giá trị chéo nhỏ để tránh Heywood
    rng = np.random.default_rng(random_state)
    latent_corr_matrix = np.eye(n_factors)
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            val = rng.uniform(-0.15, 0.15)
            latent_corr_matrix[i, j] = val
            latent_corr_matrix[j, i] = val
    # Sinh dữ liệu latent
    latent_samples = rng.multivariate_normal(mean=np.zeros(n_factors), cov=latent_corr_matrix, size=n_samples)
    return latent_samples, latent_corr_matrix
