import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from utils import nearest_positive_definite
from data_generation import generate_items_from_latent
from metrics import calculate_cronbach_alpha

# Hàm đánh giá sử dụng Y CHANG logic phạt & thưởng từ script tham chiếu người dùng cung cấp
def evaluate_parameters_wrapper(args):
    params_array, factors_config, regression_models, num_observations, rng_seed, n_latent_factors, n_latent_cor_values = args

    rng = np.random.default_rng(rng_seed)

    latent_cor_values = params_array[:n_latent_cor_values]
    error_strength = params_array[n_latent_cor_values]
    loading_strength = params_array[n_latent_cor_values + 1]

    latent_factor_names = list(factors_config.keys())

    # 1. Tái tạo ma trận tương quan tiềm ẩn
    latent_cor_matrix = np.eye(n_latent_factors)
    k = 0
    for i in range(n_latent_factors):
        for j in range(i + 1, n_latent_factors):
            latent_cor_matrix[i, j] = latent_cor_values[k]
            latent_cor_matrix[j, i] = latent_cor_values[k]
            k += 1

    # Kiểm tra tính xác định dương và Heywood cases ở cấp độ tiềm ẩn
    try:
        latent_cor_matrix_adjusted, diag_gt_one_latent = nearest_positive_definite(latent_cor_matrix)
    except RuntimeError:
        return -1_000_000, "Latent PD Error"

    if diag_gt_one_latent:
        return -1_000_000, "Heywood (Latent Diag > 1)"

    # 2. Sinh các yếu tố tiềm ẩn
    try:
        latent_samples = rng.multivariate_normal(mean=np.zeros(n_latent_factors), cov=latent_cor_matrix_adjusted, size=num_observations)
        latent_df = pd.DataFrame(latent_samples, columns=[f"{name}_latent" for name in latent_factor_names])
    except np.linalg.LinAlgError:
        return -1_000_000, "Latent MVN Error"

    # 3. Sinh các mục Likert
    generated_factors_list = {}
    for factor_name in latent_factor_names:
        config = factors_config[factor_name]
        item_names_for_factor = config["original_items"]
        num_items_in_factor = len(item_names_for_factor)

        current_latent_factor = latent_df[f"{factor_name}_latent"].values

        factor_data_transformed = generate_items_from_latent(
            latent_factor=current_latent_factor,
            num_items=num_items_in_factor,
            loading_strength=loading_strength,
            error_strength=error_strength,
            rng=rng
        )
        factor_data_transformed.columns = item_names_for_factor
        generated_factors_list[factor_name] = factor_data_transformed

    data_for_analysis = pd.concat(generated_factors_list.values(), axis=1)

    # 4. Tính điểm tổng hợp
    composite_scores = pd.DataFrame({
        f"{fac}_composite": data_for_analysis[[col for col in data_for_analysis.columns if col.startswith(fac)]].mean(axis=1)
        for fac in factors_config.keys()
    })
    composite_scores = composite_scores.dropna()

    if composite_scores.empty:
        return -1_000_000, "Empty Composite Scores"

    # Thêm biến tương tác nếu mô hình cần (không ảnh hưởng logic phạt gốc nếu không dùng)
    if {"DIS_composite", "SC_composite"}.issubset(composite_scores.columns):
        if "DISxSC" not in composite_scores.columns:
            composite_scores["DISxSC"] = composite_scores["DIS_composite"] * composite_scores["SC_composite"]
    if {"PV_composite", "SC_composite"}.issubset(composite_scores.columns):
        if "PVxSC" not in composite_scores.columns:
            composite_scores["PVxSC"] = composite_scores["PV_composite"] * composite_scores["SC_composite"]

    # --- Đánh giá Cronbach's Alpha ---
    cronbach_alpha_score = 0
    for factor_name, config in factors_config.items():
        items_for_factor = config["original_items"]
        if all(item in data_for_analysis.columns for item in items_for_factor):
            alpha = calculate_cronbach_alpha(data_for_analysis[items_for_factor])
            if not np.isnan(alpha):
                if alpha >= 0.7:
                    cronbach_alpha_score += (alpha - 0.7) * 500
                else:
                    cronbach_alpha_score -= (0.7 - alpha) * 1000
            else:
                cronbach_alpha_score -= 500
        else:
            cronbach_alpha_score -= 1000

    # --- Đánh giá EFA ---
    efa_score = 0
    data_for_efa = data_for_analysis.dropna()
    if data_for_efa.empty or data_for_efa.shape[1] < n_latent_factors:
        return -1_000_000, "EFA Not Enough Data"

    try:
        fa_promax = FactorAnalyzer(n_factors=n_latent_factors, rotation="promax", method='principal', use_smc=True)
        fa_promax.fit(data_for_efa)

        loadings = fa_promax.loadings_
        communalities = fa_promax.get_communalities()

        # Heywood cases trong loadings hoặc communalities (PHẠT RẤT NẶNG)
        if np.any(loadings > 1.0 + np.finfo(float).eps) or np.any(communalities > 1.0 + np.finfo(float).eps):
            return -10_000_000, "Heywood (EFA Loadings/Communalities > 1)"

        # Mean Item Complexity (MIC)
        if np.any(np.sum(loadings**4, axis=1) == 0):
            mean_item_complexity = 10
        else:
            item_complexity = np.sum(loadings**2, axis=1)**2 / np.sum(loadings**4, axis=1)
            mean_item_complexity = np.mean(item_complexity)

        if mean_item_complexity > 1.5:
            efa_score -= (mean_item_complexity - 1.5) * 100
        else:
            efa_score += (1.5 - mean_item_complexity) * 50

        # Ma trận tương quan nhân tố (Promax) - kiểm tra tương quan âm
        if hasattr(fa_promax, 'phi_') and fa_promax.phi_ is not None:
            factor_correlations = fa_promax.phi_
            off_diagonal_corrs = factor_correlations[np.triu_indices(n_latent_factors, k=1)]
            neg_corrs = off_diagonal_corrs[off_diagonal_corrs < 0]
            if len(neg_corrs) > 0:
                efa_score -= np.sum(np.abs(neg_corrs)) * 500
        else:
            efa_score -= 500
    except (ValueError, np.linalg.LinAlgError) as e:
        return -1_000_000, f"EFA Fitting Error: {e}"

    # --- Đánh giá Mô hình Hồi quy ---
    r_squared_score = 0
    beta_significance_score = 0
    beta_order_score = 0
    negative_beta_penalty = 0

    scaler = StandardScaler()
    composite_scores_std = pd.DataFrame(scaler.fit_transform(composite_scores), columns=composite_scores.columns)

    for model_spec in regression_models:
        dependent_var = model_spec["dependent"]
        independent_vars = model_spec["independent"]
        expected_order = model_spec["order"]

        if not all(v in composite_scores_std.columns for v in [dependent_var] + independent_vars):
            return -1_000_000, "Missing Regression Vars"

        X = sm.add_constant(composite_scores_std[independent_vars])
        y = composite_scores_std[dependent_var]

        try:
            model = sm.OLS(y, X).fit()

            # R-squared (ĐÃ ĐIỀU CHỈNH THƯỞNG)
            r_squared_score += model.rsquared * 200
            if model.rsquared >= 0.4:
                r_squared_score += (model.rsquared - 0.4) * 400
            if model.rsquared >= 0.5:
                r_squared_score += (model.rsquared - 0.5) * 1500

            # Hình phạt R-squared thấp (< 0.5)
            if model.rsquared < 0.5:
                r_squared_score -= (0.5 - model.rsquared) * 10000

            # Beta Significance & Negative Beta
            for var in independent_vars:
                p_value = model.pvalues[var]
                beta_coef = model.params[var]

                if beta_coef < 0:
                    negative_beta_penalty -= 5000

                if p_value < 0.05:
                    beta_significance_score += 100
                else:
                    beta_significance_score -= 2000

            # Thứ tự Beta
            actual_betas = model.params.drop("const").sort_values(ascending=False)
            actual_order_filtered = [var for var in actual_betas.index if var in expected_order]

            if len(actual_order_filtered) == len(expected_order):
                if actual_order_filtered == expected_order:
                    beta_order_score += 200
                else:
                    beta_order_score -= 8000
            else:
                beta_order_score -= 4000
        except (ValueError, np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError) as e:
            return -1_000_000, f"Regression Error: {e}"

    total_score = (cronbach_alpha_score + efa_score + r_squared_score +
                   beta_significance_score + beta_order_score + negative_beta_penalty)
    # Return all sub-scores for diagnostics if no hard penalty
    detail = {
        'total_score': total_score,
        'cronbach_alpha_score': cronbach_alpha_score,
        'efa_score': efa_score,
        'r_squared_score': r_squared_score,
        'beta_significance_score': beta_significance_score,
        'beta_order_score': beta_order_score,
        'negative_beta_penalty': negative_beta_penalty
    }
    return total_score, detail
