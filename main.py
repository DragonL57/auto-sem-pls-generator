

import os
import pandas as pd
import sys
import time
import numpy as np
import multiprocessing
import concurrent.futures
import warnings
from contextlib import redirect_stdout, redirect_stderr

# Ẩn toàn bộ FutureWarning và UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import (
    factors_config, regression_models, latent_factor_names, n_latent_factors, n_latent_cor_values,
    num_observations, bo_n_calls, bo_n_initial_points, bo_acq_func, bo_n_jobs, bo_early_stopping, bo_patience,
    bo_latent_cor_min, bo_latent_cor_max, bo_error_strength_min, bo_error_strength_max, bo_loading_strength_min, bo_loading_strength_max
)
from evaluation import evaluate_parameters_wrapper
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import EarlyStopper
from utils import nearest_positive_definite
from diagnostics import print_cronbach_alphas, run_kmo_bartlett, run_efa, run_regressions
from data_generation import generate_items_from_latent
from excel_export import export_multi_sheet_excel

if __name__ == '__main__':
    # Tạo thư mục output nếu chưa có
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "terminal.log")
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                try:
                    f.write(obj)
                    f.flush()
                except UnicodeEncodeError:
                    # Fallback for encoding issues
                    f.write(obj.encode('ascii', errors='replace').decode('ascii'))
                    f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open(log_path, "w", encoding="utf-8", errors='replace') as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            start_total_time = time.time()
            rng = np.random.default_rng(42)
            print("==================================================")
            print("BẮT ĐẦU QUÁ TRÌNH TỐI ƯU HÓA (BAYESIAN OPTIMIZATION)")
            print("Sử dụng Bayesian Optimization thay thế Genetic Algorithm để tối ưu hiệu suất")
            print("==================================================")
            
            # Tạo search space cho Bayesian Optimization - sử dụng tham số từ config
            search_space = []
            for i in range(n_latent_cor_values):
                search_space.append(Real(bo_latent_cor_min, bo_latent_cor_max, name=f'latent_cor_{i}'))
            search_space.append(Real(bo_error_strength_min, bo_error_strength_max, name='error_strength'))
            search_space.append(Real(bo_loading_strength_min, bo_loading_strength_max, name='loading_strength'))
            
            # Theo dõi kết quả tốt nhất
            best_container = {'score': -np.inf, 'params': None, 'reason': ''}
            evaluation_history = []
            
            def objective_function(params):
                start_time = time.time()
                try:
                    params_array = np.array(params)
                    print(f"Debug: Testing params = {params_array}")
                    rng_seed = np.random.randint(0, 1000000)
                    fitness_score, reason = evaluate_parameters_wrapper(
                        (params_array, factors_config, regression_models, num_observations, 
                         rng_seed, n_latent_factors, n_latent_cor_values)
                    )
                    
                    print(f"Debug: fitness_score = {fitness_score}, reason = {reason}")
                    
                    # Cập nhật best solution
                    if fitness_score > best_container['score']:
                        best_container['score'] = fitness_score
                        best_container['params'] = params_array.copy()
                        best_container['reason'] = reason
                        
                    evaluation_history.append(fitness_score)
                    
                    # Print progress
                    current_eval = len(evaluation_history)
                    if current_eval % 5 == 0:
                        print(f"Evaluation {current_eval}/{bo_n_calls}: Best score = {best_container['score']:.2f}, Current = {fitness_score:.2f}")
                    
                    return -fitness_score  # Negative vì gp_minimize tìm minimum
                    
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1_000_000  # Penalty score
            
            # Early stopping callback
            def early_stopper(result):
                if len(evaluation_history) < bo_patience:
                    return False
                recent_scores = evaluation_history[-bo_patience:]
                best_recent = max(recent_scores)
                if best_recent <= best_container['score']:
                    print(f"\n[Early Stopping] No improvement for {bo_patience} evaluations")
                    return True
                return False
            
            # Run Bayesian Optimization
            try:
                result = gp_minimize(
                    func=objective_function,
                    dimensions=search_space,
                    n_calls=bo_n_calls,
                    n_initial_points=bo_n_initial_points,
                    acq_func=bo_acq_func,
                    n_jobs=bo_n_jobs,
                    callback=EarlyStopper(early_stopper) if bo_early_stopping and len(evaluation_history) > 0 else None,
                    random_state=42,
                    verbose=True
                )
            except KeyboardInterrupt:
                print("\n\n[!] Đã nhận Ctrl+C - Đang xuất kết quả với tham số tốt nhất hiện tại...")
            finally:
                print("[LOG] Bắt đầu xuất kết quả cuối cùng...")
                try:
                    final_score = best_container['score']
                    final_params = best_container['params'].copy() if best_container['params'] is not None else None
                    final_reason = best_container['reason']
                    end_total_time = time.time()
                    print("\n==================================================")
                    print("QUÁ TRÌNH TỐI ƯU HÓA (BAYESIAN OPTIMIZATION) HOÀN TẤT")
                    print(f"Tổng thời gian chạy: {end_total_time - start_total_time:.2f} giây")
                    print(f"Tổng số evaluations: {len(evaluation_history)}")
                    print("==================================================")
                    print(f"Điểm số tốt nhất tìm được: {final_score:.2f}, Lý do: {final_reason}")
                    
                    if final_params is not None:
                        print("Bộ tham số tốt nhất:")
                        print(f"  Độ mạnh tải nhân tố (Loading Strength): {final_params[n_latent_cor_values + 1]:.3f}")
                        print(f"  Độ mạnh sai số (Error Strength): {final_params[n_latent_cor_values]:.3f}")
                        print("  Các giá trị tương quan tiềm ẩn (Tam giác trên):")
                        print(final_params[:n_latent_cor_values].round(3))
                        
                        # Print performance summary
                        print(f"\n=== PERFORMANCE SUMMARY ===")
                        print(f"Bayesian Optimization:")
                        print(f"  - Total evaluations: {len(evaluation_history)}")
                        print(f"  - Total time: {end_total_time - start_total_time:.2f}s")
                        print(f"  - Time per evaluation: {(end_total_time - start_total_time)/len(evaluation_history):.2f}s")
                        print(f"  - High efficiency with focused search")
                        # === Tạo dữ liệu kết quả và xuất file Excel vào output_dir ===
                        # Tái tạo ma trận tương quan tiềm ẩn từ best_params
                        best_latent_cor_matrix = np.eye(n_latent_factors)
                        k = 0
                        for i in range(n_latent_factors):
                            for j in range(i + 1, n_latent_factors):
                                best_latent_cor_matrix[i, j] = final_params[k]
                                best_latent_cor_matrix[j, i] = final_params[k]
                                k += 1
                        print("\n--- Ma trận tương quan tiềm ẩn TỐT NHẤT (Mục tiêu) ---\n")
                        print(pd.DataFrame(best_latent_cor_matrix, index=latent_factor_names, columns=latent_factor_names).round(3))
                        best_latent_cor_matrix_adjusted, diag_gt_one_latent_final = nearest_positive_definite(best_latent_cor_matrix)
                        if diag_gt_one_latent_final:
                            print("\nCẢNH BÁO: Ma trận tương quan tiềm ẩn TỐT NHẤT vẫn có Heywood cases (diagonal > 1) sau điều chỉnh nearPD.")
                        else:
                            print("\nMa trận tương quan tiềm ẩn TỐT NHẤT là xác định dương và không có Heywood cases ở cấp độ tiềm ẩn.")
                        print("Ma trận tương quan tiềm ẩn TỐT NHẤT (ĐÃ điều chỉnh nếu cần):\n")
                        print(pd.DataFrame(best_latent_cor_matrix_adjusted, index=latent_factor_names, columns=latent_factor_names).round(3))

                        # --- Tự động cập nhật latent_correlation_matrix trong config.py ---
                        try:
                            import re
                            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
                            with open(config_path, "r", encoding="utf-8") as config_file:
                                config_code = config_file.read()
                            # Tạo string ma trận mới
                            matrix_str = "[\n" + ",\n".join([
                                "    [" + ", ".join(f"{v:.3f}" for v in row) + "]" for row in best_latent_cor_matrix
                            ]) + "\n]"
                            # Regex: tìm cả trường hợp latent_correlation_matrix = None và latent_correlation_matrix = [[...]]
                            pattern = r"^latent_correlation_matrix\s*=\s*(None|\[[\s\S]*?^\])"  # ^ ở đầu dòng, không phải comment
                            new_code = re.sub(
                                pattern,
                                f"latent_correlation_matrix = {matrix_str}",
                                config_code,
                                flags=re.MULTILINE
                            )
                            with open(config_path, "w", encoding="utf-8") as config_file:
                                config_file.write(new_code)
                            print("[LOG] Đã tự động cập nhật latent_correlation_matrix trong config.py với ma trận tốt nhất!")
                        except (FileNotFoundError, PermissionError, UnicodeDecodeError, re.error) as e:
                            print(f"[WARNING] Không thể tự động cập nhật latent_correlation_matrix trong config.py: {e}")
                        rng_final = np.random.default_rng(42)
                        latent_samples_final = rng_final.multivariate_normal(mean=np.zeros(n_latent_factors), cov=best_latent_cor_matrix_adjusted, size=num_observations)
                        latent_df_final = pd.DataFrame(latent_samples_final, columns=[f"{name}_latent" for name in latent_factor_names])
                        print("\n--- Thống kê mô tả của các yếu tố tiềm ẩn chính (TỐT NHẤT) ---\n")
                        print(latent_df_final.describe().round(3))
                        print("\n--- Ma trận tương quan của các yếu tố tiềm ẩn chính (TỐT NHẤT) ---\n")
                        print(latent_df_final.corr().round(3))
                        generated_factors_list_final = {}
                        for factor_name in latent_factor_names:
                            config = factors_config[factor_name]
                            item_names_for_factor = config["original_items"]
                            num_items_in_factor = len(item_names_for_factor)
                            current_latent_factor = latent_df_final[f"{factor_name}_latent"].values
                            factor_data_transformed = generate_items_from_latent(
                                latent_factor = current_latent_factor,
                                num_items = num_items_in_factor,
                                loading_strength = final_params[n_latent_cor_values + 1],
                                error_strength = final_params[n_latent_cor_values],
                                rng = rng
                            )
                            factor_data_transformed.columns = item_names_for_factor
                            generated_factors_list_final[factor_name] = factor_data_transformed
                        data_for_analysis_final = pd.concat(generated_factors_list_final.values(), axis=1)
                        composite_scores_final = pd.DataFrame({
                            f"{fac}_composite": data_for_analysis_final[[col for col in data_for_analysis_final.columns if col.startswith(fac)]].mean(axis=1)
                            for fac in factors_config.keys()
                        })
                        # Tự động tạo biến tương tác dựa trên regression_models trong config.py
                        import re
                        interaction_pattern = re.compile(r"(.+)x(.+)")
                        # Tìm tất cả biến interaction cần thiết từ các mô hình hồi quy
                        interaction_terms = set()
                        for model in regression_models:
                            for var in model["independent"]:
                                match = interaction_pattern.fullmatch(var)
                                if match:
                                    interaction_terms.add(var)
                        # Tạo các biến interaction nếu đủ biến thành phần
                        for term in interaction_terms:
                            match = interaction_pattern.fullmatch(term)
                            if match:
                                var1, var2 = match.group(1), match.group(2)
                                if {var1, var2}.issubset(composite_scores_final.columns):
                                    composite_scores_final[term] = composite_scores_final[var1] * composite_scores_final[var2]
                        composite_scores_final = composite_scores_final.dropna()
                        output_excel_path = os.path.join(output_dir, "output.xlsx")
                        try:
                            export_multi_sheet_excel(
                                data_for_analysis_final, 
                                composite_scores_final, 
                                factors_config, 
                                regression_models, 
                                n_latent_factors, 
                                output_excel_path
                            )
                            print(f"\n--- Dữ liệu đa sheet đã được lưu vào: {output_excel_path} ---\n")
                        except (FileNotFoundError, PermissionError, ValueError) as e:
                            print(f"Lỗi khi lưu file Excel: {e}")
                        # Diagnostics chi tiết cuối cùng
                        print_cronbach_alphas(data_for_analysis_final, factors_config)
                        run_kmo_bartlett(data_for_analysis_final)
                        run_efa(data_for_analysis_final, n_latent_factors)
                        # Truyền đủ dữ liệu cho checklist tự động
                        run_regressions(composite_scores_final, regression_models, data_for_analysis_final, factors_config)
                    print("[LOG] Xuất kết quả cuối cùng hoàn tất!")
                except (AttributeError, ValueError, KeyError, IndexError) as e:
                    print(f"[ERROR] Lỗi khi xuất kết quả cuối cùng: {e}")
