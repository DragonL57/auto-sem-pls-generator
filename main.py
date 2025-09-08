

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
    num_observations, bounds_list,
    population_size, num_generations, crossover_rate, base_mutation_rate, mutation_scale, elitism_count,
    stagnation_threshold, mutation_increase_factor, mutation_decrease_factor, max_mutation_rate, min_mutation_rate
)
from genetic_algorithm import initialize_population, select_parents, crossover, mutate
from evaluation import evaluate_parameters_wrapper
from utils import nearest_positive_definite
from diagnostics import print_cronbach_alphas, run_kmo_bartlett, run_efa, run_regressions
from data_generation import generate_items_from_latent

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
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            start_total_time = time.time()
            adaptive_mutation_rate = base_mutation_rate
            rng = np.random.default_rng(42)
            print("==================================================")
            print("BẮT ĐẦU QUÁ TRÌNH TỰ ĐỘNG TỐI ƯU HÓA (GENETIC ALGORITHM)")
            print(f"Số thế hệ: {num_generations}, Kích thước quần thể: {population_size}")
            print(f"Sử dụng {multiprocessing.cpu_count()} tiến trình để đánh giá hàm mục tiêu.")
            print("==================================================")
            population = initialize_population(population_size, bounds_list)
            best_individual = None
            best_score = -np.inf
            best_reason = ""
            stagnation_counter = 0
            num_processes = max(1, multiprocessing.cpu_count() - 1)
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                    for generation in range(num_generations):
                        start_gen_time = time.time()
                        evaluation_args = [
                            (individual, factors_config, regression_models, num_observations, rng.integers(0, 1000000), n_latent_factors, n_latent_cor_values)
                            for individual in population
                        ]
                        # Submit all tasks for this generation
                        futures = [executor.submit(evaluate_parameters_wrapper, arg) for arg in evaluation_args]
                        results = [f.result() for f in concurrent.futures.as_completed(futures)]
                        # as_completed returns results in completion order, so we need to restore original order
                        # We'll use a dict to map id(arg) to result, then reconstruct in order
                        arg_id_to_result = {id(futures[i]): res for i, res in enumerate(results)}
                        ordered_results = [f.result() for f in futures]
                        fitnesses = [res[0] for res in ordered_results]
                        reasons = [res[1] for res in ordered_results]
                        current_gen_best_score = max(fitnesses)
                        current_gen_best_idx = np.argmax(fitnesses)
                        current_gen_best_reason = reasons[current_gen_best_idx]
                        if current_gen_best_score > best_score:
                            best_score = current_gen_best_score
                            best_individual = population[current_gen_best_idx].copy()
                            best_reason = current_gen_best_reason
                            stagnation_counter = 0
                            adaptive_mutation_rate = max(min_mutation_rate, adaptive_mutation_rate * mutation_decrease_factor)
                            print(f"Generation {generation+1} (Time: {time.time() - start_gen_time:.2f}s): New best score = {best_score:.2f}, Reason: {best_reason}, Mutation Rate: {adaptive_mutation_rate:.3f}")
                        else:
                            stagnation_counter += 1
                            if stagnation_counter >= stagnation_threshold:
                                adaptive_mutation_rate = min(max_mutation_rate, adaptive_mutation_rate * mutation_increase_factor)
                                stagnation_counter = 0
                            print(f"Generation {generation+1} (Time: {time.time() - start_gen_time:.2f}s): Best score in this generation = {current_gen_best_score:.2f}, Reason: {current_gen_best_reason} (Overall best: {best_score:.2f}), Mutation Rate: {adaptive_mutation_rate:.3f}")

                        # --- Detailed diagnostics for best individual in this generation ---
                        detail_score, detail_reason = evaluate_parameters_wrapper((population[current_gen_best_idx], factors_config, regression_models, num_observations, 12345, n_latent_factors, n_latent_cor_values))
                        print("  [Diagnostics] Best individual details:")
                        if isinstance(detail_reason, dict):
                            for k, v in detail_reason.items():
                                print(f"    {k}: {v}")
                        else:
                            print(f"    Penalty reason: {detail_reason}")
                        sorted_indices = np.argsort(fitnesses)[::-1]
                        new_population = [population[sorted_indices[i]].copy() for i in range(elitism_count)]
                        while len(new_population) < population_size:
                            parent1 = select_parents(population, fitnesses, 1)[0]
                            parent2 = select_parents(population, fitnesses, 1)[0]
                            if rng.random() < crossover_rate:
                                offspring1, offspring2 = crossover(parent1, parent2, bounds_list)
                            else:
                                offspring1, offspring2 = parent1.copy(), parent2.copy()
                            offspring1 = mutate(offspring1, bounds_list, adaptive_mutation_rate, mutation_scale)
                            offspring2 = mutate(offspring2, bounds_list, adaptive_mutation_rate, mutation_scale)
                            new_population.append(offspring1)
                            if len(new_population) < population_size:
                                new_population.append(offspring2)
                        population = new_population
                        if (generation + 1) % 10 == 0:
                            print(f"--- Tiến độ: {generation+1}/{num_generations} thế hệ. Điểm số tốt nhất hiện tại: {best_score:.2f} ---")
            except KeyboardInterrupt:
                print("\n\n[!] Đã nhận Ctrl+C - Đang xuất kết quả với tham số tốt nhất hiện tại...")
            finally:
                print("[LOG] Bắt đầu xuất kết quả cuối cùng...")
                try:
                    final_score = best_score
                    final_params = best_individual.copy() if best_individual is not None else None
                    final_reason = best_reason
                    end_total_time = time.time()
                    print("\n==================================================")
                    print("QUÁ TRÌNH TỐI ƯU HÓA (GENETIC ALGORITHM) HOÀN TẤT")
                    print(f"Tổng thời gian chạy: {end_total_time - start_total_time:.2f} giây")
                    print("==================================================")
                    print(f"Điểm số tốt nhất tìm được: {final_score:.2f}, Lý do: {final_reason}")
                    if final_params is not None:
                        print("Bộ tham số tốt nhất:")
                        print(f"  Độ mạnh tải nhân tố (Loading Strength): {final_params[n_latent_cor_values + 1]:.3f}")
                        print(f"  Độ mạnh sai số (Error Strength): {final_params[n_latent_cor_values]:.3f}")
                        print("  Các giá trị tương quan tiềm ẩn (Tam giác trên):")
                        print(final_params[:n_latent_cor_values].round(3))
                        # === Tạo dữ liệu kết quả và xuất file Excel vào output_dir ===
                        # Tái tạo ma trận tương quan tiềm ẩn từ best_individual
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
                            # Regex: chỉ thay thế đúng block khai báo biến (không phải comment, không bị lặp)
                            # Tìm dòng bắt đầu khai báo biến (không phải comment), thay thế block tiếp theo là list
                            pattern = r"^latent_correlation_matrix\s*=\s*\[[\s\S]*?^\]"  # ^ ở đầu dòng, không phải comment
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
                        data_for_excel_final = pd.concat([data_for_analysis_final, composite_scores_final], axis=1)
                        output_excel_path = os.path.join(output_dir, "output.xlsx")
                        try:
                            data_for_excel_final.to_excel(output_excel_path, index=False)
                            print(f"\n--- Dữ liệu giả mạo cuối cùng đã được lưu vào: {output_excel_path} ---\n")
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
