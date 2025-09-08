import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from metrics import calculate_cronbach_alpha

def print_cronbach_alphas(data_for_analysis, factors_config):
    """
    Print Cronbach's Alpha for each factor based on the provided data and factor configuration.

    Parameters:
        data_for_analysis (pd.DataFrame): The data containing all items for analysis.
        factors_config (dict): A dictionary mapping factor names to their configuration, including original items.
    """
    print("\n--- Cronbach's Alpha cho từng nhân tố (TỐT NHẤT) ---")
    for factor_name, config in factors_config.items():
        items_for_factor = config["original_items"]
        alpha = calculate_cronbach_alpha(data_for_analysis[items_for_factor])
        print(f"Cronbach's Alpha cho {factor_name}: {alpha:.3f}")

def run_kmo_bartlett(data_for_analysis):
    """
    Perform and print the results of the KMO and Bartlett's Test of Sphericity on the provided data.

    Parameters:
        data_for_analysis (pd.DataFrame): The data to be analyzed for factorability.
    """
    print("\n--- KMO và Bartlett's Test of Sphericity (TỐT NHẤT) ---")
    data_for_efa_final = data_for_analysis.dropna()
    try:
        _, kmo_model = calculate_kmo(data_for_efa_final)
        chi_square_value, p_value_bartlett = calculate_bartlett_sphericity(data_for_efa_final)
        print(f"KMO Overall: {kmo_model:.3f}")
        print(f"Bartlett's Chi-square: {chi_square_value:.2f}, p-value: {p_value_bartlett:.3f}")
    except ValueError as e:
        print(f"KMO/Bartlett ValueError: {e}")
    except ImportError as e:
        print(f"KMO/Bartlett ImportError: {e}")
    except (TypeError, AttributeError) as e:
        print(f"KMO/Bartlett Unexpected Error: {e}")

def run_efa(data_for_analysis, n_factors):
    print("\n--- EFA (Exploratory Factor Analysis) (TỐT NHẤT) ---")
    data_for_efa_final = data_for_analysis.dropna()
    try:
        fa_promax_final = FactorAnalyzer(n_factors=n_factors, rotation="promax", method='principal', use_smc=True)
        fa_promax_final.fit(data_for_efa_final)
        loadings_promax_final = pd.DataFrame(
            fa_promax_final.loadings_,
            index=data_for_efa_final.columns,
            columns=[f"Factor{i+1}" for i in range(n_factors)]
        )
        print("\nFactor Loadings (Promax - TỐT NHẤT):\n")
        print(loadings_promax_final.round(3))

        # Bổ sung: Factor Loadings đã lọc (ẩn <0.4) thành một bảng duy nhất
        print("\nFactor Loadings (|loading| >= 0.4, mỗi item chỉ giữ loading lớn nhất, nhóm theo nhân tố, dạng bậc thang):\n")
        # Lọc các giá trị < 0.4
        filtered = loadings_promax_final.where(loadings_promax_final.abs() >= 0.4, np.nan)
        # Xác định cột có trị tuyệt đối lớn nhất trên mỗi hàng
        max_col = filtered.abs().idxmax(axis=1)
        # Chỉ giữ lại loading lớn nhất trên mỗi hàng, các cột khác để NaN
        stair = filtered.copy() * np.nan
        for idx in filtered.index:
            col = max_col[idx]
            if not np.isnan(filtered.loc[idx, col]):
                stair.loc[idx, col] = filtered.loc[idx, col]
        # Gom nhóm theo từng nhân tố, sắp xếp giảm dần theo trị tuyệt đối
        stair_rows = []
        for col in stair.columns:
            # Lấy các item mà loading lớn nhất nằm ở cột này
            items = stair.index[stair[col].notna()]
            # Sắp xếp các item này theo trị tuyệt đối giảm dần
            items_sorted = stair.loc[items, col].abs().sort_values(ascending=False).index
            stair_rows.extend(items_sorted)
        stair_sorted = stair.loc[stair_rows]

        print(stair_sorted.round(3).to_string(na_rep=''))

        # --- Tự động detect nhóm item và cảnh báo lẫn lộn nhân tố ---
        import re
        # 1. Lấy prefix cho từng item (ví dụ: PI1 -> PI)
        def get_prefix(item):
            m = re.match(r"([A-Za-z]+)", item)
            return m.group(1) if m else item

        # 2. Tìm nhóm prefix và nhân tố thực tế đầu tiên xuất hiện (theo bảng rotated)
        prefix2factor = {}
        for item in stair_sorted.index:
            prefix = get_prefix(item)
            factor_actual = stair_sorted.columns[stair_sorted.loc[item].notna()][0] if stair_sorted.loc[item].notna().any() else None
            if prefix not in prefix2factor and factor_actual:
                prefix2factor[prefix] = factor_actual

        # 3. So sánh từng item: nếu nhân tố thực tế khác nhân tố lý thuyết (theo prefix) thì cảnh báo
        wrong_items = []
        for item in stair_sorted.index:
            prefix = get_prefix(item)
            factor_expected = prefix2factor.get(prefix)
            factor_actual = stair_sorted.columns[stair_sorted.loc[item].notna()][0] if stair_sorted.loc[item].notna().any() else None
            if factor_actual and factor_expected and factor_actual != factor_expected:
                wrong_items.append((item, factor_expected, factor_actual))

        if wrong_items:
            print("\n[!] CẢNH BÁO: Có item bị xoay sai nhóm nhân tố (tự động detect):")
            for item, expected, actual in wrong_items:
                print(f"  - {item}: Nhóm lý thuyết = {expected}, Xoay ra = {actual}")
        else:
            print("\n[OK] Tất cả item đều xoay đúng nhóm nhân tố lý thuyết!")

        communalities_final = fa_promax_final.get_communalities()
        print("\nCommunalities (Promax - TỐT NHẤT):\n")
        print(pd.DataFrame(communalities_final.round(3), index=data_for_efa_final.columns, columns=["Communalities"]))
        if np.any(loadings_promax_final.values > 1.0 + np.finfo(float).eps) or np.any(communalities_final > 1.0 + np.finfo(float).eps):
            print("\nCẢNH BÁO: EFA TỐT NHẤT vẫn có Heywood cases (loadings hoặc communalities > 1).")
        else:
            print("\nEFA TỐT NHẤT không có Heywood cases (loadings hoặc communalities > 1).")
        if hasattr(fa_promax_final, 'phi_') and fa_promax_final.phi_ is not None:
            print("\nFactor Correlation Matrix (Promax - TỐT NHẤT):\n")
            print(pd.DataFrame(fa_promax_final.phi_.round(3),
                               index=[f"Factor{i+1}" for i in range(n_factors)],
                               columns=[f"Factor{i+1}" for i in range(n_factors)]))
        if np.any(np.sum(loadings_promax_final.values**4, axis=1) == 0):
            mean_item_complexity_final = np.nan
        else:
            item_complexity_final = np.sum(loadings_promax_final.values**2, axis=1)**2 / np.sum(loadings_promax_final.values**4, axis=1)
            mean_item_complexity_final = np.mean(item_complexity_final)
        print(f"\nMean Item Complexity (Promax - TỐT NHẤT): {mean_item_complexity_final:.3f}")
    except (ValueError, ImportError, TypeError, AttributeError) as e:
        print(f"EFA Final Error: {e}")

def run_regressions(composite_scores_final, regression_models, data_for_analysis=None, factors_config=None):
    print("\n--- Kết quả mô hình hồi quy (TỐT NHẤT) ---")
    scaler_final = StandardScaler()
    composite_scores_std_final = pd.DataFrame(
        scaler_final.fit_transform(composite_scores_final),
        columns=composite_scores_final.columns
    )
    for model_spec in regression_models:
        dependent_var = model_spec["dependent"]
        independent_vars = model_spec["independent"]
        expected_order = model_spec["order"]
        if not all(v in composite_scores_std_final.columns for v in [dependent_var] + independent_vars):
            print(f"Bỏ qua mô hình {dependent_var}, thiếu biến.")
            continue
        X = sm.add_constant(composite_scores_std_final[independent_vars])
        y = composite_scores_std_final[dependent_var]
        try:
            model = sm.OLS(y, X).fit()
            print(f"\n--- Mô hình {dependent_var}: {dependent_var} ~ {' + '.join(independent_vars)} ---")
            print(model.summary())
            print("Standardized betas:")
            print(model.params.drop("const"))
            actual_betas = model.params.drop("const").sort_values(ascending=False)
            actual_order_filtered = [var for var in actual_betas.index if var in expected_order]
            if actual_order_filtered == expected_order:
                print(f"Thứ tự ảnh hưởng ĐÚNG: {expected_order}")
            else:
                print(f"Thứ tự ảnh hưởng SAI. Mong muốn: {expected_order}, Thực tế: {actual_order_filtered}")
        except (ValueError, sm.tools.sm_exceptions.PerfectSeparationError, KeyError) as e:
            print(f"Regression Final Error ({dependent_var}): {e}")
    # Tự động in checklist cuối cùng với đủ dữ liệu
    try:
        print_guidance(data_for_analysis, factors_config)
    except (ValueError, KeyError, ImportError, AttributeError) as e:
        print(f"[Checklist Error] {e}")

def print_guidance(data_for_analysis=None, factors_config=None):
    print("\n================== CHECKLIST KIỂM ĐỊNH ==================")
    print("1. Cronbach's Alpha >= 0.7 cho tất cả nhân tố: ", end='')
    try:
        all_alpha = []
        if data_for_analysis is not None and factors_config is not None:
            for config in factors_config.values():
                items = config["original_items"]
                alpha = calculate_cronbach_alpha(data_for_analysis[items])
                all_alpha.append(alpha)
            if all(a >= 0.7 for a in all_alpha):
                print("ĐẠT")
            else:
                print("CHƯA ĐẠT (Có nhân tố alpha < 0.7)")
        else:
            print("(Không đủ dữ liệu để kiểm tra)")
    except (ValueError, KeyError, ImportError, AttributeError) as e:
        print(f"(Lỗi kiểm tra: {e})")

    print("2. EFA phân đúng nhân tố, không có Heywood case: ", end='')
    print("Xem cảnh báo ngay dưới bảng rotated ở phần EFA")

    print("3. Thứ tự standardized beta đúng như mong muốn: ", end='')
    print("Xem từng mô hình hồi quy, dòng 'Thứ tự ảnh hưởng ĐÚNG/SAI'")

    print("4. R-squared các mô hình hồi quy: ", end='')
    print("Xem từng mô hình, giá trị R-squared >= 0.1 là chấp nhận được")

    print("5. Durbin-Watson kiểm tra tự tương quan phần dư: ", end='')
    print("Xem từng mô hình, giá trị ~2 là tốt (1.5-2.5)")

    print("6. VIF kiểm tra đa cộng tuyến: ", end='')
    print("Nếu có in VIF, tất cả < 5 là ĐẠT")

    print("==================================================")
