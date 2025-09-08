"""
Multi-Sheet Excel Export Module for SEM/PLS Data Generator
Xuất dữ liệu ra file Excel với nhiều sheet tổ chức chức nghiệp
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from metrics import calculate_cronbach_alpha


def create_statistical_analysis_sheet(data_for_analysis, factors_config):
    """
    Tạo bảng phân tích thống kê cho Excel sheet
    
    Args:
        data_for_analysis: DataFrame chứa dữ liệu phân tích
        factors_config: Cấu hình các nhân tố
        
    Returns:
        pd.DataFrame: DataFrame chứa thống kê mô tả
    """
    stats_data = []
    
    for factor_name, config in factors_config.items():
        items = config["original_items"]
        factor_data = data_for_analysis[items]
        
        # Tính toán thống kê cho từng item
        for item in items:
            item_stats = {
                'Factor': factor_name,
                'Item': item,
                'Mean': factor_data[item].mean(),
                'Std Dev': factor_data[item].std(),
                'Min': factor_data[item].min(),
                'Max': factor_data[item].max(),
                'Skewness': factor_data[item].skew(),
                'Kurtosis': factor_data[item].kurtosis()
            }
            stats_data.append(item_stats)
    
    return pd.DataFrame(stats_data)


def create_correlation_sheet(data_for_analysis):
    """
    Tạo bảng tương quan Pearson cho Excel sheet
    
    Args:
        data_for_analysis: DataFrame chứa dữ liệu phân tích
        
    Returns:
        pd.DataFrame: Ma trận tương quan
    """
    return data_for_analysis.corr().round(3)


def create_factor_analysis_sheet(data_for_analysis, n_factors):
    """
    Tạo bảng kết quả EFA cho Excel sheet
    
    Args:
        data_for_analysis: DataFrame chứa dữ liệu phân tích
        n_factors: Số lượng nhân tố tiềm ẩn
        
    Returns:
        tuple: (loadings_df, factor_corr_df, communalities_df)
    """
    data_for_efa = data_for_analysis.dropna()
    
    # Chạy EFA
    fa = FactorAnalyzer(n_factors=n_factors, rotation="promax", method='principal', use_smc=True)
    fa.fit(data_for_efa)
    
    # Factor loadings
    loadings_df = pd.DataFrame(
        fa.loadings_,
        index=data_for_efa.columns,
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    ).round(3)
    
    # Factor correlation matrix
    if hasattr(fa, 'phi_') and fa.phi_ is not None:
        factor_corr_df = pd.DataFrame(
            fa.phi_.round(3),
            index=[f"Factor{i+1}" for i in range(n_factors)],
            columns=[f"Factor{i+1}" for i in range(n_factors)]
        )
    else:
        factor_corr_df = pd.DataFrame()
    
    # Communalities
    communalities = pd.DataFrame({
        'Item': data_for_efa.columns,
        'Communality': fa.get_communalities().round(3),
        'Uniqueness': (1 - fa.get_communalities()).round(3)
    })
    
    return loadings_df, factor_corr_df, communalities


def create_regression_sheet(composite_scores_final, regression_models):
    """
    Tạo bảng kết quả hồi quy cho Excel sheet
    
    Args:
        composite_scores_final: DataFrame chứa composite scores
        regression_models: Các mô hình hồi quy
        
    Returns:
        pd.DataFrame: DataFrame chứa kết quả hồi quy
    """
    scaler = StandardScaler()
    composite_scores_std = pd.DataFrame(
        scaler.fit_transform(composite_scores_final),
        columns=composite_scores_final.columns
    )
    
    regression_results = []
    
    for model_spec in regression_models:
        dependent_var = model_spec["dependent"]
        independent_vars = model_spec["independent"]
        expected_order = model_spec["order"]
        
        if not all(v in composite_scores_std.columns for v in [dependent_var] + independent_vars):
            continue
            
        X = sm.add_constant(composite_scores_std[independent_vars])
        y = composite_scores_std[dependent_var]
        
        try:
            model = sm.OLS(y, X).fit()
            
            # Thông tin model
            model_info = {
                'Dependent_Variable': dependent_var,
                'Independent_Variables': ', '.join(independent_vars),
                'R_Squared': model.rsquared,
                'Adj_R_Squared': model.rsquared_adj,
                'F_Statistic': model.fvalue,
                'F_pvalue': model.f_pvalue,
                'Durbin_Watson': sm.stats.durbin_watson(model.resid)
            }
            regression_results.append(model_info)
            
            # Coefficients
            for i, (var, coef) in enumerate(zip(['const'] + independent_vars, model.params)):
                coef_info = {
                    'Dependent_Variable': dependent_var,
                    'Variable': var,
                    'Coefficient': coef,
                    'Std_Error': model.bse[i],
                    't_value': model.tvalues[i],
                    'p_value': model.pvalues[i],
                    'CI_Lower': model.conf_int()[0][i],
                    'CI_Upper': model.conf_int()[1][i]
                }
                regression_results.append(coef_info)
                
        except Exception:
            continue
    
    return pd.DataFrame(regression_results)


def create_diagnostics_sheet(data_for_analysis, factors_config, n_factors):
    """
    Tạo bảng kết quả diagnostics cho Excel sheet
    
    Args:
        data_for_analysis: DataFrame chứa dữ liệu phân tích
        factors_config: Cấu hình các nhân tố
        n_factors: Số lượng nhân tố tiềm ẩn
        
    Returns:
        pd.DataFrame: DataFrame chứa kết quả diagnostics
    """
    diagnostics_data = []
    
    # Cronbach's Alpha
    for factor_name, config in factors_config.items():
        items = config["original_items"]
        alpha = calculate_cronbach_alpha(data_for_analysis[items])
        diagnostics_data.append({
            'Test': 'Cronbach\'s Alpha',
            'Factor': factor_name,
            'Value': alpha,
            'Threshold': 0.7,
            'Status': 'PASS' if alpha >= 0.7 else 'FAIL'
        })
    
    # KMO và Bartlett's Test
    try:
        data_for_efa = data_for_analysis.dropna()
        _, kmo_model = calculate_kmo(data_for_efa)
        chi_square_value, p_value_bartlett = calculate_bartlett_sphericity(data_for_efa)
        
        diagnostics_data.extend([
            {
                'Test': 'KMO Overall',
                'Factor': 'Overall',
                'Value': kmo_model,
                'Threshold': 0.6,
                'Status': 'PASS' if kmo_model >= 0.6 else 'FAIL'
            },
            {
                'Test': 'Bartlett\'s Test p-value',
                'Factor': 'Overall',
                'Value': p_value_bartlett,
                'Threshold': 0.05,
                'Status': 'PASS' if p_value_bartlett < 0.05 else 'FAIL'
            }
        ])
    except Exception:
        pass
    
    return pd.DataFrame(diagnostics_data)


def export_multi_sheet_excel(data_for_analysis, composite_scores_final, factors_config, regression_models, n_factors, output_path):
    """
    Xuất dữ liệu ra file Excel multi-sheet
    
    Args:
        data_for_analysis: DataFrame chứa dữ liệu phân tích
        composite_scores_final: DataFrame chứa composite scores
        factors_config: Cấu hình các nhân tố
        regression_models: Các mô hình hồi quy
        n_factors: Số lượng nhân tố tiềm ẩn
        output_path: Đường dẫn file output
    """
    print("[LOG] Đang tạo multi-sheet Excel...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Generated Data (Raw Likert-scale data)
        generated_data = pd.concat([data_for_analysis, composite_scores_final], axis=1)
        generated_data.to_excel(writer, sheet_name='Generated_Data', index=False)
        
        # Sheet 2: Statistical Analysis
        stats_df = create_statistical_analysis_sheet(data_for_analysis, factors_config)
        stats_df.to_excel(writer, sheet_name='Statistical_Analysis', index=False)
        
        # Sheet 3: Correlation Matrix
        corr_df = create_correlation_sheet(data_for_analysis)
        corr_df.to_excel(writer, sheet_name='Correlation_Matrix', index=True)
        
        # Sheet 4: Factor Analysis - Loadings
        loadings_df, factor_corr_df, communalities_df = create_factor_analysis_sheet(data_for_analysis, n_factors)
        loadings_df.to_excel(writer, sheet_name='Factor_Loadings', index=True)
        
        # Sheet 5: Factor Analysis - Correlations
        if not factor_corr_df.empty:
            factor_corr_df.to_excel(writer, sheet_name='Factor_Correlations', index=True)
        
        # Sheet 6: Factor Analysis - Communalities
        communalities_df.to_excel(writer, sheet_name='Communalities', index=False)
        
        # Sheet 7: Regression Results
        regression_df = create_regression_sheet(composite_scores_final, regression_models)
        if not regression_df.empty:
            regression_df.to_excel(writer, sheet_name='Regression_Results', index=False)
        
        # Sheet 8: Diagnostics
        diagnostics_df = create_diagnostics_sheet(data_for_analysis, factors_config, n_factors)
        diagnostics_df.to_excel(writer, sheet_name='Diagnostics', index=False)
        
        # Sheet 9: Composite Scores Only
        composite_scores_final.to_excel(writer, sheet_name='Composite_Scores', index=False)
    
    print(f"[LOG] Multi-sheet Excel đã được lưu vào: {output_path}")


if __name__ == "__main__":
    # Test functions
    print("Excel Export Module Loaded Successfully")
    print("Available functions:")
    print("- create_statistical_analysis_sheet()")
    print("- create_correlation_sheet()")
    print("- create_factor_analysis_sheet()")
    print("- create_regression_sheet()")
    print("- create_diagnostics_sheet()")
    print("- export_multi_sheet_excel()")