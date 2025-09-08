"""
Bayesian Optimization Module for SEM/PLS Parameter Optimization
Thay thế Genetic Algorithm bằng Bayesian Optimization để tối ưu hóa hiệu suất
"""

import numpy as np
import time
import warnings
from skopt import gp_minimize
from skopt.space import Real
from skopt.callbacks import EarlyStopper
from evaluation import evaluate_parameters_wrapper
from config import (
    n_latent_factors, n_latent_cor_values, factors_config, regression_models, num_observations,
    bo_n_calls, bo_n_initial_points, bo_acq_func, bo_n_jobs, bo_early_stopping, bo_patience,
    bo_latent_cor_min, bo_latent_cor_max, bo_error_strength_min, bo_error_strength_max, bo_loading_strength_min, bo_loading_strength_max
)

# Ẩn warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class BayesianOptimizer:
    def __init__(self, optimizer_config):
        """
        Khởi tạo Bayesian Optimizer
        
        Args:
            optimizer_config: Dictionary containing optimization parameters
        """
        self.config = optimizer_config
        self.n_calls = optimizer_config.get('n_calls', bo_n_calls)  # Số lần đánh giá
        self.n_initial_points = optimizer_config.get('n_initial_points', bo_n_initial_points)  # Số điểm khởi tạo
        self.acq_func = optimizer_config.get('acq_func', bo_acq_func)  # Acquisition function
        self.n_jobs = optimizer_config.get('n_jobs', bo_n_jobs)  # Số processes (-1 = tất cả)
        self.early_stopping = optimizer_config.get('early_stopping', bo_early_stopping)
        self.patience = optimizer_config.get('patience', bo_patience)  # Số iteration chờ trước khi dừng
        
        # Khởi tạo search space - giảm upper bound để tránh Heywood cases
        self.search_space = self._create_search_space()
        
        # Theo dõi kết quả
        self.best_score = -np.inf
        self.best_params = None
        self.best_reason = ""
        self.evaluation_history = []
        self.time_history = []
        
    def _create_search_space(self):
        """
        Tạo search space cho Bayesian Optimization - giảm upper bound để tránh Heywood cases
        
        Returns:
            list: Danh sách các dimension
        """
        search_space = []
        
        # Latent correlation values - sử dụng tham số từ config
        for i in range(n_latent_cor_values):
            search_space.append(Real(bo_latent_cor_min, bo_latent_cor_max, name=f'latent_cor_{i}'))
            
        # Error strength
        search_space.append(Real(bo_error_strength_min, bo_error_strength_max, name='error_strength'))
        
        # Loading strength
        search_space.append(Real(bo_loading_strength_min, bo_loading_strength_max, name='loading_strength'))
        
        return search_space
    
    def objective_function(self, params):
        """
        Objective function cho Bayesian Optimization
        
        Args:
            params: Danh sách các tham số cần tối ưu
            
        Returns:
            float: Giá trị objective (negative fitness vì gp_minimize minimize)
        """
        start_time = time.time()
        
        try:
            # Chuyển đổi params thành array
            params_array = np.array(params)
            
            # Tạo seed ngẫu nhiên
            rng_seed = np.random.randint(0, 1000000)
            
            # Đánh giá tham số
            fitness_score, reason = evaluate_parameters_wrapper(
                (params_array, factors_config, regression_models, num_observations, 
                 rng_seed, n_latent_factors, n_latent_cor_values)
            )
            
            # Cập nhật best solution
            if fitness_score > self.best_score:
                self.best_score = fitness_score
                self.best_params = params_array.copy()
                self.best_reason = reason
                
            # Lưu lịch sử
            self.evaluation_history.append(fitness_score)
            self.time_history.append(time.time() - start_time)
            
            # Print progress
            current_eval = len(self.evaluation_history)
            if current_eval % 5 == 0:
                print(f"Evaluation {current_eval}/{self.n_calls}: Best score = {self.best_score:.2f}, Current = {fitness_score:.2f}")
            
            # Return negative vì gp_minimize tìm minimum
            return -fitness_score
            
        except (ValueError, TypeError, RuntimeError, IndexError, KeyError) as e:
            print(f"Error in evaluation: {e}")
            # Trả về penalty score nếu có lỗi
            return 1_000_000
    
    def optimize(self):
        """
        Chạy quá trình optimization
        
        Returns:
            tuple: (best_params, best_score, best_reason, evaluation_history)
        """
        print("==================================================")
        print("BẮT ĐẦU QUÁ TRÌNH TỐI ƯU HÓA (BAYESIAN OPTIMIZATION)")
        print(f"Số evaluations: {self.n_calls}")
        print(f"Số điểm khởi tạo: {self.n_initial_points}")
        print(f"Acquisition function: {self.acq_func}")
        print("==================================================")
        
        start_time = time.time()
        
        # Tạo callback cho early stopping
        callback = None
        if self.early_stopping:
            callback = self._create_early_stopper()
        
        # Run Bayesian Optimization
        _result = gp_minimize(
            func=self.objective_function,
            dimensions=self.search_space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            n_jobs=self.n_jobs,
            callback=callback,
            random_state=42,
            verbose=True
        )
        
        total_time = time.time() - start_time
        
        # In kết quả cuối cùng
        print("\n==================================================")
        print("QUÁ TRÌNH TỐI ƯU HÓA (BAYESIAN OPTIMIZATION) HOÀN TẤT")
        print(f"Tổng thời gian chạy: {total_time:.2f} giây")
        print(f"Số evaluations thực tế: {len(self.evaluation_history)}")
        print("==================================================")
        print(f"Điểm số tốt nhất tìm được: {self.best_score:.2f}")
        print(f"Lý do: {self.best_reason}")
        
        return self.best_params, self.best_score, self.best_reason, self.evaluation_history
    
    def _create_early_stopper(self):
        """
        Tạo early stopper callback
        
        Returns:
            EarlyStopper: Callback object
        """
        def early_stopper(_result):
            if len(self.evaluation_history) < self.patience:
                return False
                
            # Kiểm tra nếu không cải thiện trong 'patience' evaluations
            recent_scores = self.evaluation_history[-self.patience:]
            best_recent = max(recent_scores)
            
            if best_recent <= self.best_score:
                print(f"\n[Early Stopping] No improvement for {self.patience} evaluations")
                print(f"Best score: {self.best_score:.2f}")
                return True
                
            return False
            
        return EarlyStopper(early_stopper)
    
    def get_optimization_summary(self):
        """
        Lấy summary của quá trình optimization
        
        Returns:
            dict: Summary information
        """
        if not self.evaluation_history:
            return {}
            
        return {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'best_reason': self.best_reason,
            'total_evaluations': len(self.evaluation_history),
            'convergence_rate': self._calculate_convergence_rate(),
            'average_time_per_eval': np.mean(self.time_history) if self.time_history else 0,
            'improvement_ratio': self._calculate_improvement_ratio()
        }
    
    def _calculate_convergence_rate(self):
        """
        Tính toán convergence rate
        
        Returns:
            float: Convergence rate
        """
        if len(self.evaluation_history) < 10:
            return 0.0
            
        # Tính toán slope của 10 evaluations cuối cùng
        recent_scores = self.evaluation_history[-10:]
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        return slope
    
    def _calculate_improvement_ratio(self):
        """
        Tính toán improvement ratio
        
        Returns:
            float: Improvement ratio
        """
        if len(self.evaluation_history) < 2:
            return 0.0
            
        initial_score = self.evaluation_history[0]
        final_score = self.best_score
        
        if initial_score == 0:
            return 0.0
            
        return (final_score - initial_score) / abs(initial_score)

def create_bayesian_config():
    """
    Tạo config cho Bayesian Optimization sử dụng tham số từ config.py
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        'n_calls': bo_n_calls,
        'n_initial_points': bo_n_initial_points,
        'acq_func': bo_acq_func,
        'n_jobs': bo_n_jobs,
        'early_stopping': bo_early_stopping,
        'patience': bo_patience
    }

if __name__ == "__main__":
    # Test Bayesian Optimizer
    config = create_bayesian_config()
    optimizer = BayesianOptimizer(config)
    
    # Run optimization
    best_params, best_score, best_reason, history = optimizer.optimize()
    
    # Print summary
    summary = optimizer.get_optimization_summary()
    print("\n=== Optimization Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")