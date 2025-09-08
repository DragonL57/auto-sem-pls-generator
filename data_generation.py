import numpy as np
import pandas as pd

def generate_items_from_latent(latent_factor, num_items, mean_val=3.0, sd_val=1.0, loading_strength=0.6, error_strength=0.4, rng=None):
    rng = rng or np.random.default_rng()
    num_obs_current = len(latent_factor)
    item_data = np.empty((num_obs_current, num_items), dtype=float)
    if sd_val < np.finfo(float).eps: 
        sd_val = 0.1 
    for i in range(num_items):
        item_values = loading_strength * latent_factor + error_strength * rng.normal(loc=0, scale=1, size=num_obs_current)
        if np.std(item_values, ddof=1) < np.finfo(float).eps:
            item_values_scaled = np.full(num_obs_current, mean_val)
        else:
            item_values_scaled = (item_values - np.mean(item_values)) / np.std(item_values, ddof=1) * sd_val + mean_val
        item_data[:, i] = np.clip(np.round(item_values_scaled), 1, 5)
    return pd.DataFrame(item_data)
