import numpy as np

def calculate_cronbach_alpha(df):
    """
    Calculate Cronbach's alpha for a given DataFrame.

    Parameters:
        df (pandas.DataFrame): DataFrame where each column represents an item and each row a respondent.

    Returns:
        float: Cronbach's alpha coefficient, or np.nan if not computable.
    """
    if df.shape[1] < 2:
        return np.nan 
    df_clean = df.dropna()
    if df_clean.empty:
        return np.nan 
    num_items = df_clean.shape[1]
    item_variances = df_clean.var(axis=0, ddof=1).sum()
    total_score_variance = df_clean.sum(axis=1).var(ddof=1)
    if total_score_variance == 0 or num_items == 1: 
        return np.nan
    alpha = (num_items / (num_items - 1)) * (1 - (item_variances / total_score_variance))
    return alpha
