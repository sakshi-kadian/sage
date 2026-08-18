import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def bootstrap_ci(data, metric_fn, n_resamples=1000, ci=0.95, seed=42):
    """
    Computes the bootstrap confidence interval for a given metric.
    
    Args:
        data (np.ndarray): The data array to sample from.
        metric_fn (callable): A function that takes data and returns a scalar metric.
        n_resamples (int): Number of bootstrap resamples.
        ci (float): Confidence interval level (e.g., 0.95 for 95%).
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    bootstrapped_metrics = np.zeros(n_resamples)
    
    for i in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        bootstrapped_metrics[i] = metric_fn(sample)
        
    alpha = 1.0 - ci
    lower_bound = np.percentile(bootstrapped_metrics, alpha / 2.0 * 100)
    upper_bound = np.percentile(bootstrapped_metrics, (1.0 - alpha / 2.0) * 100)
    
    return lower_bound, upper_bound

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Performs McNemar's Chi-Squared test to compare two models.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred1 (np.ndarray): Predictions from model 1.
        y_pred2 (np.ndarray): Predictions from model 2.
        
    Returns:
        tuple: (chi2_statistic, p_value)
    """
    # Create the contingency table
    # Cell [0, 0]: Model 1 correct, Model 2 correct
    # Cell [0, 1]: Model 1 correct, Model 2 incorrect
    # Cell [1, 0]: Model 1 incorrect, Model 2 correct
    # Cell [1, 1]: Model 1 incorrect, Model 2 incorrect
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n00 = np.sum(correct1 & correct2)
    n01 = np.sum(correct1 & ~correct2)
    n10 = np.sum(~correct1 & correct2)
    n11 = np.sum(~correct1 & ~correct2)
    
    table = [[n00, n01],
             [n10, n11]]
             
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue
