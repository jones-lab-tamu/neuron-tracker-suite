# -*- coding: utf-8 -*-
"""
Cosinor Analysis Core Logic

Provides a function to perform single-component Cosinor analysis on time-series data.
"""
import numpy as np
from scipy import stats

def cosinor_analysis(data, time, period):
    """
    Performs a single-component Cosinor analysis on a time series.

    Args:
        data (np.ndarray): 1D array of the signal (e.g., intensity).
        time (np.ndarray): 1D array of time points corresponding to the data.
        period (float): The known period to fit the cosine function to.

    Returns:
        dict: A dictionary containing the results of the fit:
              'mesor', 'amplitude', 'acrophase', 'p_value', 'r_squared'.
              Returns a dict of NaNs if the fit fails.
    """
    if len(data) < 3 or len(data) != len(time):
        return {
            'mesor': np.nan, 'amplitude': np.nan, 'acrophase': np.nan,
            'p_value': np.nan, 'r_squared': np.nan
        }

    # --- Setup the linear model: y = M + A*cos(2*pi*t/T) + B*sin(2*pi*t/T) ---
    w = 2 * np.pi / period
    X = np.array([
        np.ones_like(time),
        np.cos(w * time),
        np.sin(w * time)
    ]).T
    y = data

    # --- Perform Ordinary Least Squares (OLS) regression ---
    try:
        # (X'X)^-1 * X'y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        M, A, B = beta[0], beta[1], beta[2]
    except np.linalg.LinAlgError:
        # Matrix is singular, fit cannot be performed
        return {
            'mesor': np.nan, 'amplitude': np.nan, 'acrophase': np.nan,
            'p_value': np.nan, 'r_squared': np.nan
        }

    # --- Calculate Cosinor parameters ---
    mesor = M
    amplitude = np.sqrt(A**2 + B**2)
    acrophase_rad = np.arctan2(B, A)
    acrophase = (acrophase_rad / w) % period

    # --- Calculate goodness-of-fit statistics (p-value and R^2) ---
    n = len(y)
    k = len(beta) # Number of parameters (3)
    
    y_fit = X @ beta
    residuals = y - y_fit
    
    sum_sq_total = np.sum((y - np.mean(y))**2)
    sum_sq_residuals = np.sum(residuals**2)
    
    if sum_sq_total == 0: # No variance in data
        return {
            'mesor': mesor, 'amplitude': amplitude, 'acrophase': acrophase,
            'p_value': np.nan, 'r_squared': 1.0
        }

    r_squared = 1 - (sum_sq_residuals / sum_sq_total)

    # F-statistic for zero-amplitude test
    sum_sq_model = sum_sq_total - sum_sq_residuals
    ms_model = sum_sq_model / (k - 1)
    ms_residuals = sum_sq_residuals / (n - k)

    if ms_residuals == 0: # Perfect fit
        p_value = 0.0
    else:
        f_statistic = ms_model / ms_residuals
        p_value = 1 - stats.f.cdf(f_statistic, k - 1, n - k)

    return {
        'mesor': mesor,
        'amplitude': amplitude,
        'acrophase': acrophase,
        'p_value': p_value,
        'r_squared': r_squared
    }