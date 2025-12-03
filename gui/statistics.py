# gui/statistics.py
import numpy as np
from scipy.stats import circmean

def circmean_hours(phases, period):
    """
    Circular mean of phases given in hours.
    Returns mean in hours on the same scale [0, period].
    """
    phases = np.asarray(phases, dtype=float)
    if phases.size == 0:
        return np.nan
    
    # Convert to radians [-pi, pi]
    # (Phase / Period) * 2pi
    angles = phases * (2 * np.pi / period)
    
    # Calculate circular mean in radians
    mean_angle = circmean(angles, low=-np.pi, high=np.pi)
    
    # Convert back to hours
    return mean_angle * (period / (2 * np.pi))

def watson_williams_f(group_a, group_b):
    """
    Watson-Williams F-test for two circular samples.
    Input: Arrays of phases in Hours.
    Assumption: Period is 24.0 hours (standard for group analysis).
    Returns: F-statistic (float).
    """
    a = np.asarray(group_a)
    b = np.asarray(group_b)
    
    n1 = len(a)
    n2 = len(b)
    N = n1 + n2
    
    if n1 == 0 or n2 == 0:
        return 0.0

    # Convert hours to radians (Assumes Period=24.0)
    period = 24.0 
    rad_a = a * (2 * np.pi / period)
    rad_b = b * (2 * np.pi / period)
    
    # Calculate resultant vectors
    R1 = np.abs(np.sum(np.exp(1j * rad_a)))
    R2 = np.abs(np.sum(np.exp(1j * rad_b)))
    
    combined = np.concatenate([rad_a, rad_b])
    R = np.abs(np.sum(np.exp(1j * combined)))
    
    # Watson-Williams Formula
    numerator = (N - 2) * (R1 + R2 - R)
    denominator = (N - (R1 + R2))
    
    if denominator < 1e-9:
        return 0.0
    
    f_val = numerator / denominator
    return max(0, f_val)