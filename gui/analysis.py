import numpy as np
from scipy.signal import medfilt, find_peaks
from neuron_tracker_core import DEFAULT_PERIOD_HOURS

# ------------------------------------------------------------
# Phase calculation
# ------------------------------------------------------------

RHYTHM_TREND_WINDOW_HOURS = 36.0  # default detrend timescale

def compute_median_window_frames(minutes_per_frame, trend_window_hours, T=None):
    """
    Compute an odd-length median filter window in frames for a given trend window in hours.

    Parameters:
        minutes_per_frame : float
            Sampling interval in minutes.
        trend_window_hours : float
            Desired width of the detrending window in hours.
        T : int or None
            Optional number of frames in the trace, used to cap the window.

    Returns:
        median_window_frames : int (odd, >= 3)
    """
    if minutes_per_frame <= 0:
        # Fallback: use a small default window in frames if sampling is invalid
        frames = 3
    else:
        hours_per_frame = minutes_per_frame / 60.0
        if hours_per_frame <= 0:
            frames = 3
        else:
            frames = int(round(trend_window_hours / hours_per_frame))
            if frames < 3:
                frames = 3

    # Enforce odd window size
    if frames % 2 == 0:
        frames += 1

    # Cap by T if provided and meaningful
    if T is not None and T > 0:
        if frames > T:
            frames = T if (T % 2 == 1) else (T - 1)
            if frames < 3:
                frames = 3

    return frames

def preprocess_for_rhythmicity(trace, method="running_median",
                               median_window_frames=None,
                               poly_order=2):
    """
    Preprocess a raw fluorescence trace before rhythmicity analysis.
    Returns a detrended trace suitable for FFT or cosinor.

    Parameters:
        trace: 1D numpy array
        method: "running_median", "polynomial", or "none"
        median_window_frames: int or None
            If None and method == "running_median", caller must have computed a
            suitable window; this function will fall back to a small default.
        poly_order: polynomial order for polynomial detrending

    Returns:
        detrended_trace: 1D numpy array
    """
    trace = np.asarray(trace, dtype=float)

    if method == "none":
        return trace.copy()

    x = np.arange(len(trace))

    if method == "running_median":
        # If the caller did not provide a window, fall back to a minimal safe default.
        if median_window_frames is None:
            median_window_frames = 3
        if median_window_frames % 2 == 0:
            median_window_frames += 1
        baseline = medfilt(trace, kernel_size=median_window_frames)
        detrended = trace - baseline
        return detrended

    if method == "polynomial":
        # Global polynomial baseline
        if trace.size < (poly_order + 1):
            return trace.copy()
        coeffs = np.polyfit(x, trace, poly_order)
        baseline = np.polyval(coeffs, x)
        detrended = trace - baseline
        return detrended

    # Fallback
    return trace.copy()

def estimate_cycle_count_from_trace(
    trace,
    minutes_per_frame,
    period_hours,
    detrend_method="running_median",
    median_window_frames=None,
    trend_window_hours=RHYTHM_TREND_WINDOW_HOURS,
    smoothing_window_hours=2.0,
    min_prominence_fraction=0.2,
):
    """
    Estimate how many cycles a single trace contains around a target period.

    Parameters:
        trace : 1D array-like
            Raw fluorescence trace for one cell.
        minutes_per_frame : float
            Sampling interval in minutes.
        period_hours : float
            Target period (e.g., 24.0 for circadian).
        detrend_method : str
            Method passed to preprocess_for_rhythmicity ("running_median", "polynomial", or "none").
        median_window_frames : int or None
            If None and detrend_method == "running_median", window is derived from
            minutes_per_frame and trend_window_hours.
        trend_window_hours : float
            Trend window in hours for dynamic median-window computation.
        smoothing_window_hours : float
            Width of the smoothing window (in hours) for peak detection.
        min_prominence_fraction : float
            Minimum peak prominence as a fraction of the smoothed trace amplitude range.

    Returns:
        n_cycles : int
            Estimated number of cycles (roughly number of usable peaks minus 1).
        peak_indices : 1D np.ndarray of ints
            Indices of peaks used for counting.
    """
    trace = np.asarray(trace, dtype=float)
    T = trace.shape[0]
    if T < 3:
        return 0, np.array([], dtype=int)

    # Choose median window in frames if needed
    if detrend_method == "running_median" and median_window_frames is None:
        median_window_frames = compute_median_window_frames(
            minutes_per_frame,
            trend_window_hours,
            T=T,
        )

    detr = preprocess_for_rhythmicity(
        trace,
        method=detrend_method,
        median_window_frames=median_window_frames,
    )

    hours_per_frame = minutes_per_frame / 60.0
    if hours_per_frame <= 0:
        return 0, np.array([], dtype=int)

    window_frames = int(round(smoothing_window_hours / hours_per_frame))
    if window_frames < 1:
        window_frames = 1
    if window_frames % 2 == 0:
        window_frames += 1

    if window_frames > 1:
        kernel = np.ones(window_frames, dtype=float) / float(window_frames)
        smooth = np.convolve(detr, kernel, mode="same")
    else:
        smooth = detr

    amp_range = float(smooth.max() - smooth.min())
    if amp_range <= 0:
        return 0, np.array([], dtype=int)

    prominence = amp_range * float(min_prominence_fraction)
    if prominence <= 0:
        prominence = amp_range * 0.1

    peaks, _ = find_peaks(smooth, prominence=prominence)
    if peaks.size == 0:
        return 0, peaks

    expected_frames = period_hours / hours_per_frame
    if expected_frames <= 0:
        return int(peaks.size), peaks

    min_dist = 0.5 * expected_frames
    max_dist = 1.5 * expected_frames

    good_peaks = []
    prev_peak = None
    for idx in peaks:
        if prev_peak is None:
            good_peaks.append(idx)
            prev_peak = idx
        else:
            d = idx - prev_peak
            if min_dist <= d <= max_dist:
                good_peaks.append(idx)
                prev_peak = idx
            elif d > max_dist:
                good_peaks.append(idx)
                prev_peak = idx
            else:
                continue

    good_peaks = np.array(good_peaks, dtype=int)
    n_cycles = max(int(good_peaks.size - 1), 0)
    return n_cycles, good_peaks

def strict_cycle_mask(
    traces_data,
    minutes_per_frame,
    period_hours,
    base_mask,
    min_cycles=2,
    detrend_method="running_median",
    median_window_frames=None,
    trend_window_hours=RHYTHM_TREND_WINDOW_HOURS,
    smoothing_window_hours=2.0,
    min_prominence_fraction=0.2,
):
    """
    Refine a base rhythmicity mask by requiring a minimum number of cycles.

    Parameters:
        traces_data : 2D array-like, shape (T, N+1)
            First column is time, remaining columns are per-cell traces.
        minutes_per_frame : float
            Sampling interval in minutes.
        period_hours : float
            Target period (e.g., 24.0 for circadian).
        base_mask : 1D boolean array, length N
            Initial rhythmicity decision per cell (e.g., Cosinor/FFT filter).
        min_cycles : int
            Minimum number of cycles required to keep a cell as "rhythmic".
        detrend_method, median_window_frames, trend_window_hours,
        smoothing_window_hours, min_prominence_fraction :
            Parameters passed through to estimate_cycle_count_from_trace.

    Returns:
        strict_mask : 1D boolean np.ndarray, length N
            Refined mask that incorporates both base_mask and cycle-count requirement.
    """
    traces_data = np.asarray(traces_data)
    base_mask = np.asarray(base_mask, dtype=bool)

    if traces_data.ndim != 2 or traces_data.shape[1] < 2:
        raise ValueError("traces_data must have shape (T, N+1).")

    T, cols = traces_data.shape
    N = cols - 1

    if base_mask.shape[0] != N:
        raise ValueError("base_mask length must equal number of cells (N).")

    strict_mask = base_mask.copy()
    if not np.any(base_mask):
        return strict_mask

    intensities = traces_data[:, 1:]

    for i in range(N):
        if not base_mask[i]:
            continue

        trace = intensities[:, i]
        n_cycles, _ = estimate_cycle_count_from_trace(
            trace,
            minutes_per_frame=minutes_per_frame,
            period_hours=period_hours,
            detrend_method=detrend_method,
            median_window_frames=median_window_frames,
            trend_window_hours=trend_window_hours,
            smoothing_window_hours=smoothing_window_hours,
            min_prominence_fraction=min_prominence_fraction,
        )

        if n_cycles < min_cycles:
            strict_mask[i] = False

    return strict_mask

def calculate_phases_fft(traces_data,
                         minutes_per_frame,
                         period_min=None,
                         period_max=None,
                         detrend_method="running_median",
                         detrend_window_frames=None,
                         detrend_window_hours=RHYTHM_TREND_WINDOW_HOURS):
    """
    Compute per-cell FFT phases and a sideband-based SNR rhythmicity score.

    Parameters:
        traces_data: 2D array of shape (T, N+1) where first column is time
                     and remaining columns are fluorescence traces.
        minutes_per_frame: float
        period_min, period_max: optional bounds on allowed periods (hours)
        detrend_method: "running_median", "polynomial", or "none"
        detrend_window_frames: int or None
            If None and detrend_method == "running_median", the window will be
            derived from minutes_per_frame and detrend_window_hours.
        detrend_window_hours: float
            Trend window width in hours for dynamic median-window computation.

    Returns:
        phases: array of shape (N,)
        period_hours: float, dominant period
        rhythm_snr_scores: array of shape (N,)
    """

    intensities = traces_data[:, 1:]        # (T, N)
    T, N = intensities.shape

    # Time → frequency mapping
    dt_hours = minutes_per_frame / 60.0
    freqs = np.fft.rfftfreq(T, d=dt_hours)

    # Choose median window in frames if needed
    if detrend_method == "running_median" and detrend_window_frames is None:
        detrend_window_frames = compute_median_window_frames(
            minutes_per_frame,
            detrend_window_hours,
            T=T,
        )

    # Detrend ALL traces before spectrum analysis
    detrended = np.zeros_like(intensities)
    for i in range(N):
        detrended[:, i] = preprocess_for_rhythmicity(
            intensities[:, i],
            method=detrend_method,
            median_window_frames=detrend_window_frames,
        )

    # Mean trace used to estimate dominant period
    mean_signal = detrended.mean(axis=1)
    fft_mean = np.fft.rfft(mean_signal)
    power_mean = np.abs(fft_mean) ** 2

    # Determine allowed frequency range for peak search
    if period_min is not None and period_max is not None:
        fmin = 1.0 / period_max
        fmax = 1.0 / period_min
        mask = (freqs >= fmin) & (freqs <= fmax)
    else:
        mask = np.ones_like(freqs, dtype=bool)

    masked_power = power_mean.copy()

    # Make out-of-mask bins impossible to select
    masked_power[~mask] = -np.inf

    # Make DC impossible to select
    if len(masked_power) > 0:
        masked_power[0] = -np.inf

    peak_idx = int(np.argmax(masked_power))

    # If nothing valid has positive power (or all are -inf), fall back to the closest bin to 1/DEFAULT_PERIOD_HOURS,
    # but DO NOT keep period_hours=DEFAULT unless the chosen bin truly corresponds to 24 h.
    bad_peak = (not np.isfinite(masked_power[peak_idx])) or (masked_power[peak_idx] <= 0)

    if bad_peak:
        target_freq = 1.0 / DEFAULT_PERIOD_HOURS

        # Prefer choosing within mask (excluding DC) if any bins exist
        candidate = np.where(mask & (freqs > 0))[0]
        if candidate.size > 0:
            peak_idx = int(candidate[np.argmin(np.abs(freqs[candidate] - target_freq))])
        else:
            # No masked positive-frequency candidates, fall back to any positive-frequency bin
            pos = np.where(freqs > 0)[0]
            if pos.size > 0:
                peak_idx = int(pos[np.argmin(np.abs(freqs[pos] - target_freq))])
            else:
                # Truly degenerate case: no positive frequencies exist
                peak_idx = 0

    peak_freq = freqs[peak_idx]

    # period_hours MUST correspond to peak_idx used for phase extraction
    if peak_freq > 0:
        period_hours = 1.0 / peak_freq
    else:
        # last-resort fallback: if we truly cannot pick a positive frequency bin, use DEFAULT
        period_hours = DEFAULT_PERIOD_HOURS

    # Final safety check for finiteness
    if not np.isfinite(period_hours) or period_hours <= 0:
        period_hours = DEFAULT_PERIOD_HOURS

    # Sideband SNR: signal band = ±1 bins; noise = ±10 bins outside that
    signal_band = []
    for off in (-1, 0, 1):
        idx = peak_idx + off
        if 0 <= idx < len(freqs):
            signal_band.append(idx)
    signal_band = np.array(signal_band, dtype=int)

    noise_band = []
    for off in range(2, 11):
        for sign in (-1, +1):
            idx = peak_idx + sign * off
            if 0 <= idx < len(freqs):
                noise_band.append(idx)
    noise_band = np.array(noise_band, dtype=int)

    phases = np.zeros(N)
    rhythm_snr_scores = np.zeros(N)

    for i in range(N):
        trace = detrended[:, i]
        fft_vals = np.fft.rfft(trace)
        power_vals = np.abs(fft_vals) ** 2

        complex_val = fft_vals[peak_idx]
        phases[i] = -np.angle(complex_val)

        signal_power = power_vals[signal_band].sum()
        noise_power = power_vals[noise_band].mean() if len(noise_band) > 0 else 1e-9
        snr = signal_power / (noise_power + 1e-9)
        rhythm_snr_scores[i] = snr

    return phases, period_hours, rhythm_snr_scores
