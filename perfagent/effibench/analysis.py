import numpy as np
import scipy.stats as st


def analyze_samples(
    samples: list[float],
    confidence: float = 0.95,
    trim_ratio: float = 0.05,
    iqr_k: float = 1.0,
) -> dict[str, float]:
    """
    Analyzes a list of runtimes, filtering outliers using the IQR method.

    This function first removes outliers from the runtime samples using the
    Interquartile Range (IQR) method. It then calculates various statistical
    measures on the filtered data, including mean, standard deviation, min, max,
    confidence interval, and a trimmed mean.

    Args:
        samples: A list of floats representing the runtimes of multiple executions.
        confidence: The confidence level for the confidence interval (e.g., 0.95 for 95%).
        trim_ratio: The fraction of observations to be trimmed from each end of the
                    filtered runtimes before the trimmed mean is computed.
        iqr_k: The multiplier for the IQR. Data points outside
               [Q1 - k*IQR, Q3 + k*IQR] are removed as outliers.

    Returns:
        A dictionary containing various statistical measures of the runtimes.
        Returns default values if the list is empty or all samples are filtered out.
    """
    if not samples:
        return {
            "original_n": 0,
            "n": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "max_diff": float("inf"),
            "95%_CI": (float("inf"), float("inf")),
            "trimmed_mean": float("inf"),
        }

    samples_np = np.array(samples)
    original_n = len(samples_np)

    if original_n == 1:
        return {
            "original_n": original_n,
            "n": 1,
            "mean": samples_np[0],
            "std": 0.0,
            "min": samples_np[0],
            "max": samples_np[0],
            "max_diff": 0.0,
            "95%_CI": (samples_np[0], samples_np[0]),
            "trimmed_mean": samples_np[0],
        }

    # IQR outlier removal
    Q1 = np.percentile(samples_np, 25)
    Q3 = np.percentile(samples_np, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_k * IQR
    upper_bound = Q3 + iqr_k * IQR

    filtered_samples = samples_np[(samples_np >= lower_bound) & (samples_np <= upper_bound)]

    if len(filtered_samples) == 0:
        return {
            "original_n": original_n,
            "n": 0,
            "mean": float("inf"),
            "std": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "max_diff": float("inf"),
            "95%_CI": (float("inf"), float("inf")),
            "trimmed_mean": float("inf"),
        }

    samples_np = filtered_samples

    mean = samples_np.mean()
    std = samples_np.std(ddof=1) if len(samples_np) > 1 else 0.0
    min_val = samples_np.min()
    max_val = samples_np.max()
    max_diff = max_val - min_val

    # Confidence Interval
    ci_low, ci_high = float("inf"), float("inf")
    if len(samples_np) > 1:
        ci_low, ci_high = st.t.interval(confidence, df=len(samples_np) - 1, loc=mean, scale=st.sem(samples_np))

    # Trimmed mean on filtered data
    sorted_samples = np.sort(samples_np)
    n = len(samples_np)
    k = int(n * trim_ratio)

    trimmed = sorted_samples[k : n - k] if k > 0 and (n - 2 * k) > 0 else sorted_samples
    trimmed_mean = trimmed.mean() if len(trimmed) > 0 else float("inf")

    return {
        "original_n": original_n,
        "n": len(samples_np),
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "max_diff": max_diff,
        "95%_CI": (ci_low, ci_high),
        "trimmed_mean": trimmed_mean,
    }
