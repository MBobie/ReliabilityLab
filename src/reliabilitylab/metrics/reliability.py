"""Reliability statistics for repeated experiments."""


import numpy as np
from scipy import stats


def summarize_repeated_runs(values) -> dict[str, float]:
    """Summarize a collection of repeated experiment scores.

    Parameters
    ----------
    values
        Sequence of metric values, such as accuracy scores.

    Returns
    -------
    dict
        Summary statistics describing central tendency,
        dispersion, confidence interval, and peak performance.
    """

    values = np.asarray(values, dtype=float)

    if values.size < 2:
        raise ValueError(
            "At least two experiment runs are required "
            "to calculate reliability statistics."
        )

    n = len(values)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    median = float(np.median(values))
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    score_range = maximum - minimum

    standard_error = std / np.sqrt(n)

    t_critical = stats.t.ppf(
        0.975,
        df=n - 1,
    )

    ci_margin = t_critical * standard_error

    ci_lower = mean - ci_margin
    ci_upper = mean + ci_margin

    peak_mean_gap = maximum - mean

    return {
        "n_runs": n,
        "mean": mean,
        "std": std,
        "median": median,
        "min": minimum,
        "max": maximum,
        "range": score_range,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "peak_mean_gap": peak_mean_gap,
    }