import scipy.stats as stats
from scipy.optimize import bisect

def bounded_gamma(rng, max_v, percentile=0.95):
    """
    Sample from a gamma distribution with mode=1, scale chosen so that
    max_v is the given percentile (default 95th). Returns a value in [0, max_v].
    """
    def objective(scale):
        shape = 1/scale + 1
        return stats.gamma.cdf(max_v, a=shape, scale=scale) - percentile
    fa = objective(1e-8)
    fb = objective(max_v)
    if fa * fb > 0:
        raise ValueError(f"Cannot bracket root for gamma scale: objective(1e-8)={fa}, objective(max_v)={fb}. Try a different max_v or percentile.")
    scale = bisect(objective, 1e-8, max_v)
    shape = 1/scale + 1
    for _ in range(100):
        val = rng.gammavariate(shape, scale)
        if val <= max_v:
            return val
    return max_v  # fallback
