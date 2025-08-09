import numpy as np
import pandas as pd

from gosales.pipeline.rank_whitespace import _percentile_normalize


def test_capture_at_k_math():
    # Create a dataset with known positives concentrated at top of score
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:100] = 1  # 10% positives
    score = np.concatenate([np.linspace(1, 0.6, 100), np.linspace(0.59, 0.0, 900)])
    df = pd.DataFrame({'label': y, 'score': score})
    # Capture@10% should be close to 1.0 (all positives in top decile)
    k = int(n * 0.10)
    topk = df.nlargest(k, 'score')
    capture = topk['label'].sum() / max(1, df['label'].sum())
    assert capture > 0.95


