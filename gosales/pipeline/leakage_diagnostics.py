from __future__ import annotations

"""
Quick leakage diagnostics: label permutation and importance stability.

Outputs JSON summaries and optional PNG plots under:
  gosales/outputs/leakage/<division>/<cutoff>/

Use SAFE audit defaults: apply gauntlet tail mask to windowed features.
"""

from pathlib import Path
import json
import math
import argparse
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

from gosales.utils.paths import OUTPUTS_DIR
from gosales.utils.config import load_config
from gosales.utils.db import get_curated_connection, get_db_connection
from gosales.features.engine import create_feature_matrix


def _time_aware_split(df: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rec_col = 'rfm__all__recency_days__life'
    try:
        if rec_col in df.columns:
            order = np.argsort(np.nan_to_num(pd.to_numeric(df[rec_col], errors='coerce').values.astype(float), nan=1e9))
            n = len(order)
            n_valid = max(1, int(0.2 * n))
            idx_valid = order[:n_valid]
            idx_train = order[n_valid:]
            return idx_train, idx_valid
    except Exception:
        pass
    # Fallback random split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    return idx[:split], idx[split:]


def _sanitize_numeric(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        if not (pd.api.types.is_integer_dtype(Xc[c]) or pd.api.types.is_float_dtype(Xc[c])):
            Xc[c] = pd.to_numeric(Xc[c], errors='coerce')
    Xc.replace([np.inf, -np.inf], np.nan, inplace=True)
    return Xc.fillna(0.0)


def _downsample(df: pd.DataFrame, max_rows: int = 12000, seed: int = 42) -> pd.DataFrame:
    if len(df) <= int(max_rows):
        return df
    rng = np.random.RandomState(seed)
    # Stratify by label if possible
    try:
        pos = df[df['bought_in_division'] == 1]
        neg = df[df['bought_in_division'] == 0]
        pos_n = len(pos)
        target_n = int(max_rows)
        pos_k = min(pos_n, max(1, int(target_n * (pos_n / max(1, len(df))))))
        neg_k = max(0, target_n - pos_k)
        pos_s = pos.sample(n=pos_k, random_state=seed, replace=False)
        neg_s = neg.sample(n=min(len(neg), neg_k), random_state=seed, replace=False)
        out = pd.concat([pos_s, neg_s], axis=0).sample(frac=1.0, random_state=seed)
        return out.reset_index(drop=True)
    except Exception:
        return df.sample(n=int(max_rows), random_state=seed, replace=False).reset_index(drop=True)


def _select_features(X: pd.DataFrame, max_features: int = 500) -> pd.DataFrame:
    cols = list(X.columns)
    if len(cols) <= int(max_features):
        return X
    try:
        var = X.var(axis=0, numeric_only=True)
        var = var.fillna(0.0)
        top = var.sort_values(ascending=False).head(int(max_features)).index.tolist()
        return X[top]
    except Exception:
        return X.iloc[:, : int(max_features)]


def run_label_permutation(out_dir: Path, df: pd.DataFrame, n_perm: int = 50, seed: int = 42, max_rows: int = 12000, max_features: int = 500) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df_s = _downsample(df, max_rows=max_rows, seed=seed)
    y = df_s['bought_in_division'].astype(int).values
    X = _sanitize_numeric(df_s.drop(columns=['customer_id', 'bought_in_division']))
    X = _select_features(X, max_features=max_features)
    it, iv = _time_aware_split(df_s, seed)

    # Use scaling + liblinear (binary) for faster convergence on reduced set
    clf = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('lr', LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        clf.fit(X.iloc[it], y[it])
        p = clf.predict_proba(X.iloc[iv])[:, 1]
    auc_baseline = float(roc_auc_score(y[iv], p))

    # Permute labels within time buckets if available
    rec = df_s.get('rfm__all__recency_days__life')
    if rec is not None:
        try:
            bins = pd.qcut(pd.to_numeric(rec, errors='coerce').fillna(rec.max()).astype(float), q=5, duplicates='drop')
            groups = bins.astype(str).values
        except Exception:
            groups = np.array(['all'] * len(df_s))
    else:
        groups = np.array(['all'] * len(df_s))

    auc_perm: list[float] = []
    for i in range(int(n_perm)):
        rng = np.random.RandomState(seed + i)
        y_perm = y.copy()
        try:
            # shuffle labels only on the TRAIN subset, within time buckets
            unique_groups = pd.unique(groups[it])
            for g in unique_groups:
                pos_in_train = np.where(groups[it] == g)[0]
                if len(pos_in_train) > 1:
                    idx_global = it[pos_in_train]
                    y_perm[idx_global] = y_perm[idx_global][rng.permutation(len(idx_global))]
        except Exception:
            # fallback: shuffle train labels only
            y_perm[it] = y_perm[it][rng.permutation(len(it))]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            clf.fit(X.iloc[it], y_perm[it])
            pp = clf.predict_proba(X.iloc[iv])[:, 1]
        try:
            # Evaluate against true labels to measure degradation
            a = float(roc_auc_score(y[iv], pp))
        except Exception:
            a = float('nan')
        auc_perm.append(a)

    auc_perm = [a for a in auc_perm if a == a and math.isfinite(a)]
    # one-sided p-value: P(AUC_perm >= AUC_baseline)
    p_value = None
    if auc_perm:
        arr = np.asarray(auc_perm, dtype=float)
        p_value = float(((arr >= auc_baseline).sum() + 1) / (len(arr) + 1))
    stats = {
        'baseline_auc': auc_baseline,
        'permuted_auc_mean': float(np.mean(auc_perm)) if auc_perm else None,
        'permuted_auc_std': float(np.std(auc_perm)) if auc_perm else None,
        'n_permutations': int(len(auc_perm)),
        'auc_degradation': (auc_baseline - float(np.mean(auc_perm))) if auc_perm else None,
        'p_value': p_value,
    }

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(auc_perm, bins=20, color='#4C78A8', alpha=0.9)
        ax.axvline(auc_baseline, color='#F58518', linestyle='--', label=f'baseline={auc_baseline:.3f}')
        ax.set_title('Label Permutation AUCs')
        ax.set_xlabel('AUC')
        ax.set_ylabel('Count')
        ax.legend()
        png = out_dir / 'perm_auc_hist.png'
        fig.tight_layout()
        fig.savefig(png)
        plt.close(fig)
        stats['plot'] = str(png)
    except Exception:
        pass

    path = out_dir / 'permutation_diag.json'
    path.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return {'permutation_diag': str(path), 'permutation_plot': stats.get('plot')}


def run_importance_stability(out_dir: Path, df: pd.DataFrame, n_subsets: int = 5, topk: int = 50, seed: int = 42, max_rows: int = 12000, max_features: int = 500) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from scipy.stats import spearmanr

    rng = np.random.RandomState(seed)
    df_s = _downsample(df, max_rows=max_rows, seed=seed)
    y = df_s['bought_in_division'].astype(int).values
    X = _sanitize_numeric(df_s.drop(columns=['customer_id', 'bought_in_division']))
    X = _select_features(X, max_features=max_features)

    it, iv = _time_aware_split(df_s, seed)
    # Use train indices for bootstraps
    train_idx = np.array(it)
    feats = list(X.columns)

    imp_mat = []
    for i in range(int(n_subsets)):
        bs = rng.choice(train_idx, size=len(train_idx), replace=True)
        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('lr', LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            clf.fit(X.iloc[bs], y[bs])
        base = getattr(clf, 'named_steps', {}).get('lr', clf)
        if hasattr(base, 'coef_'):
            imp = np.abs(np.ravel(base.coef_))
        else:
            imp = np.zeros(len(feats))
        imp_mat.append(imp)

    imp_mat = np.asarray(imp_mat)
    # Spearman stability across subsets
    rho_vals = []
    for i in range(len(imp_mat)):
        for j in range(i + 1, len(imp_mat)):
            try:
                r, _ = spearmanr(imp_mat[i], imp_mat[j])
            except Exception:
                r = np.nan
            if r == r and math.isfinite(r):
                rho_vals.append(float(r))
    mean_spearman = float(np.mean(rho_vals)) if rho_vals else None

    # Jaccard overlap of top-k sets
    def topk_set(vec):
        order = np.argsort(-vec)
        idx = order[: min(int(topk), len(order))]
        return set(idx.tolist())

    jac = []
    for i in range(len(imp_mat)):
        for j in range(i + 1, len(imp_mat)):
            a = topk_set(imp_mat[i])
            b = topk_set(imp_mat[j])
            inter = len(a & b)
            union = len(a | b) or 1
            jac.append(inter / union)
    mean_jaccard_topk = float(np.mean(jac)) if jac else None

    mean_imp = np.mean(imp_mat, axis=0)
    std_imp = np.std(imp_mat, axis=0)
    top_features = pd.DataFrame({
        'feature': feats,
        'mean_abs_coef': mean_imp,
        'std_abs_coef': std_imp,
        'cv': np.divide(std_imp, np.where(mean_imp == 0, 1.0, mean_imp)),
    }).sort_values('mean_abs_coef', ascending=False).head(min(50, len(feats)))

    # Plot top mean importances
    plot_path = None
    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(8, 6))
        tf = top_features.iloc[:25]
        ax.barh(tf['feature'][::-1], tf['mean_abs_coef'][::-1], color='#4C78A8')
        ax.set_title('Top Mean |Coef| (bootstrapped)')
        ax.set_xlabel('Mean |coef|')
        fig.tight_layout()
        plot_path = out_dir / 'importance_top_mean_abscoef.png'
        fig.savefig(plot_path)
        plt.close(fig)
    except Exception:
        pass

    out = {
        'mean_spearman': mean_spearman,
        'mean_jaccard_topk': mean_jaccard_topk,
        'n_subsets': int(n_subsets),
        'topk': int(topk),
        'top_features': top_features.to_dict(orient='records'),
        'plot': (str(plot_path) if plot_path else None),
    }
    p = out_dir / 'importance_stability.json'
    p.write_text(json.dumps(out, indent=2), encoding='utf-8')
    return {'importance_stability': str(p), 'importance_plot': out.get('plot')}


def run_diagnostics(division: str, cutoff: str, window_months: int, n_perm: int = 50, n_subsets: int = 5, topk: int = 50, max_rows: int = 12000, max_features: int = 500) -> dict:
    # Prepare out dir
    out_dir = OUTPUTS_DIR / 'leakage' / division / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)

    # DB engine
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()

    cfg = load_config()
    mask_tail = int(getattr(getattr(cfg, 'validation', object()), 'gauntlet_mask_tail_days', 0) or 0)
    fm = create_feature_matrix(engine, division, cutoff, window_months, mask_tail_days=mask_tail)
    if fm.is_empty():
        raise RuntimeError('Empty feature matrix')
    df = fm.to_pandas()

    art = {}
    try:
        art.update(run_label_permutation(out_dir, df, n_perm=n_perm, max_rows=max_rows, max_features=max_features))
    except Exception as e:
        (out_dir / 'permutation_diag.error').write_text(str(e), encoding='utf-8')
    try:
        art.update(run_importance_stability(out_dir, df, n_subsets=n_subsets, topk=topk, max_rows=max_rows, max_features=max_features))
    except Exception as e:
        (out_dir / 'importance_stability.error').write_text(str(e), encoding='utf-8')
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--division', required=True)
    ap.add_argument('--cutoff', required=True)
    ap.add_argument('--window-months', type=int, default=6)
    ap.add_argument('--n-perm', type=int, default=30)
    ap.add_argument('--n-subsets', type=int, default=5)
    ap.add_argument('--topk', type=int, default=50)
    ap.add_argument('--max-rows', type=int, default=12000)
    ap.add_argument('--max-features', type=int, default=500)
    args = ap.parse_args()
    # Silence convergence warnings globally
    warnings.simplefilter('ignore', ConvergenceWarning)
    run_diagnostics(
        args.division,
        args.cutoff,
        args.window_months,
        n_perm=args.n_perm,
        n_subsets=args.n_subsets,
        topk=args.topk,
        max_rows=args.max_rows,
        max_features=args.max_features,
    )


if __name__ == '__main__':
    main()
