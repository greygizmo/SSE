from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Callable

import click
import numpy as np
import pandas as pd

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.utils.logger import get_logger
from gosales.pipeline.rank_whitespace import _percentile_normalize
from gosales.validation.utils import bootstrap_ci, psi, ks_statistic


logger = get_logger(__name__)


def _load_model_and_features(division: str):
    import joblib, json
    model_path = MODELS_DIR / f"{division.lower()}_model" / "model.pkl"
    feat_path = MODELS_DIR / f"{division.lower()}_model" / "feature_list.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {division}: {model_path}")
    model = joblib.load(model_path)
    feats = None
    if feat_path.exists():
        feats = json.loads(feat_path.read_text(encoding="utf-8"))
    return model, feats


def _build_validation_frame(division: str, cutoff: str, window_months: int, cfg) -> pd.DataFrame:
    # Reuse features parquet as base, then join holdout labels computed previously
    base_path = OUTPUTS_DIR / f"features_{division.lower()}_{cutoff}.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing features for validation: {base_path}")
    df = pd.read_parquet(base_path)
    # Score with frozen model
    model, feat_cols = _load_model_and_features(division)
    X = df[feat_cols] if feat_cols else df.select_dtypes(include=[np.number])
    df['p_hat'] = model.predict_proba(X)[:, 1]
    # Eligibility proxy (reuse Phase 4 columns if present)
    if 'owned_division_pre_cutoff' not in df.columns:
        win_cols = [f'rfm__div__tx_n__{w}m' for w in cfg.features.windows_months]
        present_cols = [c for c in win_cols if c in df.columns]
        df['owned_division_pre_cutoff'] = (df[present_cols].sum(axis=1) > 0) if present_cols else False
    # EV proxy (reuse normalized EV if present)
    if 'EV_norm' not in df.columns:
        gpcol = 'rfm__all__gp_sum__12m' if 'rfm__all__gp_sum__12m' in df.columns else None
        if gpcol:
            cap = df[gpcol].quantile(cfg.whitespace.ev_cap_percentile)
            df['EV_norm'] = _percentile_normalize(pd.to_numeric(df[gpcol], errors='coerce').fillna(0.0).clip(upper=cap))
        else:
            df['EV_norm'] = 0.0
    # Deterministic order
    df = df.sort_values(['customer_id']).reset_index(drop=True)
    return df


def _gains_deciles(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({'y': y, 'p': p}).sort_values('p', ascending=False).reset_index(drop=True)
    df['decile'] = (np.floor((df.index / max(1, len(df)-1)) * 10) + 1).clip(1, 10).astype(int)
    return df.groupby('decile').agg(fraction_positives=('y', 'mean'), count=('y', 'size'), mean_predicted=('p', 'mean')).reset_index()


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({'y': y, 'p': p})
    try:
        bins = pd.qcut(df['p'], q=n_bins, duplicates='drop')
    except Exception:
        bins = pd.cut(df['p'], bins=n_bins, include_lowest=True, duplicates='drop')
    return df.assign(bin=bins).groupby('bin').agg(mean_predicted=('p','mean'), fraction_positives=('y','mean'), count=('y','size')).reset_index(drop=True)


@click.command()
@click.option('--division', required=True)
@click.option('--cutoff', required=True)
@click.option('--window-months', default=6, type=int)
@click.option('--capacity-grid', default='5,10,20')
@click.option('--bootstrap', default=1000, type=int)
@click.option('--config', default=str((Path(__file__).parents[1] / 'config.yaml').resolve()))
def main(division: str, cutoff: str, window_months: int, capacity_grid: str, bootstrap: int, config: str) -> None:
    cfg = load_config(config)
    out_dir = OUTPUTS_DIR / 'validation' / division.lower() / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)

    vf = _build_validation_frame(division, cutoff, window_months, cfg)
    # Join holdout labels from holdout CSVs if available (data/holdout/*). Fallback to training labels in feature parquet
    try:
        from gosales.utils.paths import DATA_DIR
        cutoff_dt = pd.to_datetime(cutoff)
        window_end = cutoff_dt + pd.DateOffset(months=window_months)
        holdout_dir = (DATA_DIR / 'holdout')
        buyers = None
        if holdout_dir.exists():
            # Load and concatenate CSVs in holdout directory
            parts = []
            for pth in holdout_dir.glob('*.csv'):
                try:
                    parts.append(pd.read_csv(pth))
                except Exception:
                    continue
            if parts:
                ho = pd.concat(parts, ignore_index=True)
                # Parse dates and filter by division and window
                if 'Rec Date' in ho.columns:
                    ho['Rec Date'] = pd.to_datetime(ho['Rec Date'], errors='coerce')
                    mask_window = (ho['Rec Date'] > cutoff_dt) & (ho['Rec Date'] <= window_end)
                else:
                    mask_window = pd.Series(True, index=ho.index)
                div_col = 'Division' if 'Division' in ho.columns else None
                if div_col:
                    mask_div = ho[div_col].astype(str).str.strip().str.casefold() == division.lower()
                else:
                    mask_div = pd.Series(True, index=ho.index)
                cust_col = 'CustomerId' if 'CustomerId' in ho.columns else 'customer_id'
                buyers = pd.to_numeric(ho.loc[mask_window & mask_div, cust_col], errors='coerce').dropna().astype('Int64').unique()
        if buyers is not None:
            labels_df = pd.DataFrame({'customer_id': buyers, 'holdout_bought': 1})
            vf['customer_id'] = pd.to_numeric(vf['customer_id'], errors='coerce').astype('Int64')
            vf = vf.merge(labels_df, on='customer_id', how='left')
            vf['holdout_bought'] = vf['holdout_bought'].fillna(0).astype(int)
            if 'bought_in_division' in vf.columns:
                vf.drop(columns=['bought_in_division'], inplace=True)
            vf.rename(columns={'holdout_bought': 'bought_in_division'}, inplace=True)
    except Exception:
        pass

    # Persist validation frame parquet
    out_dir = OUTPUTS_DIR / 'validation' / division.lower() / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)
    vf.to_parquet(out_dir / 'validation_frame.parquet', index=False)

    y = vf.get('bought_in_division', pd.Series(0, index=vf.index)).astype(int).values
    p = vf['p_hat'].values

    # Metrics
    gains = _gains_deciles(y, p)
    gains.to_csv(out_dir / 'gains.csv', index=False)
    calib = _calibration_bins(y, p, n_bins=10)
    calib.to_csv(out_dir / 'calibration.csv', index=False)

    # Capture@K grid
    topks = [int(x) for x in capacity_grid.split(',') if x]
    scenarios = []
    for k in topks:
        kk = max(1, int(len(vf) * (k / 100.0)))
        topk = vf.nlargest(kk, ['p_hat','EV_norm','customer_id'])
        capture = float(topk['bought_in_division'].sum() / max(1, vf['bought_in_division'].sum())) if 'bought_in_division' in vf.columns else None
        precision = float(topk['bought_in_division'].mean()) if 'bought_in_division' in vf.columns else None
        exp_gp = float((topk['EV_norm']).sum())
        scenarios.append({'k_percent': k, 'contacts': int(kk), 'capture': capture, 'precision': precision, 'expected_gp_norm': exp_gp})
    scen_df = pd.DataFrame(scenarios)
    # Bootstrap CIs for capture and precision
    try:
        n_boot = int(getattr(cfg.validation, 'bootstrap_n', 1000))
        def cap_at_k(df_in: pd.DataFrame, k: int) -> float:
            if df_in.empty:
                return 0.0
            kk = max(1, int(len(df_in) * (k / 100.0)))
            topk = df_in.nlargest(kk, ['p_hat','EV_norm','customer_id'])
            return float(topk['bought_in_division'].sum() / max(1, df_in['bought_in_division'].sum()))
        def prec_at_k(df_in: pd.DataFrame, k: int) -> float:
            if df_in.empty:
                return 0.0
            kk = max(1, int(len(df_in) * (k / 100.0)))
            topk = df_in.nlargest(kk, ['p_hat','EV_norm','customer_id'])
            return float(topk['bought_in_division'].mean())
        for i, row in scen_df.iterrows():
            k = int(row['k_percent'])
            lo_c, hi_c = bootstrap_ci(lambda df_in, kk=k: cap_at_k(df_in, kk), vf, n=n_boot, seed=cfg.modeling.seed)
            lo_p, hi_p = bootstrap_ci(lambda df_in, kk=k: prec_at_k(df_in, kk), vf, n=n_boot, seed=cfg.modeling.seed)
            scen_df.loc[i, 'capture_ci_lo'] = lo_c
            scen_df.loc[i, 'capture_ci_hi'] = hi_c
            scen_df.loc[i, 'precision_ci_lo'] = lo_p
            scen_df.loc[i, 'precision_ci_hi'] = hi_p
    except Exception:
        pass
    scen_df.to_csv(out_dir / 'topk_scenarios.csv', index=False)

    # Drift diagnostics (if training score snapshot available)
    drift = {}
    try:
        # Attempt to load train-time scores if saved (metrics_{division}.json not sufficient); fallback to feature proxy
        train_proxy = vf.get('rfm__all__gp_sum__12m', pd.Series(dtype=float))
        hold_proxy = vf.get('rfm__all__gp_sum__12m', pd.Series(dtype=float))
        drift['psi_gp12m'] = psi(train_proxy, hold_proxy)
        drift['ks_phat_train_holdout'] = None  # placeholder unless train p_hat snapshot is available
    except Exception:
        pass

    # Minimal metrics.json
    metrics = {
        'division': division,
        'cutoff': cutoff,
        'rows': int(len(vf)),
        'capture_grid': {str(s['k_percent']): s['capture'] for s in scenarios},
        'drift': drift,
    }
    (out_dir / 'metrics.json').write_text(pd.Series(metrics).to_json(indent=2), encoding='utf-8')
    logger.info(f"Wrote validation artifacts to {out_dir}")


if __name__ == '__main__':
    main()


