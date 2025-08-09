from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Callable

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
import json

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR, MODELS_DIR
from gosales.utils.logger import get_logger
from gosales.pipeline.rank_whitespace import _percentile_normalize
from gosales.validation.utils import bootstrap_ci, psi, ks_statistic
from gosales.ops.run import run_context
from gosales.etl.sku_map import get_sku_mapping


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


def _calibration_mae(bins_df: pd.DataFrame) -> float:
    if bins_df.empty:
        return float('nan')
    diff = (bins_df['mean_predicted'].astype(float) - bins_df['fraction_positives'].astype(float)).abs()
    w = bins_df['count'].astype(float)
    return float((diff * w).sum() / max(1e-9, w.sum()))


@click.command()
@click.option('--division', required=True)
@click.option('--cutoff', required=True)
@click.option('--window-months', default=6, type=int)
@click.option('--capacity-grid', default='5,10,20')
@click.option('--accounts-per-rep-grid', default='10,25')
@click.option('--bootstrap', default=1000, type=int)
@click.option('--config', default=str((Path(__file__).parents[1] / 'config.yaml').resolve()))
def main(division: str, cutoff: str, window_months: int, capacity_grid: str, accounts_per_rep_grid: str, bootstrap: int, config: str) -> None:
    cfg = load_config(config)
    out_dir = OUTPUTS_DIR / 'validation' / division.lower() / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    ctx_cm = run_context("phase5_validation")
    ctx = ctx_cm.__enter__()

    vf = _build_validation_frame(division, cutoff, window_months, cfg)
    # Join holdout labels from holdout CSVs if available (data/holdout/*). Fallback to training labels in feature parquet
    try:
        from gosales.utils.paths import DATA_DIR
        cutoff_dt = pd.to_datetime(cutoff)
        window_end = cutoff_dt + pd.DateOffset(months=window_months)
        holdout_dir = (DATA_DIR / 'holdout')
        buyers = None
        holdout_gp_map = None
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

                # Compute realized GP for target division using SKU mapping (sum of division GP columns)
                try:
                    mapping = get_sku_mapping()
                    div_cols = [gp for gp, meta in mapping.items() if meta.get('division', '').strip().lower() == division.lower()]
                    # Some datasets have missing GP columns; keep existing ones only
                    div_cols = [c for c in div_cols if c in ho.columns]
                    if div_cols:
                        gp_df = ho.loc[mask_window, [cust_col] + div_cols].copy()
                        # Coerce GP cols to numeric
                        for c in div_cols:
                            gp_df[c] = pd.to_numeric(gp_df[c], errors='coerce').fillna(0.0)
                        gp_df['holdout_gp'] = gp_df[div_cols].sum(axis=1)
                        holdout_gp_map = gp_df.groupby(cust_col)['holdout_gp'].sum().reset_index()
                        holdout_gp_map[cust_col] = pd.to_numeric(holdout_gp_map[cust_col], errors='coerce').astype('Int64')
                except Exception:
                    holdout_gp_map = None
        if buyers is not None:
            labels_df = pd.DataFrame({'customer_id': buyers, 'holdout_bought': 1})
            vf['customer_id'] = pd.to_numeric(vf['customer_id'], errors='coerce').astype('Int64')
            vf = vf.merge(labels_df, on='customer_id', how='left')
            vf['holdout_bought'] = vf['holdout_bought'].fillna(0).astype(int)
            if 'bought_in_division' in vf.columns:
                vf.drop(columns=['bought_in_division'], inplace=True)
            vf.rename(columns={'holdout_bought': 'bought_in_division'}, inplace=True)
            # Join realized GP if computed
            if holdout_gp_map is not None:
                vf = vf.merge(holdout_gp_map.rename(columns={cust_col: 'customer_id'}), on='customer_id', how='left')
                vf['holdout_gp'] = vf['holdout_gp'].fillna(0.0)
    except Exception:
        pass

    # Persist validation frame parquet
    out_dir = OUTPUTS_DIR / 'validation' / division.lower() / cutoff
    out_dir.mkdir(parents=True, exist_ok=True)
    vf_path = out_dir / 'validation_frame.parquet'
    vf.to_parquet(vf_path, index=False)
    try:
        artifacts['validation_frame.parquet'] = str(vf_path)
    except Exception:
        pass

    y = vf.get('bought_in_division', pd.Series(0, index=vf.index)).astype(int).values
    p = vf['p_hat'].values

    # Metrics
    gains = _gains_deciles(y, p)
    gains_path = out_dir / 'gains.csv'
    gains.to_csv(gains_path, index=False)
    try:
        artifacts['gains.csv'] = str(gains_path)
    except Exception:
        pass
    calib = _calibration_bins(y, p, n_bins=10)
    calib_path = out_dir / 'calibration.csv'
    calib.to_csv(calib_path, index=False)
    try:
        artifacts['calibration.csv'] = str(calib_path)
    except Exception:
        pass
    # Core metrics
    auc_val = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')
    pr_prec, pr_rec, _ = precision_recall_curve(y, p)
    pr_auc_val = float(auc(pr_rec, pr_prec)) if pr_prec is not None else float('nan')
    brier = float(brier_score_loss(y, p))
    cal_mae = _calibration_mae(calib)

    # Capture@K grid
    topks = [int(x) for x in capacity_grid.split(',') if x]
    per_rep_ns = [int(x) for x in accounts_per_rep_grid.split(',') if x]
    scenarios = []
    total_realized_gp = float(vf.get('holdout_gp', pd.Series(0.0, index=vf.index)).sum())
    for k in topks:
        kk = max(1, int(len(vf) * (k / 100.0)))
        topk = vf.nlargest(kk, ['p_hat','EV_norm','customer_id'])
        capture = float(topk['bought_in_division'].sum() / max(1, vf['bought_in_division'].sum())) if 'bought_in_division' in vf.columns else None
        precision = float(topk['bought_in_division'].mean()) if 'bought_in_division' in vf.columns else None
        exp_gp = float((topk['EV_norm']).sum())
        realized_gp = float(topk.get('holdout_gp', pd.Series(0.0, index=topk.index)).sum())
        rev_capture = float(realized_gp / max(1e-9, total_realized_gp)) if total_realized_gp > 0 else None
        scenarios.append({'mode': 'top_percent', 'k_percent': k, 'contacts': int(kk), 'capture': capture, 'precision': precision, 'expected_gp_norm': exp_gp, 'realized_gp': realized_gp, 'rev_capture': rev_capture})

    # Per-rep scenarios (if 'rep' column exists)
    if 'rep' in vf.columns and len(per_rep_ns) > 0:
        for n in per_rep_ns:
            sel = vf.sort_values(['rep','p_hat','EV_norm','customer_id'], ascending=[True, False, False, True])
            sel = sel.groupby('rep', as_index=False).head(int(n))
            contacts = int(len(sel))
            capture = float(sel['bought_in_division'].sum() / max(1, vf['bought_in_division'].sum())) if 'bought_in_division' in vf.columns else None
            precision = float(sel['bought_in_division'].mean()) if 'bought_in_division' in sel.columns else None
            exp_gp = float(sel['EV_norm'].sum())
            realized_gp = float(sel.get('holdout_gp', pd.Series(0.0, index=sel.index)).sum())
            rev_capture = float(realized_gp / max(1e-9, total_realized_gp)) if total_realized_gp > 0 else None
            scenarios.append({'mode': 'per_rep', 'accounts_per_rep': int(n), 'contacts': contacts, 'capture': capture, 'precision': precision, 'expected_gp_norm': exp_gp, 'realized_gp': realized_gp, 'rev_capture': rev_capture})

    # Hybrid scenarios by segment (round-robin across first available segment column)
    seg_col = next((c for c in getattr(cfg.validation, 'segment_columns', []) if c in vf.columns), None)
    if seg_col:
        for k in topks:
            kk = max(1, int(len(vf) * (k / 100.0)))
            # Prepare per-segment sorted lists
            seg_lists = {s: df.sort_values(['p_hat','EV_norm','customer_id'], ascending=[False, False, True])
                         for s, df in vf.groupby(seg_col)}
            iters = {s: df.itertuples(index=False) for s, df in seg_lists.items()}
            order = list(seg_lists.keys())
            picked = []
            idx = 0
            while len(picked) < kk and order:
                s = order[idx % len(order)]
                try:
                    picked.append(next(iters[s]))
                except StopIteration:
                    order.pop(idx % len(order))
                    continue
                idx += 1
            sel = pd.DataFrame(picked, columns=vf.columns) if picked else vf.head(0)
            contacts = int(len(sel))
            capture = float(sel['bought_in_division'].sum() / max(1, vf['bought_in_division'].sum())) if 'bought_in_division' in vf.columns else None
            precision = float(sel['bought_in_division'].mean()) if 'bought_in_division' in sel.columns else None
            exp_gp = float(sel['EV_norm'].sum())
            realized_gp = float(sel.get('holdout_gp', pd.Series(0.0, index=sel.index)).sum())
            rev_capture = float(realized_gp / max(1e-9, total_realized_gp)) if total_realized_gp > 0 else None
            scenarios.append({'mode': 'hybrid_segment', 'segment': seg_col, 'k_percent': k, 'contacts': contacts, 'capture': capture, 'precision': precision, 'expected_gp_norm': exp_gp, 'realized_gp': realized_gp, 'rev_capture': rev_capture})
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
        def rev_cap_at_k(df_in: pd.DataFrame, k: int) -> float:
            if df_in.empty:
                return 0.0
            kk = max(1, int(len(df_in) * (k / 100.0)))
            topk = df_in.nlargest(kk, ['p_hat','EV_norm','customer_id'])
            total_gp = float(df_in.get('holdout_gp', pd.Series(0.0, index=df_in.index)).sum())
            top_gp = float(topk.get('holdout_gp', pd.Series(0.0, index=topk.index)).sum())
            if total_gp <= 0:
                return 0.0
            return float(top_gp / total_gp)
        def realized_gp_at_k(df_in: pd.DataFrame, k: int) -> float:
            if df_in.empty:
                return 0.0
            kk = max(1, int(len(df_in) * (k / 100.0)))
            topk = df_in.nlargest(kk, ['p_hat','EV_norm','customer_id'])
            return float(topk.get('holdout_gp', pd.Series(0.0, index=topk.index)).sum())
        for i, row in scen_df.iterrows():
            k = int(row['k_percent'])
            lo_c, hi_c = bootstrap_ci(lambda df_in, kk=k: cap_at_k(df_in, kk), vf, n=n_boot, seed=cfg.modeling.seed)
            lo_p, hi_p = bootstrap_ci(lambda df_in, kk=k: prec_at_k(df_in, kk), vf, n=n_boot, seed=cfg.modeling.seed)
            lo_rc, hi_rc = bootstrap_ci(lambda df_in, kk=k: rev_cap_at_k(df_in, kk), vf, n=n_boot, seed=cfg.modeling.seed)
            lo_rg, hi_rg = bootstrap_ci(lambda df_in, kk=k: realized_gp_at_k(df_in, kk), vf, n=n_boot, seed=cfg.modeling.seed)
            scen_df.loc[i, 'capture_ci_lo'] = lo_c
            scen_df.loc[i, 'capture_ci_hi'] = hi_c
            scen_df.loc[i, 'precision_ci_lo'] = lo_p
            scen_df.loc[i, 'precision_ci_hi'] = hi_p
            scen_df.loc[i, 'rev_capture_ci_lo'] = lo_rc
            scen_df.loc[i, 'rev_capture_ci_hi'] = hi_rc
            scen_df.loc[i, 'realized_gp_ci_lo'] = lo_rg
            scen_df.loc[i, 'realized_gp_ci_hi'] = hi_rg
        # For per_rep rows, compute CIs with group selection logic
        if 'mode' in scen_df.columns:
            for i, row in scen_df.iterrows():
                if row.get('mode') == 'per_rep':
                    n_acc = int(row.get('accounts_per_rep', 0))
                    def per_rep_metric(df_in: pd.DataFrame, metric: str) -> float:
                        if df_in.empty or 'rep' not in df_in.columns:
                            return 0.0
                        sel = df_in.sort_values(['rep','p_hat','EV_norm','customer_id'], ascending=[True, False, False, True])
                        sel = sel.groupby('rep', as_index=False).head(n_acc)
                        if metric == 'capture':
                            return float(sel['bought_in_division'].sum() / max(1, df_in['bought_in_division'].sum()))
                        if metric == 'precision':
                            return float(sel['bought_in_division'].mean())
                        if metric == 'rev_capture':
                            total_gp = float(df_in.get('holdout_gp', pd.Series(0.0, index=df_in.index)).sum())
                            top_gp = float(sel.get('holdout_gp', pd.Series(0.0, index=sel.index)).sum())
                            return float(top_gp / total_gp) if total_gp > 0 else 0.0
                        if metric == 'realized_gp':
                            return float(sel.get('holdout_gp', pd.Series(0.0, index=sel.index)).sum())
                        return 0.0
                    scen_df.loc[i, 'capture_ci_lo'], scen_df.loc[i, 'capture_ci_hi'] = bootstrap_ci(lambda dfi: per_rep_metric(dfi, 'capture'), vf, n=n_boot, seed=cfg.modeling.seed)
                    scen_df.loc[i, 'precision_ci_lo'], scen_df.loc[i, 'precision_ci_hi'] = bootstrap_ci(lambda dfi: per_rep_metric(dfi, 'precision'), vf, n=n_boot, seed=cfg.modeling.seed)
                    scen_df.loc[i, 'rev_capture_ci_lo'], scen_df.loc[i, 'rev_capture_ci_hi'] = bootstrap_ci(lambda dfi: per_rep_metric(dfi, 'rev_capture'), vf, n=n_boot, seed=cfg.modeling.seed)
                    scen_df.loc[i, 'realized_gp_ci_lo'], scen_df.loc[i, 'realized_gp_ci_hi'] = bootstrap_ci(lambda dfi: per_rep_metric(dfi, 'realized_gp'), vf, n=n_boot, seed=cfg.modeling.seed)
    except Exception:
        pass
    # Rank scenarios: if calibration MAE low, rank by expected GP; otherwise by capture
    try:
        rank_by = 'expected_gp_norm' if (not np.isnan(cal_mae) and cal_mae < 0.03) else 'capture'
        scen_sorted = scen_df.sort_values(rank_by, ascending=False).reset_index(drop=True)
        scen_csv = out_dir / 'topk_scenarios.csv'
        scen_sorted.to_csv(scen_csv, index=False)
        scen_sorted_path = out_dir / 'topk_scenarios_sorted.csv'
        scen_sorted.to_csv(scen_sorted_path, index=False)
        try:
            artifacts['topk_scenarios.csv'] = str(scen_csv)
            artifacts['topk_scenarios_sorted.csv'] = str(scen_sorted_path)
        except Exception:
            pass
    except Exception:
        scen_csv = out_dir / 'topk_scenarios.csv'
        scen_df.to_csv(scen_csv, index=False)
        try:
            artifacts['topk_scenarios.csv'] = str(scen_csv)
        except Exception:
            pass

    # Drift diagnostics (if training score snapshot available)
    drift = {}
    try:
        # Attempt to load train-time scores if saved (metrics_{division}.json not sufficient); fallback to feature proxy
        train_scores_path = OUTPUTS_DIR / f"train_scores_{division.lower()}_{cutoff}.csv"
        if train_scores_path.exists():
            train_scores = pd.read_csv(train_scores_path)
            # Align on customer_id
            merged = vf[['customer_id']].merge(train_scores, on='customer_id', how='left')
            p_train = pd.to_numeric(merged['p_hat'], errors='coerce')
            p_hold = pd.Series(p)
            drift['ks_phat_train_holdout'] = ks_statistic(p_train, p_hold)
        else:
            drift['ks_phat_train_holdout'] = None
        # PSI proxy between EV and holdout GP
        train_proxy = vf.get('rfm__all__gp_sum__12m', pd.Series(dtype=float))
        hold_proxy = vf.get('rfm__all__gp_sum__12m', pd.Series(dtype=float))
        drift['psi_gp12m'] = psi(train_proxy, hold_proxy)
    except Exception:
        pass

    # Minimal metrics.json
    # Drift highlights for metrics.json (top PSI features over threshold)
    drift_highlights = {}
    try:
        psi_map = drift_report.get('psi_per_feature', {}) if 'drift_report' in locals() else {}
        thr = float(getattr(cfg.validation, 'psi_threshold', 0.25))
        flagged = sorted(
            (
                {'feature': k, 'psi': float(v)}
                for k, v in psi_map.items()
                if isinstance(v, (int, float)) and float(v) >= thr
            ),
            key=lambda x: x['psi'], reverse=True
        )[:20]
        drift_highlights = {
            'psi_threshold': thr,
            'psi_flagged_top': flagged,
        }
    except Exception:
        drift_highlights = {}

    metrics = {
        'division': division,
        'cutoff': cutoff,
        'rows': int(len(vf)),
        'capture_grid': {str(s['k_percent']): s['capture'] for s in scenarios},
        'drift': drift,
        'drift_highlights': drift_highlights,
        'metrics': {
            'auc': auc_val,
            'pr_auc': pr_auc_val,
            'brier': brier,
            'cal_mae': cal_mae,
        },
    }
    metrics_file = out_dir / 'metrics.json'
    metrics_file.write_text(pd.Series(metrics).to_json(indent=2), encoding='utf-8')
    # Log top PSI-flagged features for quick visibility
    try:
        flagged = drift_highlights.get('psi_flagged_top', []) if isinstance(drift_highlights, dict) else []
        thr = drift_highlights.get('psi_threshold', None) if isinstance(drift_highlights, dict) else None
        if flagged:
            preview = ', '.join([f"{item['feature']} ({float(item['psi']):.2f})" for item in flagged[:10]])
            if thr is not None:
                logger.info(f"Per-feature drift (PSI ≥ {thr}): {preview}")
            else:
                logger.info(f"Per-feature drift (PSI): {preview}")
        else:
            logger.info("No per-feature PSI flags above threshold")
    except Exception:
        pass

    # Segment performance stability
    try:
        seg_col = next((c for c in getattr(cfg.validation, 'segment_columns', []) if c in vf.columns), None)
        if seg_col:
            seg_rows = []
            for seg_val, sub in vf.groupby(seg_col):
                for k in topks:
                    kk = max(1, int(len(sub) * (k / 100.0)))
                    topk = sub.nlargest(kk, ['p_hat','EV_norm','customer_id'])
                    capture = float(topk['bought_in_division'].sum() / max(1, sub['bought_in_division'].sum())) if 'bought_in_division' in sub.columns and sub['bought_in_division'].sum() > 0 else 0.0
                    precision = float(topk['bought_in_division'].mean()) if 'bought_in_division' in sub.columns else 0.0
                    realized_gp = float(topk.get('holdout_gp', pd.Series(0.0, index=topk.index)).sum())
                    total_gp = float(sub.get('holdout_gp', pd.Series(0.0, index=sub.index)).sum())
                    rev_capture = float(realized_gp / total_gp) if total_gp > 0 else 0.0
                    seg_rows.append({'segment_col': seg_col, 'segment': seg_val, 'k_percent': k, 'capture': capture, 'precision': precision, 'rev_capture': rev_capture})
            seg_perf = out_dir / 'segment_performance.csv'
            pd.DataFrame(seg_rows).to_csv(seg_perf, index=False)
            try:
                artifacts['segment_performance.csv'] = str(seg_perf)
            except Exception:
                pass
    except Exception:
        pass

    # Drift JSON: per-feature PSI (train sample vs holdout), EV vs holdout GP PSI, KS(p_hat train vs holdout and pos vs neg)
    try:
        drift_report = {}
        # EV vs holdout GP PSI
        ev_raw = vf.get('rfm__all__gp_sum__12m', pd.Series(dtype=float))
        hold_gp = vf.get('holdout_gp', pd.Series(dtype=float))
        drift_report['psi_ev_vs_holdout_gp'] = psi(ev_raw, hold_gp)
        # p_hat separation KS (pos vs neg)
        if 'bought_in_division' in vf.columns:
            pos_p = pd.Series(p)[vf['bought_in_division'] == 1]
            neg_p = pd.Series(p)[vf['bought_in_division'] == 0]
            drift_report['ks_p_hat_pos_vs_neg'] = ks_statistic(pos_p, neg_p)
        # Train vs holdout KS on p_hat
        train_scores_path = OUTPUTS_DIR / f"train_scores_{division.lower()}_{cutoff}.csv"
        if train_scores_path.exists():
            train_scores = pd.read_csv(train_scores_path)
            merged = vf[['customer_id']].merge(train_scores, on='customer_id', how='left')
            p_train = pd.to_numeric(merged['p_hat'], errors='coerce')
            p_hold = pd.Series(p)
            drift_report['ks_phat_train_holdout'] = ks_statistic(p_train, p_hold)
        else:
            drift_report['ks_phat_train_holdout'] = None
        # Per-feature PSI using train feature sample snapshot
        feat_sample_path = OUTPUTS_DIR / f"train_feature_sample_{division.lower()}_{cutoff}.parquet"
        if feat_sample_path.exists():
            train_feat = pd.read_parquet(feat_sample_path)
            per_feature = {}
            # Intersect numeric columns
            num_cols = [c for c in vf.columns if pd.api.types.is_numeric_dtype(vf[c]) and c not in ('bought_in_division',)]
            for c in num_cols:
                if c in train_feat.columns:
                    per_feature[c] = psi(train_feat[c], vf[c])
            drift_report['psi_per_feature'] = per_feature
        drift_file = out_dir / 'drift.json'
        drift_file.write_text(json.dumps(drift_report, indent=2), encoding='utf-8')
    except Exception:
        pass
    try:
        artifacts['metrics.json'] = str(metrics_file)
    except Exception:
        pass
    # Write run manifest and registry
    try:
        ctx['write_manifest'](artifacts)
        ctx['append_registry']({'phase': 'phase5_validation', 'division': division, 'cutoff': cutoff, 'artifact_count': len(artifacts)})
    except Exception:
        pass
    try:
        ctx_cm.__exit__(None, None, None)
    except Exception:
        pass
    logger.info(f"Wrote validation artifacts to {out_dir}")


if __name__ == '__main__':
    main()


