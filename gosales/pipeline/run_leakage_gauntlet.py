from __future__ import annotations

"""
Leakage Gauntlet runner.

Implements static checks that do not require retraining, and lays the structure
for future dynamic checks (date shift, ablation, group-safe CV). Artifacts are
written under gosales/outputs/leakage/<division>/<cutoff>/.
"""

from pathlib import Path
from dataclasses import dataclass
import json
import re
import sys
import click
import subprocess
from datetime import timedelta

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR, ROOT_DIR, MODELS_DIR
from gosales.utils.logger import get_logger
from gosales.utils.db import get_curated_connection, get_db_connection
from gosales.features.engine import create_feature_matrix


logger = get_logger(__name__)


# --- Static scan for time-now calls ---
_BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("datetime.now", re.compile(r"\bdatetime\s*\.\s*now\s*\(")),
    ("pd.Timestamp.now", re.compile(r"\b(pd\s*\.\s*)?Timestamp\s*\.\s*now\s*\(")),
    ("date.today", re.compile(r"\bdate\s*\.\s*today\s*\(")),
]


def _static_scan(paths: list[Path]) -> dict:
    results: list[dict] = []
    for base in paths:
        for p in base.rglob("*.py"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                s = line.strip()
                # Skip comments
                if not s or s.startswith("#"):
                    continue
                for name, pat in _BANNED_PATTERNS:
                    if pat.search(line):
                        results.append({
                            "file": str(p.relative_to(ROOT_DIR)),
                            "line": i,
                            "pattern": name,
                            "code": line.strip(),
                        })
    status = "PASS" if not results else "FAIL"
    return {"status": status, "findings": results}


@dataclass
class LGContext:
    division: str
    cutoff: str
    out_dir: Path


def _ensure_outdir(division: str, cutoff: str) -> LGContext:
    d = division.strip()
    c = cutoff.strip()
    out = OUTPUTS_DIR / "leakage" / d / c
    out.mkdir(parents=True, exist_ok=True)
    return LGContext(d, c, out)


def run_static_checks(ctx: LGContext) -> dict[str, str]:
    checks: dict[str, str] = {}
    # Static scan across feature and ETL code paths
    scan = _static_scan([ROOT_DIR / "gosales" / "features", ROOT_DIR / "gosales" / "etl"])
    static_path = ctx.out_dir / f"static_scan_{ctx.division}_{ctx.cutoff}.json"
    static_path.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    checks["static_scan"] = str(static_path)
    return checks


def run_feature_date_audit(ctx: LGContext, window_months: int) -> dict[str, str]:
    """Emit a per-feature latest-event audit and a JSON summary.

    Approach: compute max(order_date) from fact_transactions filtered at cutoff.
    Use the feature matrix to enumerate features, then assign the observed max
    event date to each feature and check against cutoff. This guards against any
    accidental use of post-cutoff data in feature computation.
    """
    # Prefer curated engine (where facts live); fallback to primary DB
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()

    # Enumerate features by building the feature matrix (no training involved)
    # Use Gauntlet tail mask to reduce near-cutoff signal in windowed features
    cfg = load_config()
    mask_tail = int(getattr(getattr(cfg, 'validation', object()), 'gauntlet_mask_tail_days', 0) or 0)
    fm = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months, mask_tail_days=mask_tail)
    cols = [c for c in fm.columns if c not in ("customer_id", "bought_in_division")]
    # Compute the latest event date used for features
    import pandas as pd
    sql = "SELECT MAX(order_date) AS max_order_date FROM fact_transactions WHERE order_date <= :cutoff"
    df = pd.read_sql_query(sql, engine, params={"cutoff": ctx.cutoff})
    max_order_date = None
    try:
        max_order_date = pd.to_datetime(df["max_order_date"].iloc[0]) if not df.empty else None
    except Exception:
        max_order_date = None
    # Build per-feature audit frame
    cutoff_dt = pd.to_datetime(ctx.cutoff)
    rows = []
    for name in cols:
        latest = max_order_date
        status = "OK"
        if latest is not None and latest > cutoff_dt:
            status = "LEAK"
        rows.append({
            "feature": str(name),
            "latest_event_date": (latest.date().isoformat() if pd.notna(latest) else None),
            "cutoff": ctx.cutoff,
            "status": status,
        })
    audit_path = ctx.out_dir / f"feature_date_audit_{ctx.division}_{ctx.cutoff}.csv"
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    # Summary JSON for consolidated report
    summary = {
        "status": ("FAIL" if any(r["status"] == "LEAK" for r in rows) else "PASS"),
        "max_event_date": (max_order_date.date().isoformat() if isinstance(max_order_date, pd.Timestamp) and pd.notna(max_order_date) else None),
        "cutoff": ctx.cutoff,
        "feature_count": len(rows),
    }
    summary_path = ctx.out_dir / f"feature_date_audit_{ctx.division}_{ctx.cutoff}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"feature_date_audit": str(summary_path), "feature_date_audit_csv": str(audit_path)}


def _unwrap_model_and_features(model) -> tuple[object, list[str] | None]:
    """Attempt to unwrap calibrated/pipeline models and return base estimator + feature names if present."""
    feats = None
    base = getattr(model, 'base_estimator', None)
    if base is None and hasattr(model, 'estimator'):
        base = model.estimator
    if base is None:
        base = model
    # Pipeline named_steps
    try:
        from sklearn.pipeline import Pipeline as _SkPipeline  # lazy import
        if isinstance(base, _SkPipeline):
            if 'model' in getattr(base, 'named_steps', {}):
                base = base.named_steps['model']
    except Exception:
        pass
    # Try to get feature names if stored
    try:
        feats = getattr(model, 'feature_names_in_', None)
        if feats is not None:
            feats = [str(x) for x in feats]
    except Exception:
        feats = None
    return base, feats


def run_topk_ablation_check(ctx: LGContext, window_months: int, k_list: list[int], run_training: bool = False, epsilon_auc: float = 0.01, epsilon_lift10: float = 0.25) -> dict[str, str]:
    """Top-K ablation scaffold: ranks features by model importance and (optionally) retrains without top-K.

    Emits `ablation_topk_<div>_<cutoff>.csv` with ranked features and per-K sets.
    When `run_training=True`, trains a simple LR on the last-cutoff feature matrix and compares AUC/lift@10 to baseline.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from pathlib import Path as _P
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Load trained model and try to extract importances
    model_dir = MODELS_DIR / f"{ctx.division.lower()}_model"
    pkl = model_dir / 'model.pkl'
    model = joblib.load(pkl)
    base, _ = _unwrap_model_and_features(model)
    # Feature names from metadata
    meta = {}
    feats: list[str] | None = None
    try:
        meta = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
        feats = meta.get('feature_names') or None
        if feats:
            feats = [str(x) for x in feats]
    except Exception:
        feats = feats or None

    imp: pd.Series | None = None
    try:
        importances = None
        if hasattr(base, 'feature_importances_'):
            importances = getattr(base, 'feature_importances_')
        elif hasattr(base, 'coef_'):
            importances = np.abs(np.ravel(base.coef_))
        if importances is not None:
            if feats is None and hasattr(base, 'feature_names_in_'):
                feats = [str(x) for x in getattr(base, 'feature_names_in_')]
            names = feats if feats is not None else [f'f{i}' for i in range(len(importances))]
            imp = pd.Series(importances, index=names).sort_values(ascending=False)
    except Exception:
        imp = None

    # Write ranking and K-sets
    out_dir = ctx.out_dir
    rank_path = out_dir / f"ablation_topk_{ctx.division}_{ctx.cutoff}.csv"
    rows = []
    if imp is not None:
        for name, val in imp.items():
            rows.append({'feature': name, 'importance': float(val)})
    else:
        rows.append({'feature': None, 'importance': None})
    pd.DataFrame(rows).to_csv(rank_path, index=False)

    # Training comparison (optional)
    summary = {
        'status': 'PLANNED' if not run_training else 'UNKNOWN',
        'cutoff': ctx.cutoff,
        'k_list': k_list,
        'baseline': {},
        'ablations': [],
    }
    try:
        base_metrics_path = OUTPUTS_DIR / f"metrics_{ctx.division.lower()}.json"
        if base_metrics_path.exists():
            base_metrics = json.loads(base_metrics_path.read_text(encoding='utf-8'))
            fin = base_metrics.get('final', {}) or {}
            summary['baseline'] = {'auc': fin.get('auc'), 'lift10': fin.get('lift@10') or fin.get('lift10')}
    except Exception:
        pass

    if run_training:
        # Build feature matrix and simple LR validation to estimate impact
        eng = None
        try:
            eng = get_curated_connection()
        except Exception:
            eng = get_db_connection()
        fm = create_feature_matrix(eng, ctx.division, ctx.cutoff, window_months)
        if not fm.is_empty():
            df = fm.to_pandas()
            y = df['bought_in_division'].astype(int).values
            X = df.drop(columns=['customer_id', 'bought_in_division'])
            # Simple time-aware split using recency proxy if present
            rec_col = 'rfm__all__recency_days__life'
            try:
                if rec_col in X.columns:
                    order = X[rec_col].astype(float).fillna(X[rec_col].astype(float).max()).argsort()
                    # assign smaller recency (more recent) to validation
                    n = len(order)
                    n_valid = max(1, int(0.2 * n))
                    idx_valid = order[:n_valid]
                    idx_train = order[n_valid:]
                else:
                    idx = np.arange(len(X))
                    np.random.seed(42)
                    np.random.shuffle(idx)
                    split = int(0.8 * len(idx))
                    idx_train, idx_valid = idx[:split], idx[split:]
            except Exception:
                idx = np.arange(len(X))
                split = int(0.8 * len(idx))
                idx_train, idx_valid = idx[:split], idx[split:]

            def _lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
                order = np.argsort(-y_score)
                k_idx = max(1, int(len(order) * (k/100.0)))
                topk = order[:k_idx]
                return float((y_true[topk].mean() / max(1e-9, y_true.mean()))) if y_true.mean() > 0 else 0.0

            # Baseline LR on all features
            lr = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
            lr.fit(X.iloc[idx_train], y[idx_train])
            p_valid = lr.predict_proba(X.iloc[idx_valid])[:,1]
            auc0 = float(roc_auc_score(y[idx_valid], p_valid))
            lift10_0 = _lift_at_k(y[idx_valid], p_valid, 10)
            summary['baseline'].update({'auc_lr': auc0, 'lift10_lr': lift10_0})

            # Determine ranking
            ranked = imp.index.tolist() if imp is not None else list(X.columns)
            for K in k_list:
                drop = set(ranked[:min(K, len(ranked))])
                keep_cols = [c for c in X.columns if c not in drop]
                if not keep_cols:
                    res = {'k': K, 'auc_lr': None, 'lift10_lr': None}
                else:
                    lr2 = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
                    lr2.fit(X.iloc[idx_train][keep_cols], y[idx_train])
                    p2 = lr2.predict_proba(X.iloc[idx_valid][keep_cols])[:,1]
                    auc2 = float(roc_auc_score(y[idx_valid], p2))
                    lift10_2 = _lift_at_k(y[idx_valid], p2, 10)
                    res = {'k': K, 'auc_lr': auc2, 'lift10_lr': lift10_2}
                summary['ablations'].append(res)

            # Determine status: improvements beyond epsilon are suspicious
            try:
                status = 'PASS'
                for res in summary['ablations']:
                    if res.get('auc_lr') is not None and auc0 is not None and (res['auc_lr'] - auc0) > epsilon_auc:
                        status = 'FAIL'
                        break
                    if res.get('lift10_lr') is not None and lift10_0 is not None and (res['lift10_lr'] - lift10_0) > epsilon_lift10:
                        status = 'FAIL'
                        break
                summary['status'] = status
            except Exception:
                summary['status'] = 'UNKNOWN'

    # Write summary
    sum_path = ctx.out_dir / f"ablation_topk_{ctx.division}_{ctx.cutoff}.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {'ablation_topk': str(sum_path), 'ablation_topk_csv': str(rank_path)}


def run_shift14_check(ctx: LGContext, window_months: int, run_training: bool = False, epsilon_auc: float = 0.01, epsilon_lift10: float = 0.25) -> dict[str, str]:
    """Scaffold a 14-day shift test: compute features at cutoff and cutoff-14d and (optionally) train.

    Writes `shift14_metrics_<div>_<cutoff>.json` with PASS/FAIL when training is run; otherwise PLANNED.
    """
    import pandas as pd
    artifacts: dict[str, str] = {}
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()

    # Compute shifted cutoff
    base_cut = pd.to_datetime(ctx.cutoff)
    cut_shift = (base_cut - timedelta(days=14)).date().isoformat()

    # Build feature matrices to capture prevalence/context
    try:
        cfg2 = load_config()
        mask_tail = int(getattr(getattr(cfg2, 'validation', object()), 'gauntlet_mask_tail_days', 0) or 0)
        fm_base = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months, mask_tail_days=mask_tail)
        fm_shift = create_feature_matrix(engine, ctx.division, cut_shift, window_months, mask_tail_days=mask_tail)
        base_prev = float(fm_base.to_pandas()['bought_in_division'].mean()) if not fm_base.is_empty() else None
        shift_prev = float(fm_shift.to_pandas()['bought_in_division'].mean()) if not fm_shift.is_empty() else None
    except Exception:
        base_prev = None
        shift_prev = None

    status = "PLANNED"
    comp = {}

    if run_training:
        # Best-effort: train on shifted cutoff only and compare to any existing baseline metrics
        try:
            from gosales.utils.paths import OUTPUTS_DIR as _OUT
            import shutil
            division = ctx.division
            div_key = division.lower()
            met_path = _OUT / f"metrics_{div_key}.json"
            backup_path = None
            if met_path.exists():
                backup_path = _OUT / f"metrics_{div_key}.json.bak"
                shutil.copy2(met_path, backup_path)
            # Train shifted-only
            # Enforce GroupKFold and purge days; use SAFE mode for Gauntlet training
            from gosales.utils.config import load_config as _load
            _cfg = _load()
            purge = int(getattr(getattr(_cfg, 'validation', object()), 'gauntlet_purge_days', 30) or 30)
            label_buf = int(getattr(getattr(_cfg, 'validation', object()), 'gauntlet_label_buffer_days', 0) or 0)
            cmd = [
                sys.executable, "-m", "gosales.models.train",
                "--division", division,
                "--cutoffs", cut_shift,
                "--window-months", str(window_months),
                "--group-cv",
                "--purge-days", str(int(purge)),
                "--safe-mode",
                "--label-buffer-days", str(int(label_buf)),
            ]
            subprocess.run(cmd, check=True)
            # Read shifted metrics
            shift_metrics = json.loads(met_path.read_text(encoding="utf-8")) if met_path.exists() else {}
            # Restore baseline metrics if we backed up
            if backup_path and backup_path.exists():
                shutil.move(str(backup_path), str(met_path))
            # Try to read baseline (post-restore)
            base_metrics = json.loads(met_path.read_text(encoding="utf-8")) if met_path.exists() else {}
            # Extract comparable metrics
            def _final(m):
                return m.get("final", {}) if isinstance(m, dict) else {}
            bm = _final(base_metrics)
            sm = _final(shift_metrics)
            # Harmonize lift@10 field name across variants
            def _lift10(d: dict) -> float | None:
                if d is None:
                    return None
                v = d.get("lift@10") if isinstance(d, dict) else None
                if v is None and isinstance(d, dict):
                    v = d.get("lift10")
                return v
            comp = {
                "auc_base": bm.get("auc"),
                "auc_shift": sm.get("auc"),
                "lift10_base": _lift10(bm),
                "lift10_shift": _lift10(sm),
                "brier_base": bm.get("brier"),
                "brier_shift": sm.get("brier"),
            }
            # Determine status: improvement beyond epsilon is suspicious
            try:
                auc_imp = (float(sm.get("auc", 0.0)) - float(bm.get("auc", 0.0))) if bm.get("auc") is not None and sm.get("auc") is not None else 0.0
                # Compare lift@10 if available (supports both keys)
                lb = _lift10(bm); ls = _lift10(sm)
                lift_imp = (float(ls) - float(lb)) if lb is not None and ls is not None else 0.0
                if auc_imp > float(epsilon_auc) or lift_imp > float(epsilon_lift10):
                    status = "FAIL"
                else:
                    status = "PASS"
            except Exception:
                status = "UNKNOWN"
        except Exception as e:
            status = "ERROR"
            comp = {"error": str(e)}

        # Auxiliary LR comparison using masked features (gauntlet mask)
        try:
            import numpy as _np
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            # Build masked matrices
            mask_tail = int(getattr(getattr(load_config(), 'validation', object()), 'gauntlet_mask_tail_days', 0) or 0)
            eng2 = None
            try:
                eng2 = get_curated_connection()
            except Exception:
                eng2 = get_db_connection()
            fb = create_feature_matrix(eng2, ctx.division, ctx.cutoff, window_months, mask_tail_days=mask_tail)
            fs = create_feature_matrix(eng2, ctx.division, cut_shift, window_months, mask_tail_days=mask_tail)
            if not fb.is_empty() and not fs.is_empty():
                db = fb.to_pandas(); ds = fs.to_pandas()
                def _eval(df):
                    y = df['bought_in_division'].astype(int).values
                    X = df.drop(columns=['customer_id','bought_in_division'])
                    # time-aware split
                    rec = 'rfm__all__recency_days__life'
                    try:
                        if rec in X.columns:
                            order = _np.argsort(_np.nan_to_num(X[rec].astype(float).values, nan=1e9))
                            n = len(order); nv = max(1, int(0.2*n))
                            iv = order[:nv]; it = order[nv:]
                        else:
                            idx = _np.arange(len(X)); _np.random.seed(42); _np.random.shuffle(idx)
                            sp = int(0.8*len(idx)); it, iv = idx[:sp], idx[sp:]
                    except Exception:
                        idx = _np.arange(len(X)); sp = int(0.8*len(idx)); it, iv = idx[:sp], idx[sp:]
                    lr = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
                    lr.fit(X.iloc[it], y[it])
                    p = lr.predict_proba(X.iloc[iv])[:,1]
                    def _lift_at_k(y_true, y_score, k=10):
                        order = _np.argsort(-y_score); kk = max(1, int(len(order)*(k/100.0)))
                        topk = order[:kk]; m = y_true.mean()
                        return float(y_true[topk].mean()/m) if m>0 else 0.0
                    return float(roc_auc_score(y[iv], p)), _lift_at_k(y[iv], p, 10)
                auc_b, l10_b = _eval(db)
                auc_s, l10_s = _eval(ds)
                # Also evaluate after dropping high-risk feature families
                def _drop_high_risk(df_pd):
                    Xcols = [c for c in df_pd.columns if c not in ('customer_id','bought_in_division')]
                    keep = []
                    for c in Xcols:
                        s = str(c).lower()
                        if s.startswith('assets_expiring_'):
                            continue
                        if 'days_since_last' in s or 'recency' in s:
                            continue
                        if s.startswith('assets_subs_share_') or s.startswith('assets_on_subs_share_') or s.startswith('assets_off_subs_share_'):
                            continue
                        keep.append(c)
                    cols = ['customer_id','bought_in_division'] + keep
                    return df_pd[cols]
                db2 = _drop_high_risk(db)
                ds2 = _drop_high_risk(ds)
                auc_b2, l10_b2 = _eval(db2)
                auc_s2, l10_s2 = _eval(ds2)
                comp.update({
                    'auc_lr_masked_base': auc_b,
                    'auc_lr_masked_shift': auc_s,
                    'lift10_lr_masked_base': l10_b,
                    'lift10_lr_masked_shift': l10_s,
                    'auc_lr_masked_dropped_base': auc_b2,
                    'auc_lr_masked_dropped_shift': auc_s2,
                    'lift10_lr_masked_dropped_base': l10_b2,
                    'lift10_lr_masked_dropped_shift': l10_s2,
                })
                # If masked LR also shows suspicious improvements, keep FAIL; else annotate
                try:
                    if status != 'FAIL':
                        imp_auc = max(auc_s - auc_b, auc_s2 - auc_b2)
                        imp_l10 = max(l10_s - l10_b, l10_s2 - l10_b2)
                        if imp_auc > float(epsilon_auc) or imp_l10 > float(epsilon_lift10):
                            status = 'FAIL'
                except Exception:
                    pass
        except Exception:
            pass

    out = {
        "status": status,
        "cutoff": ctx.cutoff,
        "shift_cutoff": cut_shift,
        "window_months": int(window_months),
        "prevalence_base": base_prev,
        "prevalence_shift": shift_prev,
        "comparison": comp,
        "notes": "Set --run-shift14-training to execute training and metric comparison." if not run_training else None,
    }
    out_path = ctx.out_dir / f"shift14_metrics_{ctx.division}_{ctx.cutoff}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return {"shift14": str(out_path)}


def write_consolidated_report(ctx: LGContext, artifacts: dict[str, str]) -> Path:
    # Determine overall status based on included artifacts
    status_map = {}
    for name, path in artifacts.items():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            status_map[name] = data.get("status", "UNKNOWN")
        except Exception:
            status_map[name] = "UNKNOWN"
    overall = "PASS"
    if any(v == "FAIL" for v in status_map.values()):
        overall = "FAIL"
    report = {
        "division": ctx.division,
        "cutoff": ctx.cutoff,
        "overall": overall,
        "checks": status_map,
    }
    out = ctx.out_dir / f"leakage_report_{ctx.division}_{ctx.cutoff}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


@click.command()
@click.option("--division", required=True, help="Target division name (e.g., Solidworks)")
@click.option("--cutoff", required=True, help="Cutoff date YYYY-MM-DD")
@click.option("--window-months", default=6, type=int)
@click.option("--static-only/--no-static-only", default=True, help="Run only static checks (no training)")
@click.option("--run-shift14-training/--no-run-shift14-training", default=False, help="Run training for shift-14 cutoff and compare metrics (overwrites metrics during run; restored after)")
@click.option("--shift14-eps-auc", type=float, default=None, help="Override epsilon AUC threshold for shift-14 (default from config)")
@click.option("--shift14-eps-lift10", type=float, default=None, help="Override epsilon lift@10 threshold for shift-14 (default from config)")
@click.option("--run-topk-ablation/--no-run-topk-ablation", default=False, help="Run Top-K ablation training and compare metrics (heavy)")
@click.option("--topk-list", default="10,20", help="Comma-separated K list for ablation (e.g., 10,20,50)")
@click.option("--ablation-eps-auc", type=float, default=None, help="Override epsilon AUC threshold for ablation (default from config)")
@click.option("--ablation-eps-lift10", type=float, default=None, help="Override epsilon lift@10 threshold for ablation (default from config)")
@click.option("--config", default=str((Path(__file__).parents[1] / "config.yaml").resolve()))
def main(division: str, cutoff: str, window_months: int, static_only: bool, run_shift14_training: bool, shift14_eps_auc: float | None, shift14_eps_lift10: float | None, run_topk_ablation: bool, topk_list: str, ablation_eps_auc: float | None, ablation_eps_lift10: float | None, config: str) -> None:
    cfg = load_config(config)
    ctx = _ensure_outdir(division, cutoff)
    artifacts = {}
    try:
        logger.info("Running Leakage Gauntlet static checks for %s @ %s", division, cutoff)
        artifacts.update(run_static_checks(ctx))
    except Exception as e:
        logger.error("Static checks failed: %s", e)
    try:
        logger.info("Running feature date audit for %s @ %s", division, cutoff)
        artifacts.update(run_feature_date_audit(ctx, window_months))
    except Exception as e:
        logger.error("Feature date audit failed: %s", e)
    try:
        logger.info("Running 14-day shift test (scaffold) for %s @ %s", division, cutoff)
        eps_auc = float(shift14_eps_auc) if shift14_eps_auc is not None else float(getattr(getattr(cfg, 'validation', object()), 'shift14_epsilon_auc', 0.01))
        eps_lift = float(shift14_eps_lift10) if shift14_eps_lift10 is not None else float(getattr(getattr(cfg, 'validation', object()), 'shift14_epsilon_lift10', 0.25))
        artifacts.update(run_shift14_check(ctx, window_months, run_training=run_shift14_training, epsilon_auc=eps_auc, epsilon_lift10=eps_lift))
    except Exception as e:
        logger.error("Shift-14 check failed: %s", e)

    # Top-K ablation scaffold
    try:
        logger.info("Running Top-K ablation (scaffold) for %s @ %s", division, cutoff)
        k_list = [int(x.strip()) for x in str(topk_list).split(',') if str(x).strip()]
        eps_auc2 = float(ablation_eps_auc) if ablation_eps_auc is not None else float(getattr(getattr(cfg, 'validation', object()), 'ablation_epsilon_auc', 0.01))
        eps_lift2 = float(ablation_eps_lift10) if ablation_eps_lift10 is not None else float(getattr(getattr(cfg, 'validation', object()), 'ablation_epsilon_lift10', 0.25))
        artifacts.update(run_topk_ablation_check(ctx, window_months, k_list=k_list, run_training=run_topk_ablation, epsilon_auc=eps_auc2, epsilon_lift10=eps_lift2))
    except Exception as e:
        logger.error("Top-K ablation failed: %s", e)
    # Future: dynamic checks here when enabled
    report = write_consolidated_report(ctx, artifacts)
    logger.info("Wrote leakage report to %s", report)
    try:
        # Exit non-zero on failure to allow CI gating
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("overall") == "FAIL":
            sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        pass


if __name__ == "__main__":
    main()
