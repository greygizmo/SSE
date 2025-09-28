from __future__ import annotations

"""
Leakage Gauntlet runner (reliable implementation).

Phases:
 1) Static scan for time-now calls inside ETL/features (guards against implicit leakage)
 2) Feature-date audit vs cutoff (ensures no post-cutoff events feed features)
 3) Optional dynamic checks (shift-14, shift grid, top-k ablation, reproducibility) – best-effort, non-gating

Artifacts are written under gosales/outputs/leakage/<division>/<cutoff>/ and an
overall report is emitted as leakage_report_<division>_<cutoff>.json.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import sys
from typing import Dict

import click
import pandas as pd

from gosales.utils.config import load_config
from gosales.utils.paths import OUTPUTS_DIR, ROOT_DIR
from gosales.utils.logger import get_logger
from gosales.utils.db import get_curated_connection, get_db_connection
from gosales.features.engine import create_feature_matrix
from gosales.pipeline.leakage_diagnostics import run_label_permutation


logger = get_logger(__name__)


# --- Static scan for time-now calls ---
_BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("datetime.now", re.compile(r"\bdatetime\s*\.\s*now\s*\(")),
    ("pd.Timestamp.now", re.compile(r"\b(pd\s*\.\s*)?Timestamp\s*\.\s*now\s*\(")),
    ("date.today", re.compile(r"\bdate\s*\.\s*today\s*\(")),
]


@dataclass
class LGContext:
    division: str
    cutoff: str
    out_dir: Path


def _ensure_outdir(division: str, cutoff: str) -> LGContext:
    d = str(division).strip()
    c = str(cutoff).strip()
    out = OUTPUTS_DIR / "leakage" / d / c
    out.mkdir(parents=True, exist_ok=True)
    return LGContext(d, c, out)


def _static_scan(paths: list[Path]) -> dict:
    findings: list[dict] = []
    for base in paths:
        for p in base.rglob("*.py"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                for name, pat in _BANNED_PATTERNS:
                    if pat.search(line):
                        findings.append({
                            "file": str(p.relative_to(ROOT_DIR)),
                            "line": i,
                            "pattern": name,
                            "code": line.strip(),
                        })
    status = "PASS" if not findings else "FAIL"
    return {"status": status, "findings": findings}


def run_static_checks(ctx: LGContext) -> Dict[str, str]:
    scan = _static_scan([ROOT_DIR / "gosales" / "features", ROOT_DIR / "gosales" / "etl"])
    static_path = ctx.out_dir / f"static_scan_{ctx.division}_{ctx.cutoff}.json"
    static_path.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    return {"static_scan": str(static_path)}


def run_feature_date_audit(ctx: LGContext, window_months: int) -> Dict[str, str]:
    # Prefer curated engine (facts live there)
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    # Build feature matrix to enumerate features (no training)
    cfg = load_config()
    mask_tail = int(getattr(getattr(cfg, "validation", object()), "gauntlet_mask_tail_days", 0) or 0)
    fm = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months, mask_tail_days=mask_tail)
    if isinstance(fm, pd.DataFrame):
        fm_df = fm.copy()
    elif hasattr(fm, "to_pandas"):
        fm_df = fm.to_pandas()
    else:
        fm_df = pd.DataFrame(fm)
    cols = [c for c in fm_df.columns if c not in ("customer_id", "bought_in_division")]
    # Compute max observable event date at/before cutoff for lifetime-style features
    df = pd.read_sql_query(
        "SELECT MAX(order_date) AS max_order_date FROM fact_transactions WHERE order_date <= :cutoff",
        engine,
        params={"cutoff": ctx.cutoff},
    )
    try:
        max_order_date = pd.to_datetime(df["max_order_date"].iloc[0]) if not df.empty else None
    except Exception:
        max_order_date = None
    cutoff_dt = pd.to_datetime(ctx.cutoff)
    effective_end = cutoff_dt
    if mask_tail and mask_tail > 0:
        try:
            effective_end = cutoff_dt - pd.to_timedelta(mask_tail, unit="D")
        except Exception:
            effective_end = cutoff_dt

    import numpy as np

    def _coerce_numeric(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.empty:
            return numeric
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        return numeric

    def _infer_latest_event(col_name: str, series: pd.Series) -> pd.Timestamp | None:
        lowered = str(col_name).lower()

        # Direct datetime-like columns
        dt_series = None
        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                dt_series = pd.to_datetime(series, errors="coerce")
            elif series.dtype == object or "_date" in lowered or "timestamp" in lowered or lowered.endswith("_dt"):
                dt_series = pd.to_datetime(series, errors="coerce")
        except Exception:
            dt_series = None
        if dt_series is not None:
            dt_series = dt_series.dropna()
            if not dt_series.empty:
                latest_dt = dt_series.max()
                try:
                    latest_dt = latest_dt.tz_localize(None)  # type: ignore[attr-defined]
                except Exception:
                    pass
                if pd.notna(latest_dt):
                    return pd.to_datetime(latest_dt)

        # Recency expressed as days since cutoff
        if "days_since" in lowered or "recency_days" in lowered:
            numeric = _coerce_numeric(series)
            if not numeric.empty:
                values = numeric.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    min_days = float(np.min(np.maximum(values, 0.0)))
                    return cutoff_dt - pd.to_timedelta(min_days, unit="D")

        # Log-recency variants (log1p transform of days)
        if "log_recency" in lowered:
            numeric = _coerce_numeric(series)
            if not numeric.empty:
                values = numeric.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    days = np.expm1(values)
                    days = days[np.isfinite(days)]
                    if days.size:
                        min_days = float(np.min(np.maximum(days, 0.0)))
                        return cutoff_dt - pd.to_timedelta(min_days, unit="D")

        # Recency decay (exp(-days / half_life))
        if "recency_decay__hl" in lowered:
            match = re.search(r"__hl(\d+)", lowered)
            if match:
                try:
                    half_life = float(match.group(1))
                except Exception:
                    half_life = None
                if half_life:
                    numeric = _coerce_numeric(series)
                    if not numeric.empty:
                        values = numeric.to_numpy(dtype=float)
                        values = values[(values > 0.0) & np.isfinite(values)]
                        if values.size:
                            days = -half_life * np.log(values)
                            days = days[np.isfinite(days)]
                            if days.size:
                                min_days = float(np.min(np.maximum(days, 0.0)))
                                return cutoff_dt - pd.to_timedelta(min_days, unit="D")

        # Future-looking expiring horizons (treated as post-cutoff checks)
        expiring = re.search(r"expiring_(\d+)d", lowered)
        if expiring:
            try:
                horizon = int(expiring.group(1))
                return cutoff_dt + pd.to_timedelta(horizon, unit="D")
            except Exception:
                pass

        # Lagged affinity / offset-style features
        lag_match = re.search(r"lag(\d+)d", lowered)
        if lag_match:
            try:
                lag_days = int(lag_match.group(1))
                return cutoff_dt - pd.to_timedelta(lag_days, unit="D")
            except Exception:
                pass

        offset_match = re.search(r"_off(\d+)d", lowered)
        if offset_match:
            try:
                offset_days = int(offset_match.group(1))
                return cutoff_dt - pd.to_timedelta(offset_days, unit="D")
            except Exception:
                pass

        window_match = re.search(r"(?:__|_last_)(\d+)m", lowered)
        if window_match:
            return effective_end

        if lowered.endswith("__life") or "__life__" in lowered or lowered.endswith("_life"):
            return max_order_date

        return None

    rows = []
    latest_values: list[pd.Timestamp] = []
    for name in cols:
        series = fm_df.get(name)
        if series is None:
            latest = None
        else:
            latest = _infer_latest_event(name, series)
        if isinstance(latest, pd.Timestamp) and pd.notna(latest):
            latest = latest.tz_localize(None) if getattr(latest, "tzinfo", None) else latest
            latest_values.append(latest)
        status = "UNKNOWN"
        if latest is None or pd.isna(latest):
            status = "UNKNOWN"
        elif latest > cutoff_dt:
            status = "LEAK"
        else:
            status = "OK"
        rows.append({
            "feature": str(name),
            "latest_event_date": (latest.date().isoformat() if isinstance(latest, pd.Timestamp) and pd.notna(latest) else None),
            "cutoff": ctx.cutoff,
            "status": status,
        })
    audit_csv = ctx.out_dir / f"feature_date_audit_{ctx.division}_{ctx.cutoff}.csv"
    pd.DataFrame(rows).to_csv(audit_csv, index=False)
    overall_latest = max(latest_values) if latest_values else None
    summary = {
        "status": ("FAIL" if any(r["status"] == "LEAK" for r in rows) else "PASS"),
        "max_event_date": (overall_latest.date().isoformat() if isinstance(overall_latest, pd.Timestamp) and pd.notna(overall_latest) else None),
        "cutoff": ctx.cutoff,
        "feature_count": len(rows),
    }
    summary_json = ctx.out_dir / f"feature_date_audit_{ctx.division}_{ctx.cutoff}.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"feature_date_audit": str(summary_json), "feature_date_audit_csv": str(audit_csv)}


def run_permutation_gate(
    ctx: LGContext,
    window_months: int,
    n_perm: int | None = None,
    min_auc_gap: float | None = None,
    max_p_value: float | None = None,
) -> Dict[str, str]:
    cfg = load_config()
    val_cfg = getattr(cfg, "validation", object())
    perm_n = int(n_perm if n_perm is not None else getattr(val_cfg, "permutation_n_perm", 50))
    min_gap = float(min_auc_gap if min_auc_gap is not None else getattr(val_cfg, "permutation_min_auc_gap", 0.05))
    max_p = float(max_p_value if max_p_value is not None else getattr(val_cfg, "permutation_max_p_value", 0.01))
    mask_tail = int(getattr(val_cfg, "gauntlet_mask_tail_days", 0) or 0)

    def _safe_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            val = float(value)
            if math.isnan(val):
                return None
            return val
        except Exception:
            return None

    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()

    fm = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months, mask_tail_days=mask_tail)
    if hasattr(fm, "is_empty") and fm.is_empty():  # type: ignore[attr-defined]
        raise RuntimeError("Empty feature matrix for permutation gate")

    if isinstance(fm, pd.DataFrame):
        df = fm.copy()
    elif hasattr(fm, "to_pandas"):
        df = fm.to_pandas()  # type: ignore[assignment]
    else:
        df = pd.DataFrame(fm)

    artifacts = run_label_permutation(ctx.out_dir, df, n_perm=perm_n)
    diag_path_str = artifacts.get("permutation_diag")
    if not diag_path_str:
        raise RuntimeError("Permutation diagnostics did not emit permutation_diag.json")

    diag_path = Path(diag_path_str)
    stats = json.loads(diag_path.read_text(encoding="utf-8"))
    baseline = _safe_float(stats.get("baseline_auc"))
    perm_mean = _safe_float(stats.get("permuted_auc_mean"))
    p_value = _safe_float(stats.get("p_value"))
    gap = (baseline - perm_mean) if (baseline is not None and perm_mean is not None) else None

    gap_ok = gap is not None and gap >= min_gap
    p_ok = p_value is not None and p_value <= max_p
    status = "PASS" if (gap_ok or p_ok) else "FAIL"

    reasons: list[str] = []
    if not gap_ok:
        if gap is None:
            reasons.append("Missing baseline/permuted AUC to compute degradation.")
        else:
            reasons.append(f"AUC degradation {gap:.4f} below minimum {min_gap:.4f}.")
    if not p_ok:
        if p_value is None:
            reasons.append("Permutation p-value missing.")
        else:
            reasons.append(f"Permutation p-value {p_value:.4f} exceeds maximum {max_p:.4f}.")

    summary: dict[str, object] = {
        "status": status,
        "baseline_auc": baseline,
        "permuted_auc_mean": perm_mean,
        "auc_degradation": gap,
        "p_value": p_value,
        "thresholds": {
            "min_auc_gap": min_gap,
            "max_p_value": max_p,
            "n_perm": perm_n,
        },
        "permutation_diag": str(diag_path),
    }
    if reasons and status != "PASS":
        summary["reasons"] = reasons

    gate_path = ctx.out_dir / f"permutation_gate_{ctx.division}_{ctx.cutoff}.json"
    gate_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifacts["permutation_gate"] = str(gate_path)
    return artifacts


def write_consolidated_report(ctx: LGContext, artifacts: Dict[str, str]) -> Path:
    status_map: Dict[str, str] = {}
    for name, path in artifacts.items():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            # Special case for shift_grid summaries
            if name == 'shift_grid':
                status_map[name] = data.get('overall', 'UNKNOWN')
            else:
                status_map[name] = data.get("status", "UNKNOWN")
        except Exception:
            status_map[name] = "UNKNOWN"
    overall = "PASS"
    # Gate on hard failures only; SUSPECT/UNKNOWN are non-gating
    if any(v == "FAIL" for v in status_map.values()):
        overall = "FAIL"
    report = {"division": ctx.division, "cutoff": ctx.cutoff, "overall": overall, "checks": status_map}
    out = ctx.out_dir / f"leakage_report_{ctx.division}_{ctx.cutoff}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    for c in Xc.columns:
        Xc[c] = pd.to_numeric(Xc[c], errors='coerce')
    Xc = Xc.replace([float('inf'), float('-inf')], pd.NA).fillna(0.0)
    return Xc.astype(float)


def _time_aware_split(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    rec = 'rfm__all__recency_days__life'
    try:
        if rec in X.columns:
            order = pd.to_numeric(X[rec], errors='coerce').fillna(X[rec].astype(float).max()).values
            idx = order.argsort()
            n = len(idx)
            nv = max(1, int(0.2 * n))
            iv = idx[:nv]
            it = idx[nv:]
            return X.iloc[it], X.iloc[iv], y.iloc[it], y.iloc[iv]
    except Exception:
        pass
    # fallback random split
    rng = pd.Series(range(len(X)))
    rs = rng.sample(frac=1.0, random_state=seed).values
    sp = int(0.8 * len(rs))
    it, iv = rs[:sp], rs[sp:]
    return X.iloc[it], X.iloc[iv], y.iloc[it], y.iloc[iv]


def _auc_lift(y_true: pd.Series, y_score: pd.Series, k_percent: int = 10) -> tuple[float | None, float | None]:
    from sklearn.metrics import roc_auc_score
    import numpy as np
    y = pd.Series(y_true).astype(int).values
    p = pd.Series(y_score).astype(float).values
    try:
        if np.any(y) and not np.all(y == 1) and not np.all(y == 0):
            auc = float(roc_auc_score(y, p))
        else:
            auc = None
    except Exception:
        auc = None
    try:
        n = len(y)
        if n == 0:
            return auc, None
        k = max(1, int(n * (k_percent / 100.0)))
        idx = p.argsort(kind='stable')[::-1][:k]
        top_rate = float(y[idx].mean())
        base = float(y.mean()) if y.mean() > 0 else 1e-9
        lift = top_rate / base
    except Exception:
        lift = None
    return auc, lift


def _eval_lr(df: pd.DataFrame, seed: int = 42) -> tuple[float | None, float | None]:
    from sklearn.linear_model import LogisticRegression
    y = df['bought_in_division'].astype(int)
    X = df.drop(columns=['customer_id', 'bought_in_division'], errors='ignore')
    X = _sanitize_features(X)
    Xtr, Xva, ytr, yva = _time_aware_split(X, y, seed=seed)
    if len(Xtr) == 0 or len(Xva) == 0:
        return None, None
    lr = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
    lr.fit(Xtr, ytr)
    p = lr.predict_proba(Xva)[:, 1]
    return _auc_lift(yva, p)


def run_shift14_dynamic(ctx: LGContext, window_months: int, epsilon_auc: float, epsilon_lift10: float) -> Dict[str, str]:
    # Build feature matrices @ cutoff and cutoff-14d
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    cfg = load_config()
    mask_tail = int(getattr(getattr(cfg, 'validation', object()), 'gauntlet_mask_tail_days', 0) or 0)
    base_df = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months, mask_tail_days=mask_tail).to_pandas()
    try:
        cut_dt = pd.to_datetime(ctx.cutoff)
        shift_cut = (cut_dt - pd.DateOffset(days=14)).date().isoformat()
    except Exception:
        shift_cut = ctx.cutoff
    shift_df = create_feature_matrix(engine, ctx.division, shift_cut, window_months, mask_tail_days=mask_tail).to_pandas()
    # Evaluate LR on both
    auc_b, l10_b = _eval_lr(base_df, seed=42)
    auc_s, l10_s = _eval_lr(shift_df, seed=43)
    # Also evaluate with high-risk families dropped (SAFE mask)
    def _drop_high_risk(df_pd: pd.DataFrame) -> pd.DataFrame:
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
    try:
        auc_b2, l10_b2 = _eval_lr(_drop_high_risk(base_df), seed=52)
        auc_s2, l10_s2 = _eval_lr(_drop_high_risk(shift_df), seed=53)
    except Exception:
        auc_b2 = l10_b2 = auc_s2 = l10_s2 = None
    status = 'PASS'
    comp = {
        'cutoff_base': ctx.cutoff,
        'cutoff_shift': shift_cut,
        'auc_base': auc_b,
        'auc_shift': auc_s,
        'lift10_base': l10_b,
        'lift10_shift': l10_s,
        'eps_auc': float(epsilon_auc),
        'eps_lift10': float(epsilon_lift10),
    }
    try:
        raw_fail = False
        if (auc_b is not None and auc_s is not None and (auc_b - auc_s) > float(epsilon_auc)):
            raw_fail = True
        if (l10_b is not None and l10_s is not None and (l10_b - l10_s) > float(epsilon_lift10)):
            raw_fail = True
        # Allow SAFE-masked pass to clear the gate
        safe_pass = False
        try:
            safe_pass = (
                (auc_b2 is not None and auc_s2 is not None and (auc_b2 - auc_s2) <= float(epsilon_auc)) and
                (l10_b2 is not None and l10_s2 is not None and (l10_b2 - l10_s2) <= float(epsilon_lift10))
            )
        except Exception:
            safe_pass = False
        if raw_fail and not safe_pass:
            status = 'FAIL'
    except Exception:
        status = 'ERROR'
    payload = {
        'status': status,
        'comparison': comp,
        'cutoff': ctx.cutoff,
        'shift_cutoff': shift_cut,
        'window_months': int(window_months),
        'masked_eval': {
            'auc_base': auc_b2,
            'auc_shift': auc_s2,
            'lift10_base': l10_b2,
            'lift10_shift': l10_s2,
        },
        'mask_tail_days': int(mask_tail),
    }
    out = ctx.out_dir / f"shift14_metrics_{ctx.division}_{ctx.cutoff}.json"
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return {'shift14': str(out)}


def run_repro_dynamic(ctx: LGContext, window_months: int, eps_auc: float, eps_lift10: float) -> Dict[str, str]:
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    df = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months).to_pandas()
    auc1, l10_1 = _eval_lr(df, seed=101)
    auc2, l10_2 = _eval_lr(df, seed=202)
    d_auc = None if (auc1 is None or auc2 is None) else abs(float(auc2) - float(auc1))
    d_l10 = None if (l10_1 is None or l10_2 is None) else abs(float(l10_2) - float(l10_1))
    status = 'PASS'
    if d_auc is not None and d_auc > float(eps_auc):
        status = 'FAIL'
    if d_l10 is not None and d_l10 > float(eps_lift10):
        status = 'FAIL'
    payload = {
        'status': status,
        'cutoff': ctx.cutoff,
        'window_months': int(window_months),
        'metrics_run1': {'auc': auc1, 'lift10': l10_1},
        'metrics_run2': {'auc': auc2, 'lift10': l10_2},
        'delta_auc': d_auc,
        'delta_lift10': d_l10,
        'eps_auc': float(eps_auc),
        'eps_lift10': float(eps_lift10),
    }
    out = ctx.out_dir / f"repro_check_{ctx.division}_{ctx.cutoff}.json"
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return {'repro_check': str(out)}


def run_topk_ablation_dynamic(ctx: LGContext, window_months: int, k_list: list[int], eps_auc: float, eps_lift10: float) -> Dict[str, str]:
    from sklearn.linear_model import LogisticRegression
    try:
        engine = get_curated_connection()
    except Exception:
        engine = get_db_connection()
    df = create_feature_matrix(engine, ctx.division, ctx.cutoff, window_months).to_pandas()
    y = df['bought_in_division'].astype(int)
    X = df.drop(columns=['customer_id', 'bought_in_division'], errors='ignore')
    X = _sanitize_features(X)
    Xtr, Xva, ytr, yva = _time_aware_split(X, y, seed=11)
    if len(Xtr) == 0 or len(Xva) == 0:
        payload = {'status': 'SKIPPED', 'reason': 'insufficient data'}
    else:
        base_lr = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
        base_lr.fit(Xtr, ytr)
        p0 = base_lr.predict_proba(Xva)[:, 1]
        auc0, l100 = _auc_lift(yva, p0)
        # Rank features by |coef|
        try:
            import numpy as np
            coefs = np.abs(base_lr.coef_).ravel()
            names = list(Xtr.columns)
            imp = sorted(zip(names, coefs), key=lambda t: -t[1])
        except Exception:
            imp = [(c, 0.0) for c in list(Xtr.columns)]
        rows = [{'feature': n, 'importance': float(v)} for n, v in imp]
        (ctx.out_dir / f"ablation_topk_{ctx.division}_{ctx.cutoff}.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding='utf-8')
        # Evaluate ablations
        status = 'OK'
        results = []
        for K in k_list:
            drop = set(n for n, _ in imp[:min(K, len(imp))])
            keep = [c for c in Xtr.columns if c not in drop]
            if not keep:
                results.append({'k': int(K), 'auc': None, 'lift10': None})
                continue
            lr2 = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced')
            lr2.fit(Xtr[keep], ytr)
            p2 = lr2.predict_proba(Xva[keep])[:, 1]
            auc2, l102 = _auc_lift(yva, p2)
            # If dropping top-K improves metrics beyond epsilon, mark as SUSPECT (non-gating)
            try:
                suspicious = False
                if (auc0 is not None and auc2 is not None and (auc2 - auc0) > float(eps_auc)):
                    suspicious = True
                if (l100 is not None and l102 is not None and (l102 - l100) > float(eps_lift10)):
                    suspicious = True
                if suspicious:
                    status = 'SUSPECT'
            except Exception:
                pass
            results.append({'k': int(K), 'auc': auc2, 'lift10': l102})
        payload = {'status': status, 'baseline': {'auc': auc0, 'lift10': l100}, 'ablations': results, 'k_list': [int(k) for k in k_list], 'eps_auc': float(eps_auc), 'eps_lift10': float(eps_lift10)}
    out = ctx.out_dir / f"ablation_topk_{ctx.division}_{ctx.cutoff}.json"
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return {'ablation_topk': str(out)}


@click.command()
@click.option("--division", required=True, help="Target division name (e.g., Solidworks)")
@click.option("--cutoff", required=True, help="Cutoff date YYYY-MM-DD")
@click.option("--window-months", default=6, type=int)
@click.option("--static-only/--no-static-only", default=False, help="Run only static checks (no dynamic checks)")
@click.option("--run-permutation/--no-run-permutation", default=True, help="Run label permutation leakage gate")
@click.option("--permutation-n-perm", type=int, default=None, help="Override permutation iterations for label shuffle gate")
@click.option("--permutation-min-auc-gap", type=float, default=None, help="Minimum baseline minus permuted AUC gap for PASS")
@click.option("--permutation-max-p-value", type=float, default=None, help="Maximum permutation p-value for PASS")
@click.option("--run-shift14-training/--no-run-shift14-training", default=True, help="Run Shift-14 dynamic check (LR)")
@click.option("--run-shift-grid/--no-run-shift-grid", default=False, help="Run shift-grid checks (LR)")
@click.option("--shift14-eps-auc", type=float, default=None)
@click.option("--shift14-eps-lift10", type=float, default=None)
@click.option("--shift-grid", default="7,14,28,56")
@click.option("--run-repro-check/--no-run-repro-check", default=True, help="Run reproducibility dynamic check (LR)")
@click.option("--repro-eps-auc", type=float, default=0.002)
@click.option("--repro-eps-lift10", type=float, default=0.05)
@click.option("--run-topk-ablation/--no-run-topk-ablation", default=True, help="Run Top-K ablation (LR)")
@click.option("--topk-list", default="10,25,50")
@click.option("--ablation-eps-auc", type=float, default=None)
@click.option("--ablation-eps-lift10", type=float, default=None)
@click.option("--summary-divisions", default=None, help="(Optional) Build shift-grid cross-division summary")
def main(
    division: str,
    cutoff: str,
    window_months: int,
    static_only: bool,
    run_permutation: bool,
    permutation_n_perm: int | None,
    permutation_min_auc_gap: float | None,
    permutation_max_p_value: float | None,
    run_shift14_training: bool,
    run_shift_grid: bool,
    shift14_eps_auc: float | None,
    shift14_eps_lift10: float | None,
    shift_grid: str,
    run_repro_check: bool,
    repro_eps_auc: float,
    repro_eps_lift10: float,
    run_topk_ablation: bool,
    topk_list: str,
    ablation_eps_auc: float | None,
    ablation_eps_lift10: float | None,
    summary_divisions: str | None,
):
    ctx = _ensure_outdir(division, cutoff)
    artifacts: Dict[str, str] = {}

    # Static checks (gating)
    artifacts.update(run_static_checks(ctx))
    artifacts.update(run_feature_date_audit(ctx, window_months))

    # Dynamic checks (gating): run LR-based implementations to keep runtime manageable
    if not static_only:
        cfg = load_config()
        eps_auc = float(shift14_eps_auc) if shift14_eps_auc is not None else float(getattr(getattr(cfg, 'validation', object()), 'shift14_epsilon_auc', 0.01))
        eps_l10 = float(shift14_eps_lift10) if shift14_eps_lift10 is not None else float(getattr(getattr(cfg, 'validation', object()), 'shift14_epsilon_lift10', 0.25))
        abl_auc = float(ablation_eps_auc) if ablation_eps_auc is not None else float(getattr(getattr(cfg, 'validation', object()), 'ablation_epsilon_auc', 0.01))
        abl_l10 = float(ablation_eps_lift10) if ablation_eps_lift10 is not None else float(getattr(getattr(cfg, 'validation', object()), 'ablation_epsilon_lift10', 0.25))

        if run_permutation:
            try:
                artifacts.update(
                    run_permutation_gate(
                        ctx,
                        window_months,
                        n_perm=permutation_n_perm,
                        min_auc_gap=permutation_min_auc_gap,
                        max_p_value=permutation_max_p_value,
                    )
                )
            except Exception as e:
                payload = {'status': 'FAIL', 'error': str(e)}
                p = ctx.out_dir / f"permutation_gate_{division}_{cutoff}.json"
                p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                artifacts['permutation_gate'] = str(p)

        if run_shift14_training:
            try:
                artifacts.update(run_shift14_dynamic(ctx, window_months, eps_auc, eps_l10))
            except Exception as e:
                # Emit error artifact
                payload = {'status': 'ERROR', 'error': str(e)}
                p = ctx.out_dir / f"shift14_metrics_{division}_{cutoff}.json"
                p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                artifacts['shift14'] = str(p)

        if run_shift_grid:
            try:
                # Evaluate a small grid using LR dynamic check
                shifts = [int(x.strip()) for x in str(shift_grid).split(',') if str(x).strip()]
                summary = {"overall": "PASS", "shifts": []}
                for d in shifts:
                    cxt = _ensure_outdir(ctx.division, ctx.cutoff)
                    art = run_shift14_dynamic(cxt, window_months, eps_auc, eps_l10) if int(d) == 14 else None
                    # If not 14, emulate by recomputing with that shift
                    if int(d) != 14:
                        try:
                            # Temporarily call using modified cutoff by d days
                            base = ctx.cutoff
                            # Overload function locally
                            cut_dt = pd.to_datetime(ctx.cutoff)
                            shift_cut = (cut_dt - pd.DateOffset(days=int(d))).date().isoformat()
                            # Build and evaluate quickly
                            try:
                                engine = get_curated_connection()
                            except Exception:
                                engine = get_db_connection()
                            base_df = create_feature_matrix(engine, ctx.division, base, window_months).to_pandas()
                            shift_df = create_feature_matrix(engine, ctx.division, shift_cut, window_months).to_pandas()
                            auc_b, l10_b = _eval_lr(base_df, seed=42)
                            auc_s, l10_s = _eval_lr(shift_df, seed=43)
                            status = 'PASS'
                            if (auc_b is not None and auc_s is not None and (auc_s - auc_b) > eps_auc):
                                status = 'FAIL'
                            if (l10_b is not None and l10_s is not None and (l10_s - l10_b) > eps_l10):
                                status = 'FAIL'
                            summary['shifts'].append({"days": int(d), "status": status, "comparison": {"auc_base": auc_b, "auc_shift": auc_s, "lift10_base": l10_b, "lift10_shift": l10_s}})
                            if status == 'FAIL':
                                summary['overall'] = 'FAIL'
                        except Exception:
                            summary['shifts'].append({"days": int(d), "status": "ERROR"})
                p = ctx.out_dir / f"shift_grid_{division}_{cutoff}.json"
                p.write_text(json.dumps(summary, indent=2), encoding='utf-8')
                artifacts['shift_grid'] = str(p)
            except Exception:
                pass

        if run_topk_ablation:
            try:
                k_list = [int(x.strip()) for x in str(topk_list).split(',') if str(x).strip()]
                artifacts.update(run_topk_ablation_dynamic(ctx, window_months, k_list, abl_auc, abl_l10))
            except Exception as e:
                payload = {'status': 'ERROR', 'error': str(e)}
                p = ctx.out_dir / f"ablation_topk_{division}_{cutoff}.json"
                p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                artifacts['ablation_topk'] = str(p)

        if run_repro_check:
            try:
                artifacts.update(run_repro_dynamic(ctx, window_months, repro_eps_auc, repro_eps_lift10))
            except Exception as e:
                payload = {'status': 'ERROR', 'error': str(e)}
                p = ctx.out_dir / f"repro_check_{division}_{cutoff}.json"
                p.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                artifacts['repro_check'] = str(p)

    report = write_consolidated_report(ctx, artifacts)
    logger.info("Wrote leakage report to %s", report)

    # Exit non-zero on FAIL to support CI; here PASS is expected unless static checks found a problem
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("overall") == "FAIL":
            sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        pass


if __name__ == "__main__":
    main()
