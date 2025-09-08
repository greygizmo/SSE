from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from gosales.utils.paths import ROOT_DIR


@dataclass
class Paths:
    raw: Path
    staging: Path
    curated: Path
    outputs: Path


@dataclass
class Database:
    engine: str = "sqlite"  # sqlite | duckdb | azure
    sqlite_path: Path = ROOT_DIR.parent / "gosales.db"
    curated_target: str = "db"  # 'db' | 'sqlite'
    curated_sqlite_path: Path = ROOT_DIR.parent / "gosales_curated.db"
    # Enforce external DB presence; if True and AZSQL_* env vars missing/unhealthy, fail instead of falling back
    strict_db: bool = False
    # Optional mapping of logical table names -> concrete source (e.g., "dbo.saleslog" or "csv")
    source_tables: Dict[str, str] = field(default_factory=dict)
    # Optional explicit allow-list of schema-qualified DB objects permitted in dynamic SQL
    allowed_identifiers: list[str] = field(default_factory=list)


@dataclass
class Run:
    cutoff_date: str = "2024-12-31"
    prediction_window_months: int = 6
    lookback_years: int = 3


@dataclass
class ETL:
    coerce_dates_tz: str = "UTC"
    currency: str = "USD"
    fail_on_contract_breach: bool = True
    allow_unknown_columns: bool = False
    # Industry enrichment fuzzy-match controls
    enable_industry_fuzzy: bool = True
    fuzzy_min_unmatched: int = 50
    fuzzy_skip_if_coverage_ge: float = 0.95
    # Source column mapping: use exact DB headers
    source_columns: Dict[str, str] = field(default_factory=dict)


@dataclass
class Logging:
    level: str = "INFO"
    jsonl: bool = True


@dataclass
class Labels:
    gp_min_threshold: float = 0.0
    denylist_skus_csv: Optional[Path] = None
    # Per-division window overrides for sparse groups
    per_division_window_months: Dict[str, int] = field(default_factory=dict)
    # Sparse division label widening targets
    sparse_min_positive_target: Optional[int] = None
    sparse_max_window_months: int = 12


@dataclass
class Features:
    windows_months: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    gp_winsor_p: float = 0.99
    add_missingness_flags: bool = True
    use_eb_smoothing: bool = True
    use_market_basket: bool = True
    use_als_embeddings: bool = False
    als_lookback_months: int = 12
    use_item2vec: bool = False
    use_text_tags: bool = False
    # Toggle Moneyball-based asset features at cutoff (rollups, expiring windows, subs shares)
    use_assets: bool = True
    # Guard days for look-ahead expiration windows; exclude [cutoff, cutoff+guard]
    expiring_guard_days: int = 14
    # Floor for recency features to avoid near-cutoff signals (e.g., 14 days)
    recency_floor_days: int = 0


@dataclass
class ModelingConfig:
    seed: int = 42
    folds: int = 3
    models: list[str] = field(default_factory=lambda: ["logreg", "lgbm"])
    lr_grid: Dict[str, Any] = field(default_factory=lambda: {"l1_ratio": [0.0, 0.2, 0.5], "C": [0.1, 1.0, 10.0]})
    lgbm_grid: Dict[str, Any] = field(default_factory=lambda: {"num_leaves": [31, 63], "min_data_in_leaf": [50, 100], "learning_rate": [0.05, 0.1], "feature_fraction": [0.7, 0.9], "bagging_fraction": [0.7, 0.9]})
    calibration_methods: list[str] = field(default_factory=lambda: ["platt", "isotonic"])
    top_k_percents: list[int] = field(default_factory=lambda: [5, 10, 20])
    capacity_percent: int = 10
    # Threshold of positives above which to prefer isotonic; otherwise sigmoid
    sparse_isotonic_threshold_pos: int = 1000
    # Max rows allowed for SHAP computation; skip if exceeded
    shap_max_rows: int = 50000
    # Class imbalance controls
    class_weight: str = "balanced"  # 'balanced' or 'none'
    use_scale_pos_weight: bool = True
    scale_pos_weight_cap: float = 10.0


@dataclass
class WhitespaceEligibilityConfig:
    exclude_if_owned_ever: bool = True
    exclude_if_recent_contact_days: int = 0
    exclude_if_open_deal: bool = False
    require_region_match: bool = False


@dataclass
class WhitespaceConfig:
    weights: list[float] = field(default_factory=lambda: [0.60, 0.20, 0.10, 0.10])
    normalize: str = "percentile"
    eligibility: WhitespaceEligibilityConfig = field(default_factory=WhitespaceEligibilityConfig)
    capacity_mode: str = "top_percent"
    accounts_per_rep: int = 25
    ev_cap_percentile: float = 0.95
    als_coverage_threshold: float = 0.30
    bias_division_max_share_topN: float = 0.6
    cooldown_days: int = 30
    cooldown_factor: float = 0.75
    # Optional challenger meta-learner over [p_icp_pct, lift_norm, als_norm, EV_norm]
    challenger_enabled: bool = False
    challenger_model: str = "lr"  # currently only 'lr'
    # Shadow mode emits legacy heuristic whitespace for comparison
    shadow_mode: bool = False


@dataclass
class ValidationConfig:
    bootstrap_n: int = 1000
    top_k_percents: list[int] = field(default_factory=lambda: [5, 10, 20])
    capacity_grid: list[int] = field(default_factory=lambda: [5, 10, 20])
    ev_cap_percentile: float = 0.95
    segment_columns: list[str] = field(default_factory=lambda: ["industry", "industry_sub", "region", "territory"])
    ks_threshold: float = 0.15
    psi_threshold: float = 0.25
    cal_mae_threshold: float = 0.03
    # Leakage Gauntlet thresholds for shift-14 check
    shift14_epsilon_auc: float = 0.01
    shift14_epsilon_lift10: float = 0.25
    # Leakage Gauntlet thresholds for Top-K ablation
    ablation_epsilon_auc: float = 0.01
    ablation_epsilon_lift10: float = 0.25
    # Gauntlet-only masking: exclude last N days inside windowed aggregations
    gauntlet_mask_tail_days: int = 14
    # Gauntlet-only: purge/embargo days between train and validation
    gauntlet_purge_days: int = 30
    # Gauntlet-only: start labels at cutoff+buffer_days (horizon buffer)
    gauntlet_label_buffer_days: int = 0


@dataclass
class Config:
    paths: Paths
    database: Database = field(default_factory=Database)
    run: Run = field(default_factory=Run)
    etl: ETL = field(default_factory=ETL)
    logging: Logging = field(default_factory=Logging)
    labels: Labels = field(default_factory=Labels)
    features: Features = field(default_factory=Features)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    whitespace: WhitespaceConfig = field(default_factory=WhitespaceConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> Dict[str, Any]:
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, '__dataclass_fields__'):
                d = asdict(obj)
                return {k: _convert(v) for k, v in d.items()}
            return obj

        return {
            "paths": _convert(self.paths),
            "database": _convert(self.database),
            "run": _convert(self.run),
            "etl": _convert(self.etl),
            "logging": _convert(self.logging),
            "labels": _convert(self.labels),
            "features": _convert(self.features),
            "modeling": _convert(self.modeling),
            "whitespace": _convert(self.whitespace),
            "validation": _convert(self.validation),
        }


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return base

    def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    return deep_merge(base, overrides)


def _paths_from_dict(d: Dict[str, Any]) -> Paths:
    return Paths(
        raw=Path(d["raw"]).resolve(),
        staging=Path(d["staging"]).resolve(),
        curated=Path(d["curated"]).resolve(),
        outputs=Path(d["outputs"]).resolve(),
    )


def load_config(config_path: Optional[str | Path] = None, cli_overrides: Optional[Dict[str, Any]] = None) -> Config:
    if config_path is None:
        config_path = ROOT_DIR / "config.yaml"

    cfg_path_obj = Path(config_path)
    cfg_dict = _load_yaml(cfg_path_obj) if cfg_path_obj.exists() else {}

    allowed_top = {"paths","database","run","etl","logging","labels","features","modeling","whitespace","validation"}
    unknown_top = set(cfg_dict.keys()) - allowed_top
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown_top)}. Allowed: {sorted(allowed_top)}")

    env_db_engine = os.getenv("GOSALES_DB_ENGINE")
    env_sqlite_path = os.getenv("GOSALES_SQLITE_PATH")
    env_use_assets = os.getenv("GOSALES_FEATURES_USE_ASSETS")
    env_exp_guard = os.getenv("GOSALES_FEATURES_EXPIRING_GUARD_DAYS")
    env_rec_floor = os.getenv("GOSALES_FEATURES_RECENCY_FLOOR_DAYS")
    if env_db_engine:
        cfg_dict.setdefault("database", {})["engine"] = env_db_engine
    if env_sqlite_path:
        cfg_dict.setdefault("database", {})["sqlite_path"] = env_sqlite_path
    if env_use_assets is not None:
        # Accept truthy strings: '1', 'true', 'yes'
        truthy = {"1", "true", "yes", "on"}
        val = str(env_use_assets).strip().lower() in truthy
        cfg_dict.setdefault("features", {})["use_assets"] = val
    if env_exp_guard is not None:
        try:
            cfg_dict.setdefault("features", {})["expiring_guard_days"] = int(env_exp_guard)
        except Exception:
            pass
    if env_rec_floor is not None:
        try:
            cfg_dict.setdefault("features", {})["recency_floor_days"] = int(env_rec_floor)
        except Exception:
            pass

    cfg_dict = _merge_overrides(cfg_dict, cli_overrides)

    paths_dict = cfg_dict.get("paths") or {
        "raw": str(ROOT_DIR / "data" / "raw"),
        "staging": str(ROOT_DIR / "data" / "staging"),
        "curated": str(ROOT_DIR / "data" / "curated"),
        "outputs": str(ROOT_DIR / "outputs"),
    }

    database = cfg_dict.get("database", {})
    run_cfg = cfg_dict.get("run", {})
    etl_cfg = cfg_dict.get("etl", {})
    log_cfg = cfg_dict.get("logging", {})
    labels_cfg = cfg_dict.get("labels", {})
    feat_cfg = cfg_dict.get("features", {})
    mdl_cfg = cfg_dict.get("modeling", {})
    ws_cfg = cfg_dict.get("whitespace", {})
    val_cfg = cfg_dict.get("validation", {})

    ws_eligibility_cfg = ws_cfg.get("eligibility", {})

    # Parse and validate whitespace weights
    raw_ws_weights = ws_cfg.get("weights", [0.60, 0.20, 0.10, 0.10])
    try:
        ws_weights = [float(w) for w in raw_ws_weights]
    except Exception as e:  # pragma: no cover - defensive
        raise ValueError("whitespace.weights must be a list of numbers") from e
    if len(ws_weights) != 4:
        raise ValueError("whitespace.weights must have 4 entries")
    if any((not math.isfinite(w)) or w < 0 for w in ws_weights):
        raise ValueError("whitespace.weights must be finite and non-negative")
    total_w = sum(ws_weights)
    if total_w <= 0:
        raise ValueError("whitespace.weights must sum to a positive number")
    ws_weights = [w / total_w for w in ws_weights]

    cfg = Config(
        paths=_paths_from_dict(paths_dict),
        database=Database(
            engine=str(database.get("engine", "sqlite")),
            sqlite_path=Path(database.get("sqlite_path", ROOT_DIR.parent / "gosales.db")).resolve(),
            curated_target=str(database.get("curated_target", "db")),
            curated_sqlite_path=Path(database.get("curated_sqlite_path", ROOT_DIR.parent / "gosales_curated.db")).resolve(),
            strict_db=bool(database.get("strict_db", False)),
            source_tables=dict(database.get("source_tables", {})),
            allowed_identifiers=list(database.get("allowed_identifiers", [])),
        ),
        run=Run(
            cutoff_date=str(run_cfg.get("cutoff_date", "2024-12-31")),
            prediction_window_months=int(run_cfg.get("prediction_window_months", 6)),
            lookback_years=int(run_cfg.get("lookback_years", 3)),
        ),
        etl=ETL(
            coerce_dates_tz=str(etl_cfg.get("coerce_dates_tz", "UTC")),
            currency=str(etl_cfg.get("currency", "USD")),
            fail_on_contract_breach=bool(etl_cfg.get("fail_on_contract_breach", True)),
            allow_unknown_columns=bool(etl_cfg.get("allow_unknown_columns", False)),
            enable_industry_fuzzy=bool(etl_cfg.get("enable_industry_fuzzy", True)),
            fuzzy_min_unmatched=int(etl_cfg.get("fuzzy_min_unmatched", 50)),
            fuzzy_skip_if_coverage_ge=float(etl_cfg.get("fuzzy_skip_if_coverage_ge", 0.95)),
            source_columns=dict(etl_cfg.get("source_columns", {})),
        ),
        logging=Logging(
            level=str(log_cfg.get("level", "INFO")),
            jsonl=bool(log_cfg.get("jsonl", True)),
        ),
        labels=Labels(
            gp_min_threshold=float(labels_cfg.get("gp_min_threshold", 0.0)),
            denylist_skus_csv=(Path(labels_cfg["denylist_skus_csv"]).resolve() if labels_cfg.get("denylist_skus_csv") else None),
            per_division_window_months=dict(labels_cfg.get("per_division_window_months", {})),
            sparse_min_positive_target=(int(labels_cfg.get("sparse_min_positive_target")) if labels_cfg.get("sparse_min_positive_target") is not None else None),
            sparse_max_window_months=int(labels_cfg.get("sparse_max_window_months", 12)),
        ),
        features=Features(
            windows_months=list(feat_cfg.get("windows_months", [3, 6, 12, 24])),
            gp_winsor_p=float(feat_cfg.get("gp_winsor_p", 0.99)),
            add_missingness_flags=bool(feat_cfg.get("add_missingness_flags", True)),
            use_eb_smoothing=bool(feat_cfg.get("use_eb_smoothing", True)),
            use_market_basket=bool(feat_cfg.get("use_market_basket", True)),
            use_als_embeddings=bool(feat_cfg.get("use_als_embeddings", False)),
            als_lookback_months=int(feat_cfg.get("als_lookback_months", 12)),
            use_item2vec=bool(feat_cfg.get("use_item2vec", False)),
            use_text_tags=bool(feat_cfg.get("use_text_tags", False)),
            use_assets=bool(feat_cfg.get("use_assets", True)),
            expiring_guard_days=int(feat_cfg.get("expiring_guard_days", 14)),
            recency_floor_days=int(feat_cfg.get("recency_floor_days", 0)),
        ),
        modeling=ModelingConfig(
            seed=int(mdl_cfg.get("seed", 42)),
            folds=int(mdl_cfg.get("folds", 3)),
            models=list(mdl_cfg.get("models", ["logreg", "lgbm"])),
            lr_grid=dict(mdl_cfg.get("lr_grid", {"l1_ratio": [0.0, 0.2, 0.5], "C": [0.1, 1.0, 10.0]})),
            lgbm_grid=dict(mdl_cfg.get("lgbm_grid", {"num_leaves": [31, 63], "min_data_in_leaf": [50, 100], "learning_rate": [0.05, 0.1], "feature_fraction": [0.7, 0.9], "bagging_fraction": [0.7, 0.9]})),
            calibration_methods=list(mdl_cfg.get("calibration_methods", ["platt", "isotonic"])),
            top_k_percents=list(mdl_cfg.get("top_k_percents", [5, 10, 20])),
            capacity_percent=int(mdl_cfg.get("capacity_percent", 10)),
            sparse_isotonic_threshold_pos=int(mdl_cfg.get("sparse_isotonic_threshold_pos", 1000)),
            class_weight=str(mdl_cfg.get("class_weight", "balanced")),
            use_scale_pos_weight=bool(mdl_cfg.get("use_scale_pos_weight", True)),
            scale_pos_weight_cap=float(mdl_cfg.get("scale_pos_weight_cap", 10.0)),
        ),
        whitespace=WhitespaceConfig(
            weights=ws_weights,
            normalize=str(ws_cfg.get("normalize", "percentile")),
            eligibility=WhitespaceEligibilityConfig(
                exclude_if_owned_ever=bool(ws_eligibility_cfg.get("exclude_if_owned_ever", True)),
                exclude_if_recent_contact_days=int(ws_eligibility_cfg.get("exclude_if_recent_contact_days", 0)),
                exclude_if_open_deal=bool(ws_eligibility_cfg.get("exclude_if_open_deal", False)),
                require_region_match=bool(ws_eligibility_cfg.get("require_region_match", False)),
            ),
            capacity_mode=str(ws_cfg.get("capacity_mode", "top_percent")),
            accounts_per_rep=int(ws_cfg.get("accounts_per_rep", 25)),
            ev_cap_percentile=float(ws_cfg.get("ev_cap_percentile", 0.95)),
            als_coverage_threshold=float(ws_cfg.get("als_coverage_threshold", 0.30)),
            bias_division_max_share_topN=float(ws_cfg.get("bias_division_max_share_topN", 0.6)),
            cooldown_days=int(ws_cfg.get("cooldown_days", 30)),
            cooldown_factor=float(ws_cfg.get("cooldown_factor", 0.75)),
            challenger_enabled=bool(ws_cfg.get("challenger_enabled", False)),
            challenger_model=str(ws_cfg.get("challenger_model", "lr")),
            shadow_mode=bool(ws_cfg.get("shadow_mode", False)),
        ),
        validation=ValidationConfig(
            bootstrap_n=int(val_cfg.get("bootstrap_n", 1000)),
            top_k_percents=list(val_cfg.get("top_k_percents", [5, 10, 20])),
            capacity_grid=list(val_cfg.get("capacity_grid", [5, 10, 20])),
            ev_cap_percentile=float(val_cfg.get("ev_cap_percentile", 0.95)),
            segment_columns=list(val_cfg.get("segment_columns", ["industry", "industry_sub", "region", "territory"])),
            ks_threshold=float(val_cfg.get("ks_threshold", 0.15)),
            psi_threshold=float(val_cfg.get("psi_threshold", 0.25)),
            cal_mae_threshold=float(val_cfg.get("cal_mae_threshold", 0.03)),
            shift14_epsilon_auc=float(val_cfg.get("shift14_epsilon_auc", 0.01)),
            shift14_epsilon_lift10=float(val_cfg.get("shift14_epsilon_lift10", 0.25)),
            ablation_epsilon_auc=float(val_cfg.get("ablation_epsilon_auc", 0.01)),
            ablation_epsilon_lift10=float(val_cfg.get("ablation_epsilon_lift10", 0.25)),
            gauntlet_mask_tail_days=int(val_cfg.get("gauntlet_mask_tail_days", 14)),
            gauntlet_purge_days=int(val_cfg.get("gauntlet_purge_days", 30)),
            gauntlet_label_buffer_days=int(val_cfg.get("gauntlet_label_buffer_days", 0)),
        ),
    )

    try:
        if str(cfg.whitespace.normalize).lower() not in {"percentile", "pooled"}:
            raise ValueError("whitespace.normalize must be 'percentile' or 'pooled'")
        if any(k <= 0 or k > 100 for k in cfg.modeling.top_k_percents):
            raise ValueError("modeling.top_k_percents must be integers in (0,100]")
        if not (0.0 < cfg.validation.ev_cap_percentile <= 1.0):
            raise ValueError("validation.ev_cap_percentile must be in (0,1]")
    except Exception:
        raise

    for p in [cfg.paths.raw, cfg.paths.staging, cfg.paths.curated, cfg.paths.outputs]:
        Path(p).mkdir(parents=True, exist_ok=True)

    return cfg
