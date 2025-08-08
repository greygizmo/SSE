from __future__ import annotations

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


@dataclass
class Logging:
    level: str = "INFO"
    jsonl: bool = True


@dataclass
class Labels:
    gp_min_threshold: float = 0.0
    denylist_skus_csv: Optional[Path] = None


@dataclass
class Config:
    paths: Paths
    database: Database = field(default_factory=Database)
    run: Run = field(default_factory=Run)
    etl: ETL = field(default_factory=ETL)
    logging: Logging = field(default_factory=Logging)
    labels: Labels = field(default_factory=Labels)

    def to_dict(self) -> Dict[str, Any]:
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (Paths, Database, Run, ETL, Logging)):
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
    """Load configuration from YAML with optional overrides.

    Precedence: YAML -> environment -> CLI overrides.
    """
    if config_path is None:
        config_path = ROOT_DIR / "config.yaml"

    cfg_dict = _load_yaml(Path(config_path)) if Path(config_path).exists() else {}

    # Environment overrides (flat small set for now)
    env_db_engine = os.getenv("GOSALES_DB_ENGINE")
    env_sqlite_path = os.getenv("GOSALES_SQLITE_PATH")
    if env_db_engine:
        cfg_dict.setdefault("database", {})["engine"] = env_db_engine
    if env_sqlite_path:
        cfg_dict.setdefault("database", {})["sqlite_path"] = env_sqlite_path

    # CLI overrides (nested)
    cfg_dict = _merge_overrides(cfg_dict, cli_overrides)

    # Default paths if missing
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

    cfg = Config(
        paths=_paths_from_dict(paths_dict),
        database=Database(
            engine=str(database.get("engine", "sqlite")),
            sqlite_path=Path(database.get("sqlite_path", ROOT_DIR.parent / "gosales.db")).resolve(),
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
        ),
        logging=Logging(
            level=str(log_cfg.get("level", "INFO")),
            jsonl=bool(log_cfg.get("jsonl", True)),
        ),
        labels=Labels(
            gp_min_threshold=float(labels_cfg.get("gp_min_threshold", 0.0)),
            denylist_skus_csv=(Path(labels_cfg["denylist_skus_csv"]).resolve() if labels_cfg.get("denylist_skus_csv") else None),
        ),
    )

    # Ensure directories exist (non-destructive)
    for p in [cfg.paths.raw, cfg.paths.staging, cfg.paths.curated, cfg.paths.outputs]:
        Path(p).mkdir(parents=True, exist_ok=True)

    return cfg


