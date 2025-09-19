from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from gosales.etl.sku_map import get_model_targets
from gosales.utils.grades import letter_grade_from_percentile

TARGET_DIVISIONS: Sequence[Tuple[str, str]] = [
    ("CPE", "cpe"),
    ("Solidworks", "solidworks"),
    ("Printers", "printers"),
]
GRADE_ORDER: Dict[str, int] = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1, "N/A": 0}
WHITESPACE_BASE_COLS: Sequence[str] = (
    "whitespace_score",
    "whitespace_score_pct",
    "whitespace_grade",
    "whitespace_p_icp",
    "whitespace_p_icp_pct",
    "whitespace_p_icp_grade",
    "whitespace_EV_norm",
    "whitespace_lift_norm",
    "whitespace_als_norm",
    "whitespace_reason",
)


def extract_division_slice(df: pd.DataFrame, division_name: str, suffix: str) -> pd.DataFrame:
    mask = df["division_name"].str.lower() == division_name.lower()
    subset = df.loc[
        mask,
        ["customer_id", "icp_score", "icp_percentile", "icp_grade", "bought_in_division"],
    ].copy()
    subset["customer_id"] = pd.to_numeric(subset["customer_id"], errors="coerce").astype("Int64")
    subset = subset.dropna(subset=["customer_id"]).groupby("customer_id", as_index=False).first()
    return subset.rename(
        columns={
            "icp_score": f"icp_score_{suffix}",
            "icp_percentile": f"icp_percentile_{suffix}",
            "icp_grade": f"icp_grade_{suffix}",
            "bought_in_division": f"bought_{suffix}",
        }
    )


def extract_whitespace_slice(df: pd.DataFrame, division_name: str, suffix: str) -> pd.DataFrame | None:
    mask = df["division_name"].str.lower() == division_name.lower()
    if not mask.any():
        return None
    subset = df.loc[
        mask,
        [
            "customer_id",
            "score",
            "score_pct",
            "score_grade",
            "p_icp",
            "p_icp_pct",
            "p_icp_grade",
            "EV_norm",
            "lift_norm",
            "als_norm",
            "nba_reason",
        ],
    ].copy()
    subset["customer_id"] = pd.to_numeric(subset["customer_id"], errors="coerce").astype("Int64")
    subset = subset.dropna(subset=["customer_id"]).groupby("customer_id", as_index=False).first()
    return subset.rename(
        columns={
            "score": f"whitespace_score_{suffix}",
            "score_pct": f"whitespace_score_pct_{suffix}",
            "score_grade": f"whitespace_grade_{suffix}",
            "p_icp": f"whitespace_p_icp_{suffix}",
            "p_icp_pct": f"whitespace_p_icp_pct_{suffix}",
            "p_icp_grade": f"whitespace_p_icp_grade_{suffix}",
            "EV_norm": f"whitespace_EV_norm_{suffix}",
            "lift_norm": f"whitespace_lift_norm_{suffix}",
            "als_norm": f"whitespace_als_norm_{suffix}",
            "nba_reason": f"whitespace_reason_{suffix}",
        }
    )


def load_cpe_icp(path: Path) -> pd.DataFrame:
    cols = ["customer_id", "customer_name", "division_name", "icp_score", "bought_in_division"]
    df = pd.read_csv(path, usecols=cols)
    mask = df["division_name"].str.lower() == "cpe"
    df = df.loc[mask].copy()
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["customer_id"]).groupby("customer_id", as_index=False).first()
    df["icp_percentile_cpe"] = df["icp_score"].rank(method="average", pct=True)
    df["icp_grade_cpe"] = df["icp_percentile_cpe"].map(letter_grade_from_percentile)
    return df.rename(
        columns={
            "icp_score": "icp_score_cpe",
            "bought_in_division": "bought_cpe",
            "customer_name": "customer_name_cpe",
        }
    )


def load_division_coverage(db_path: Path) -> Tuple[pd.DataFrame, str, str]:
    conn = sqlite3.connect(db_path)
    try:
        max_row = pd.read_sql_query("SELECT MAX(order_date) AS max_dt FROM fact_transactions", conn)
        max_date = max_row["max_dt"].iloc[0]
        if pd.isna(max_date):
            raise RuntimeError("fact_transactions has no data.")
        max_ts = pd.to_datetime(max_date)
        cutoff_ts = max_ts - pd.DateOffset(months=12)
        cutoff_str = cutoff_ts.strftime("%Y-%m-%d %H:%M:%S")
        printer_skus = tuple(get_model_targets("Printers"))
        placeholders = ",".join(["?"] * len(printer_skus))
        query = f"""
            SELECT
                customer_id,
                MAX(CASE WHEN product_division = 'Solidworks' THEN 1 ELSE 0 END) AS has_solidworks,
                MAX(CASE WHEN product_division = 'CPE' THEN 1 ELSE 0 END) AS has_cpe,
                MAX(
                    CASE
                        WHEN product_division = 'Hardware' AND product_sku IN ({placeholders})
                        THEN 1 ELSE 0
                    END
                ) AS has_printers,
                SUM(CASE WHEN order_date >= ? THEN gross_profit ELSE 0 END) AS gp_12m,
                SUM(gross_profit) AS gp_all_time,
                COUNT(DISTINCT CASE WHEN order_date >= ? THEN invoice_id END) AS invoices_12m
            FROM fact_transactions
            GROUP BY customer_id
        """
        params = [*printer_skus, cutoff_str, cutoff_str]
        coverage = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    coverage["customer_id"] = pd.to_numeric(coverage["customer_id"], errors="coerce").astype("Int64")
    return coverage, cutoff_str, max_ts.strftime("%Y-%m-%d")


def compute_enterprise_threshold(df: pd.DataFrame, target_count: int) -> float:
    gp = df.loc[df["gp_12m"] > 0, "gp_12m"]
    if gp.empty:
        return 0.0
    for quantile in (0.90, 0.75, 0.50, 0.25, 0.0):
        threshold = float(gp.quantile(quantile))
        if (df["gp_12m"] >= threshold).sum() >= target_count or quantile == 0.0:
            return threshold
    return float(gp.quantile(0.0))


def load_whitespace_map(outputs_dir: Path) -> Dict[str, pd.DataFrame]:
    ws_path = outputs_dir / "whitespace.csv"
    if not ws_path.exists():
        return {}
    ws_df = pd.read_csv(ws_path)
    ws_df["customer_id"] = pd.to_numeric(ws_df["customer_id"], errors="coerce").astype("Int64")
    ws_df = ws_df.dropna(subset=["customer_id"])
    ws_df["division_name"] = ws_df["division_name"].astype(str).str.strip()
    slices: Dict[str, pd.DataFrame] = {}
    for division_name, suffix in TARGET_DIVISIONS:
        slice_df = extract_whitespace_slice(ws_df, division_name, suffix)
        if slice_df is not None:
            slices[suffix] = slice_df
    return slices


def prepare_customer_base(outputs_dir: Path, db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, str, str]:
    icp_path = outputs_dir / "icp_scores.csv"
    cpe_icp_path = outputs_dir / "icp_scores-ReidPC.csv"

    icp_cols = [
        "customer_id",
        "division_name",
        "icp_score",
        "icp_percentile",
        "icp_grade",
        "customer_name",
        "bought_in_division",
        "rfm__all__gp_sum__12m",
        "total_gp_all_time",
        "total_transactions_all_time",
    ]
    icp_df = pd.read_csv(icp_path, usecols=icp_cols)
    icp_df["customer_id"] = pd.to_numeric(icp_df["customer_id"], errors="coerce").astype("Int64")
    icp_df = icp_df.dropna(subset=["customer_id"])

    meta = (
        icp_df[
            [
                "customer_id",
                "customer_name",
                "rfm__all__gp_sum__12m",
                "total_gp_all_time",
                "total_transactions_all_time",
            ]
        ]
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )

    solidworks_slice = extract_division_slice(icp_df, "Solidworks", "solidworks")
    printers_slice = extract_division_slice(icp_df, "Printers", "printers")
    cpe_icp = load_cpe_icp(cpe_icp_path)
    coverage, cutoff_12m, max_order_date = load_division_coverage(db_path)
    whitespace_map = load_whitespace_map(outputs_dir)

    ids = pd.Index(meta["customer_id"].dropna().astype("Int64"))
    ids = ids.union(pd.Index(cpe_icp["customer_id"].dropna().astype("Int64")))
    ids = ids.union(pd.Index(coverage["customer_id"].dropna().astype("Int64")))
    customers = pd.DataFrame({"customer_id": ids})

    customers = customers.merge(meta, on="customer_id", how="left")
    customers = customers.merge(cpe_icp, on="customer_id", how="left")
    if "customer_name_cpe" in customers.columns:
        customers["customer_name"] = customers["customer_name"].fillna(customers["customer_name_cpe"])
        customers = customers.drop(columns=["customer_name_cpe"], errors="ignore")
    customers = customers.merge(solidworks_slice, on="customer_id", how="left")
    customers = customers.merge(printers_slice, on="customer_id", how="left")
    customers = customers.merge(coverage, on="customer_id", how="left")

    for division_name, suffix in TARGET_DIVISIONS:
        slice_df = whitespace_map.get(suffix)
        if slice_df is not None:
            customers = customers.merge(slice_df, on="customer_id", how="left")
        else:
            for base in WHITESPACE_BASE_COLS:
                col = f"{base}_{suffix}"
                customers[col] = pd.NA

    customers["customer_name"] = customers["customer_name"].fillna("Unknown Account")
    customers["rfm__all__gp_sum__12m"] = pd.to_numeric(
        customers["rfm__all__gp_sum__12m"], errors="coerce"
    ).fillna(0.0)
    customers["total_gp_all_time"] = pd.to_numeric(
        customers["total_gp_all_time"], errors="coerce"
    ).fillna(0.0)
    customers["total_transactions_all_time"] = pd.to_numeric(
        customers["total_transactions_all_time"], errors="coerce"
    ).fillna(0).astype(int)

    for col in ["gp_12m", "gp_all_time"]:
        customers[col] = pd.to_numeric(customers[col], errors="coerce").fillna(0.0)
    customers["invoices_12m"] = pd.to_numeric(customers["invoices_12m"], errors="coerce").fillna(0).astype(int)

    for _, suffix in TARGET_DIVISIONS:
        has_col = f"has_{suffix}"
        if has_col not in customers.columns:
            customers[has_col] = 0
        customers[has_col] = pd.to_numeric(customers[has_col], errors="coerce").fillna(0).astype(int)
        score_col = f"icp_score_{suffix}"
        if score_col not in customers.columns:
            customers[score_col] = 0.0
        customers[score_col] = pd.to_numeric(customers[score_col], errors="coerce").fillna(0.0)
        grade_col = f"icp_grade_{suffix}"
        if grade_col not in customers.columns:
            customers[grade_col] = "N/A"
        customers[grade_col] = customers[grade_col].fillna("N/A")
        pct_col = f"icp_percentile_{suffix}"
        if pct_col in customers.columns:
            customers[pct_col] = pd.to_numeric(customers[pct_col], errors="coerce").fillna(0.0)

    division_meta = [
        ("CPE", "has_cpe", "icp_score_cpe", "icp_grade_cpe"),
        ("Solidworks", "has_solidworks", "icp_score_solidworks", "icp_grade_solidworks"),
        ("Printers", "has_printers", "icp_score_printers", "icp_grade_printers"),
    ]

    customers["current_divisions"] = customers.apply(
        lambda row: ", ".join([name for name, has_col, _, _ in division_meta if row.get(has_col, 0) >= 1]) or "None",
        axis=1,
    )
    customers["missing_divisions"] = customers.apply(
        lambda row: ", ".join([name for name, has_col, _, _ in division_meta if row.get(has_col, 0) < 1]) or "None",
        axis=1,
    )

    def build_recommendations(row: pd.Series) -> str:
        pieces: List[str] = []
        for name, has_col, score_col, grade_col in sorted(
            division_meta,
            key=lambda meta: row.get(meta[2], 0.0),
            reverse=True,
        ):
            if row.get(has_col, 0) >= 1:
                continue
            score = float(row.get(score_col, 0.0))
            grade = row.get(grade_col, "N/A")
            pieces.append(f"{name} (ICP {score:.3f}, grade {grade})")
        return "; ".join(pieces)

    customers["top_recommendations"] = customers.apply(build_recommendations, axis=1)
    customers["top_recommendations"] = customers["top_recommendations"].replace("", "Complete coverage")
    customers["missing_divisions"] = customers["missing_divisions"].replace("", "None")
    customers["own_count"] = customers[["has_cpe", "has_solidworks", "has_printers"]].sum(axis=1)

    eligible = customers[customers["own_count"] >= 1].copy()
    if eligible.empty:
        raise RuntimeError("No customers with transactions in the target divisions were found.")

    threshold = compute_enterprise_threshold(eligible, target_count=150)
    eligible["enterprise_flag"] = eligible["gp_12m"] >= threshold
    enterprise = eligible[eligible["enterprise_flag"]].copy()
    if enterprise.empty:
        threshold = 0.0
        enterprise = eligible.copy()

    return customers, eligible, enterprise, threshold, cutoff_12m, max_order_date


def assign_group_labels(df: pd.DataFrame) -> pd.DataFrame:
    def assign_group(row: pd.Series) -> Tuple[str, str]:
        if row["own_count"] >= 3:
            return "A", "A. Owns all three today"
        if row["own_count"] == 2:
            return "B", f"B. Owns two (missing {row['missing_divisions']})"
        return "C", f"C. Owns one (expand into {row['missing_divisions']})"

    group_df = df.apply(assign_group, axis=1, result_type="expand")
    df["group_key"] = group_df[0]
    df["group_label"] = group_df[1]
    group_priority: Dict[str, int] = {"B": 1, "C": 2, "A": 3}
    df["group_priority"] = df["group_key"].map(group_priority).fillna(4)
    return df


def build_icp_top100(enterprise: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = assign_group_labels(enterprise.copy())
    df["opportunity_score"] = df.apply(
        lambda row: sum(
            float(row.get(f"icp_score_{suffix}", 0.0))
            for _, suffix in TARGET_DIVISIONS
            if row.get(f"has_{suffix}", 0) < 1
        ),
        axis=1,
    )
    df = df.sort_values(
        ["group_priority", "opportunity_score", "gp_12m"], ascending=[True, False, False]
    )
    df["group_rank"] = df.groupby("group_key").cumcount() + 1

    top100 = df.head(100).copy()
    top100["overall_rank"] = range(1, len(top100) + 1)

    for col in [c for c in top100.columns if c.startswith("icp_score")]:
        top100[col] = pd.to_numeric(top100[col], errors="coerce").fillna(0.0).round(4)
    for col in [c for c in top100.columns if c.startswith("icp_percentile")]:
        top100[col] = pd.to_numeric(top100[col], errors="coerce").fillna(0.0).round(4)
    top100["opportunity_score"] = pd.to_numeric(top100["opportunity_score"], errors="coerce").fillna(0.0).round(4)
    top100["gp_12m"] = pd.to_numeric(top100["gp_12m"], errors="coerce").fillna(0.0).round(2)
    top100["gp_all_time"] = pd.to_numeric(top100["gp_all_time"], errors="coerce").fillna(0.0).round(2)
    top100["invoices_12m"] = pd.to_numeric(top100["invoices_12m"], errors="coerce").fillna(0).astype(int)
    top100["total_transactions_all_time"] = (
        pd.to_numeric(top100["total_transactions_all_time"], errors="coerce").fillna(0).astype(int)
    )
    for col in [c for c in top100.columns if c.startswith("icp_grade")]:
        top100[col] = top100[col].fillna("N/A")

    top100 = top100.rename(
        columns={
            "gp_12m": "gross_profit_12m",
            "gp_all_time": "gross_profit_lifetime",
            "total_transactions_all_time": "transactions_lifetime",
        }
    )

    ordered_cols = [
        "overall_rank",
        "group_key",
        "group_rank",
        "group_label",
        "customer_id",
        "customer_name",
        "current_divisions",
        "missing_divisions",
        "top_recommendations",
        "icp_score_cpe",
        "icp_percentile_cpe",
        "icp_grade_cpe",
        "icp_score_solidworks",
        "icp_percentile_solidworks",
        "icp_grade_solidworks",
        "icp_score_printers",
        "icp_percentile_printers",
        "icp_grade_printers",
        "opportunity_score",
        "gross_profit_12m",
        "gross_profit_lifetime",
        "transactions_lifetime",
        "invoices_12m",
    ]
    top100 = top100[ordered_cols]

    group_counts = top100["group_key"].value_counts().to_dict()
    return top100, group_counts


def build_whitespace_top100(enterprise: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    df = assign_group_labels(enterprise.copy())

    def collect_opportunities(row: pd.Series) -> List[Dict[str, object]]:
        opportunities: List[Dict[str, object]] = []
        for division_name, suffix in TARGET_DIVISIONS:
            if row.get(f"has_{suffix}", 0) >= 1:
                continue
            score = row.get(f"whitespace_score_{suffix}")
            grade = row.get(f"whitespace_grade_{suffix}")
            reason = row.get(f"whitespace_reason_{suffix}")
            p_icp = row.get(f"whitespace_p_icp_{suffix}")
            p_pct = row.get(f"whitespace_p_icp_pct_{suffix}")
            ev_norm = row.get(f"whitespace_EV_norm_{suffix}")
            lift_norm = row.get(f"whitespace_lift_norm_{suffix}")
            als_norm = row.get(f"whitespace_als_norm_{suffix}")
            source = "whitespace"
            if pd.isna(score):
                score = row.get(f"icp_score_{suffix}", 0.0)
                grade = row.get(f"icp_grade_{suffix}", "N/A")
                p_icp = row.get(f"icp_score_{suffix}", 0.0)
                p_pct = row.get(f"icp_percentile_{suffix}", 0.0)
                ev_norm = pd.NA
                lift_norm = pd.NA
                als_norm = pd.NA
                reason = "ICP fallback (whitespace unavailable)"
                source = "ICP fallback"
            opportunities.append(
                {
                    "division": division_name,
                    "suffix": suffix,
                    "score": float(score) if pd.notna(score) else 0.0,
                    "grade": str(grade) if pd.notna(grade) else "N/A",
                    "reason": reason if isinstance(reason, str) else "",
                    "source": source,
                    "p_icp": float(p_icp) if pd.notna(p_icp) else 0.0,
                    "p_pct": float(p_pct) if pd.notna(p_pct) else 0.0,
                    "ev": float(ev_norm) if pd.notna(ev_norm) else 0.0,
                    "lift": float(lift_norm) if pd.notna(lift_norm) else 0.0,
                    "als": float(als_norm) if pd.notna(als_norm) else 0.0,
                }
            )
        return opportunities

    df["_ws_records"] = df.apply(collect_opportunities, axis=1)
    df["whitespace_score_total"] = df["_ws_records"].apply(lambda recs: sum(r["score"] for r in recs))

    def best_record(recs: List[Dict[str, object]]) -> Dict[str, object]:
        if not recs:
            return {
                "division": "None",
                "score": 0.0,
                "grade": "N/A",
                "reason": "Complete coverage",
                "source": "None",
                "p_icp": 0.0,
                "p_pct": 0.0,
                "ev": 0.0,
                "lift": 0.0,
                "als": 0.0,
            }
        return max(
            recs,
            key=lambda r: (
                r["score"],
                GRADE_ORDER.get(str(r.get("grade", "N/A")), 0),
            ),
        )

    df["_best_ws"] = df["_ws_records"].apply(best_record)
    df["whitespace_best_division"] = df["_best_ws"].apply(lambda r: r["division"])
    df["whitespace_best_score"] = df["_best_ws"].apply(lambda r: float(r["score"]))
    df["whitespace_best_grade"] = df["_best_ws"].apply(lambda r: str(r["grade"]))
    df["whitespace_best_reason"] = df["_best_ws"].apply(lambda r: str(r["reason"]))
    df["whitespace_best_source"] = df["_best_ws"].apply(lambda r: str(r["source"]))
    df["whitespace_best_p_icp"] = df["_best_ws"].apply(lambda r: float(r["p_icp"]))
    df["whitespace_best_p_icp_pct"] = df["_best_ws"].apply(lambda r: float(r["p_pct"]))
    df["whitespace_best_EV_norm"] = df["_best_ws"].apply(lambda r: float(r["ev"]))
    df["whitespace_best_lift_norm"] = df["_best_ws"].apply(lambda r: float(r["lift"]))
    df["whitespace_best_als_norm"] = df["_best_ws"].apply(lambda r: float(r["als"]))
    df["whitespace_fallback_count"] = df["_ws_records"].apply(
        lambda recs: sum(1 for r in recs if r.get("source") != "whitespace")
    )

    def summarize(recs: List[Dict[str, object]]) -> str:
        if not recs:
            return "Complete coverage"
        ordered = sorted(
            recs,
            key=lambda r: (
                r["score"],
                GRADE_ORDER.get(str(r.get("grade", "N/A")), 0),
            ),
            reverse=True,
        )
        parts = []
        for r in ordered[:3]:
            reason = r["reason"] or ("Whitespace model" if r["source"] == "whitespace" else "ICP fallback")
            parts.append(
                f"{r['division']} (score {r['score']:.3f}, grade {r['grade']}, {reason})"
            )
        return "; ".join(parts)

    df["whitespace_opportunity_summary"] = df["_ws_records"].apply(summarize)

    df = df.sort_values(
        [
            "group_priority",
            "whitespace_score_total",
            "whitespace_best_score",
            "gp_12m",
        ],
        ascending=[True, False, False, False],
    )
    df["group_rank"] = df.groupby("group_key").cumcount() + 1

    top100 = df.head(100).copy()
    top100["overall_rank"] = range(1, len(top100) + 1)

    for col in [
        "whitespace_score_total",
        "whitespace_best_score",
        "whitespace_best_p_icp",
        "whitespace_best_p_icp_pct",
        "whitespace_best_EV_norm",
        "whitespace_best_lift_norm",
        "whitespace_best_als_norm",
    ]:
        top100[col] = pd.to_numeric(top100[col], errors="coerce").fillna(0.0).round(4)
    top100["gp_12m"] = pd.to_numeric(top100["gp_12m"], errors="coerce").fillna(0.0).round(2)
    top100["gp_all_time"] = pd.to_numeric(top100["gp_all_time"], errors="coerce").fillna(0.0).round(2)
    top100["invoices_12m"] = pd.to_numeric(top100["invoices_12m"], errors="coerce").fillna(0).astype(int)
    top100["total_transactions_all_time"] = (
        pd.to_numeric(top100["total_transactions_all_time"], errors="coerce").fillna(0).astype(int)
    )

    top100 = top100.rename(
        columns={
            "gp_12m": "gross_profit_12m",
            "gp_all_time": "gross_profit_lifetime",
            "total_transactions_all_time": "transactions_lifetime",
        }
    )

    ordered_cols = [
        "overall_rank",
        "group_key",
        "group_rank",
        "group_label",
        "customer_id",
        "customer_name",
        "current_divisions",
        "missing_divisions",
        "whitespace_best_division",
        "whitespace_best_score",
        "whitespace_best_grade",
        "whitespace_best_reason",
        "whitespace_best_source",
        "whitespace_best_p_icp",
        "whitespace_best_p_icp_pct",
        "whitespace_best_EV_norm",
        "whitespace_best_lift_norm",
        "whitespace_best_als_norm",
        "whitespace_score_total",
        "whitespace_opportunity_summary",
        "icp_score_cpe",
        "icp_grade_cpe",
        "icp_score_solidworks",
        "icp_grade_solidworks",
        "icp_score_printers",
        "icp_grade_printers",
        "gross_profit_12m",
        "gross_profit_lifetime",
        "transactions_lifetime",
        "invoices_12m",
    ]
    top100 = top100[ordered_cols]

    group_counts = top100["group_key"].value_counts().to_dict()
    fallback_usage = int((top100["whitespace_best_source"] == "ICP fallback").sum())
    return top100, group_counts, fallback_usage


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs_dir = root / "gosales" / "outputs"
    db_path = root / "gosales" / "gosales_curated.db"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    customers, eligible, enterprise, threshold, cutoff_12m, max_order_date = prepare_customer_base(
        outputs_dir, db_path
    )

    icp_top100, icp_group_counts = build_icp_top100(enterprise)
    ws_top100, ws_group_counts, ws_fallback_usage = build_whitespace_top100(enterprise)

    legacy_csv = reports_dir / "gosales_cross_sell_top100_accounts.csv"
    icp_csv = reports_dir / "gosales_cross_sell_top100_accounts_icp.csv"
    ws_csv = reports_dir / "gosales_cross_sell_top100_accounts_whitespace.csv"
    summary_path = reports_dir / "gosales_cross_sell_top100_summary.md"

    icp_top100.to_csv(icp_csv, index=False)
    icp_top100.to_csv(legacy_csv, index=False)
    ws_top100.to_csv(ws_csv, index=False)

    enterprise_counts = enterprise["own_count"].value_counts()
    enterprise_group_counts = assign_group_labels(enterprise.copy())["group_key"].value_counts().to_dict()

    summary_lines = [
        "# Cross-Sell Target Accounts (Top 100)",
        f"- Latest invoice date: {max_order_date}",
        f"- 12-month window starts: {cutoff_12m.split(' ')[0]}",
        f"- Enterprise threshold (12-month gross profit): ${threshold:,.2f}",
        f"- Enterprise accounts meeting threshold: {len(enterprise)} of {len(eligible)} relevant customers",
        "",
        "## ICP-ranked list",
        f"- Output: {icp_csv.name}",
        f"- Top-100 group mix (ICP): "
        + ", ".join([f"{key}: {icp_group_counts.get(key, 0)}" for key in ("A", "B", "C")]),
        "",
        "## Whitespace-ranked list",
        f"- Output: {ws_csv.name}",
        f"- Top-100 group mix (Whitespace): "
        + ", ".join([f"{key}: {ws_group_counts.get(key, 0)}" for key in ("A", "B", "C")]),
        f"- Accounts using ICP fallback (no whitespace coverage): {ws_fallback_usage}",
        "",
        "## Additional context",
        "- Enterprise group counts (all qualifying accounts): "
        + ", ".join([f"{key}: {enterprise_group_counts.get(key, 0)}" for key in ("A", "B", "C")]),
        "- Ownership distribution (count of divisions currently owned): "
        + ", ".join([
            f"{int(idx)} divisions: {cnt}"
            for idx, cnt in sorted(enterprise_counts.items(), key=lambda x: int(x[0]))
        ]),
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote {len(icp_top100)} accounts to {icp_csv}")
    print(f"Wrote {len(ws_top100)} accounts to {ws_csv}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
