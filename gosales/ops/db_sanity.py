"""Run ad-hoc SQL sanity checks against the curated warehouse."""

from __future__ import annotations

import pandas as pd
from gosales.utils.db import get_db_connection


def main() -> None:
    e = get_db_connection()
    print("Window counts by product_division (2024-07-01..2024-12-30):")
    q1 = (
        """
        SELECT product_division, COUNT(*) AS rows
        FROM fact_transactions
        WHERE order_date > '2024-06-30' AND order_date <= '2024-12-30'
        GROUP BY product_division
        ORDER BY rows DESC;
        """
    )
    print(pd.read_sql(q1, e).to_string(index=False))

    print("\nDistinct like AM/CPE/Post (raw values with lengths):")
    q2 = (
        """
        SELECT DISTINCT product_division, LENGTH(product_division) AS len
        FROM fact_transactions
        WHERE product_division IN ('AM_Software','CPE','Post_Processing')
           OR product_division LIKE '%Software%'
           OR product_division LIKE '%Post%'
           OR product_division LIKE '%CPE%';
        """
    )
    print(pd.read_sql(q2, e).to_string(index=False))

    print("\nDistinct TRIM(product_division) values (sample):")
    q3 = (
        """
        SELECT DISTINCT TRIM(product_division) AS trimmed, LENGTH(TRIM(product_division)) AS len
        FROM fact_transactions;
        """
    )
    df3 = pd.read_sql(q3, e)
    print(df3.head(50).to_string(index=False))


if __name__ == "__main__":
    main()


