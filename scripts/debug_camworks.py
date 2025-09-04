import pandas as pd
import sqlite3
from pathlib import Path

curated = Path('gosales/gosales_curated.db')
print('Curated DB:', curated.resolve(), 'exists=', curated.exists())
if not curated.exists():
    raise SystemExit('Curated DB missing')
con = sqlite3.connect(str(curated))

def q(name, sql):
    try:
        df = pd.read_sql(sql, con)
        print(f"\n[{name}]\n", df.head(50).to_string(index=False))
    except Exception as e:
        print('Query failed', name, e)

q('divisions', "SELECT product_division, COUNT(*) n FROM fact_transactions GROUP BY product_division ORDER BY n DESC LIMIT 20")
q('camworks_gp_qty', "SELECT SUM(gross_profit) gp, SUM(quantity) qty FROM fact_transactions WHERE product_sku='CAMWorks'")
q('camworks_rows', "SELECT COUNT(*) AS n FROM fact_transactions WHERE product_division='CAMWorks'")
q('cam_like', "SELECT product_sku, product_division, SUM(quantity) qty, SUM(gross_profit) gp FROM fact_transactions WHERE product_sku LIKE '%CAM%' OR product_division LIKE '%CAM%' GROUP BY product_sku, product_division ORDER BY gp DESC LIMIT 50")

con.close()

