import pandas as pd
import sqlite3
con = sqlite3.connect('gosales/gosales_curated.db')
q1 = "SELECT CompanyId AS customer_id FROM fact_sales_line WHERE Rec_Date > '2025-03-31' AND Rec_Date <= '2025-09-30' AND LOWER(TRIM(item_rollup))='draftsight' LIMIT 20"
rl = pd.read_sql_query(q1, con)
print('line buyers sample n=',len(rl))
print(rl.head().to_dict(orient='records'))
q2 = "SELECT customer_id FROM dim_customer LIMIT 5"
rc = pd.read_sql_query(q2, con)
print('dim sample', rc.head().to_dict(orient='records'))
con.close()
