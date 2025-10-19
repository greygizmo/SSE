import sqlite3, os
p = r'gosales/gosales_curated.db'
print('curated path:', p, 'exists:', os.path.exists(p))
con = sqlite3.connect(p)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('tables sample:', [r[0] for r in cur.fetchall() if r[0] in ('fact_sales_line','fact_sales_header','fact_transactions','dim_customer')])
con.close()
