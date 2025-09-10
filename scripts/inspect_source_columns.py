from gosales.utils.db import get_db_connection
import pandas as pd

engine = get_db_connection()
try:
    df = pd.read_sql("SELECT TOP 1 * FROM dbo.saleslog", engine)
except Exception:
    # Fallback for non-SQL Server syntax
    df = pd.read_sql("SELECT * FROM dbo.saleslog WHERE 1=0", engine)
cols = list(map(str, df.columns))
print('Total columns:', len(cols))
def find(pat):
    hits = [c for c in cols if pat.lower() in c.lower()]
    print(f"Columns containing '{pat}':", hits)
for pat in ['cam', 'camworks', 'electrical', 'inspection', 'formlabs', 'metals', 'polyjet', 'p3', 'saf', 'sla', 'fdm']:
    find(pat)
