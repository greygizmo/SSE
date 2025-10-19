import sqlite3, os
p = r'gosales/gosales.db'
print('db path:', p, 'exists:', os.path.exists(p))
con = sqlite3.connect(p)
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
rows = cur.fetchall()
print('tables count:', len(rows))
print('tables sample:', [r[0] for r in rows[:30]])
con.close()
