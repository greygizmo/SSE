import re
s=open('gosales/etl/sku_map.py','r',encoding='utf-8').read()
vals=re.findall(r"['\"]division['\"]\s*:\s*['\"]([^'\"]+)['\"]", s)
print(sorted(set(vals)))
