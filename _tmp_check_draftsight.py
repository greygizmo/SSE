import pandas as pd
p='gosales/outputs/features_draftsight_2025-03-31.parquet'
df=pd.read_parquet(p)
print('rows',len(df),'positives',int(df.get('bought_in_division',pd.Series()).sum()))
print(df.columns[:10].tolist())
