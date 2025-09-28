"""Quickly inspect whitespace scoring output for manual QA.

The whitespace scoring flow exports a consolidated CSV that contains per-customer
scores, probabilities, and qualitative reasons. This helper script reads that
CSV (defaulting to the most recent dated file in ``gosales/outputs``) and prints
human-readable summaries so a sales analyst can sanity check the results without
opening a spreadsheet:

* overall row count and the list of available divisions in the drop
* the top 20 customers globally by score
* the top 10 customers for each strategic division of interest

The output provides a high-signal view of how the ranking behaves and whether
expected context columns made it through the scoring pipeline.
"""

import pandas as pd
from pathlib import Path

WS = Path('gosales/outputs/whitespace_20240630.csv')

def _num(s):
    return pd.to_numeric(s, errors='coerce')

def main():
    if not WS.exists():
        print('Whitespace not found:', WS)
        return
    df = pd.read_csv(WS)
    # Coerce dtypes
    for c in ['score','score_challenger','p_icp','p_icp_pct','lift_norm','als_norm','EV_norm']:
        if c in df.columns:
            df[c] = _num(df[c])
    # Display basic coverage
    print('Rows:', len(df), 'Divisions:', sorted(df['division_name'].dropna().unique().tolist()))
    # Top 20 overall by score
    top_overall = df.sort_values(['score','p_icp','EV_norm'], ascending=[False, False, False]).head(20)
    cols = [c for c in ['division_name','customer_id','customer_name','score','p_icp','p_icp_pct','EV_norm','lift_norm','als_norm','nba_reason'] if c in df.columns]
    print('\n=== Top 20 Overall (by score) ===')
    print(top_overall[cols].to_string(index=False))
    # Top 10 per selected divisions/models
    targets = ['Printers','SWX_Seats','PDM_Seats','Scanning','CAMWorks','SW_Electrical','SW_Inspection','Training','Services','Simulation','Success_Plan']
    for t in targets:
        sub = df[df['division_name'].astype(str).str.strip()==t]
        if sub.empty:
            continue
        top = sub.sort_values(['score','p_icp','EV_norm'], ascending=[False, False, False]).head(10)
        print(f"\n=== Top 10 for {t} ===")
        print(top[cols].to_string(index=False))

if __name__ == '__main__':
    main()

