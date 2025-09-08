from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from gosales.utils.paths import OUTPUTS_DIR


def main() -> Path:
    root = OUTPUTS_DIR / 'leakage'
    rows: list[tuple[str, str, str, Path]] = []
    if not root.exists():
        return root / f'summary_{datetime.utcnow().strftime("%Y%m%d%H%M%S")}.md'
    for div_dir in root.iterdir():
        if not div_dir.is_dir():
            continue
        for cut_dir in div_dir.iterdir():
            if not cut_dir.is_dir():
                continue
            rep = cut_dir / f'leakage_report_{div_dir.name}_{cut_dir.name}.json'
            if rep.exists():
                try:
                    data = json.loads(rep.read_text(encoding='utf-8'))
                    rows.append((div_dir.name, cut_dir.name, str(data.get('overall')), rep))
                except Exception:
                    rows.append((div_dir.name, cut_dir.name, 'UNKNOWN', rep))
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out = root / f'summary_{now}.md'
    lines = [f'# Leakage Gauntlet Summary ({now} UTC)\n']
    for div, cut, status, rep in sorted(rows):
        try:
            data = json.loads(rep.read_text(encoding='utf-8'))
            checks = data.get('checks', {})
        except Exception:
            checks = {}
        lines.append(f'- Division: {div}, Cutoff: {cut}, Overall: {status}  ')
        lines.append(f'  Report: {rep}')
        if 'shift14' in checks:
            lines.append(f"  Shift-14: {checks.get('shift14')}")
        if 'ablation_topk' in checks:
            lines.append(f"  Top-K Ablation: {checks.get('ablation_topk')}")
        lines.append('')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote', out)
    return out


if __name__ == '__main__':
    main()

