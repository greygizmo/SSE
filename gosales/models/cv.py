from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import numpy as np
import pandas as pd


@dataclass
class BlockedPurgedGroupCV:
    n_splits: int = 5
    purge_days: int = 30
    seed: int = 42

    def split(
        self,
        X,
        y,
        groups: Iterable[str],
        *,
        anchor_days_from_cutoff: Iterable[float],
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        groups = np.asarray(list(groups))
        anchor = np.asarray(list(anchor_days_from_cutoff), dtype=float)
        df = pd.DataFrame({"group": groups, "anchor": anchor})
        g_agg = df.groupby("group", as_index=False)["anchor"].median().rename(columns={"anchor": "anchor_median"})
        g_agg = g_agg.sort_values("anchor_median", ascending=True).reset_index(drop=True)
        blocks = np.array_split(g_agg, self.n_splits)

        idx_by_group: dict[str, list[int]] = {}
        for i, grp in enumerate(groups.astype(str)):
            idx_by_group.setdefault(grp, []).append(i)

        for k in range(self.n_splits):
            val_groups = set(blocks[k]["group"].tolist())
            val_min = float(blocks[k]["anchor_median"].min())
            val_max = float(blocks[k]["anchor_median"].max())

            train_groups: list[str] = []
            for j, block in enumerate(blocks):
                if j == k:
                    continue
                mask = (block["anchor_median"] >= (val_max + self.purge_days)) | (
                    block["anchor_median"] <= (val_min - self.purge_days)
                )
                safe_block = block[mask]
                train_groups.extend(safe_block["group"].tolist())

            train_idx: list[int] = []
            val_idx: list[int] = []
            for grp in train_groups:
                train_idx.extend(idx_by_group.get(str(grp), []))
            for grp in val_groups:
                val_idx.extend(idx_by_group.get(str(grp), []))
            yield np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

