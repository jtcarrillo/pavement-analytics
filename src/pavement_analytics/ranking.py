"""Condition-normalized, interval-weighted group rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import condition_predict


def rank(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for material, frame in pairs.groupby("material", sort=True):
        frame = frame.copy()
        frame["residual"] = frame.DetRatePerYear - condition_predict(
            frame.PCI_start, frame.DetRatePerYear, frame.delta_days, frame.PCI_start)
        population = np.average(frame.residual, weights=frame.delta_days)
        groups = []
        for group, block in frame.groupby("group_id", sort=True):
            groups.append({"material": material, "group_id": group, "n_obs": len(block),
                           "n_sections": len(block[["_source_file_start", "section_id"]].drop_duplicates()),
                           "total_interval_years": float(block.delta_days.sum() / 365),
                           "raw_mean_rate": float(np.average(block.DetRatePerYear, weights=block.delta_days)),
                           "mean_pci": float(np.average(block.PCI_start, weights=block.delta_days)),
                           "raw_residual": float(np.average(block.residual, weights=block.delta_days))})
        table = pd.DataFrame(groups)
        grid = np.linspace(1, 50, 201)
        n = table.n_obs.to_numpy()[:, None]
        residuals = table.raw_residual.to_numpy()[:, None]
        scores = (n * residuals + grid[None, :] * population) / (n + grid[None, :])
        table["score"] = np.trapezoid(scores, grid, axis=1) / (grid[-1] - grid[0])
        rank_grid = pd.DataFrame(scores).rank(axis=0, ascending=False, method="average")
        table["rank_swing"] = rank_grid.max(axis=1) - rank_grid.min(axis=1)
        table["stable"] = table.rank_swing.lt(0.5)
        table["rank"] = table.score.rank(ascending=False, method="average")
        rows.append(table.sort_values(["rank", "group_id"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
