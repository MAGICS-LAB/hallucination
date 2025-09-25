"""
Per-prompt plots: hallucination rate vs. epoch.

Inputs
------
--results_csv : path to all_checkpoints_results.csv (from HCDR evaluator)
--losses_csv  : optional path to training_losses.csv to map global_step -> epoch
                (should contain at least columns: epoch and global_step;
                 if missing, we fall back to binning steps into pseudo-epochs)
--out_dir     : output directory for figures

Output
------
For each unique prompt, saves:
    _hallucination_vs_epoch.png
    _hallucination_vs_epoch.pdf
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return s[:maxlen] if s else "prompt"


def load_rate_by_step(results_csv: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    needed = {"global_step", "prompt", "inside_hcdr"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{results_csv} must contain columns {needed}")

    # hallucination rate per (prompt, step) = 1 - mean(inside_hcdr)
    grp = (
        df.groupby(["prompt", "global_step"], as_index=False)["inside_hcdr"]
          .mean()
          .rename(columns={"inside_hcdr": "inside_mean"})
          .sort_values(["prompt", "global_step"])
    )
    grp["hallucination_rate"] = 1.0 - grp["inside_mean"]
    return grp[["prompt", "global_step", "hallucination_rate"]]


def load_step_epoch_map(losses_csv: str) -> pd.DataFrame:
    """
    Build a step->epoch map if possible from a flexible loss CSV.
    Requires columns: epoch and global_step (case-insensitive).
    Returns DataFrame with columns ['global_step','epoch'] or empty if not available.
    """
    L = pd.read_csv(losses_csv)
    cols = {c.lower(): c for c in L.columns}
    if "epoch" in cols and "global_step" in cols:
        m = L[[cols["global_step"], cols["epoch"]]].dropna()
        m.columns = ["global_step", "epoch"]
        # keep the last epoch seen for a step if duplicates
        m = m.sort_values(["global_step", "epoch"]).drop_duplicates("global_step", keep="last")
        return m
    return pd.DataFrame(columns=["global_step", "epoch"])


def attach_epochs(rate_step: pd.DataFrame, step_epoch_map: pd.DataFrame, num_bins: int | None) -> pd.DataFrame:
    """
    Add an 'epoch' column to the per-(prompt,step) rates.
    Priority: use provided step->epoch map. Fallback: quantile binning by global_step rank.
    """
    out = rate_step.sort_values("global_step").copy()

    if not step_epoch_map.empty:
        out = pd.merge_asof(
            out.sort_values("global_step"),
            step_epoch_map.sort_values("global_step"),
            on="global_step",
            direction="backward"
        )
        out["epoch"] = out["epoch"].ffill().bfill()
        return out

    # Fallback: create pseudo-epochs by binning steps uniformly.
    steps_order = out["global_step"].rank(method="first")
    if num_bins is None:
        # pick a reasonable number of bins (epochs) from distinct steps
        distinct_steps = out["global_step"].nunique()
        num_bins = max(3, min(20, distinct_steps))  # clamp between 3 and 20
    out["epoch"] = pd.qcut(steps_order, q=num_bins, labels=range(1, num_bins + 1)).astype(int)
    return out


def plot_per_prompt(rate_epoch: pd.DataFrame, prompt: str, out_dir: Path):
    """
    rate_epoch: DataFrame with columns ['epoch','hallucination_rate'] for a single prompt
    """
    rate_epoch = rate_epoch.sort_values("epoch")
    fig = plt.figure(figsize=(7.0, 4.2))
    ax = plt.gca()

    ax.plot(rate_epoch["epoch"], rate_epoch["hallucination_rate"],
            color="green", marker="o")
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Hallucination Rate", fontsize=20)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(prompt, fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    name = slug(prompt)
    # png = out_dir / f"{name}_hallucination_vs_epoch.png"
    pdf = out_dir / f"{name}_hallucination_vs_epoch.pdf"
    # fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, default="./hcdr_report/all_checkpoints_results.csv",
                    help="Path to all_checkpoints_results.csv")
    ap.add_argument("--losses_csv", type=str, default=None,
                    help="Optional path to training_losses.csv (to map global_step to epoch)")
    ap.add_argument("--out_dir", type=str, default="./per_prompt_plots",
                    help="Directory to save per-prompt figures")
    ap.add_argument("--pseudo_epochs", type=int, default=None,
                    help="If no losses_csv, number of pseudo-epoch bins (default: auto)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rate_step = load_rate_by_step(args.results_csv)

    # Step->epoch mapping
    step_epoch_map = pd.DataFrame(columns=["global_step", "epoch"])
    if args.losses_csv:
        try:
            step_epoch_map = load_step_epoch_map(args.losses_csv)
        except Exception:
            step_epoch_map = pd.DataFrame(columns=["global_step", "epoch"])

    rate_with_epochs = attach_epochs(rate_step, step_epoch_map, args.pseudo_epochs)

    # Compute per-prompt per-epoch hallucination rate
    by_prompt_epoch = (
        rate_with_epochs.groupby(["prompt", "epoch"], as_index=False)["hallucination_rate"]
                        .mean()
                        .sort_values(["prompt", "epoch"])
    )

    # Plot one figure per prompt
    for prompt, dfp in by_prompt_epoch.groupby("prompt"):
        plot_per_prompt(dfp[["epoch", "hallucination_rate"]], prompt, out_dir)

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
