### Text-to-image Plottings
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_rate_by_step(results_csv: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    needed = {"global_step", "inside_hcdr"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{results_csv} must contain columns {needed}")
    g = (
        df.groupby("global_step", as_index=False)["inside_hcdr"]
        .mean()
        .rename(columns={"inside_hcdr": "inside_mean"})
        .sort_values("global_step")
    )
    g["hallucination_rate"] = 1.0 - g["inside_mean"]
    return g[["global_step", "hallucination_rate"]]


def load_losses(losses_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (loss_epoch, step_epoch_map)
      - loss_epoch: columns ['epoch','avg_loss'] (epoch-level)
      - step_epoch_map: columns ['global_step','epoch'] (for mapping steps to epochs)
    Accepts flexible CSVs containing some subset of: epoch, global_step, avg_loss, step_loss.
    """
    L = pd.read_csv(losses_csv)

    # normalize common column names
    cols = {c.lower(): c for c in L.columns}
    def col(name): return cols.get(name, None)

    # Build epoch-level loss table
    loss_epoch = pd.DataFrame(columns=["epoch", "avg_loss"])
    if col("avg_loss") and col("epoch"):
        tmp = L[[cols["epoch"], cols["avg_loss"]]].dropna()
        tmp.columns = ["epoch", "avg_loss"]
        # if duplicates, keep last or mean—use last by default
        loss_epoch = tmp.drop_duplicates("epoch", keep="last").sort_values("epoch")
    elif col("step_loss") and col("epoch"):
        # Fallback: average step losses per epoch
        tmp = L[[cols["epoch"], cols["step_loss"]]].dropna()
        tmp.columns = ["epoch", "step_loss"]
        loss_epoch = (
            tmp.groupby("epoch", as_index=False)["step_loss"].mean()
            .rename(columns={"step_loss": "avg_loss"})
            .sort_values("epoch")
        )

    # Build step→epoch map if present
    step_epoch_map = pd.DataFrame(columns=["global_step", "epoch"])
    if col("global_step") and col("epoch"):
        tmp = L[[cols["global_step"], cols["epoch"]]].dropna()
        tmp.columns = ["global_step", "epoch"]
        step_epoch_map = (
            tmp.sort_values(["global_step", "epoch"])
               .drop_duplicates("global_step", keep="last")
        )

    return loss_epoch, step_epoch_map


def assign_epochs_to_steps(rate_step: pd.DataFrame,
                           step_epoch_map: pd.DataFrame,
                           loss_epoch: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'epoch' column to rate_step.
    Primary: merge_asof with step_epoch_map.
    Fallback: if no mapping rows, infer epochs by rank-binning steps into len(unique_epochs) bins.
    """
    rs = rate_step.sort_values("global_step").copy()

    if not step_epoch_map.empty:
        rs = pd.merge_asof(
            rs, step_epoch_map.sort_values("global_step"),
            on="global_step", direction="backward"
        )
        # still NA? forward fill then back fill
        rs["epoch"] = rs["epoch"].ffill().bfill()
    else:
        # Fallback heuristic: evenly bin steps into the number of epochs we have
        uniq_epochs = sorted(loss_epoch["epoch"].unique()) if not loss_epoch.empty else []
        if len(uniq_epochs) >= 2:
            rs["epoch"] = pd.qcut(
                rs["global_step"].rank(method="first"),
                q=len(uniq_epochs), labels=uniq_epochs
            ).astype(int)
        else:
            # If we truly have no epoch info, make a dummy epoch (1)
            rs["epoch"] = 1

    return rs


def plot_rate_and_loss(rate_epoch: pd.DataFrame, loss_epoch: pd.DataFrame,
                       out_png: str, out_pdf: str, title: str):
    # Common epoch index
    epochs = sorted(set(rate_epoch["epoch"]).union(set(loss_epoch["epoch"])))
    rate_plot = rate_epoch.set_index("epoch").reindex(epochs).reset_index()
    loss_plot = loss_epoch.set_index("epoch").reindex(epochs).reset_index()

    fig = plt.figure(figsize=(8.5, 5.0))
    ax1 = plt.gca()

    # Hallucination rate (green)
    line1, = ax1.plot(rate_plot["epoch"], rate_plot["hallucination_rate"],
                      color="green", marker="o", label="Hallucination Rate", linewidth=2)
    ax1.set_xlabel("Epochs", fontsize=19)
    ax1.set_ylabel("Hallucination rate", color="green", fontsize=19)
    ax1.tick_params(axis="both", labelsize=16)
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xticks(epochs)  # force integer ticks

    # Training loss (blue) on twin axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(loss_plot["epoch"], loss_plot["avg_loss"],
                      color="blue", marker="s", linestyle="--",
                      label="Training Loss (avg)", linewidth=2)
    ax2.set_ylabel("Training Loss (avg)", color="blue", fontsize=19)
    ax2.tick_params(axis="y", labelsize=16, labelcolor="blue")

    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=14)

    plt.title(title, fontsize=21)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(out_png, dpi=220, bbox_inches="tight")  # save first
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}\nSaved: {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", type=str, default="./hcdr_report/all_checkpoints_results.csv",
                    help="Path to all_checkpoints_results.csv")
    ap.add_argument("--losses_csv", type=str, default="./training_losses.csv",
                    help="Path to training_losses.csv")
    ap.add_argument("--out_png", type=str, default="rate_loss_epochs.png")
    ap.add_argument("--out_pdf", type=str, default="rate_loss_epochs.pdf")
    ap.add_argument("--title", type=str,
                    default="Hallucination Rate and Training Loss")
    args = ap.parse_args()

    rate_step = load_rate_by_step(args.results_csv)
    loss_epoch, step_epoch_map = load_losses(args.losses_csv)
    rate_step = assign_epochs_to_steps(rate_step, step_epoch_map, loss_epoch)

    # Average hallucination rate per epoch
    rate_epoch = (
        rate_step.groupby("epoch", as_index=False)["hallucination_rate"]
        .mean()
        .sort_values("epoch")
    )

    # If loss_epoch is empty, create a NaN series aligned to the epochs we have (plot will show only rate)
    if loss_epoch.empty:
        loss_epoch = pd.DataFrame({"epoch": rate_epoch["epoch"], "avg_loss": np.nan})

    plot_rate_and_loss(rate_epoch, loss_epoch, args.out_png, args.out_pdf, args.title)


if __name__ == "__main__":
    main()