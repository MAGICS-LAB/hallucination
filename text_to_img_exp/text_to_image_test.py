#!/usr/bin/env python3
"""
End-to-end HCDR test on AFHQ (cats/dogs) for T2I prompts — with:
- CSV logging
- Plots
- Optional fine-tuned UNet override at eval time

Paper alignment (very short):
- Estimator under squared loss is the expectation A*(X) = E[A(X)] (Lemma A.1).
- δ-hallucination: A*(X) falls outside the Highest(-Conditional) Density Region(s) of mass M.
This script instantiates that in a CLIP feature space, with consistent preprocessing.

"""

import os, math, random, argparse, csv, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

warnings.filterwarnings("ignore")

# -----------------------------
# Utility: deterministic seeding
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------
# Data: AFHQ cats/dogs pulled from HF hub
# ----------------------------------------
class AFHQCatsDogs(Dataset):
    """
    Materializes AFHQ (train split) locally and gives PIL images + labels for {cat, dog}.
    """
    def __init__(self, root: str, split: str = "train", max_per_class: int = None, image_size: int = 256):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.items: List[Tuple[str, str]] = []  # (path, label)
        self._prepare_if_needed(max_per_class=max_per_class)

    def _prepare_if_needed(self, max_per_class: int):
        split_dir = self.root / self.split
        if not split_dir.exists() or not any(split_dir.iterdir()):
            print("Downloading AFHQ (huggan/AFHQ) and materializing...")
            ds = load_dataset("huggan/AFHQ", split=self.split)  # single 'train' split
            label_names = ds.features["label"].names  # ['cat','dog','wild']
            (split_dir / "cat").mkdir(parents=True, exist_ok=True)
            (split_dir / "dog").mkdir(parents=True, exist_ok=True)

            counters = {"cat": 0, "dog": 0}
            for row in ds:
                name = label_names[int(row["label"])]
                if name not in ("cat", "dog"):
                    continue
                if max_per_class is not None and counters[name] >= max_per_class:
                    continue
                img: Image.Image = row["image"]
                outdir = split_dir / name
                outpath = outdir / f"{name}_{counters[name]:06d}.jpg"
                img.save(outpath, quality=95)
                counters[name] += 1

        # index
        for cls in ("cat", "dog"):
            cls_dir = split_dir / cls
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    self.items.append((str(f), cls))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, label = self.items[idx]
        img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size), Image.LANCZOS)
        return img, label


# -------------------------------------
# Feature preprocessor: L2 -> PCA -> Z
# -------------------------------------
class FeaturePreproc:
    def __init__(self, pca_dim: int = 128):
        self.pca_dim = pca_dim
        self.pca = None
        self.zs = None

    def fit(self, feats: np.ndarray):
        # L2 normalize
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        # PCA
        self.pca = PCA(n_components=min(self.pca_dim, feats.shape[1]), random_state=0)
        Xp = self.pca.fit_transform(feats)
        # Z-score
        self.zs = StandardScaler().fit(Xp)
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        Xp = self.pca.transform(feats)
        Xz = self.zs.transform(Xp)
        return Xz


# --------------------------------------
# Core: HCDR Builder + Evaluation logic
# --------------------------------------
class HCDRRunner:
    def __init__(self, args):
        self.args = args
        self.outdir = Path(args.output_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(args.seed)

        # CLIP (keep in fp32 for numerically stable features)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_model.eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # SD pipeline (optionally override UNet with fine-tuned weights)
        dtype = torch.float16 if (self.device.type == "cuda" and args.fp16) else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            args.sd_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        ).to(self.device)

        if args.unet_path:
            print(f"Loading fine-tuned UNet from: {args.unet_path}")
            ft_unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=dtype)
            self.pipe.unet = ft_unet.to(self.device)

        # Data
        self.ds = AFHQCatsDogs(root=args.data_root, split="train",
                               max_per_class=args.max_per_class, image_size=args.image_size)

        # Containers
        self.preproc = FeaturePreproc(pca_dim=args.pca_dim)
        self.gmms: Dict[str, GaussianMixture] = {}           # per-state density
        self.log_thr: Dict[str, float] = {}                   # per-state log threshold
        self.labels = ("cat", "dog")

        # CSV setup
        self.csv_path = self.outdir / "hcdr_results.csv"
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "prompt", "K", "hdr_percentile",
                "log_thr_cat", "log_thr_dog",
                "logp_mean_cat", "logp_mean_dog",
                "inside_hcdr", "decision"
            ])

    # ---------- Feature extraction ----------
    @torch.no_grad()
    def clip_embed(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.clip_proc(images=images, return_tensors="pt").to(self.device)
        feats = self.clip_model.get_image_features(**inputs)  # (N, D)
        return feats.float().cpu().numpy()

    # ---------- Build densities + thresholds ----------
    def build_reference(self):
        # Split AFHQ per-class into fit and calibration sets
        by_cls: Dict[str, List[Image.Image]] = {c: [] for c in self.labels}
        for img, cls in self.ds:
            by_cls[cls].append(img)

        # Extract features per class, stack for preproc fit
        raw_all = []
        raw_by_cls = {}
        for c in self.labels:
            feats_c = self.clip_embed(by_cls[c])
            raw_by_cls[c] = feats_c
            raw_all.append(feats_c)
        raw_all = np.concatenate(raw_all, axis=0)

        # Fit preproc ONCE over pooled reference
        self.preproc.fit(raw_all)

        # Fit per-class GMMs and calibrate thresholds on HELD-OUT split
        self.cal_ll: Dict[str, np.ndarray] = {}
        for c in self.labels:
            feats = self.preproc.transform(raw_by_cls[c])
            # split 80/20 for fit/cal
            N = len(feats)
            idx = np.arange(N)
            rng = np.random.RandomState(self.args.seed)
            rng.shuffle(idx)
            s = int(0.8 * N)
            fit_idx, cal_idx = idx[:s], idx[s:]
            X_fit, X_cal = feats[fit_idx], feats[cal_idx]

            gmm = GaussianMixture(
                n_components=min(6, max(2, len(X_fit)//200)),
                covariance_type="diag",
                reg_covar=1e-3,
                max_iter=300,
                random_state=self.args.seed
            ).fit(X_fit)
            self.gmms[c] = gmm

            cal_ll = gmm.score_samples(X_cal)  # LOG densities
            self.cal_ll[c] = cal_ll
            q = np.percentile(cal_ll, self.args.hdr_percentile)  # e.g., 10th → HDR mass ≈ 90%
            self.log_thr[c] = float(q)

        # Diagnostics to console
        print("Per-class log-thresholds (HDR @ p%):")
        for c in self.labels:
            print(f"  {c}: p{self.args.hdr_percentile} -> {self.log_thr[c]:.3f}")

        # Save a quick summary JSON (optional)
        with open(self.outdir / "thresholds.txt", "w") as f:
            for c in self.labels:
                f.write(f"{c}\tp{self.args.hdr_percentile}\t{self.log_thr[c]:.6f}\n")

    # ---------- Expectation estimator ----------
    @torch.no_grad()
    def estimate_mean_embedding(self, prompt: str, K: int) -> Tuple[np.ndarray, np.ndarray]:
        imgs = []
        for k in range(K):
            img = self.pipe(
                prompt,
                num_inference_steps=self.args.sd_steps,
                guidance_scale=self.args.guidance,
                generator=torch.Generator(self.device).manual_seed(self.args.seed + k)
            ).images[0]
            imgs.append(img)
        feats = self.clip_embed(imgs)             # raw
        feats_z = self.preproc.transform(feats)   # SAME preproc as reference
        m_hat = feats_z.mean(axis=0)              # ≈ E[φ(Y)|X]
        return m_hat, feats_z

    # ---------- HCDR membership test ----------
    def in_hcdr(self, vec_z: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        per_class_ll = {}
        for c in self.labels:
            ll = float(self.gmms[c].score_samples(vec_z[None, :])[0])  # LOG density
            per_class_ll[c] = ll
        inside_any = any(per_class_ll[c] > self.log_thr[c] for c in self.labels)
        return inside_any, per_class_ll

    # ---------- Plot helpers ----------
    def _plot_calibration_hist(self):
        # Histograms of calibration log-densities + thresholds
        plt.figure(figsize=(10,5))
        for c in self.labels:
            plt.hist(self.cal_ll[c], bins=50, alpha=0.5, label=f"{c} cal log p")
            plt.axvline(self.log_thr[c], linestyle="--", linewidth=2, label=f"{c} thr@p{self.args.hdr_percentile}")
        plt.title("Calibration log-densities & HDR thresholds")
        plt.xlabel("log density")
        plt.ylabel("count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.outdir / "calibration_histograms.png", dpi=200)
        plt.close()

    def _plot_prompt_scores(self, logs: List[dict]):
        # Scatter of per-prompt m_hat log p vs thresholds, and pass/fail bars
        # 1) Scatter
        plt.figure(figsize=(10,5))
        xs = np.arange(len(logs))
        cat_ll = [d["logp_mean_cat"] for d in logs]
        dog_ll = [d["logp_mean_dog"] for d in logs]
        plt.scatter(xs, cat_ll, marker="o", label="log p(m_hat|cat)")
        plt.scatter(xs, dog_ll, marker="s", label="log p(m_hat|dog)")
        plt.axhline(self.log_thr["cat"], linestyle="--", label=f"cat thr@p{self.args.hdr_percentile}")
        plt.axhline(self.log_thr["dog"], linestyle="--", label=f"dog thr@p{self.args.hdr_percentile}")
        plt.xticks(xs, [d["prompt_short"] for d in logs], rotation=30, ha="right")
        plt.title("Per-prompt m_hat log densities vs HDR thresholds")
        plt.ylabel("log density")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.outdir / "per_prompt_scores.png", dpi=200)
        plt.close()

        # 2) Pass/Fail bars
        plt.figure(figsize=(8,4))
        pf = [1 if d["inside_hcdr"] else 0 for d in logs]
        plt.bar(xs, pf)
        plt.xticks(xs, [d["prompt_short"] for d in logs], rotation=30, ha="right")
        plt.yticks([0,1], ["δ-hallucination", "non-hallucinating"])
        plt.title("HCDR decision per prompt")
        plt.tight_layout()
        plt.savefig(self.outdir / "per_prompt_decision.png", dpi=200)
        plt.close()

    # ---------- End-to-end for a prompt ----------
    def run_for_prompt(self, prompt: str) -> dict:
        print(f"\nPrompt: {prompt}")
        m_hat, sample_feats = self.estimate_mean_embedding(prompt, self.args.K)
        inside, scores = self.in_hcdr(m_hat)
        decision = "IN HCDR (non-hallucinating)" if inside else "OUTSIDE HCDR → δ-HALLUCINATION"
        print(f"Status: {decision}")
        print("Per-class log p( m_hat | Z=i ):")
        for c in self.labels:
            print(f"  {c}: {scores[c]:.3f}  (thr {self.log_thr[c]:.3f})")

        row = {
            "prompt": prompt,
            "prompt_short": (prompt[:40] + "…") if len(prompt) > 40 else prompt,
            "K": self.args.K,
            "hdr_percentile": self.args.hdr_percentile,
            "log_thr_cat": self.log_thr["cat"],
            "log_thr_dog": self.log_thr["dog"],
            "logp_mean_cat": scores["cat"],
            "logp_mean_dog": scores["dog"],
            "inside_hcdr": int(inside),
            "decision": decision
        }

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                row["prompt"], row["K"], row["hdr_percentile"],
                row["log_thr_cat"], row["log_thr_dog"],
                row["logp_mean_cat"], row["logp_mean_dog"],
                row["inside_hcdr"], row["decision"]
            ])

        # Also save the K sample features for possible diagnostics
        np.save(self.outdir / f"sample_feats_{row['prompt_short'].replace(' ','_')}.npy", sample_feats)
        return row

    # ---------- Script entry ----------
    def run(self):
        self.build_reference()
        logs = []
        for prompt in self.args.prompts:
            logs.append(self.run_for_prompt(prompt))

        # Plots
        self._plot_calibration_hist()
        self._plot_prompt_scores(logs)

        # Summary
        rate = 1.0 - (sum([d["inside_hcdr"] for d in logs]) / max(1, len(logs)))
        with open(self.outdir / "summary.txt", "w") as f:
            f.write(f"Num prompts: {len(logs)}\n")
            f.write(f"HDR percentile: {self.args.hdr_percentile}\n")
            f.write(f"δ-hallucination rate (expectation outside HCDR): {rate:.3f}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./hcdr_afhq_results")
    p.add_argument("--data_root", type=str, default="./afhq_data")
    p.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--unet_path", type=str, default="./runs/sd_ft", help="Path to fine-tuned UNet folder (e.g., .../final_model/unet)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 for the diffusion UNet/pipe when on CUDA")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--max_per_class", type=int, default=2000, help="cap AFHQ per class for speed")
    p.add_argument("--pca_dim", type=int, default=128)
    p.add_argument("--hdr_percentile", type=float, default=10.0,
                   help="percentile of log-density on held-out real features; 10 → HDR mass ≈ 90%")
    p.add_argument("--K", type=int, default=64, help="samples to estimate mean embedding")
    p.add_argument("--sd_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts", nargs="+", default=[
        "a realistic photo of a friendly dog",      # dog-ish
        "a fluffy cat sitting on a sofa",           # cat-ish
        "a cute pet animal"                         # ambiguous → union HCDR
    ])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = HCDRRunner(args)
    runner.run()
