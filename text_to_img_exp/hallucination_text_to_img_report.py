"""
HCDR report over training checkpoints (AFHQ cats/dogs, Stable Diffusion UNets).

---------------------
1) One-time calibration on REAL AFHQ:
   - Fix embedding (CLIP image encoder).
   - Consistent preproc: L2 → PCA(D) -> z-score.
   - Per-class (cat/dog) diagonal-cov GMM in preproc space.
   - HDR thresholds from HELD-OUT real images on the *log* scale (percentile p).

2) For each UNet checkpoint:
   - Swap the UNet into a Stable Diffusion pipeline (base text encoder/VAE/scheduler).
   - For each prompt X, draw K images, embed with e, average to get m(X) ≈ E[e(Y)|X]).
   - Score log p(m|Z=i) for i\in{cat,dog}, check HCDR membership (union of HDRs).
   - Log CSV rows and per-ckpt summary (delta-hallucination rate over prompts).

3) Aggregate & visualize:
   - Join with training loss (if provided).
   - Plots: calibration histograms; per-prompt score vs thresholds; pass/fail bars;
            time-series of hallucination rate; loss vs hallucination.
   - A compact HTML report linking all artifacts.

-------------
- Under squared loss, the optimal estimator is the *expectation* A*(X)=E[A(X)].
- delta-hallucination ⇔ A*(X) lies *outside* all conditional HDRs (HCDR union test).
"""

import os, re, csv, glob, json, math, random, argparse, warnings, base64
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# --------------- utils ---------------

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def b64img(path: Path) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

# --------------- data ---------------

class AFHQCatsDogs(Dataset):
    """
    Materialize AFHQ (train) {cat,dog} locally. Returns PIL + label.
    """
    def __init__(self, root: str, split: str = "train", max_per_class: int = None, image_size: int = 256):
        super().__init__()
        self.root = Path(root); self.split = split; self.image_size = image_size
        self.items: List[Tuple[str,str]] = []
        self._prepare_if_needed(max_per_class)

    def _prepare_if_needed(self, max_per_class:int):
        split_dir = self.root / self.split
        if not split_dir.exists() or not any(split_dir.iterdir()):
            print("Downloading AFHQ (huggan/AFHQ) and materializing...")
            ds = load_dataset("huggan/AFHQ", split=self.split)  # one split: 'train'
            names = ds.features["label"].names  # ['cat','dog','wild']
            (split_dir/"cat").mkdir(parents=True, exist_ok=True)
            (split_dir/"dog").mkdir(parents=True, exist_ok=True)
            ctr = {"cat":0, "dog":0}
            for row in ds:
                name = names[int(row["label"])]
                if name not in ("cat","dog"): continue
                if max_per_class is not None and ctr[name] >= max_per_class: continue
                img: Image.Image = row["image"]
                out = split_dir/name/f"{name}_{ctr[name]:06d}.jpg"
                img.save(out, quality=95); ctr[name]+=1

        for cls in ("cat","dog"):
            for f in sorted((split_dir/cls).iterdir()):
                if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp"):
                    self.items.append((str(f), cls))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        p, label = self.items[idx]
        img = Image.open(p).convert("RGB").resize((self.image_size,self.image_size), Image.LANCZOS)
        return img, label

# --------------- preproc ---------------

class FeaturePreproc:
    """
    L2 → PCA(d) → z-score ; fit ONCE on pooled real features, reuse everywhere.
    """
    def __init__(self, pca_dim:int=128):
        self.pca_dim=pca_dim; self.pca=None; self.zs=None

    def fit(self, feats: np.ndarray):
        feats = feats / (np.linalg.norm(feats,axis=1,keepdims=True)+1e-12)
        self.pca = PCA(n_components=min(self.pca_dim, feats.shape[1]), random_state=0)
        Xp = self.pca.fit_transform(feats)
        self.zs = StandardScaler().fit(Xp)
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        feats = feats / (np.linalg.norm(feats,axis=1,keepdims=True)+1e-12)
        Xp = self.pca.transform(feats)
        return self.zs.transform(Xp)

    def dump(self, path: Path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, 
                 pca_components_=self.pca.components_, 
                 pca_mean_=self.pca.mean_,
                 pca_explained_variance_=self.pca.explained_variance_,
                 zs_mean_=self.zs.mean_,
                 zs_scale_=self.zs.scale_)

    @staticmethod
    def load(path: Path):
        blob = np.load(path, allow_pickle=True)
        obj = FeaturePreproc(pca_dim=blob["pca_components_"].shape[0])
        obj.pca = PCA(n_components=obj.pca_dim)
        obj.pca.components_ = blob["pca_components_"]
        obj.pca.mean_ = blob["pca_mean_"]
        obj.pca.explained_variance_ = blob["pca_explained_variance_"]
        obj.pca.n_features_in_ = obj.pca.mean_.shape[0]
        obj.zs = StandardScaler()
        obj.zs.mean_ = blob["zs_mean_"]
        obj.zs.scale_ = blob["zs_scale_"]
        obj.zs.var_ = obj.zs.scale_**2
        return obj

# --------------- evaluator core ---------------

class HCDREvaluator:
    def __init__(self, args):
        self.a = args
        self.outdir = Path(args.output_dir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.cache = self.outdir / "frame_cache"
        self.cache.mkdir(exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(args.seed)

        # CLIP in fp32 for stable features
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip.eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Base SD pipeline; we’ll swap UNet per checkpoint
        dtype = torch.float16 if (self.device.type=="cuda" and args.fp16) else torch.float32
        self.base_pipe = StableDiffusionPipeline.from_pretrained(
            args.sd_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        ).to(self.device)

        # AFHQ data
        self.ds = AFHQCatsDogs(root=args.data_root, split="train",
                               max_per_class=args.max_per_class, image_size=args.image_size)
        self.labels = ("cat","dog")

        # Calibration artifacts
        self.pre = None
        self.gmms: Dict[str, GaussianMixture] = {}
        self.cal_ll: Dict[str, np.ndarray] = {}
        self.log_thr: Dict[str, float] = {}

        # Logging/plots
        self.csv_all = self.outdir / "all_checkpoints_results.csv"
        with open(self.csv_all,"w",newline="") as f:
            csv.writer(f).writerow([
                "global_step","prompt","K","hdr_percentile",
                "log_thr_cat","log_thr_dog","logp_mean_cat","logp_mean_dog",
                "inside_hcdr","decision"
            ])

    # ----- features -----
    @torch.no_grad()
    def clip_embed(self, images: List[Image.Image]) -> np.ndarray:
        inp = self.proc(images=images, return_tensors="pt").to(self.device)
        feats = self.clip.get_image_features(**inp).float().cpu().numpy()
        return feats

    # ----- build HCDR frame (once) -----
    def build_or_load_frame(self):
        pre_npz = self.cache / "preproc.npz"
        gmm_dir = self.cache / "gmms"; gmm_dir.mkdir(exist_ok=True)
        thr_json = self.cache / "thresholds.json"
        cal_npz = self.cache / "cal_ll.npz"

        if pre_npz.exists() and thr_json.exists() and cal_npz.exists() and all((gmm_dir/f"{c}.npz").exists() for c in self.labels):
            print("Loading HCDR frame from cache...")
            self.pre = FeaturePreproc.load(pre_npz)
            self.log_thr = json.loads(thr_json.read_text())
            cal = np.load(cal_npz, allow_pickle=True)
            self.cal_ll = {c: cal[c] for c in self.labels}
            for c in self.labels:
                blob = np.load(gmm_dir/f"{c}.npz", allow_pickle=True)
                gmm = GaussianMixture(n_components=int(blob["n_components"]),
                                      covariance_type="diag", reg_covar=float(blob["reg_covar"]))
                gmm.means_ = blob["means"]; gmm.covariances_ = blob["covs"]; gmm.weights_ = blob["weights"]
                gmm.precisions_cholesky_ = blob["prec_chol"]
                self.gmms[c]=gmm
            return

        print("Calibrating HCDR frame on AFHQ (one time)...")
        by_cls: Dict[str, List[Image.Image]] = {c:[] for c in self.labels}
        for img, lbl in self.ds: by_cls[lbl].append(img)

        raw_all, raw_by = [], {}
        for c in self.labels:
            feats_c = self.clip_embed(by_cls[c])
            raw_by[c]=feats_c; raw_all.append(feats_c)
        raw_all = np.concatenate(raw_all, axis=0)

        self.pre = FeaturePreproc(pca_dim=self.a.pca_dim).fit(raw_all)

        self.cal_ll={}
        for c in self.labels:
            feats = self.pre.transform(raw_by[c])
            N = len(feats); idx = np.arange(N); rng=np.random.RandomState(self.a.seed); rng.shuffle(idx)
            s = int(0.8*N); fit_idx, cal_idx = idx[:s], idx[s:]
            X_fit, X_cal = feats[fit_idx], feats[cal_idx]

            gmm = GaussianMixture(
                n_components=min(6, max(2, len(X_fit)//200)),
                covariance_type="diag", reg_covar=1e-3, max_iter=300, random_state=self.a.seed
            ).fit(X_fit)
            self.gmms[c]=gmm
            cal_ll = gmm.score_samples(X_cal)
            self.cal_ll[c]=cal_ll

        self.log_thr = {c: float(np.percentile(self.cal_ll[c], self.a.hdr_percentile)) for c in self.labels}

        # save cache
        self.pre.dump(pre_npz)
        np.savez(cal_npz, **self.cal_ll)
        thr_json.write_text(json.dumps(self.log_thr, indent=2))
        for c in self.labels:
            gmm = self.gmms[c]
            np.savez(gmm_dir/f"{c}.npz",
                     n_components=gmm.n_components, reg_covar=1e-3,
                     means=gmm.means_, covs=gmm.covariances_,
                     weights=gmm.weights_, prec_chol=gmm.precisions_cholesky_)

        # quick diag plot
        self._plot_calibration_hist()

    # ----- expectation estimator -----
    @torch.no_grad()
    def estimate_mean_embedding(self, pipe, prompt:str, K:int) -> Tuple[np.ndarray, np.ndarray]:
        imgs=[]
        for k in range(K):
            img = pipe(prompt,
                       num_inference_steps=self.a.sd_steps,
                       guidance_scale=self.a.guidance,
                       generator=torch.Generator(self.device).manual_seed(self.a.seed+k)
                       ).images[0]
            imgs.append(img)
        feats = self.clip_embed(imgs)
        feats_z = self.pre.transform(feats)
        return feats_z.mean(axis=0), feats_z, imgs
    

    # --- small helper to make a grid ---
    def _save_grid(self, images, path, cols=8):
        rows = (len(images) + cols - 1) // cols
        w, h = images[0].size
        canvas = Image.new("RGB", (cols*w, rows*h), (255,255,255))
        for i, im in enumerate(images):
            canvas.paste(im, ((i % cols)*w, (i // cols)*h))
        canvas.save(path, quality=95)

    # ----- HCDR membership -----
    def in_hcdr(self, vec_z: np.ndarray) -> Tuple[bool, Dict[str,float]]:
        per = {c: float(self.gmms[c].score_samples(vec_z[None,:])[0]) for c in self.labels}
        return any(per[c] > self.log_thr[c] for c in self.labels), per

    # ----- evaluate a single checkpoint -----
    def eval_checkpoint(self, unet_path: Path, step:int) -> Dict:
        dtype = next(self.base_pipe.unet.parameters()).dtype
        ft_unet = UNet2DConditionModel.from_pretrained(str(unet_path), torch_dtype=dtype).to(self.device)
        pipe = self.base_pipe
        pipe.unet = ft_unet

        ckpt_dir = self.outdir / f"eval_ckpt_{step}"
        ckpt_dir.mkdir(exist_ok=True)
        csv_ckpt = ckpt_dir / "hcdr_results.csv"
        with open(csv_ckpt,"w",newline="") as f:
            csv.writer(f).writerow([
                "prompt","K","hdr_percentile",
                "log_thr_cat","log_thr_dog","logp_mean_cat","logp_mean_dog",
                "inside_hcdr","decision"
            ])

        logs=[]
        for prompt in self.a.prompts:
            mhat, feats_k, imgs_k = self.estimate_mean_embedding(pipe, prompt, self.a.K)
            inside, scores = self.in_hcdr(mhat)
            decision = "IN HCDR" if inside else "OUTSIDE HCDR"
            row = dict(prompt=prompt, K=self.a.K, hdr_percentile=self.a.hdr_percentile,
                       log_thr_cat=self.log_thr["cat"], log_thr_dog=self.log_thr["dog"],
                       logp_mean_cat=scores["cat"], logp_mean_dog=scores["dog"],
                       inside_hcdr=int(inside), decision=decision)
            logs.append(row)
            with open(csv_ckpt,"a",newline="") as f: csv.writer(f).writerow(list(row.values()))
            with open(self.csv_all,"a",newline="") as f:
                csv.writer(f).writerow([step, prompt, self.a.K, self.a.hdr_percentile,
                                        self.log_thr["cat"], self.log_thr["dog"],
                                        scores["cat"], scores["dog"], int(inside), decision])
            np.save(ckpt_dir / f"sample_feats_{re.sub(r'[^a-zA-Z0-9]+','_',prompt)[:40]}.npy", feats_k)
            # save only the “hallucinating” case
            if not inside:
                safe_name = re.sub(r'[^a-zA-Z0-9]+','_', prompt)[:60]
                out_img_dir = ckpt_dir / "hallucinated_samples"
                out_img_dir.mkdir(exist_ok=True)
                # save a grid and individual frames
                self._save_grid(imgs_k, out_img_dir / f"{safe_name}_grid.png", cols=8)
                for i, im in enumerate(imgs_k):
                    im.save(out_img_dir / f"{safe_name}_{i:03d}.jpg", quality=92)

        # plots per checkpoint
        self._plot_prompt_scores(logs, ckpt_dir, step)

        # summary
        hall_rate = 1.0 - (sum([r["inside_hcdr"] for r in logs]) / max(1,len(logs)))
        (ckpt_dir/"summary.txt").write_text(f"δ-hallucination rate: {hall_rate:.4f}\n")
        return {"step": step, "rate": hall_rate}

    # ----- sweep over checkpoints -----
    def run(self):
        self.build_or_load_frame()

        # collect checkpoints
        ckpts = []
        for g in self.a.checkpoints_glob:
            for p in glob.glob(g):
                if os.path.isdir(p):
                    # allow either .../checkpoint-XXXX or the unet folder itself
                    if os.path.basename(p)=="unet":
                        step = int(re.findall(r"checkpoint-([0-9]+)", p)[0])
                        ckpts.append((Path(p), step))
                    else:
                        if (Path(p)/"unet").exists():
                            step = int(re.findall(r"checkpoint-([0-9]+)", p)[0])
                            ckpts.append((Path(p)/"unet", step))
        ckpts = sorted(list({(str(p),s) for p,s in ckpts}), key=lambda x: x[1])
        ckpts = [(Path(p), s) for p,s in ckpts]
        if not ckpts:
            raise RuntimeError("No checkpoints found. Pass e.g. --checkpoints_glob 'finetuned_sd_models/checkpoint-*/unet'")

        summary=[]
        for p, s in ckpts:
            print(f"\n== Evaluating checkpoint step {s} ==")
            summary.append(self.eval_checkpoint(p, s))

        # aggregate with loss if provided
        joined = self._aggregate_with_loss(summary)

        # plots across steps
        self._plot_rate_over_steps(joined)
        self._plot_loss_vs_rate(joined)

        # HTML report
        self._emit_report(joined)

    # ----- plotting -----
    def _plot_calibration_hist(self):
        fig = plt.figure(figsize=(10,5))
        for c in self.labels:
            plt.hist(self.cal_ll[c], bins=60, alpha=0.55, label=f"{c} cal log p")
            plt.axvline(self.log_thr[c], linestyle="--", linewidth=2, label=f"{c} thr@p{self.a.hdr_percentile}")
        plt.title("Calibration log-densities & HDR thresholds")
        plt.xlabel("log density"); plt.ylabel("count")
        plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout()
        path = self.outdir/"calibration_histograms.png"
        fig.savefig(path, dpi=220); plt.close(fig)

    def _plot_prompt_scores(self, logs: List[dict], ckpt_dir: Path, step:int):
        # scatter of m̂ log p vs thresholds, and pass/fail
        xs = np.arange(len(logs))
        names = [r["prompt"] if len(r["prompt"])<=40 else r["prompt"][:37]+"…" for r in logs]
        cat_ll = [r["logp_mean_cat"] for r in logs]
        dog_ll = [r["logp_mean_dog"] for r in logs]

        fig = plt.figure(figsize=(10,5))
        plt.scatter(xs, cat_ll, marker="o", label="log p(m̂|cat)")
        plt.scatter(xs, dog_ll, marker="s", label="log p(m̂|dog)")
        plt.axhline(self.log_thr["cat"], linestyle="--", label=f"cat thr@p{self.a.hdr_percentile}")
        plt.axhline(self.log_thr["dog"], linestyle="--", label=f"dog thr@p{self.a.hdr_percentile}")
        plt.xticks(xs, names, rotation=30, ha="right")
        plt.title(f"Step {step}: m̂ log densities vs thresholds")
        plt.ylabel("log density"); plt.grid(True,alpha=0.3); plt.legend(); plt.tight_layout()
        fig.savefig(ckpt_dir/"per_prompt_scores.png", dpi=220); plt.close(fig)

        fig = plt.figure(figsize=(8,3.8))
        pf = [1 if r["inside_hcdr"]==1 else 0 for r in logs]
        plt.bar(xs, pf)
        plt.xticks(xs, names, rotation=30, ha="right")
        plt.yticks([0,1], ["δ-hallucination","non-hallucinating"])
        plt.title(f"Step {step}: HCDR decision per prompt")
        plt.tight_layout()
        fig.savefig(ckpt_dir/"per_prompt_decision.png", dpi=220); plt.close(fig)

    def _aggregate_with_loss(self, summary: List[Dict]) -> List[Dict]:
        joined = {s["step"]:{**s, "loss": None} for s in summary}
        if self.a.train_loss_csv and Path(self.a.train_loss_csv).exists():
            import pandas as pd
            loss = pd.read_csv(self.a.train_loss_csv)  # columns: global_step, epoch, avg_loss
            for _, r in loss.iterrows():
                step = int(r["global_step"])
                if step in joined:
                    joined[step]["loss"] = float(r["avg_loss"])
            # forward/backfill if needed
            last = None
            for step in sorted(joined.keys()):
                if joined[step]["loss"] is None and last is not None:
                    joined[step]["loss"] = last
                elif joined[step]["loss"] is not None:
                    last = joined[step]["loss"]
        return [joined[k] for k in sorted(joined.keys())]

    def _plot_rate_over_steps(self, joined: List[Dict]):
        fig = plt.figure(figsize=(7,4))
        steps = [j["step"] for j in joined]; rates = [j["rate"] for j in joined]
        plt.plot(steps, rates, marker="o")
        plt.xlabel("global_step"); plt.ylabel("δ-hallucination rate")
        plt.title("Expectation outside HCDR over training")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        fig.savefig(self.outdir/"rate_over_steps.png", dpi=220); plt.close(fig)

    def _plot_loss_vs_rate(self, joined: List[Dict]):
        xs=[]; ys=[]; labels=[]
        for j in joined:
            if j["loss"] is not None:
                xs.append(j["loss"]); ys.append(j["rate"]); labels.append(j["step"])
        if not xs: return
        rho, p = spearmanr(xs, ys)
        fig = plt.figure(figsize=(7,5))
        plt.scatter(xs, ys)
        for x,y,l in zip(xs,ys,labels):
            plt.text(x, y, str(l), fontsize=8, ha='left', va='bottom')
        plt.xlabel("training loss (avg)"); plt.ylabel("δ-hallucination rate")
        plt.title(f"Loss vs HCDR (Spearman ρ={rho:.2f}, p={p:.3g})")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        fig.savefig(self.outdir/"loss_vs_hcdr.png", dpi=220); plt.close(fig)
        (self.outdir/"loss_vs_hcdr_stats.json").write_text(json.dumps({"spearman_rho":rho, "p_value":p}, indent=2))

    def _emit_report(self, joined: List[Dict]):
        # simple standalone HTML
        html = []
        html.append("<html><head><meta charset='utf-8'><title>HCDR Report</title>")
        html.append("<style>body{font-family:Inter,system-ui,Arial;margin:24px} img{max-width:100%} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px}</style>")
        html.append("</head><body>")
        html.append("<h1>HCDR Report (AFHQ cats/dogs)</h1>")
        html.append(f"<p>HDR percentile = <b>{self.a.hdr_percentile}</b> (mass ≈ {100 - self.a.hdr_percentile:.0f}%). K (samples for expectation) = <b>{self.a.K}</b>.</p>")
        html.append("<h2>Calibration</h2>")
        html.append(f"<img src='{b64img(self.outdir/'calibration_histograms.png')}'/>")
        html.append("<h2>Across checkpoints</h2>")
        if (self.outdir/"rate_over_steps.png").exists():
            html.append(f"<img src='{b64img(self.outdir/'rate_over_steps.png')}'/>")
        if (self.outdir/"loss_vs_hcdr.png").exists():
            html.append(f"<img src='{b64img(self.outdir/'loss_vs_hcdr.png')}'/>")
        html.append("<h2>Per-checkpoint artifacts</h2><ul>")
        for d in sorted(self.outdir.glob("eval_ckpt_*")):
            if not d.is_dir(): continue
            step = d.name.split("_")[-1]
            html.append(f"<li><h3>Checkpoint step {step}</h3>")
            for name in ("per_prompt_scores.png","per_prompt_decision.png"):
                p=d/name
                if p.exists(): html.append(f"<img src='{b64img(p)}'/>")
            if (d/"hcdr_results.csv").exists():
                html.append(f"<p><a href='{d/'hcdr_results.csv'}'>hcdr_results.csv</a></p>")
            if (d/"summary.txt").exists():
                html.append("<pre>"+(d/"summary.txt").read_text()+"</pre>")
            html.append("</li>")
        html.append("</ul>")
        html.append("</body></html>")
        (self.outdir/"report.html").write_text("\n".join(html), encoding="utf-8")

# --------------- args & main ---------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./hcdr_afhq_results/report")
    p.add_argument("--data_root", type=str, default="./afhq_data")
    p.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--fp16", action="store_true", help="use fp16 for diffusion when on CUDA")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--max_per_class", type=int, default=2000)
    p.add_argument("--pca_dim", type=int, default=128)
    p.add_argument("--hdr_percentile", type=float, default=10.0)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--sd_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts", nargs="+", default=[
        "a realistic photo of a friendly dog",
        "a fluffy cat sitting on a sofa",
        "a cute pet animal"
    ])
    p.add_argument("--checkpoints_glob", nargs="+", required=True, default="./runs/sd_ft/checkpoint-*/unet",
                   help="Glob(s) to checkpoint folders (either .../checkpoint-*/unet or .../checkpoint-*)")
    p.add_argument("--train_loss_csv", type=str, default="./runs/sd_ft/training_log.csv",
                   help="Optional path to your training loss CSV (columns: global_step,epoch,avg_loss)")
    return p.parse_args()

def main():
    args = parse_args()
    ev = HCDREvaluator(args)
    ev.run()
    print(f"\nAll done. Open: {ev.outdir/'report.html'}")

if __name__ == "__main__":
    main()
