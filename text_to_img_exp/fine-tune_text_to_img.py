#!/usr/bin/env python3
"""
Fine-tune the UNet of Stable Diffusion v1-5 on AFHQ (cats/dogs) with
Option A captions (class-generic, mismatch-free) + CFG dropout.

Saves:
  - baseline: <output_dir>/checkpoint-0000/unet/
  - checkpoints: <output_dir>/checkpoint-<global_step>/unet/
  - final: <output_dir>/final_model/unet/
  - training log: <output_dir>/train_loss.csv  (global_step, epoch, avg_loss)
"""

import os, csv, math, random, argparse, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

warnings.filterwarnings("ignore")

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def exists(x): return x is not None


# -------------------------------------
# Option A: class-generic captions
# -------------------------------------

CAPTION_TEMPLATES = [
    "a photo of a {cls}",
    "a high quality photo of a {cls}",
    "a realistic photo of a {cls}",
    "a close-up photo of a {cls}",
    "{cls}",
]

class AFHQCatsDogs(Dataset):
    """
    AFHQ (huggan/AFHQ) materialized to disk; only {cat, dog}. 
    Option A captions: always class-correct; deterministic template per image.
    Includes classifier-free guidance dropout via empty captions with prob p_uncond.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 512,
        max_per_class: int = None,
        p_uncond: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.p_uncond = float(p_uncond)
        self.seed = int(seed)

        self.items: List[Tuple[str, str]] = []
        self._prepare_if_needed(max_per_class)

        self.tx = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.LANCZOS),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),                                   # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # -> [-1,1]
        ])

    def _prepare_if_needed(self, max_per_class: int):
        split_dir = self.root / self.split
        if not split_dir.exists() or not any(split_dir.iterdir()):
            print("Downloading AFHQ (huggan/AFHQ) and materializing...")
            ds = load_dataset("huggan/AFHQ", split=self.split)  # single 'train' split
            names = ds.features["label"].names                  # ['cat','dog','wild']
            (split_dir/"cat").mkdir(parents=True, exist_ok=True)
            (split_dir/"dog").mkdir(parents=True, exist_ok=True)

            ctr = {"cat": 0, "dog": 0}
            for row in ds:
                name = names[int(row["label"])]
                if name not in ("cat", "dog"):
                    continue
                if max_per_class is not None and ctr[name] >= max_per_class:
                    continue
                img: Image.Image = row["image"]
                out = split_dir / name / f"{name}_{ctr[name]:06d}.jpg"
                img.save(out, quality=95)
                ctr[name] += 1

        # index files
        for cls in ("cat", "dog"):
            cls_dir = split_dir / cls
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    self.items.append((str(f), cls))

    def __len__(self): 
        return len(self.items)

    def _det_template(self, path: str, cls: str) -> str:
        # deterministic template index per image to avoid caption flip-flop
        # stay within range of CAPTION_TEMPLATES
        idx = (hash(path) ^ hash(cls) ^ self.seed) % len(CAPTION_TEMPLATES)
        return CAPTION_TEMPLATES[idx].format(cls=cls)

    def __getitem__(self, idx):
        p, cls = self.items[idx]
        img = Image.open(p).convert("RGB")
        img = self.tx(img)

        # classifier-free guidance dropout (empty caption)
        if random.random() < self.p_uncond:
            caption = ""
        else:
            caption = self._det_template(p, cls)

        return img, caption


# ---------------------------
# Training config
# ---------------------------

@dataclass
class TrainConfig:
    sd_model: str = "runwayml/stable-diffusion-v1-5"
    output_dir: str = "runs/sd_ft"
    seed: int = 42

    image_size: int = 512
    max_per_class: int = None       # cap AFHQ per class; None = all
    p_uncond: float = 0.1           # prob of empty caption
    num_workers: int = 4

    # Optim / schedule
    train_batch_size: int = 8
    grad_accum_steps: int = 1
    num_epochs: int = 3
    lr: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Checkpointing / logging
    save_steps: int = 1000
    log_every: int = 50

    # Precision
    use_fp16: bool = True


# ---------------------------
# Trainer
# ---------------------------

class UNetFineTuner:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load SD components ---
        print("Loading Stable Diffusion v1-5 components...")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.sd_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.sd_model, subfolder="text_encoder").to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        self.vae = AutoencoderKL.from_pretrained(cfg.sd_model, subfolder="vae").to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.unet = UNet2DConditionModel.from_pretrained(cfg.sd_model, subfolder="unet").to(self.device)
        self.unet.train()
        self.unet.enable_gradient_checkpointing()

        self.noise_scheduler = DDPMScheduler.from_pretrained(cfg.sd_model, subfolder="scheduler")

        # --- Data ---
        self.ds = AFHQCatsDogs(
            root="./afhq_data",
            split="train",
            image_size=cfg.image_size,
            max_per_class=cfg.max_per_class,
            p_uncond=cfg.p_uncond,
            seed=cfg.seed,
        )
        self.loader = DataLoader(
            self.ds,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # --- Optimizer & AMP ---
        self.opt = torch.optim.AdamW(
            self.unet.parameters(),
            lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, eps=cfg.eps
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and cfg.use_fp16))

        # --- Output & logging ---
        self.outdir = Path(cfg.output_dir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.loss_csv = self.outdir / "train_loss.csv"
        with open(self.loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["global_step", "epoch", "avg_loss"])

        # Save baseline UNet as "checkpoint-0000"
        self._save_unet_checkpoint(step=0)

    @torch.no_grad()
    def _encode_text(self, captions: List[str]) -> torch.Tensor:
        ids = self.tokenizer(
            captions, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).input_ids.to(self.device)
        enc = self.text_encoder(ids)[0]  # (B, 77, 768)
        return enc

    @torch.no_grad()
    def _encode_vae(self, pixels: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(pixels).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def train(self):
        print(f"Dataset size: {len(self.ds)} | batch: {self.cfg.train_batch_size}")
        global_step = 0

        for epoch in range(self.cfg.num_epochs):
            epoch_losses = []

            for i, (imgs, captions) in enumerate(self.loader):
                imgs = imgs.to(self.device, non_blocking=True)

                with torch.no_grad():
                    text_emb = self._encode_text(list(captions))
                    latents = self._encode_vae(imgs)

                # Sample noise & timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # ε-prediction MSE loss
                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda" and self.cfg.use_fp16)):
                    noise_pred = self.unet(noisy, timesteps, encoder_hidden_states=text_emb).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") / self.cfg.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (i + 1) % self.cfg.grad_accum_steps == 0:
                    # step
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % self.cfg.log_every == 0:
                        print(f"[epoch {epoch+1}] step {global_step} loss {loss.item()*self.cfg.grad_accum_steps:.6f}")

                    if global_step % self.cfg.save_steps == 0:
                        self._save_unet_checkpoint(step=global_step)

                epoch_losses.append(loss.item() * self.cfg.grad_accum_steps)

            # end epoch: log avg
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            with open(self.loss_csv, "a", newline="") as f:
                csv.writer(f).writerow([global_step, epoch + 1, avg_loss])
            print(f"Epoch {epoch+1}/{self.cfg.num_epochs} avg loss: {avg_loss:.6f}")

        # final save
        self._save_final()

    def _save_unet_checkpoint(self, step: int):
        ck = self.outdir / f"checkpoint-{step}" / "unet"
        ck.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(ck)
        print(f"Saved UNet checkpoint → {ck}")

    def _save_final(self):
        fin = self.outdir / "final_model" / "unet"
        fin.mkdir(parents=True, exist_ok=True)
        self.unet.save_pretrained(fin)
        print(f"Saved final UNet → {fin}")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="runs/sd_ft")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--p_uncond", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--use_fp16", action="store_true")

    args = p.parse_args()
    return TrainConfig(
        sd_model=args.sd_model,
        output_dir=args.output_dir,
        seed=args.seed,
        image_size=args.image_size,
        max_per_class=args.max_per_class,
        p_uncond=args.p_uncond,
        num_workers=args.num_workers,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        lr=args.lr,
        save_steps=args.save_steps,
        log_every=args.log_every,
        use_fp16=args.use_fp16,
    )


if __name__ == "__main__":
    cfg = parse_args()
    tuner = UNetFineTuner(cfg)
    tuner.train()
