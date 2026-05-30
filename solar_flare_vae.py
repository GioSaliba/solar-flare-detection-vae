"""
Solar Flare Detection with a Spatio-Temporal VAE
================================================

A hybrid ConvLSTM + Variational Autoencoder for detecting rare solar flares
from NASA SDO/AIA multi-channel EUV imagery.

The model combines four ideas:
  * Spatial attention   - learns where to look within each frame.
  * ConvLSTM encoder     - captures how active regions evolve over time.
  * Temporal attention   - weights the most informative frames in a sequence.
  * A variational latent  - learns a compact "normal Sun" representation.

Detection is then driven by a Regulator block that fuses three complementary
anomaly signals into a single flare score:
    1. reconstruction error      - how poorly the VAE reproduces the input,
    2. latent Mahalanobis distance - how far the sequence sits from the normal
                                    Sun latent distribution,
    3. a supervised classifier probability.

NOTE ON DATA
------------
The NASA SDO/AIA imagery is NOT included in this repository (it is large and
is distributed by NASA, not by this project). See the README for how to obtain
it and for the expected CSV / folder layout. Paths are passed on the command
line, so nothing machine-specific is hard-coded here.

USAGE
-----
    # Load saved backbone weights and run the full evaluation pipeline:
    python solar_flare_vae.py --csv path/to/labels.csv --data-dir path/to/data

    # Train the VAE backbone from scratch first:
    python solar_flare_vae.py --csv path/to/labels.csv --data-dir path/to/data --train
"""

import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
def set_seed(seed: int = 42):
    """Fix the random seed across Python, NumPy and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# 1. LOGGING
# ============================================================
class TeeLogger:
    """Mirror everything printed to stdout into a log file as well."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ============================================================
# 2. CONFIGURATION (hyper-parameters; paths come from argparse)
# ============================================================
SEED = 42

# SDO/AIA channels (wavelengths in Angstrom) used as input "colour" channels.
CHANNELS = ["94A", "131A", "171A", "193A"]

# Image / model dimensions.
IMG_SIZE = 64       # each frame is resized to IMG_SIZE x IMG_SIZE
SEQ_LEN = 10        # frames per sequence (~1 hour at a 6-minute cadence)
LATENT_DIM = 256
HIDDEN_DIM = 64

# Backbone training.
BATCH_SIZE = 32
MAX_EPOCHS = 25
LR = 2e-4
KLD_WEIGHT = 1e-4   # weight on the VAE KL-divergence term

# Supervised calibration of the heads.
CAL_EPOCHS = 5
N_QUIET_CAL = 500   # quiet sequences sampled as negatives during calibration
N_BASELINE = 600    # quiet sequences used to model the "normal Sun" latent space

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "solar_flare_vae_weights.pth"
MAX_WINDOW_SECONDS = 3600   # a valid sequence must span at most one hour


# ============================================================
# 3. DATA PIPELINE
# ============================================================
class SolarSequenceDataset(Dataset):
    """
    Builds fixed-length sequences of multi-channel solar images from a CSV.

    The CSV must contain at least:
        dir         - folder holding one <channel>.npy file per channel
        timestamp   - capture time of that folder's frame
        label       -  1 = flare, 0 = quiet, -1 = unusable / excluded
        flare_type  - e.g. "M", "X"  (optional; NaN for quiet frames)

    A sequence is SEQ_LEN consecutive frames spanning at most one hour. It is
    labelled "flare" (1) if any frame in the window is a flare, and is dropped
    entirely if it contains an unusable (-1) frame.

    If `data_dir` is given, frame folders are looked up as
    `data_dir/<folder_basename>`, which makes the dataset portable across
    machines regardless of the absolute paths stored in the CSV.
    """

    def __init__(self, csv_path, data_dir=None):
        self.df = pd.read_csv(csv_path)
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        if data_dir is not None:
            self.path_map = {
                os.path.basename(d): os.path.join(data_dir, os.path.basename(d))
                for d in self.df["dir"]
            }
        else:
            self.path_map = (
                self.df.set_index(self.df["dir"].apply(os.path.basename))["dir"].to_dict()
            )

        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self):
        sequences = []
        df_sorted = self.df.sort_values("timestamp").reset_index(drop=True)
        for i in range(len(df_sorted) - SEQ_LEN):
            window = df_sorted.iloc[i: i + SEQ_LEN]
            span = (window["timestamp"].iloc[-1]
                    - window["timestamp"].iloc[0]).total_seconds()
            if span > MAX_WINDOW_SECONDS:        # skip non-contiguous windows
                continue
            if -1 in window["label"].values:     # skip windows with bad frames
                continue
            label = 1 if 1 in window["label"].values else 0
            flare_types = window[window["flare_type"].notna()]["flare_type"].values
            sequences.append({
                "dirs": [os.path.basename(d) for d in window["dir"]],
                "label": label,
                "type": flare_types[0] if len(flare_types) > 0 else "Quiet",
            })
        return sequences

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, folder):
        """Load and normalise one multi-channel frame -> tensor (C, H, W)."""
        channel_imgs = []
        for c in CHANNELS:
            raw = np.load(os.path.join(folder, f"{c}.npy"))
            resized = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
            # Solar intensities span many orders of magnitude, so log-scale
            # first, then squash into roughly [0, 1].
            normed = np.clip(np.log10(resized + 1.0) / 4.0, 0, 1)
            channel_imgs.append(torch.from_numpy(normed))
        return torch.stack(channel_imgs)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        frames = [self._load_frame(self.path_map[d]) for d in item["dirs"]]
        sequence = torch.stack(frames)           # (SEQ_LEN, C, H, W)
        return sequence.float(), item["label"], item["type"]


# ============================================================
# 4. MODEL ARCHITECTURE
# ============================================================
class TemporalAttention(nn.Module):
    """Learn a weight for each timestep, then return their weighted sum."""

    def __init__(self, feature_dim):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)

    def forward(self, x):                          # x: (B, T, feature_dim)
        weights = F.softmax(self.score(x), dim=1)  # (B, T, 1)
        context = torch.sum(x * weights, dim=1)    # (B, feature_dim)
        return context, weights


class RegulatorBlock(nn.Module):
    """
    Tiny network that turns the three anomaly signals into three positive
    fusion weights. Softplus guarantees non-negative weights. The bias is
    initialised to give the latent-distance signal (feature index 1) a high
    starting weight, since it is usually the most reliable flare indicator.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 3)
        )
        with torch.no_grad():
            self.net[2].bias.fill_(0.0)
            self.net[2].bias[1] = 4.6              # index 1 == Mahalanobis distance

    def forward(self, x):                          # x: (B, 3)
        return F.softplus(self.net(x))             # (B, 3), all > 0


class SolarFlareVAE(nn.Module):
    """Spatio-temporal VAE with spatial + temporal attention and two heads."""

    def __init__(self):
        super().__init__()
        flat_dim = HIDDEN_DIM * IMG_SIZE * IMG_SIZE
        n_channels = len(CHANNELS)

        # --- Encoder ---
        self.spatial_attn = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.enc_cell = nn.Conv2d(n_channels + HIDDEN_DIM, 4 * HIDDEN_DIM, 3, padding=1)
        self.temporal_attn = TemporalAttention(flat_dim)
        self.fc_mu = nn.Linear(flat_dim, LATENT_DIM)
        self.fc_logvar = nn.Linear(flat_dim, LATENT_DIM)

        # --- Heads ---
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.regulator = RegulatorBlock()

        # --- Decoder ---
        self.dec_fc = nn.Linear(LATENT_DIM, flat_dim)
        self.dec_cell = nn.Conv2d(2 * HIDDEN_DIM, 4 * HIDDEN_DIM, 3, padding=1)
        self.out_conv = nn.Conv2d(HIDDEN_DIM, n_channels, 1)

    def encode(self, x):
        """x: (B, T, C, H, W) -> (mu, logvar)."""
        b, t, c, h, w = x.shape
        ht = torch.zeros(b, HIDDEN_DIM, h, w, device=x.device)
        ct = torch.zeros(b, HIDDEN_DIM, h, w, device=x.device)

        frame_states = []
        for i in range(t):
            frame = x[:, i]
            # Spatial attention: pool across channels, learn where to look.
            avg_pool = torch.mean(frame, dim=1, keepdim=True)
            max_pool, _ = torch.max(frame, dim=1, keepdim=True)
            attn = torch.sigmoid(self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1)))
            frame = frame * attn

            # One ConvLSTM step.
            gates = self.enc_cell(torch.cat([frame, ht], dim=1))
            i_g, f_g, g_g, o_g = torch.split(gates, HIDDEN_DIM, dim=1)
            ct = torch.sigmoid(f_g) * ct + torch.sigmoid(i_g) * torch.tanh(g_g)
            ht = torch.sigmoid(o_g) * torch.tanh(ct)
            frame_states.append(ht.view(b, -1))

        # Temporal attention across all timesteps.
        context, _ = self.temporal_attn(torch.stack(frame_states, dim=1))
        return self.fc_mu(context), self.fc_logvar(context)

    def reparameterise(self, mu, logvar):
        """Sample z during training; use the mean during eval (deterministic)."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        """z: (B, LATENT_DIM) -> reconstruction (B, T, C, H, W)."""
        ht = self.dec_fc(z).view(z.size(0), HIDDEN_DIM, IMG_SIZE, IMG_SIZE)
        ct = torch.zeros_like(ht)
        outputs = []
        for _ in range(SEQ_LEN):
            gates = self.dec_cell(torch.cat([ht, ht], dim=1))
            i_g, f_g, g_g, o_g = torch.split(gates, HIDDEN_DIM, dim=1)
            ct = torch.sigmoid(f_g) * ct + torch.sigmoid(i_g) * torch.tanh(g_g)
            ht = torch.sigmoid(o_g) * torch.tanh(ct)
            outputs.append(self.out_conv(ht))
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        reconstruction = self.decode(z)
        logits = self.classifier(mu)               # classifier uses the mean, not z
        return reconstruction, mu, logvar, logits


# ============================================================
# 5. ANOMALY SIGNALS & FUSION
# ============================================================
def reconstruction_error(x, reconstruction):
    """Per-sample mean-squared reconstruction error, scaled for readability."""
    return torch.mean((x - reconstruction) ** 2, dim=(1, 2, 3, 4)) * 1000.0


def mahalanobis_distance(z, mean, inv_cov):
    """
    Vectorised Mahalanobis distance of each row of z from `mean`.
        z: (B, D) tensor, mean: (D,) tensor, inv_cov: (D, D) tensor.
    """
    diff = z - mean
    dist_sq = torch.sum((diff @ inv_cov) * diff, dim=1).clamp(min=0)
    return torch.sqrt(dist_sq)


def fuse_features(model, x, stats):
    """
    Run the model and produce the three standardised anomaly signals plus the
    single fused flare score.

    IMPORTANT: the feature order is fixed as
        [ reconstruction error , Mahalanobis distance , classifier probability ]
    and the fused score is simply weight_i * feature_i summed over i. The exact
    same features and the exact same fusion are used during calibration,
    validation and test, so the Regulator always weights the signal it was
    trained on. (In an earlier version the calibration and inference stages fed
    the Regulator different signals; that inconsistency is fixed here.)

    Reconstruction error and distance are standardised using statistics of the
    quiet Sun, so the learned weights are meaningful rather than dominated by
    whichever raw signal happens to have the largest numerical scale.

    Returns (features, fused_score, logits).
    """
    reconstruction, mu, _, logits = model(x)
    mse = reconstruction_error(x, reconstruction)
    dist = mahalanobis_distance(mu, stats["mean"], stats["inv_cov"])
    prob = torch.sigmoid(logits).squeeze(1)

    mse_z = (mse - stats["mse_mean"]) / stats["mse_std"]
    dist_z = (dist - stats["maha_mean"]) / stats["maha_std"]

    features = torch.stack([mse_z, dist_z, prob], dim=1)   # (B, 3)
    weights = model.regulator(features)                    # (B, 3)
    fused = (weights * features).sum(dim=1)                # (B,)
    return features, fused, logits


@torch.no_grad()
def compute_normal_statistics(model, dataset, quiet_indices):
    """
    Characterise the "normal Sun" using quiet sequences only. Returns the
    statistics needed to (a) measure latent Mahalanobis distance and
    (b) standardise the reconstruction-error and distance signals.
    """
    model.eval()
    latents, recon_errors = [], []
    for idx in tqdm(quiet_indices, desc="Modelling normal Sun"):
        x, _, _ = dataset[idx]
        x = x.unsqueeze(0).to(DEVICE)
        reconstruction, mu, _, _ = model(x)
        latents.append(mu.cpu().numpy())
        recon_errors.append(reconstruction_error(x, reconstruction).item())

    latents = np.concatenate(latents, axis=0)
    mean = np.mean(latents, axis=0)
    # Regularised pseudo-inverse covariance for a numerically stable distance.
    cov = np.cov(latents, rowvar=False) + np.eye(LATENT_DIM) * 1e-6
    inv_cov = np.linalg.pinv(cov)

    mean_t = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
    inv_cov_t = torch.tensor(inv_cov, dtype=torch.float32, device=DEVICE)

    # Distance of each quiet sample from the quiet mean -> distance scale.
    latents_t = torch.tensor(latents, dtype=torch.float32, device=DEVICE)
    quiet_dist = mahalanobis_distance(latents_t, mean_t, inv_cov_t).cpu().numpy()

    return {
        "mean": mean_t,
        "inv_cov": inv_cov_t,
        "mse_mean": float(np.mean(recon_errors)),
        "mse_std": float(np.std(recon_errors) + 1e-6),
        "maha_mean": float(np.mean(quiet_dist)),
        "maha_std": float(np.std(quiet_dist) + 1e-6),
    }


# ============================================================
# 6. TRAINING STAGES
# ============================================================
def train_backbone(model, dataset, train_indices):
    """Unsupervised VAE training on quiet sequences (reconstruction + KL)."""
    loader = DataLoader(Subset(dataset, train_indices),
                        batch_size=BATCH_SIZE, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)

    use_amp = (DEVICE.type == "cuda")             # mixed precision only on GPU
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(MAX_EPOCHS):
        for x, _, _ in tqdm(loader, desc=f"Backbone epoch {epoch + 1}/{MAX_EPOCHS}"):
            x = x.to(DEVICE)
            optimiser.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                reconstruction, mu, logvar, _ = model(x)
                recon_loss = F.mse_loss(reconstruction, x)
                kld = -0.5 * torch.mean(
                    torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                )
                loss = recon_loss + KLD_WEIGHT * kld
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved backbone weights to {WEIGHTS_PATH}")


def calibrate(model, dataset, quiet_indices, flare_indices, stats):
    """
    Supervised fine-tuning of ONLY the classifier and Regulator heads; the VAE
    backbone stays frozen. A class-balanced sampler compensates for the rarity
    of flares.
    """
    cal_quiet = random.sample(quiet_indices, min(N_QUIET_CAL, len(quiet_indices)))
    cal_indices = cal_quiet + flare_indices
    sample_weights = [
        2.0 if dataset.sequences[i]["label"] == 1 else 0.5 for i in cal_indices
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
    loader = DataLoader(Subset(dataset, cal_indices), batch_size=BATCH_SIZE, sampler=sampler)

    # Freeze the backbone; train only the two heads.
    for name, p in model.named_parameters():
        p.requires_grad = ("classifier" in name) or ("regulator" in name)

    params = list(model.classifier.parameters()) + list(model.regulator.parameters())
    optimiser = torch.optim.Adam(params, lr=2e-4)

    # eval() keeps the latent deterministic (no VAE sampling) while still
    # allowing gradients to flow into the two trainable heads.
    model.eval()
    for epoch in range(CAL_EPOCHS):
        for x, label, _ in tqdm(loader, desc=f"Calibration epoch {epoch + 1}/{CAL_EPOCHS}"):
            x = x.to(DEVICE)
            label = label.to(DEVICE).float()
            optimiser.zero_grad()
            _, fused, logits = fuse_features(model, x, stats)
            # Two complementary losses: train the classifier head directly, and
            # train the Regulator to make the fused score separate the classes.
            loss = (
                F.binary_cross_entropy_with_logits(logits.squeeze(1), label)
                + F.binary_cross_entropy_with_logits(fused, label)
            )
            loss.backward()
            optimiser.step()


# ============================================================
# 7. SCORING & EVALUATION
# ============================================================
@torch.no_grad()
def score_indices(model, dataset, indices, stats):
    """Compute the fused flare score for every sequence in `indices`."""
    model.eval()
    rows = []
    for idx in tqdm(indices, desc="Scoring"):
        x, label, flare_type = dataset[idx]
        x = x.unsqueeze(0).to(DEVICE)
        _, fused, _ = fuse_features(model, x, stats)
        rows.append({"score": fused.item(), "label": label, "type": flare_type})
    return pd.DataFrame(rows)


def select_threshold(val_df):
    """
    Choose the operating threshold that maximises a blend of F1 and accuracy,
    using the VALIDATION set only. The chosen threshold is then applied
    unchanged to the held-out test set, so the reported test metrics are not
    tuned on the data they are measured on.
    """
    search_space = np.linspace(val_df["score"].min(), val_df["score"].max(), 200)
    best_blend, best_thresh = -1.0, float(search_space[0])
    for t in search_space:
        preds = (val_df["score"] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            val_df["label"], preds, labels=[0, 1]
        ).ravel()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        _, _, f1, _ = precision_recall_fscore_support(
            val_df["label"], preds, average="binary", zero_division=0
        )
        blend = 0.6 * f1 + 0.4 * acc
        if blend > best_blend:
            best_blend, best_thresh = blend, float(t)
    return best_thresh


def evaluate(test_df, threshold, warning_threshold):
    """Print final test metrics and save the summary figures."""
    auc = roc_auc_score(test_df["label"], test_df["score"])
    preds = (test_df["score"] >= threshold).astype(int)
    cm = confusion_matrix(test_df["label"], preds, labels=[0, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_df["label"], preds, average="binary", zero_division=0
    )
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
    fpr, tpr, _ = roc_curve(test_df["label"], test_df["score"])

    print("\n" + "=" * 44)
    print("            TEST-SET PERFORMANCE")
    print("=" * 44)
    print(f"  ROC-AUC           : {auc:.4f}   (threshold-free)")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1-score          : {f1:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Threshold (val.)  : {threshold:.4f}")
    print("-" * 44)
    print("  Confusion matrix:")
    print(f"     TN={cm[0, 0]:<6} FP={cm[0, 1]:<6}")
    print(f"     FN={cm[1, 0]:<6} TP={cm[1, 1]:<6}")
    print("=" * 44)

    # --- Summary figure: ROC + confusion matrix + score distribution ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1, 3, 2)
    ConfusionMatrixDisplay(cm, display_labels=["Quiet", "Flare"]).plot(
        ax=plt.gca(), cmap="Blues", colorbar=False
    )
    plt.title("Confusion Matrix")

    plt.subplot(1, 3, 3)
    plt.hist(test_df[test_df["label"] == 0]["score"], bins=20,
             alpha=0.5, density=True, label="Quiet")
    plt.hist(test_df[test_df["label"] == 1]["score"], bins=20,
             alpha=0.5, density=True, label="Flare")
    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.axvline(warning_threshold, color="orange", linestyle=":", label="2-sigma warning")
    plt.xlabel("Fused flare score")
    plt.title("Score Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig("performance_summary.png", dpi=150)
    plt.close()

    # --- Flare-class separation (Quiet vs M vs X) ---
    plt.figure(figsize=(8, 5))
    for flare_class in ["Quiet", "M", "X"]:
        subset = test_df[test_df["type"] == flare_class]
        if not subset.empty:
            plt.hist(subset["score"], bins=20, alpha=0.5,
                     density=True, label=f"{flare_class}-class")
    plt.xlabel("Fused flare score")
    plt.ylabel("Density")
    plt.title("Score by Flare Class")
    plt.legend()
    plt.savefig("flare_class_separation.png", dpi=150)
    plt.close()


@torch.no_grad()
def save_anomaly_heatmaps(model, dataset, flare_indices, channel=2):
    """
    For a few flare sequences, plot the per-frame reconstruction error of one
    channel (default 171A) as a heatmap, visualising where and when the model
    finds the input anomalous.
    """
    model.eval()
    for n, idx in enumerate(flare_indices):
        x, _, _ = dataset[idx]
        x_in = x.unsqueeze(0).to(DEVICE)
        reconstruction, _, _, _ = model(x_in)
        diff = torch.abs(x[:, channel] - reconstruction[0, :, channel].cpu()).numpy()

        fig, axes = plt.subplots(1, SEQ_LEN, figsize=(2.4 * SEQ_LEN, 3))
        fig.suptitle(
            f"Temporal anomaly evolution - flare sample {n} "
            f"(channel {CHANNELS[channel]})",
            y=1.05,
        )
        im = None
        for t in range(SEQ_LEN):
            im = axes[t].imshow(diff[t], cmap="jet")
            axes[t].set_title(f"t={t * 6} min")
            axes[t].axis("off")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                     label="Reconstruction error")
        plt.savefig(f"anomaly_heatmap_flare_{n}.png", bbox_inches="tight", dpi=150)
        plt.close()


# ============================================================
# 8. PIPELINE
# ============================================================
def run(args):
    set_seed(SEED)

    dataset = SolarSequenceDataset(args.csv, data_dir=args.data_dir)
    n_flare = sum(s["label"] == 1 for s in dataset.sequences)
    n_quiet = sum(s["label"] == 0 for s in dataset.sequences)
    print(f"Built {len(dataset)} sequences ({n_flare} flare, {n_quiet} quiet).")

    quiet_idx = [i for i, s in enumerate(dataset.sequences) if s["label"] == 0]
    flare_idx = [i for i, s in enumerate(dataset.sequences) if s["label"] == 1]

    # ---- Disjoint splits ----
    # Quiet sequences are abundant -> 70 / 15 / 15  train / val / test.
    quiet_train, quiet_tmp = train_test_split(quiet_idx, test_size=0.30, random_state=SEED)
    quiet_val, quiet_test = train_test_split(quiet_tmp, test_size=0.50, random_state=SEED)
    # Flares are rare -> split into calibration / validation / test.
    flare_trainval, flare_test = train_test_split(flare_idx, test_size=0.40, random_state=SEED)
    flare_cal, flare_val = train_test_split(flare_trainval, test_size=0.25, random_state=SEED)

    model = SolarFlareVAE().to(DEVICE)

    # ---- Phase 1: VAE backbone (unsupervised) ----
    if args.train:
        print("\nPhase 1: training VAE backbone (unsupervised, quiet only)")
        train_backbone(model, dataset, quiet_train)
    else:
        if not os.path.exists(WEIGHTS_PATH):
            sys.exit(f"Weights file '{WEIGHTS_PATH}' not found. Run with --train first.")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"\nLoaded backbone weights from {WEIGHTS_PATH}")

    # ---- Phase 2: normal-Sun statistics (distance + standardisation scales) ----
    print("\nPhase 2: modelling the normal-Sun latent distribution")
    baseline_quiet = random.sample(quiet_train, min(N_BASELINE, len(quiet_train)))
    stats = compute_normal_statistics(model, dataset, baseline_quiet)

    # ---- Phase 3: supervised calibration of the two heads ----
    print("\nPhase 3: calibrating classifier + Regulator heads")
    calibrate(model, dataset, quiet_train, flare_cal, stats)

    # ---- Phase 4: threshold selection on the validation set ----
    print("\nPhase 4: scoring validation set and selecting threshold")
    val_df = score_indices(model, dataset, quiet_val + flare_val, stats)
    threshold = select_threshold(val_df)
    quiet_val_scores = val_df[val_df["label"] == 0]["score"]
    warning_threshold = quiet_val_scores.mean() + 2 * quiet_val_scores.std()

    # ---- Phase 5: final evaluation on the held-out test set ----
    print("\nPhase 5: evaluating on the held-out test set")
    test_df = score_indices(model, dataset, quiet_test + flare_test, stats)
    evaluate(test_df, threshold, warning_threshold)

    # ---- Qualitative figure ----
    save_anomaly_heatmaps(model, dataset, flare_test[:3])
    print("\nDone. Figures and log saved to the working directory.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solar flare detection with a spatio-temporal VAE"
    )
    parser.add_argument("--csv", required=True,
                        help="Path to the labels CSV file")
    parser.add_argument("--data-dir", default=None,
                        help="Base directory of the frame folders. If omitted, "
                             "the absolute paths stored in the CSV are used.")
    parser.add_argument("--train", action="store_true",
                        help="Train the VAE backbone from scratch (otherwise "
                             "load saved weights)")
    parser.add_argument("--log", default="training_log.txt",
                        help="Path for the run log file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.stdout = TeeLogger(args.log)
    run(args)
