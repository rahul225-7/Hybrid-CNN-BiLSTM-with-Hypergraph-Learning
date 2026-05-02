# ============================================================
# Cross-Cyclone Model v9 — Inductive Multi-Scale Learnable
#                          Attention Dual Hypergraph
# ============================================================
# Architecture:
#   CNN-BiLSTM + Multi-Scale Learnable Attention Dual Hypergraph
# ============================================================

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans


# ============================================================
# Hyperparameters
# ============================================================
N_EDGES_FLOOR        = 16
N_EDGES_COARSE_RATIO = 0.5
N_EDGES_FINE_RATIO   = 1.5

SEQ_LEN          = 12
BATCH_SIZE       = 32
GRAD_ACCUM_STEPS = 2

N_PROXY = 256
HG_GRAD_FREQ = 50

EPOCHS           = 30
PATIENCE         = 5

LR_MAIN = 1e-4
LR_HG   = 5e-4

HG_EDGE_DROPOUT = 0.1

# Prototype diversity regularisation weight
LAMBDA_DIVERSITY = 0.01

# Adaptive embedding refresh schedule
REFRESH_EARLY      = 2  
REFRESH_LATE       = 4   
REFRESH_SWITCH     = 10  

# ============================================================
# Prototype Diversity Regularisation
# ============================================================
def prototype_diversity_loss(prototypes):
    P = F.normalize(prototypes, p=2, dim=1)          # (K, D)
    sim_matrix = P @ P.t()                            # (K, K)
    K = sim_matrix.shape[0]
    # Exclude diagonal (self-similarity = 1.0 always)
    mask = ~torch.eye(K, dtype=torch.bool, device=prototypes.device)
    return sim_matrix[mask].mean()
# ============================================================
# Learnable Hyperedge Generator — Inductive Version
# ============================================================
class LearnableHyperedgeGenerator(nn.Module):
    def __init__(self, embed_dim, n_edges, top_k=3):
        super().__init__()
        self.n_edges   = n_edges
        self.top_k     = min(top_k, n_edges)
        self.embed_dim = embed_dim
        self.prototypes = nn.Parameter(torch.randn(n_edges, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def init_from_kmeans(self, embeddings_np, train_mask=None, n_init=15):
        emb_norm = normalize(embeddings_np, norm='l2')
        # INDUCTIVE: fit only on train nodes
        fit_emb  = emb_norm[train_mask] if train_mask is not None else emb_norm
        km = MiniBatchKMeans(
            n_clusters=self.n_edges, random_state=42,
            batch_size=min(4096, len(fit_emb)), n_init=n_init,
        )
        km.fit(fit_emb)
        centres = normalize(km.cluster_centers_, norm='l2')
        with torch.no_grad():
            self.prototypes.copy_(
                torch.from_numpy(centres.astype(np.float32)))
        n_fit = train_mask.sum() if train_mask is not None else len(embeddings_np)
        print(f"    Prototypes warm-started ({self.n_edges} edges, "
              f"fitted on {n_fit} train nodes only)")

    def forward(self, X, train_mask_tensor=None):
        assert X.dim() == 2 and X.shape[1] == self.embed_dim, \
            f"X must be (N, {self.embed_dim}), got {tuple(X.shape)}"
        if train_mask_tensor is not None:
            assert train_mask_tensor.shape == (X.shape[0],), \
                f"train_mask_tensor must be (N,) with N={X.shape[0]}, got {tuple(train_mask_tensor.shape)}"

        X_norm = F.normalize(X, p=2, dim=1)
        P_norm = F.normalize(self.prototypes, p=2, dim=1)
        sim    = X_norm @ P_norm.t()                   # (N, K)

        # INDUCTIVE: non-train nodes never influence prototype gradients.
        # In v9, train_mask_tensor = hg_train_mask_tensor which is False
        # for BOTH val and test nodes, so neither set contributes to
        # prototype gradient flow.
        if train_mask_tensor is not None:
            non_train_mask = ~train_mask_tensor
            sim_detached   = sim.detach()
            sim = torch.where(non_train_mask.unsqueeze(1), sim_detached, sim)

        topk_vals, topk_idx = sim.topk(self.top_k, dim=1)
        topk_weights        = F.softmax(topk_vals, dim=1)

        H = torch.zeros(X.shape[0], self.n_edges, device=X.device)
        H.scatter_add_(1, topk_idx, topk_weights)
        return H

# ============================================================
# Attention-based Hypergraph Convolution
# ============================================================
class AttentionHypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, n_edges, edge_dropout=0.0, n_heads=4):
        super().__init__()
        self.n_edges      = n_edges
        self.edge_dropout = edge_dropout
        self.n_heads      = n_heads
        self.head_dim     = in_dim // n_heads
        assert in_dim % n_heads == 0, \
            f"in_dim ({in_dim}) must be divisible by n_heads ({n_heads})"

        self.edge_weight = nn.Parameter(torch.ones(n_edges))
        self.W_query     = nn.Linear(in_dim, in_dim, bias=False)
        self.W_key       = nn.Linear(in_dim, in_dim, bias=False)
        self.W_value     = nn.Linear(in_dim, in_dim, bias=False)
        self.theta       = nn.Linear(in_dim, out_dim, bias=False)
        self.norm        = nn.LayerNorm(out_dim)
        self.residual    = (nn.Identity() if in_dim == out_dim
                            else nn.Linear(in_dim, out_dim, bias=False))

    def forward(self, X, H, prototypes, train_mask_tensor=None):
        assert X.dim() == 2, \
            f"X must be (N, D), got {tuple(X.shape)}"
        assert H.dim() == 2 and H.shape[0] == X.shape[0], \
            f"H must be (N, K) with N={X.shape[0]}, got {tuple(H.shape)}"
        assert prototypes.dim() == 2 and prototypes.shape[0] == H.shape[1], \
            f"prototypes must be (K, D) with K={H.shape[1]}, got {tuple(prototypes.shape)}"

        N, D    = X.shape
        K       = H.shape[1]
        W       = torch.relu(self.edge_weight) + 1e-6

        if self.training and self.edge_dropout > 0:
            drop_mask = (torch.rand_like(W) > self.edge_dropout).float()
            W         = W * drop_mask

        # INDUCTIVE: zero out non-train node features before aggregation.
        # In v9, train_mask_tensor = hg_train_mask_tensor so this zeroes
        # BOTH val and test nodes (val was active in v8, now closed).
        if train_mask_tensor is not None:
            X_agg = X * train_mask_tensor.unsqueeze(1).float()
        else:
            X_agg = X

        Q       = self.W_query(prototypes).view(K, self.n_heads, self.head_dim)
        K_proj  = self.W_key(X_agg).view(N, self.n_heads, self.head_dim)
        V_proj  = self.W_value(X_agg).view(N, self.n_heads, self.head_dim)

        attn_logits = torch.einsum(
            'khd,nhd->hkn', Q, K_proj) / (self.head_dim ** 0.5)

        H_mask      = (H.t() > 1e-8).unsqueeze(0).expand(
            self.n_heads, -1, -1)
        attn_logits = attn_logits.masked_fill(~H_mask, -1e9)
        attn_logits = attn_logits + torch.log(
            H.t().unsqueeze(0).clamp(min=1e-8))
        attn_weights = F.softmax(attn_logits, dim=-1)

        HtX_attn = torch.einsum('hkn,nhd->khd', attn_weights, V_proj)
        HtX_attn = HtX_attn.reshape(K, D)
        WBHtX    = HtX_attn * W.unsqueeze(1)

        HW   = H * W.unsqueeze(0)
        Dv   = HW.sum(dim=1, keepdim=True).clamp(min=1e-6)
        Z_hg = (H @ WBHtX) / Dv

        Z = self.norm(self.theta(Z_hg) + self.residual(X))
        return torch.relu(Z)

# ============================================================
# Multi-Scale Hypergraph Module
# ============================================================
class MultiScaleHypergraph(nn.Module):
    def __init__(self, in_dim, out_dim_per_scale,
                 n_edges_coarse, n_edges_fine,
                 edge_dropout=0.0, n_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.hg_gen_coarse  = LearnableHyperedgeGenerator(
            in_dim, n_edges_coarse, top_k=3)
        self.hg_gen_fine    = LearnableHyperedgeGenerator(
            in_dim, n_edges_fine,   top_k=5)
        self.hg_conv_coarse = AttentionHypergraphConv(
            in_dim, out_dim_per_scale, n_edges_coarse,
            edge_dropout=edge_dropout, n_heads=n_heads)
        self.hg_conv_fine   = AttentionHypergraphConv(
            in_dim, out_dim_per_scale, n_edges_fine,
            edge_dropout=edge_dropout, n_heads=n_heads)

    def init_prototypes(self, embeddings_np, train_mask=None):
        print("  [Coarse scale]")
        self.hg_gen_coarse.init_from_kmeans(embeddings_np, train_mask)
        print("  [Fine scale]")
        self.hg_gen_fine.init_from_kmeans(embeddings_np, train_mask)

    def diversity_loss(self):
        """Combined prototype diversity loss for both scales."""
        return (prototype_diversity_loss(self.hg_gen_coarse.prototypes) +
                prototype_diversity_loss(self.hg_gen_fine.prototypes))

    def forward(self, X, train_mask_tensor=None):
        H_coarse = self.hg_gen_coarse(X, train_mask_tensor)
        H_fine   = self.hg_gen_fine(X, train_mask_tensor)

        P_coarse = F.normalize(self.hg_gen_coarse.prototypes, p=2, dim=1)
        P_fine   = F.normalize(self.hg_gen_fine.prototypes,   p=2, dim=1)

        Z_coarse = self.hg_conv_coarse(X, H_coarse, prototypes=P_coarse,
                                       train_mask_tensor=train_mask_tensor)
        Z_fine   = self.hg_conv_fine(X, H_fine,     prototypes=P_fine,
                                     train_mask_tensor=train_mask_tensor)

        return torch.cat([Z_coarse, Z_fine], dim=-1), H_coarse, H_fine

# ============================================================
# Smoothing Ratio Monitor — 4 independent per-scale ratios
# ============================================================
def smoothing_ratio_check(Z_v, Z_m, V_cpu, M_cpu, device):
    V_dim = V_cpu.shape[1]
    M_dim = M_cpu.shape[1]

    def ratio(Z_scale, X_ref_cpu):
        X_ref = X_ref_cpu.to(Z_scale.device)
        if Z_scale.shape != X_ref.shape:
            print(f"  [SMOOTH WARN] Shape mismatch "
                  f"{Z_scale.shape} vs {X_ref.shape} — skipping")
            return torch.tensor(0.0, device=Z_scale.device)
        return (Z_scale - X_ref).norm() / X_ref.norm().clamp(min=1e-8)

    sr = {
        "v_coarse": ratio(Z_v[:, :V_dim],  V_cpu).item(),
        "v_fine":   ratio(Z_v[:, V_dim:],  V_cpu).item(),
        "m_coarse": ratio(Z_m[:, :M_dim],  M_cpu).item(),
        "m_fine":   ratio(Z_m[:, M_dim:],  M_cpu).item(),
    }

    for name, val in sr.items():
        status = "OK" if 0.05 < val < 3.0 else "WARN"
        if status == "WARN":
            print(f"  [SMOOTH {status}] {name}: {val:.3f} "
                  f"(outside [0.05, 3.0])")

    return sr

# ============================================================
# Embedding Networks
# ============================================================
class VisualEmbedder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),         nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        return torch.relu(self.fc(self.conv(x).view(x.size(0), -1)))


class MetaEmbedder(nn.Module):
    def __init__(self, meta_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),       nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# CNN Encoder + Main Model
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),         nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        return torch.relu(self.fc(self.conv(x).view(x.size(0), -1)))


class CrossCycloneModel(nn.Module):
    def __init__(self, img_channels, hg_out_dim):
        super().__init__()
        self.cnn     = CNNEncoder(img_channels)
        self.lstm    = nn.LSTM(
            128 + hg_out_dim, 64,
            batch_first=True, bidirectional=True)
        self.fc1     = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.out     = nn.Linear(64, 1)

    def forward(self, img_seq, Z, img_indices):
        B, T, C, H, W = img_seq.shape
        assert img_indices.shape == (B, T), \
            f"img_indices must be (B, T)=({B}, {T}), got {tuple(img_indices.shape)}"
        assert Z.dim() == 2, \
            f"Z must be (N, hg_out_dim), got {tuple(Z.shape)}"
        img_features  = self.cnn(
            img_seq.view(B*T, C, H, W)).view(B, T, -1)
        z_seq    = Z[img_indices.view(-1)].view(B, T, -1)
        combined = torch.cat([img_features, z_seq], dim=-1)
        last     = self.lstm(combined)[0][:, -1, :]
        return self.out(
            self.dropout(torch.relu(self.fc1(last)))).squeeze(-1)

# ============================================================
# Dataset
# ============================================================
class CycloneDataset(Dataset):
    def __init__(self, sequences, h5_path, y_all, seq_len):
        self.sequences = sequences
        self.h5_path   = h5_path
        self.y_all     = y_all
        self.seq_len   = seq_len
        self.hf = self.images = None

    def _init_h5(self):
        if self.hf is None:
            self.hf     = h5py.File(
                self.h5_path, "r",
                rdcc_nbytes=512*1024*1024,
                rdcc_nslots=10007, rdcc_w0=1.0)
            self.images = self.hf["matrix"]

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        self._init_h5()
        img_indices, target_idx = self.sequences[idx]
        sort_order  = np.argsort(img_indices)
        sorted_idx  = img_indices[sort_order]
        restore_idx = np.argsort(sort_order)
        imgs = np.nan_to_num(
            self.images[sorted_idx.tolist()][restore_idx])
        if imgs.shape[1] == 201:
            imgs = imgs[:, 63:138, 63:138, :]
        imgs = torch.from_numpy(
            imgs.astype(np.float32)).permute(0, 3, 1, 2)
        return (
            imgs,
            torch.from_numpy(img_indices.copy()).long(),
            torch.tensor(self.y_all[target_idx], dtype=torch.float32),
        )


# ============================================================
# Z Computation Helpers
# ============================================================
@torch.no_grad()
def compute_Z_eval(V_cpu, M_cpu, ms_hg_v, ms_hg_m, device,
                   hg_train_mask_tensor=None):
    
    ms_hg_v.eval(); ms_hg_m.eval()

    V_dev = V_cpu.to(device)
    Z_v, _, _ = ms_hg_v(V_dev, hg_train_mask_tensor)
    del V_dev; torch.cuda.empty_cache()

    M_dev = M_cpu.to(device)
    Z_m, _, _ = ms_hg_m(M_dev, hg_train_mask_tensor)
    del M_dev; torch.cuda.empty_cache()

    Z = torch.cat([Z_v, Z_m], dim=-1)
    return Z, Z_v, Z_m


def compute_Z_train(V_dev, M_dev, ms_hg_v, ms_hg_m,
                    hg_train_mask_tensor=None):
    
    Z_v, _, _ = ms_hg_v(V_dev, hg_train_mask_tensor)
    Z_m, _, _ = ms_hg_m(M_dev, hg_train_mask_tensor)
    return torch.cat([Z_v, Z_m], dim=-1)


def build_embeddings(visual_embedder, meta_embedder,
                     h5_path, X_meta_scaled, n_total, device,
                     batch_size=256):

    ve_was_training = visual_embedder.training
    me_was_training = meta_embedder.training
    visual_embedder.eval(); meta_embedder.eval()
    V_list = []
    with h5py.File(h5_path, "r") as hf:
        images = hf["matrix"]
        for start in range(0, n_total, batch_size):
            end   = min(start + batch_size, n_total)
            batch = np.nan_to_num(
                images[start:end]).astype(np.float32)
            if batch.shape[1] == 201:
                batch = batch[:, 63:138, 63:138, :]
            with torch.no_grad():
                t = torch.from_numpy(batch).permute(
                    0, 3, 1, 2).to(device)
                V_list.append(visual_embedder(t).cpu())
                del t
            if (start // batch_size) % 20 == 0:
                print(f"  Visual embeddings: "
                      f"{min(end, n_total)}/{n_total}")
    V_cpu = torch.cat(V_list, dim=0)
    del V_list; torch.cuda.empty_cache()

    M_list = []
    for start in range(0, n_total, 4096):
        end = min(start + 4096, n_total)
        with torch.no_grad():
            t = torch.from_numpy(
                X_meta_scaled[start:end]).to(device)
            M_list.append(meta_embedder(t).cpu())
            del t
    M_cpu = torch.cat(M_list, dim=0)
    del M_list; torch.cuda.empty_cache()

    # Restore prior training state so callers mid-training are not
    # left with embedders silently stuck in eval mode.
    visual_embedder.train(ve_was_training)
    meta_embedder.train(me_was_training)

    return V_cpu, M_cpu


def should_refresh(epoch, switch=REFRESH_SWITCH,
                   early=REFRESH_EARLY, late=REFRESH_LATE):
    freq = early if epoch < switch else late
    return (epoch + 1) % freq == 0


# ============================================================
# GPU Memory Estimator
# ============================================================
def estimate_gpu_usage(n_total, n_edges_coarse, n_edges_fine,
                       hg_out_dim):
    z_mb    = n_total * hg_out_dim * 4 / 1e6
    v_mb    = n_total * 128 * 4 / 1e6
    m_mb    = n_total * 64  * 4 / 1e6
    h_mb    = n_total * (n_edges_coarse + n_edges_fine) * 4 * 2 / 1e6
    attn_mb = 4 * max(n_edges_coarse, n_edges_fine) * n_total * 4 / 1e6
    total   = z_mb + v_mb + m_mb + h_mb + attn_mb

    print(f"\n[GPU ESTIMATE] N={n_total}")
    print(f"  Z tensor:     {z_mb:.0f} MB")
    print(f"  V on GPU:     {v_mb:.0f} MB")
    print(f"  M on GPU:     {m_mb:.0f} MB")
    print(f"  H matrices:   {h_mb:.0f} MB")
    print(f"  Attn peak:    {attn_mb:.0f} MB")
    print(f"  ESTIMATED PEAK: {total:.0f} MB  "
          f"(+ model params + batch images)")
    if total > 6000:
        print("  ⚠ WARNING: May exceed 8GB VRAM. "
              "Consider reducing N_EDGES or BATCH_SIZE.")
    else:
        print("  ✓ Should fit in 8GB VRAM.")
    return total


# ============================================================
# MAIN
# ============================================================
def main():
    total_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"DEVICE : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU    : {gpu_name}  ({gpu_mem:.1f} GB)")
    print("MODEL  : CNN-BiLSTM + Inductive Multi-Scale "
          "Learnable Attention Dual HG v9")
    print("  Full inductive isolation (val+test nodes zeroed in attn keys/values)")
    print("  y_scaler fitted on unique train targets (no storm-length bias)")
    print("  HG LR protected from scheduler decay (always learns)")
    print("  N_PROXY=256 proxy batch (4x stronger HG gradient signal)")
    print("  Forecast loss shapes HG parameters (no decoupling)")
    print("  Prototype diversity regularisation (geometry only)")
    print("  Aligned adaptive embedding + prototype refresh")
    print("  Per-scale smoothing monitor (4 independent ratios)")
    print("=" * 60)

    H5_PATH      = r"add your path"
    PARQUET_PATH = (r"add your path")

    # ── Load metadata ────────────────────────────────────────
    print("Loading metadata...")
    df             = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)
    target_col     = "Vmax_smooth"
    cyclone_id_col = "ID"

    numeric_cols = df.select_dtypes(
        include=['float64','int64','float32','int32']
    ).columns.tolist()
    meta_cols    = [c for c in numeric_cols if c != target_col]
    y_all        = np.nan_to_num(df[target_col].values)
    X_meta_all   = np.nan_to_num(
        df[meta_cols].values).astype(np.float32)
    n_total      = len(df)

    print(f"Total timestamps    : {n_total}")
    print(f"Total meta features : {len(meta_cols)}")
    print(f"Meta features used  : {meta_cols}")

    # Multi-scale edge counts
    sqrt_n         = int(np.sqrt(n_total))
    N_EDGES_COARSE = max(N_EDGES_FLOOR,
                         int(sqrt_n * N_EDGES_COARSE_RATIO))
    N_EDGES_FINE   = max(N_EDGES_FLOOR,
                         int(sqrt_n * N_EDGES_FINE_RATIO))
    print(f"N_EDGES coarse: {N_EDGES_COARSE}  "
          f"fine: {N_EDGES_FINE}  (sqrt={sqrt_n})")

    # HG output: visual(2*128) + meta(2*64) = 384
    hg_out_dim = 2 * 128 + 2 * 64

    estimate_gpu_usage(n_total, N_EDGES_COARSE, N_EDGES_FINE,
                       hg_out_dim)

    # ── Temporal storm split (strict chronological 3-way) ────
    print("\nSplitting storms (chronological 3-way: 64/16/20)...")
    storm_first_time = (
        df.groupby(cyclone_id_col)["time"]
        .min()
        .sort_values()
    )
    unique_storms_sorted = storm_first_time.index.tolist()
    n_storms  = len(unique_storms_sorted)
    n_train   = int(n_storms * 0.64)
    n_val     = int(n_storms * 0.16)
    train_storms = set(unique_storms_sorted[:n_train])
    val_storms   = set(unique_storms_sorted[n_train:n_train + n_val])
    test_storms  = set(unique_storms_sorted[n_train + n_val:])
    print(f"  Total storms : {n_storms}")
    print(f"  Train storms : {len(train_storms)}  "
          f"(up to {storm_first_time.iloc[n_train - 1]})")
    print(f"  Val   storms : {len(val_storms)}  "
          f"({storm_first_time.iloc[n_train]} - "
          f"{storm_first_time.iloc[n_train + n_val - 1]})")
    print(f"  Test  storms : {len(test_storms)}  "
          f"(from {storm_first_time.iloc[n_train + n_val]})")

    # ── Build sequences ──────────────────────────────────────
    print("Building sequences...")

    def build_index_sequences(storm_set):
        seqs         = []
        storm_groups = df.groupby(cyclone_id_col)
        for sid in storm_set:
            if sid not in storm_groups.groups:
                continue
            idx = (storm_groups.get_group(sid)
                   .sort_values("time").index.values)
            if len(idx) <= SEQ_LEN:
                continue
            for row in np.lib.stride_tricks.sliding_window_view(
                    idx, SEQ_LEN + 1):
                seqs.append((row[:-1].copy(), row[-1]))
        return seqs

    train_sequences = build_index_sequences(train_storms)
    val_sequences   = build_index_sequences(val_storms)
    test_sequences  = build_index_sequences(test_storms)
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val   sequences: {len(val_sequences)}")
    print(f"Test  sequences: {len(test_sequences)}")

    # ── Scale ────────────────────────────────────────────────
    print("Scaling data...")

    storm_id_series = df[cyclone_id_col].astype(str)
    train_mask      = storm_id_series.isin(train_storms).values   # (N,) bool

    hg_train_mask        = train_mask.copy()                       # (N,) bool
    hg_train_mask_tensor = torch.from_numpy(hg_train_mask).bool().to(device)

    # meta_scaler: fit on storm-pure train frames only
    meta_scaler   = StandardScaler()
    meta_scaler.fit(X_meta_all[train_mask])
    X_meta_scaled = meta_scaler.transform(
        X_meta_all).astype(np.float32)

    # y_scaler: fit on unique storm-pure train frame targets
    train_pure_indices = np.where(train_mask)[0]
    y_scaler = StandardScaler()
    y_scaler.fit(y_all[train_pure_indices].reshape(-1, 1))
    y_scaled = y_scaler.transform(
        y_all.reshape(-1, 1)).flatten().astype(np.float32)

    # ── Image channels ───────────────────────────────────────
    with h5py.File(H5_PATH, "r") as hf:
        img_channels = hf["matrix"].shape[-1]

    # ── Networks ─────────────────────────────────────────────
    print("\nInitialising networks...")
    visual_embedder = VisualEmbedder(img_channels).to(device)
    meta_embedder   = MetaEmbedder(X_meta_scaled.shape[1]).to(device)

    ms_hg_v = MultiScaleHypergraph(
        in_dim=128, out_dim_per_scale=128,
        n_edges_coarse=N_EDGES_COARSE, n_edges_fine=N_EDGES_FINE,
        edge_dropout=HG_EDGE_DROPOUT, n_heads=4
    ).to(device)

    ms_hg_m = MultiScaleHypergraph(
        in_dim=64, out_dim_per_scale=64,
        n_edges_coarse=N_EDGES_COARSE, n_edges_fine=N_EDGES_FINE,
        edge_dropout=HG_EDGE_DROPOUT, n_heads=4
    ).to(device)

    # ── Build embeddings + warm-start prototypes ─────────────
    print("\n[Global HG] Building embeddings...")
    V_cpu, M_cpu = build_embeddings(
        visual_embedder, meta_embedder,
        H5_PATH, X_meta_scaled, n_total, device)

    # Prototype warm-starting uses train_mask (train storms only).
    # Val and test nodes excluded — HG geometry never influenced by
    # held-out distributions.
    print("\n  Warm-starting visual HG prototypes (train nodes only)...")
    ms_hg_v.init_prototypes(V_cpu.numpy(), train_mask)
    print("  Warm-starting meta HG prototypes (train nodes only)...")
    ms_hg_m.init_prototypes(M_cpu.numpy(), train_mask)

    # ── Model ────────────────────────────────────────────────
    model = CrossCycloneModel(
        img_channels=img_channels,
        hg_out_dim=hg_out_dim).to(device)

    hg_params   = (list(ms_hg_v.parameters()) +
                   list(ms_hg_m.parameters()))
    main_params = (list(model.parameters()) +
                   list(visual_embedder.parameters()) +
                   list(meta_embedder.parameters()))

    optimizer = torch.optim.Adam([
        {"params": main_params, "lr": LR_MAIN},
        {"params": hg_params,   "lr": LR_HG},
    ])

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5)

    # ── Dataloaders ──────────────────────────────────────────
    train_loader = DataLoader(
        CycloneDataset(train_sequences, H5_PATH, y_scaled, SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        CycloneDataset(val_sequences, H5_PATH, y_scaled, SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        CycloneDataset(test_sequences, H5_PATH, y_scaled, SEQ_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Diagnostic ───────────────────────────────────────────
    print("\n[DIAG] Timing first batch...")
    t0    = time.time()
    _iter = iter(train_loader)
    _b    = next(_iter)
    print(f"[DIAG] First batch in {time.time()-t0:.1f}s  "
          f"(img: {_b[0].shape})")
    del _iter, _b; gc.collect(); torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"[DIAG] GPU allocated: "
              f"{torch.cuda.memory_allocated()/1e6:.0f} MB  "
              f"reserved: {torch.cuda.memory_reserved()/1e6:.0f} MB")

    # ── Training ─────────────────────────────────────────────
    print(f"\nStarting training...")
    print(f"  Diversity regularisation weight: {LAMBDA_DIVERSITY}")
    print(f"  Refresh schedule: every {REFRESH_EARLY} epochs "
          f"(epochs 1-{REFRESH_SWITCH}), "
          f"every {REFRESH_LATE} after")

    best_val, patience_counter = np.inf, 0
    train_losses, val_losses   = [], []
    smoothing_history          = []

    storm_to_seqs = {}
    for i, (img_idx, _) in enumerate(train_sequences):
        sid = df.iloc[img_idx[-1]][cyclone_id_col]
        storm_to_seqs.setdefault(sid, []).append(i)
    train_storm_list = list(train_storms)
    
    for epoch in range(EPOCHS):
        model.train()
        visual_embedder.train()
        meta_embedder.train()
        ms_hg_v.train()
        ms_hg_m.train()

        total_loss  = 0
        t_epoch     = time.time()
        accum_count = 0

        # ── Move embeddings to GPU once per epoch ────────────
        try:
            V_dev = V_cpu.to(device)
            M_dev = M_cpu.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  ⚠ OOM moving V/M to GPU. "
                      "Clearing cache and retrying...")
                torch.cuda.empty_cache(); gc.collect()
                V_dev = V_cpu.to(device)
                M_dev = M_cpu.to(device)
            else:
                raise

        # ── Compute Z with gradients (epoch-start proxy) ──────
        optimizer.zero_grad()
        Z_with_grad = compute_Z_train(
            V_dev, M_dev, ms_hg_v, ms_hg_m, hg_train_mask_tensor)

        sampled_storms = np.random.choice(train_storm_list, min(N_PROXY, len(train_storm_list)), replace=False)
        proxy_indices = [np.random.choice(storm_to_seqs[s])for s in sampled_storms if s in storm_to_seqs]
        
        

        proxy_imgs, proxy_idx, proxy_y = zip(
            *[train_loader.dataset[i] for i in proxy_indices])
        proxy_imgs = torch.stack(proxy_imgs).to(device)
        proxy_idx  = torch.stack(proxy_idx).to(device)
        proxy_y    = torch.stack(proxy_y).to(device)

        proxy_pred     = model(proxy_imgs, Z_with_grad, proxy_idx)
        proxy_loss     = criterion(proxy_pred, proxy_y)
        div_loss_epoch = (ms_hg_v.diversity_loss() +
                          ms_hg_m.diversity_loss())
        hg_loss = proxy_loss + LAMBDA_DIVERSITY * div_loss_epoch
        hg_loss.backward()
        nn.utils.clip_grad_norm_(
            main_params + hg_params, max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()

        del proxy_imgs, proxy_idx, proxy_y
        del proxy_pred, proxy_loss, hg_loss

        Z_epoch = Z_with_grad.detach()
        del Z_with_grad, V_dev, M_dev
        torch.cuda.empty_cache()

        print(f"  Epoch {epoch+1}: Z computed + HG grads collected. "
              f"Shape: {Z_epoch.shape}  "
              f"({time.time()-t_epoch:.1f}s)")

        # ── Batch training loop ───────────────────────────────
        for batch_idx, (img, img_idx, y_batch) in enumerate(
                train_loader):
            img     = img.to(device)
            img_idx = img_idx.to(device)
            y_batch = y_batch.to(device)

            pred      = model(img, Z_epoch, img_idx)
            loss_pred = criterion(pred, y_batch)
            loss      = loss_pred / GRAD_ACCUM_STEPS

            loss.backward()

            accum_count += 1
            total_loss  += loss_pred.item()

            if accum_count >= GRAD_ACCUM_STEPS:
                nn.utils.clip_grad_norm_(
                    main_params, max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            # ── Periodic in-loop HG gradient update ──────────
            if (batch_idx + 1) % HG_GRAD_FREQ == 0:
                # Flush partial accumulation before HG update
                if accum_count > 0:
                    nn.utils.clip_grad_norm_(main_params, max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
                optimizer.zero_grad()
                try:
                    V_dev_hg = V_cpu.to(device)
                    M_dev_hg = M_cpu.to(device)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache(); gc.collect()
                        V_dev_hg = V_cpu.to(device)
                        M_dev_hg = M_cpu.to(device)
                    else:
                        raise
                # Pass hg_train_mask_tensor so val+test nodes are zeroed
                Z_hg = compute_Z_train(
                    V_dev_hg, M_dev_hg, ms_hg_v, ms_hg_m,
                    hg_train_mask_tensor)
                del V_dev_hg, M_dev_hg

                sampled_storms_loop = np.random.choice(train_storm_list, min(N_PROXY, len(train_storm_list)), replace=False)
                lp_idx = [np.random.choice(storm_to_seqs[s])for s in sampled_storms_loop if s in storm_to_seqs]
                
                lp_imgs, lp_ii, lp_y = zip(
                    *[train_loader.dataset[i] for i in lp_idx])
                lp_imgs = torch.stack(lp_imgs).to(device)
                lp_ii   = torch.stack(lp_ii).to(device)
                lp_y    = torch.stack(lp_y).to(device)

                lp_pred = model(lp_imgs, Z_hg, lp_ii)
                lp_loss = (criterion(lp_pred, lp_y) +
                           LAMBDA_DIVERSITY * (ms_hg_v.diversity_loss() +
                                               ms_hg_m.diversity_loss()))
                lp_loss.backward()
                nn.utils.clip_grad_norm_(hg_params, max_norm=5.0)
                optimizer.step()
                optimizer.param_groups[1]['lr'] = LR_HG
                optimizer.zero_grad()

                Z_epoch = Z_hg.detach()
                del Z_hg, lp_imgs, lp_ii, lp_y, lp_pred, lp_loss
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                gpu_mb   = (torch.cuda.memory_allocated() / 1e6
                            if torch.cuda.is_available() else 0)
                print(f"  Epoch {epoch+1} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss {avg_loss:.4f} | "
                      f"GPU {gpu_mb:.0f}MB | "
                      f"{time.time()-t_epoch:.0f}s elapsed")

        if accum_count > 0:
            nn.utils.clip_grad_norm_(
                main_params + hg_params, max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        del Z_epoch
        torch.cuda.empty_cache()

        # ── Aligned adaptive refresh ─────────────────────────
        if should_refresh(epoch):
            print(f"  [Epoch {epoch+1}] Aligned refresh: "
                  f"embeddings + prototypes...")
            V_cpu, M_cpu = build_embeddings(
                visual_embedder, meta_embedder,
                H5_PATH, X_meta_scaled, n_total, device)
            ms_hg_v.init_prototypes(V_cpu.numpy(), train_mask)
            ms_hg_m.init_prototypes(M_cpu.numpy(), train_mask)

        # ── Eval Z (no gradients) ─────────────────────────────
        model.eval()
        visual_embedder.eval()
        meta_embedder.eval()
        ms_hg_v.eval()
        ms_hg_m.eval()

        with torch.no_grad():
            Z_eval, Z_v, Z_m = compute_Z_eval(
                V_cpu, M_cpu, ms_hg_v, ms_hg_m,
                device, hg_train_mask_tensor)

        # ── Smoothing monitor ─────────────────────────────────
        sr = smoothing_ratio_check(Z_v, Z_m, V_cpu, M_cpu, device)
        smoothing_history.append(sr)
        print(f"  Smoothing — "
              f"V_coarse: {sr['v_coarse']:.3f}  "
              f"V_fine: {sr['v_fine']:.3f}  "
              f"M_coarse: {sr['m_coarse']:.3f}  "
              f"M_fine: {sr['m_fine']:.3f}")
        del Z_v, Z_m

        # ── Validation ────────────────────────────────────────
        val_loss = 0
        with torch.no_grad():
            for img, img_idx, y_batch in val_loader:
                val_loss += criterion(
                    model(img.to(device),
                          Z_eval,
                          img_idx.to(device)),
                    y_batch.to(device)
                ).item()
        val_loss /= len(val_loader)

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        optimizer.param_groups[1]['lr'] = LR_HG

        epoch_time = time.time() - t_epoch
        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"Train: {train_loss:.4f}  "
              f"Val: {val_loss:.4f}  "
              f"({epoch_time:.0f}s)")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model":           model.state_dict(),
                "visual_embedder": visual_embedder.state_dict(),
                "meta_embedder":   meta_embedder.state_dict(),
                "ms_hg_v":         ms_hg_v.state_dict(),
                "ms_hg_m":         ms_hg_m.state_dict(),
            }, "best_model_crosscyclone_v9.pt")
            patience_counter = 0
            print("  ✓ Saved best model")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

        model.train()
        visual_embedder.train()
        meta_embedder.train()
        ms_hg_v.train()
        ms_hg_m.train()

        del Z_eval
        gc.collect(); torch.cuda.empty_cache()

    # ── Evaluation ───────────────────────────────────────────
    print("\n[EVAL] Loading best model...")
    ckpt = torch.load("best_model_crosscyclone_v9.pt",
                      map_location=device,
                      weights_only=True)
    model.load_state_dict(ckpt["model"])
    visual_embedder.load_state_dict(ckpt["visual_embedder"])
    meta_embedder.load_state_dict(ckpt["meta_embedder"])
    ms_hg_v.load_state_dict(ckpt["ms_hg_v"])
    ms_hg_m.load_state_dict(ckpt["ms_hg_m"])

    model.eval(); visual_embedder.eval()
    meta_embedder.eval(); ms_hg_v.eval(); ms_hg_m.eval()

    with torch.no_grad():
        V_cpu, M_cpu = build_embeddings(
            visual_embedder, meta_embedder,
            H5_PATH, X_meta_scaled, n_total, device)
        Z_eval, _, _ = compute_Z_eval(
            V_cpu, M_cpu, ms_hg_v, ms_hg_m,
            device, hg_train_mask_tensor=None)

    preds_all = []
    with torch.no_grad():
        for img, img_idx, _ in test_loader:
            preds_all.extend(
                model(img.to(device), Z_eval,
                      img_idx.to(device)).cpu().numpy())

    test_target_indices = np.array(
        [s[1] for s in test_sequences])
    y_pred = y_scaler.inverse_transform(
        np.array(preds_all).reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(
        y_scaled[test_target_indices].reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"FINAL METRICS")
    print(f"MAE : {mae:.3f} kt")
    print(f"RMSE: {rmse:.3f} kt")
    print(f"R2  : {r2:.3f}")
    print(f"{'='*50}")
    print(f"\nTotal execution time: {total_time/60:.2f} minutes "
          f"({total_time/3600:.2f} hours)")

    # ── Plots ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(train_losses, label="Train")
    axes[0, 0].plot(val_losses,   label="Val")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()

    axes[0, 1].scatter(y_true, y_pred, alpha=0.4, s=10)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    axes[0, 1].plot([lo, hi], [lo, hi], 'r--')
    axes[0, 1].set_title(f"True vs Predicted (R²={r2:.3f})")
    axes[0, 1].set_xlabel("True Vmax")
    axes[0, 1].set_ylabel("Predicted Vmax")

    residuals = y_true - y_pred
    axes[0, 2].scatter(y_pred, residuals, alpha=0.4, s=10)
    axes[0, 2].axhline(0, color='r', linestyle='--')
    axes[0, 2].set_title("Residuals")
    axes[0, 2].set_xlabel("Predicted")
    axes[0, 2].set_ylabel("Residual")

    if smoothing_history:
        for key, color in zip(
                ["v_coarse", "v_fine", "m_coarse", "m_fine"],
                ["blue", "cyan", "green", "lime"]):
            vals = [s[key] for s in smoothing_history]
            axes[1, 0].plot(vals, color=color, label=key)
        axes[1, 0].axhline(0.05, color='r', linestyle=':',
                           alpha=0.5, label="Lower bound")
        axes[1, 0].axhline(3.0,  color='r', linestyle=':',
                           alpha=0.5, label="Upper bound")
        axes[1, 0].set_title("Per-Scale Smoothing Monitor")
        axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_visible(False)
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig("crosscyclone_v9_results.png", dpi=150)
    print("Plots saved to crosscyclone_v9_results.png")

    print(f"\nTOTAL EXECUTION TIME: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
