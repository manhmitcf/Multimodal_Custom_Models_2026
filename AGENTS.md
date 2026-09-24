# AGENTS.md — Master Architecture Specification & Operational Guidelines
# Branch: ablation/dual_tie_breakers | Multimodal Tournament Network with Dual (Video + Audio) Tie-Breakers (~4.17M Params)

This document defines the invariant architectural constraints, operational guidelines, and verification procedures for AI agents (Antigravity, Gemini, Claude, Cursor) working on the **Fish Feeding Intensity Assessment** multimodal codebase on branch `ablation/dual_tie_breakers`.

---

## 1. System Architecture Overview

```text
========================================================================================
     HIERARCHICAL 2-LEVEL TOURNAMENT WITH DUAL TIE-BREAKERS (~4.17M PARAMS)
========================================================================================

   [Video Input: T=2 Frames]                           [Audio Input: 2.0s @ 256 kHz]
   Shape: (B, 2, 3, 224, 224)                          Shape: (B, 512000)
             │                                                    │
             ▼                                                    ▼
   [MotionKinematics7Ch]                               [TKEO-STFT Audio Frontend]
   - Spatial RGB (3 ch)                                - TKEO Adaptive Pre-Emphasis (alpha=0.99)
   - Farneback Optical Flow (u, v) (2 ch)              - cuFFT RFFT (n_fft=4096, hop=2048)
   - Velocity Magnitude |V| (1 ch)                     - Log Magnitude: log(|X| + 1e-8)
   - Fluid Vorticity omega (1 ch)                      - Temporal Mean Pooling -> 2049 bins
   Shape: (B, 2, 7, 224, 224)                          - Spectral 1D Aug (Cutout & Jitter)
             │                                         Shape: (B, 2049)
             ▼                                                    │
   [ConvNeXt-Nano Video Backbone]                                 ▼
   - 7-Channel Stem: Conv2d(7->48, k=4, s=4)           [Audio MLP Backbone]
   - 4 ConvNeXt Stages: [48, 96, 192, 384]             - FC1: Linear(2049 -> 512) + GELU + LN + Drop
   - LayerScale (1e-6) + Linear DropPath [0.0 -> 0.1]  - FC2: Linear(512 -> 224) + LN
   Shape: f_video (B, 224) [~2.702M params]            Shape: f_audio (B, 224) [~1.166M params]
             │                                                    │
             └─────────────────────────┬──────────────────────────┘
                                       ▼
                     [CROSS-MODAL RELIABILITY GATING]
                     - Reliability Gating: g = sigma(W[f_V || f_A])
                     - Fused Representation: f_fused = LayerNorm(g * f_V + (1-g) * f_A)
                     - Projected Joint Representation: f_joint = ProjJoint(f_fused) (dim=224)
                                       │
                                       ▼
                     [MULTIMODAL TOURNAMENT FUSION ENGINE]
                     - Level 1: Feeding Activity Gate Head (None vs Active Feeding)
                     - Level 2: 3 Specialized Pairwise Subspace Heads with Dual Referees:
                         * B12: Weak vs Medium (Base Joint + Video Referee + Audio Referee)
                         * B23: Medium vs Strong (Base Joint + Video Referee + Audio Referee)
                         * B13: Weak vs Strong (Base Joint + Video Referee + Audio Referee)
                      - Dynamic Dual-Referee Intervention Formula:
                          u_tie = exp(-|logit_base| / 2.0)
                          logit = logit_base + u_tie * (gamma_v * logit_video + gamma_a * logit_audio)
                      - Tournament Borda Voting -> Final Calibrated Probabilities
                      [~0.296M params | FLOPs: ~1.71 GFLOPs]
                                        │
                                        ▼
                      [4 Feeding Intensity Predictions]
                      None (0), Strong (1), Medium (2), Weak (3)
```

### Parameter Budget Breakdown (Strict < 5.0M Limit)
- **Video Backbone (ConvNeXt-Nano 7-ch)**: `2,702,416` (~`2.702M`)
- **Audio Backbone (TKEO-STFT-MLP 256k)**: `1,165,984` (~`1.166M`)
- **Audio Frontend (TKEO-STFT LayerNorm)**: `4,098` (~`0.004M`)
- **Tournament Decision Head (Pairwise Base + 6 Dual Referees + Borda)**: `296,049` (~`0.296M`)
- **Auxiliary Heads (Deep Supervision)**: `1,800` (~`0.002M`)
- **Total Trainable Parameters**: `4,170,347` (~`4.170M`) with all 3 dual tie-breakers enabled.
- **Remaining Headroom**: `829,653` parameters below the 5.0M budget limit.
- **Inference Complexity**: `~1.71 GFLOPs` (profiled via native PyTorch `FlopCounterMode`).

---

## 2. Invariant Subsystem Specifications

### 2.1 Video Pipeline
- **Input**: Multi-frame video clip with T=2 frames at 224x224 resolution.
- **Motion Kinematics 7-Channel**:
  - Channels 0-2: Normalized RGB.
  - Channels 3-4: Dense Optical Flow (u, v) via Farneback algorithm.
  - Channel 5: Instantaneous Velocity Magnitude |V| = sqrt(u^2 + v^2).
  - Channel 6: Fluid Vorticity omega = dv/dx - du/dy.
- **ConvNeXt-Nano Video Backbone (`ConvNeXtNanoVideoBackbone`)**:
  - 4 stages: [48, 96, 192, 384] with block depths (1, 1, 3, 1).
  - LayerScale: Learnable channel-wise scale parameter $\gamma$ initialized to $10^{-6}$ for each of the 6 residual blocks.
  - Stochastic Depth (`DropPath`): Linear schedule [0.0 -> 0.1] across 6 residual blocks during training; identity pass-through during evaluation.
  - Weight Initialization: Meta AI Truncated Normal $\mathcal{N}(0, 0.02)$, bias $= 0$.
- **Consistent Video Transform (`ConsistentVideoTransform`)**:
  - Random Horizontal Flip: Decided once per clip (p=0.5), applied identically to all frames.
  - Random Rotation: Angle sampled once per clip (theta in [-15 deg, +15 deg]), applied identically to all frames.
  - Color Jitter: Brightness and contrast factors sampled once per clip ([0.85, 1.15]), applied identically to all frames.
  - Random Erasing / Cutout: Rectangular region sampled once per clip (scale [0.05, 0.15], ratio [0.5, 2.0], p=0.3), applied identically to all frames in training mode (producing zero temporal optical flow difference).
  - Bilinear Resize (224x224) and ImageNet normalization.

### 2.2 Audio Pipeline
- **Input**: Raw 1D acoustic waveform sampled at 256,000 Hz (2.0s = 512,000 samples).
- **TKEO-STFT Frontend (`AudioFrontend`)**:
  - Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis: psi[x_n] = x_n^2 - x_{n-1}x_{n+1}.
  - Hann-windowed cuFFT Real FFT: n_fft = 4096, hop_size = 2048 -> 2049 linear frequency bins.
  - Log Magnitude: log(|X| + 1e-8) followed by temporal mean pooling -> [B, 2049].
  - Spectral 1D Augmentation (`Spectral1DAugmentation`): Cutout (24 bins, p=0.5) and Gaussian Jitter (std=0.02) during training; identity pass-through during evaluation.
- **Backbone (`AudioMLPBackbone`)**:
  - 2-layer MLP projection with LayerNorm and GELU (2049 -> 512 -> 224).

### 2.3 Tournament Fusion Engine (`MultimodalTournamentFusion`)
- **Reliability Gating**: alpha = sigma(W_gate[f_V || f_A]).
- **Fused & Joint Projection**: f_fused = LayerNorm(g * f_V + (1-g) * f_A), f_joint = ProjJoint(f_fused).
- **2-Level Tournament Decision Hierarchy**:
  - **Level 1**: Activity Gate Head classifies P(Feeding) vs P(None).
  - **Level 2**: 3 specialized pairwise subspace heads with 3 Configurable Dual Tie-Breakers:
    - B12: Weak vs Medium (Base Joint on $f_{\text{joint}}$ + Video Referee on $f_{\text{video}}$ + Audio Referee on $f_{\text{audio}}$).
    - B23: Medium vs Strong (Base Joint on $f_{\text{joint}}$ + Video Referee on $f_{\text{video}}$ + Audio Referee on $f_{\text{audio}}$).
    - B13: Weak vs Strong (Base Joint on $f_{\text{joint}}$ + Video Referee on $f_{\text{video}}$ + Audio Referee on $f_{\text{audio}}$).
  - **Dynamic Dual-Referee Intervention Formulation**:
    $$u_{\text{tie}} = \exp(-|\text{logit}_{\text{base}}| / 2.0)$$
    $$\text{logit} = \text{logit}_{\text{base}} + u_{\text{tie}} \cdot (\gamma_v \cdot \text{logit}_{\text{video}} + \gamma_a \cdot \text{logit}_{\text{audio}})$$
    where $\gamma_v, \gamma_a$ are learnable scalars initialized to $1.0$.
  - **Borda Voting**: Derives calibrated multi-class distribution from tournament matchup scores:
    $$V_c = \sum_{k \neq c} P(c > k)$$
    with exact algebraic invariant $V_{\text{Weak}} + V_{\text{Medium}} + V_{\text{Strong}} = 3.0$.

---

## 3. Training & Optimization Policy

Configurations are defined in `config/train_config.json` and validated by `config/train_config.py`:
- **Training Strategy**: Single-Phase End-to-End simultaneously optimizing all parameter tensors.
- **Optimizer**: AdamW (learning_rate = 1e-3, weight_decay = 0.05).
- **Learning Rate Schedule**: OneCycleLR (batch-level, epochs = 400, pct_start = 0.05, div_factor = 25, final_div_factor = 1000).
- **Gradient Clipping**: max_norm = 5.0.
- **Loss Function (`PairwiseTournamentLoss`)**:
  $$\mathcal{L}_{\text{total}} = 0.5 \mathcal{L}_{\text{act}} + 0.5 \mathcal{L}_{\text{pairwise}} + 1.0 \mathcal{L}_{\text{CE}} + 0.3 \mathcal{L}_{\text{aux}}$$
- **DataLoader Workers**: Fixed strictly to `8`.
- **Evaluation Monitor**: 3 configurable modes supported in `train_config.json`:
  * `"val_acc"` / `"accuracy"` (Default): Single-track monitoring peak validation Accuracy (`best_model.pth`, `best_video_backbone.pth`, `best_audio_backbone.pth`). Uses peak Val QWK as tie-breaker when validation accuracies match.
  * `"both"`: Dual-track monitoring simultaneously tracking Peak QWK (`*_qwk.pth`) and Peak Accuracy (`*_acc.pth`) throughout training without discarding either. In the Test Split evaluation phase, both candidate models are independently evaluated head-to-head. The model achieving higher Test Accuracy (tie-breaker: Test QWK) is declared the winner and copied to canonical checkpoints (`best_model.pth`, `best_video_backbone.pth`, `best_audio_backbone.pth`), with a comprehensive comparison table logged and exported to `evaluation_detailed_report.txt` and `.json`.
  * `"qwk"`: Single-track monitoring peak validation Quadratic Weighted Kappa. Uses peak Val Accuracy as tie-breaker when QWKs match.

---

## 4. Checkpoint & Artifact Management

### 4.1 Checkpoint Saving Hierarchy
Every run automatically exports checkpoints in `checkpoint/MultimodalSOTANet/`:
1. `best_model.pth`: Full multimodal model weights achieving peak performance (in `both` mode, copied from the winning candidate after Test Split evaluation).
2. `best_video_backbone.pth`: Peak weights of ConvNeXt-Nano video backbone + video aux head.
3. `best_audio_backbone.pth`: Peak weights of STFT-MLP audio backbone + frontend + audio aux head.
4. `last_model.pth`: Full resumption state (model, optimizer, scheduler, epoch, metrics).
5. **In Dual-Track Mode (`"monitor": "both"`)**:
   - `best_model_qwk.pth`, `best_video_backbone_qwk.pth`, `best_audio_backbone_qwk.pth`: Peak validation QWK candidate checkpoints.
   - `best_model_acc.pth`, `best_video_backbone_acc.pth`, `best_audio_backbone_acc.pth`: Peak validation Accuracy candidate checkpoints.

### 4.2 Logging Files
- `history.csv`: 38 columns recorded per epoch (runtime, learning rate, train metrics including video & audio backbone accuracies, val metrics, per-class AUC/AP, and flattened 4x4 confusion matrix).
- `summary.csv`: Single-row consolidated metrics, latency, parameters, and GFLOPs.
- `evaluation_detailed_report.txt` and `.json`: Comprehensive classification reports for Fusion, Video, and Audio branches.
- `learning_curves.png` & `confusion_matrix_heatmaps.png`: High-resolution evaluation visual assets.

### 4.3 Hugging Face Integration & Security
- Remote dataset repository: `manhmitcf/Results_ablation_dual_tie_breakers`.
- Token Discovery Order:
  1. `HF_TOKEN` environment variable.
  2. Local `token.txt` (or `/marimo/token.txt`).
- **CRITICAL SECURITY RULE**: `token.txt`, `*.secret`, and private keys are listed in `.gitignore` and **MUST NEVER BE COMMITTED** to version control. Use `token.txt.template` as a reference.

---

## 5. Mandatory Verification Checklist

Before proposing or committing any code changes on branch `ablation/dual_tie_breakers`, agents **MUST** execute and pass:

```bash
cd U_FFIA27K_multimodal
python test_tournament_architecture.py
python main.py --dry-run
```

- [x] **Parameter Budget**: Trainable parameters < 5,000,000 (Current: 4,170,347 with 3 dual tie-breakers).
- [x] **Complexity Budget**: Inference FLOPs < 2.0 GFLOPs (Current: ~1.71 GFLOPs).
- [x] **Gradient Propagation**: 100% of trainable parameters receive active gradients (164/164 tensors).
- [x] **Ablation 8 Combinations**: Full support for toggling B12, B23, B13 Dual Referees ($2^3 = 8$ combinations).
- [x] **Temporal Kinematics**: Video transforms must be clip-synchronized.
- [x] **Clean Exit**: Dry-run completes with exit code 0 on both CPU and CUDA.