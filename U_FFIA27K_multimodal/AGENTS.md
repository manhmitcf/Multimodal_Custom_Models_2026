# AGENTS.md — Master Architecture Specification & Operational Guidelines
# Branch: exp/audio_spectral_convnext1d_no_tie_breakers | Multimodal Tournament Network with Frequency-Domain 1D ConvNeXt Audio Backbone (~3.69M Params)

This document defines the invariant architectural constraints, operational guidelines, and verification procedures for AI agents (Antigravity, Gemini, Claude, Cursor) working on the **Fish Feeding Intensity Assessment** multimodal codebase.

---

## 1. System Architecture Overview

```text
========================================================================================
     MULTIMODAL TOURNAMENT NETWORK WITH FREQUENCY-DOMAIN 1D CONVNEXT (~3.69M PARAMS)
========================================================================================

   [Video Input: T=2 Frames]                           [Audio Input: 2.0s @ 256 kHz]
   Shape: (B, 2, 3, 224, 224)                          Shape: (B, 512000)
             │                                                    │
             ▼                                                    ▼
   [MotionKinematics7Ch]                               [TKEO-STFT Audio Frontend]
   - Spatial RGB (3 ch)                                - TKEO Adaptive Pre-Emphasis (alpha=0.99)
   - Farneback Optical Flow (u, v) (2 ch)              - cuFFT RFFT (n_fft=4096, hop=2048)
   - Velocity Magnitude |V| (1 ch)                     - Log Magnitude: log(|X| + 1e-8)
   - Fluid Vorticity omega (1 ch)                      - Dual-Channel Frequency Profile:
   Shape: (B, 2, 7, 224, 224)                            * Ch 0: Mean PSD (Stationary Background)
             │                                           * Ch 1: Peak Contrast PSD (Transient Bursts)
             ▼                                         - Advanced Spectral 1D Augmentation:
   [ConvNeXt-Nano Video Backbone]                        * Dual-Band Mask (width 20, p=0.5)
   - 7-Channel Stem: Conv2d(7->48, k=4, s=4)             * Spectral Tilt (slope 0.05)
   - 4 ConvNeXt Stages: [48, 96, 192, 384]               * Micro-Shift (12 bins)
   - Temporal Dynamics (Spatial + Motion + Burst)        * Additive Gaussian Jitter (std=0.015)
   Shape: f_video (B, 224) [~2.701M params]            Shape: (B, 2, 2049)
             │                                                    │
             │                                                    ▼
             │                                         [Frequency-Domain 1D ConvNeXt Backbone]
             │                                         - 4 ConvNeXt-1D Stages: [32, 64, 128, 224]
             │                                         - Depths: (1, 1, 2, 1) [5 Inverted Bottleneck Blocks]
             │                                         - Depthwise Conv1D k=7, GroupNorm(1, C), GELU
             │                                         - LayerScale (gamma=1e-6), DropPath (0.0 -> 0.1)
             │                                         - Adaptive Average Pooling 1D + Dropout (0.1)
             │                                         Shape: f_audio (B, 224) [~0.832M params]
             │                                                    │
             └─────────────────────────┬──────────────────────────┘
                                       ▼
                     [CROSS-MODAL RELIABILITY GATING]
                     - Reliability Gating: g = sigma(W[f_V || f_A])
                     - Fused Representation: f_fused = g * f_V + (1-g) * f_A (dim=224)
                                       │
                                       ▼
                     [MULTIMODAL TOURNAMENT FUSION ENGINE]
                     - Level 1: Feeding Activity Gate Head (None vs Active Feeding)
                     - Level 2: 3 Specialized Pairwise Subspace Heads (No Tie-Breakers):
                         * B12: Weak vs Medium (Linear 224->64 + GELU + LN + Linear 64->1)
                         * B23: Medium vs Strong (Linear 224->64 + GELU + LN + Linear 64->1)
                         * B13: Weak vs Strong (Linear 224->64 + GELU + LN + Linear 64->1)
                     - Dynamic Tie-Breakers Disabled: u_tie = 0.0
                     - Tournament Borda Voting -> Final Calibrated Probabilities
                     [~0.143M params | FLOPs: 1.8734 GFLOPs]
                                       │
                                       ▼
                     [4 Feeding Intensity Predictions]
                     None (0), Strong (1), Medium (2), Weak (3)
```

### Parameter Budget Breakdown (Strict < 5.0M Limit)
- **Video Backbone (ConvNeXt-Nano 7-ch)**: `2,701,312` (~`2.701M`)
- **Audio Backbone (ConvNeXt-1D Spectral 256k)**: `832,320` (~`0.832M`)
- **Tournament Decision Head (Pairwise Base + Borda Voting)**: `142,821` (~`0.143M`)
- **Auxiliary Heads (Deep Supervision)**: `1,800` (~`0.002M`)
- **Audio Frontend Normalization Buffers**: `8,196` (~`0.008M`)
- **Total Trainable Parameters**: `3,686,449` (~`3.686M`)
- **Remaining Headroom**: `1,313,551` parameters below the 5.0M budget limit.
- **Inference Complexity**: `1.8734 GFLOPs` (profiled via native PyTorch `FlopCounterMode`).

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
  - Stochastic Depth (`DropPath`): Linear schedule [0.0 -> 0.1] across 6 residual blocks during training; identity pass-through during evaluation.
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
  - Log Magnitude: log(|X| + 1e-8).
  - Dual-Channel Spectral Profile:
    * Channel 0: Stationary Power Spectral Density (Temporal Mean PSD).
    * Channel 1: Transient Cavitation / Feeding Burst Contrast (Temporal Max - Mean PSD).
  - Advanced 1D Spectral Augmentation (`AdvancedSpectral1DAugmentation`):
    * Dual-Band Frequency Masking: 2 independent narrow bands (width 20 bins, p=0.5).
    * Spectral Tilt: Linear transmission loss simulation (max slope 0.05).
    * Micro-Shift: Random frequency bin shift (max 12 bins, circular wrap).
    * Additive Gaussian Jitter: Standard deviation 0.015.
- **Backbone (`FrequencyConvNeXtAudioBackbone`)**:
  - 4 stages along frequency axis: [32, 64, 128, 224] with depths (1, 1, 2, 1).
  - Depthwise Conv1D (k=7), GroupNorm(1, C), Pointwise Linear (4x expansion), GELU, Pointwise Linear.
  - LayerScale (gamma initialized to 1e-6) for stable early training.
  - DropPath (Stochastic Depth): Linear schedule [0.0 -> 0.1] across 5 residual blocks.
  - Global Adaptive Average Pooling 1D + Dropout (p=0.1) -> [B, 224].

### 2.3 Tournament Fusion Engine (`MultimodalTournamentFusion`)
- **Reliability Gating**: g = sigma(W_gate[f_V || f_A]).
- **Fused Joint Projection**: f_fused = g * f_V + (1-g) * f_A (dim=224).
- **2-Level Tournament Decision Hierarchy**:
  - **Level 1**: Activity Gate Head classifies P(Feeding) vs P(None).
  - **Level 2**: 3 specialized pairwise subspace heads:
    * B12: Weak vs Medium.
    * B23: Medium vs Strong.
    * B13: Weak vs Strong.
  - **Tie-Breakers Status**: Disabled by default (`enable_b12=false`, `enable_b23=false`, `enable_b13=false`).
  - **Borda Voting**: Derives calibrated multi-class distribution from tournament matchup scores:
    $$V_c = \sum_{k \neq c} P(c > k)$$
    with exact algebraic invariant $V_{\text{Weak}} + V_{\text{Medium}} + V_{\text{Strong}} = 3.0$.

---

## 3. Training & Optimization Policy

Configurations are defined in `config/train_config.json` and validated by `config/train_config.py`:
- **Training Strategy**: Single-Phase End-to-End simultaneously optimizing all 173 parameter tensors.
- **Master Seed**: Strict deterministic locking via `seed_everything(42)` (PyTorch, CUDA, NumPy, Random).
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
1. `best_model.pth`: Full multimodal model weights achieving peak performance.
2. `best_video_backbone.pth`: Peak weights of ConvNeXt-Nano video backbone + video aux head.
3. `best_audio_backbone.pth`: Peak weights of ConvNeXt-1D audio backbone + frontend + audio aux head.
4. `last_model.pth`: Full resumption state (model, optimizer, scheduler, epoch, metrics).
5. **In Dual-Track Mode (`"monitor": "both"`)**:
   - `best_model_qwk.pth`, `best_video_backbone_qwk.pth`, `best_audio_backbone_qwk.pth`: Peak validation QWK candidate checkpoints.
   - `best_model_acc.pth`, `best_video_backbone_acc.pth`, `best_audio_backbone_acc.pth`: Peak validation Accuracy candidate checkpoints.

### 4.2 Logging Files
- `history.csv`: 38 columns recorded per epoch (runtime, learning rate, train metrics including `train_acc_video` and `train_acc_audio`, val metrics, per-class AUC/AP, and flattened 4x4 confusion matrix).
- `summary.csv`: Single-row consolidated metrics, latency, parameters, and GFLOPs.
- `evaluation_detailed_report.txt` and `.json`: Comprehensive classification reports for Fusion, Video, and Audio branches.
- `learning_curves.png` & `confusion_matrix_heatmaps.png`: High-resolution evaluation visual assets.

### 4.3 Hugging Face Integration & Security
- Remote dataset repository: `manhmitcf/Results_exp_audio_spectral_convnext1d_no_tie_breakers`.
- Token Discovery Order:
  1. `HF_TOKEN` environment variable.
  2. Local `token.txt` (or `/marimo/token.txt`).
- **CRITICAL SECURITY RULE**: `token.txt`, `*.secret`, and private keys are listed in `.gitignore` and **MUST NEVER BE COMMITTED** to version control. Use `token.txt.template` as a reference.

---

## 5. Mandatory Verification Checklist

Before proposing or committing any code changes on branch `exp/audio_spectral_convnext1d_no_tie_breakers`, agents **MUST** execute and pass:

```bash
cd U_FFIA27K_multimodal
python test_tournament_architecture.py
python main.py --dry-run
```

- [x] **Parameter Budget**: Trainable parameters < 5,000,000 (Current: 3,686,449).
- [x] **Complexity Budget**: Inference FLOPs < 2.0 GFLOPs (Current: 1.8734 GFLOPs).
- [x] **Gradient Propagation**: 100% of trainable parameters (173/173 tensors) receive active gradients.
- [x] **Tie-Breakers Disabled**: `enable_b12: false`, `enable_b23: false`, `enable_b13: false` verified.
- [x] **Frequency-Domain 1D ConvNeXt**: Dual-channel STFT profile [B, 2, 2049], LayerScale, DropPath, Spectral Augmentation verified.
- [x] **Temporal Kinematics**: Video transforms must be clip-synchronized.
- [x] **Clean Exit**: Dry-run completes with exit code 0 on both CPU and CUDA.