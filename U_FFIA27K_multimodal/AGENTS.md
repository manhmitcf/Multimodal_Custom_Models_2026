# AGENTS.md — Master Architecture Specification & Operational Guidelines
# Branch: main_architecture/flat_4class_video_tie_breakers | Flat 4-Class Round-Robin Tournament Network with Video Kinematics Tie-Breakers (~4.16M Params)

This document defines the invariant architectural constraints, operational guidelines, and verification procedures for AI agents (Antigravity, Gemini, Claude, Cursor) working on the **Fish Feeding Intensity Assessment** multimodal codebase on branch `main_architecture/flat_4class_video_tie_breakers`.

---

## 1. System Architecture Overview

```text
========================================================================================
 FLAT 4-CLASS ROUND-ROBIN WITH VIDEO KINEMATICS TIE-BREAKERS (~4.16M PARAMS)
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
   - Temporal Dynamics (Spatial + Motion + Burst)      - FC2: Linear(512 -> 224) + LN
   Shape: f_video (B, 224) [~2.701M params]            Shape: f_audio (B, 224) [~1.166M params]
             │                                                    │
             └─────────────────────────┬──────────────────────────┘
                                       ▼
                     [MULTIMODAL TOURNAMENT FUSION ENGINE]
                     - Cross-Modal Reliability Gating: g = sigma(W[f_V || f_A])
                     - Fused Representation: f_fused = g * f_V + (1-g) * f_A (dim=224)
                     - 6 Direct Pairwise Expert Heads (Binomial(4, 2) = 6):
                         * B01: None (0) <-> Strong (1) (+ Optional Video Kinematics Tie-Breaker)
                         * B02: None (0) <-> Medium (2) (+ Optional Video Kinematics Tie-Breaker)
                         * B03: None (0) <-> Weak (3)   (+ Optional Video Kinematics Tie-Breaker)
                         * B12: Strong (1) <-> Medium (2) (+ Video Kinematics Tie-Breaker) [Default: Enabled]
                         * B23: Medium (2) <-> Weak (3)   (+ Video Kinematics Tie-Breaker) [Default: Enabled]
                         * B13: Strong (1) <-> Weak (3)   (+ Video Kinematics Tie-Breaker) [Default: Enabled]
                     - Video Referee Intervention:
                         u_tie = exp(-|logit_base|)
                         logit = logit_base + gamma * u_tie * logit_video
                     - 4-Class Tournament Borda Voting:
                         * V_c = Sum_{k != c} P(c > k), Sum(V_c) = 6.0
                         * P_final = Softmax([V_0, V_1, V_2, V_3] * tau)
                     [~0.282M params | FLOPs: 1.7087 GFLOPs]
                                       │
                                       ▼
                     [4 Feeding Intensity Predictions]
                     None (0), Strong (1), Medium (2), Weak (3)
```

### Parameter Budget Breakdown (Strict < 5.0M Limit)
- **Video Backbone (ConvNeXt-Nano 7-ch)**: `2,701,312` (~`2.701M`)
- **Audio Backbone (TKEO-STFT-MLP 256k)**: `1,165,984` (~`1.166M`)
- **Tournament Decision Head (6 Base Heads + 3 Active Video Tie-Breakers B12, B23, B13)**: `281,581` (~`0.282M`)
- **Auxiliary Heads (Deep Supervision)**: `1,800`
- **Total Trainable Parameters (Default 3 Video Tie-Breakers)**: `4,154,775` (~`4.155M`) [151 parameter tensors]
- **Remaining Headroom**: `845,225` parameters below the 5.0M budget limit.
- **Inference Complexity**: `1.7087 GFLOPs` (profiled via native PyTorch `FlopCounterMode`).
*(Note: Range spans from 4,078,161 (~4.078M, 130 tensors) with 0 tie-breakers up to 4,231,389 (~4.231M, 172 tensors) if all 6 video tie-breakers are enabled).*

---

## 2. Invariant Subsystem Specifications

### 2.1 Video Pipeline
- **Input**: Multi-frame video clip with T=2 frames at 224x224 resolution.
- **Motion Kinematics 7-Channel**:
  - Channels 0-2: Normalized RGB.
  - Channels 3-4: Dense Optical Flow (u, v) via Farneback algorithm.
  - Channel 5: Instantaneous Velocity Magnitude |V| = sqrt(u^2 + v^2).
  - Channel 6: Fluid Vorticity omega = dv/dx - du/dy.
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

### 2.3 Flat Tournament Fusion Engine (`MultimodalTournamentFusion`)
- **Reliability Gating**: alpha = sigma(W_gate[f_V || f_A]).
- **Flat 4-Class Round-Robin Matchups**:
  - 6 Pairwise Heads: B01, B02, B03, B12, B23, B13.
  - Configurable Video Kinematics Tie-Breakers: $u_{\text{tie}} = \exp(-|\text{logit}_{\text{base}}|)$, $\text{logit} = \text{logit}_{\text{base}} + \gamma \cdot u_{\text{tie}} \cdot \text{logit}_{\text{video}}$.
- **Borda Voting**:
  - V_c = Sum_{k != c} P(c > k). Sum of all 4 scores equals strictly 6.0.
  - Final probabilities derived via Softmax with temperature tau=2.0.

---

## 3. Training & Optimization Policy

Configurations are defined in `config/train_config.json` and validated by `config/train_config.py`:
- **Training Strategy**: Single-Phase End-to-End simultaneously optimizing all parameter tensors.
- **Optimizer**: AdamW (learning_rate = 1e-3, weight_decay = 0.05).
- **Learning Rate Schedule**: OneCycleLR (batch-level, epochs = 400, pct_start = 0.05, div_factor = 25, final_div_factor = 1000).
- **Gradient Clipping**: max_norm = 5.0.
- **Loss Function (`PairwiseTournamentLoss`)**:
  $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{CE}} + 1.0 \cdot \mathcal{L}_{\text{pairwise}} + 0.3 \cdot \mathcal{L}_{\text{aux}}$$
  with boundary-weighted pairwise matchups:
  $$\mathcal{L}_{\text{pairwise}} = 0.25 \mathcal{L}_{03} + 0.25 \mathcal{L}_{23} + 0.25 \mathcal{L}_{12} + 0.10 \mathcal{L}_{02} + 0.10 \mathcal{L}_{13} + 0.05 \mathcal{L}_{01}$$

---

## 4. Checkpoint & Artifact Management

### 4.1 Checkpoint Saving Hierarchy
Every run automatically exports checkpoints in `checkpoint/MultimodalSOTANet/`:
1. `best_model.pth`: Full multimodal model weights achieving peak performance.
2. `best_video_backbone.pth`: Peak weights of ConvNeXt-Nano video backbone + video aux head.
3. `best_audio_backbone.pth`: Peak weights of STFT-MLP audio backbone + frontend + audio aux head.
4. `last_model.pth`: Full resumption state (model, optimizer, scheduler, epoch, metrics).

### 4.2 Logging Files
- `history.csv`: 36 columns recorded per epoch.
- `summary.csv`: Single-row consolidated metrics, latency, parameters, and GFLOPs.
- `evaluation_detailed_report.txt` and `.json`: Comprehensive classification reports.

### 4.3 Hugging Face Integration & Security
- Remote dataset repository: `manhmitcf/Results_main_architecture_flat_4class_video_tie_breakers`.
- Token Discovery Order:
  1. `HF_TOKEN` environment variable.
  2. Local `token.txt` (or `/marimo/token.txt`).
- **CRITICAL SECURITY RULE**: `token.txt`, `*.secret`, and private keys are listed in `.gitignore` and **MUST NEVER BE COMMITTED** to version control.

---

## 5. Mandatory Verification Checklist

Before proposing or committing any code changes on branch `main_architecture/flat_4class_video_tie_breakers`, agents **MUST** execute and pass:

```bash
cd U_FFIA27K_multimodal
python test_tournament_architecture.py
python main.py --dry-run
```

- [x] **Parameter Budget**: Trainable parameters < 5,000,000 (Current Default: 4,154,775).
- [x] **Complexity Budget**: Inference FLOPs < 2.0 GFLOPs (Current Default: 1.7087 GFLOPs).
- [x] **Gradient Propagation**: 100% of trainable parameters receive active gradients.
- [x] **Configurable Video Tie-Breakers**: Full support for toggling any subset of 6 Video Tie-Breakers via `train_config.json` (defaults to B12, B23, B13 enabled).
- [x] **Algebraic Invariant**: Sum of Borda votes across 4 classes strictly equals 6.0.
- [x] **Clean Exit**: Dry-run completes with exit code 0 on both CPU and CUDA.
