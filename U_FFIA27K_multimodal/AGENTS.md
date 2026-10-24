# AGENTS.md — Master Architecture Specification & Operational Guidelines
# Branch: final/video_tie_breakers | Multimodal Tournament Network with 5-ch Video Kinematics & AudioMLP L2 (~5.16M Params)

This document defines the invariant architectural constraints, operational guidelines, and verification procedures for AI agents (Antigravity, Gemini, Claude, Cursor) working on the **Fish Feeding Intensity Assessment** multimodal codebase on branch `final/video_tie_breakers`.

---

## 1. System Architecture Overview

```text
========================================================================================
 HIERARCHICAL 2-LEVEL TOURNAMENT WITH 5-CH VIDEO & AUDIO MLP L2 (384-DIM, ~5.16M PARAMS)
========================================================================================

   [Video Input: T=2 Frames]                           [Audio Input: 2.0s @ 256 kHz]
   Shape: (B, 2, 3, 224, 224)                          Shape: (B, 512000)
             │                                                    │
             ▼                                                    ▼
   [FishMotionKinematics5Ch]                           [TKEO-STFT Audio Frontend]
   - Spatial RGB (3 ch)                                - TKEO Adaptive Pre-Emphasis (alpha=0.99)
   - Differentiable Optical Flow (u, v) (2 ch)         - cuFFT RFFT (n_fft=4096, hop=2048)
   Shape: (B, 2, 5, 224, 224)                          - Log Magnitude: log(|X| + 1e-8)
             │                                         - Temporal Mean Pooling -> 2049 bins
             │                                         - Spectral 1D Aug (Cutout & Jitter)
             │                                         Shape: (B, 2049) [4,098 params]
             ▼                                                    │
   [ConvNeXt-Nano Video Backbone]                                 ▼
   - 5-Channel Stem: Conv2d(5->48, k=4, s=4)           [Audio MLP Backbone]
   - 4 ConvNeXt Stages: [48, 96, 192, 384]             - FC1: Linear(2049 -> 768) + GELU + LN + Drop
   - Stochastic Depth (Linear DropPath [0.0 -> 0.1])   - FC2: Linear(768 -> 512) + GELU + LN + Drop
   - Composite LayerNorm Aggregation (composite_ln):   - FC3: Linear(512 -> 384) + LN
     LN((v0 + v1) + |v1 - v0|)                         Shape: f_audio (B, 384) [~2.168M params]
   Shape: f_video (B, 384) [~2.613M params]                       │
             │                                                    │
             └─────────────────────────┬──────────────────────────┘
                                       ▼
                     [DIRECT CROSS-MODAL RELIABILITY GATING]
                     - Reliability Gating: g = sigma(W[f_V || f_A])
                     - Fused Representation: f_fused = LayerNorm(g * f_V + (1-g) * f_A)
                     - Directly feeds Tournament Head (dim=384, no intermediate proj_joint)
                                       │
                                       ▼
                     [MULTIMODAL TOURNAMENT FUSION ENGINE]
                     - Level 1: Feeding Activity Gate Head (None vs Active Feeding, 192-dim)
                     - Level 2: 3 Specialized Pairwise Subspace Heads with Video Referees:
                         * B12: Weak vs Medium (Base on f_fused + Video Kinematics Referee on f_video)
                         * B23: Medium vs Strong (Base on f_fused + Video Kinematics Referee on f_video)
                         * B13: Weak vs Strong (Base on f_fused + Video Kinematics Referee on f_video)
                      - Dynamic Referee Intervention Formula (Sigmoid Reparameterization):
                          u_tie = exp(-|logit_base| / 2.0)
                          gamma_eff = sigmoid(gamma_raw) in (0, 1) [init raw=0.0 -> eff=0.5]
                          logit = logit_base + gamma_eff * u_tie * logit_video
                      - Tournament Borda Voting -> Final Calibrated Probabilities
                      [~0.374M params (B23 default) | FLOPs: ~1.69 GFLOPs]
                                        │
                                        ▼
                      [4 Feeding Intensity Predictions]
                      None (0), Strong (1), Medium (2), Weak (3)
```

### Parameter Budget Breakdown
- **Video Backbone (ConvNeXt-Nano 5-ch, composite_ln)**: `2,613,408` (~`2.613M`)
- **Audio Backbone (TKEO-STFT-MLP [768, 512] -> 384)**: `2,168,448` (~`2.168M`)
- **Audio Frontend (TKEO-STFT LayerNorm)**: `4,098` (~`0.004M`)
- **Tournament Decision Head (Direct Gated + 192-dim heads)**:
  * Default (B23 enabled): `374,023` (~`0.374M`)
  * All 3 Referees enabled: `523,019` (~`0.523M`)
  * Pure Baseline (0 Referees): `299,525` (~`0.300M`)
- **Auxiliary Heads (Video + Audio 384 -> 4)**: `3,080` (~`0.003M`)
- **Total Trainable Parameters**:
  * Default (`enable_b23: true`, B12=False, B13=False): `5,163,057` (~`5.163M`) | 121 parameter tensors.
  * All 3 Referees enabled (`enable_b12: true, enable_b23: true, enable_b13: true`): `5,312,053` (~`5.312M`) | 135 parameter tensors.
  * Pure Baseline (0 Referees): `5,088,559` (~`5.089M`) | 114 parameter tensors.
- **Inference Complexity**: `~1.688 GFLOPs` (profiled via native PyTorch `FlopCounterMode`, strictly < 2.0 GFLOPs).

---

## 2. Invariant Subsystem Specifications

### 2.1 Video Pipeline
- **Input**: Multi-frame video clip with T=2 frames at 224x224 resolution.
- **Motion Kinematics 5-Channel (`FishMotionKinematics5Ch`)**:
  - Channels 0-2: Normalized RGB.
  - Channels 3-4: Dense Optical Flow (u, v) via differentiable gradient formulation.
  - Omitted $|V|$ and $\omega$ (vorticity) as demonstrated by unimodal ablation to eliminate gradient noise.
- **ConvNeXt-Nano Video Backbone (`ConvNeXtNanoVideoBackbone`)**:
  - 5-Channel Stem: Conv2d(5->48, k=4, s=4) (3,888 params).
  - 4 stages: [48, 96, 192, 384] with block depths (1, 1, 3, 1).
  - Stochastic Depth (`DropPath`): Linear schedule [0.0 -> 0.1] across 6 residual blocks during training; identity pass-through during evaluation.
  - Temporal Aggregation (`composite_ln`): $\text{LayerNorm}((v_0 + v_1) + |v_1 - v_0|)$ yielding native 384-dim latent representation.
  - Weight Initialization: Default PyTorch (Kaiming Uniform / He) initialization without calling _init_weights.
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
  - 3-layer MLP projection with LayerNorm and GELU (2049 -> 768 -> 512 -> 384).

### 2.3 Tournament Fusion Engine (`MultimodalTournamentFusion`)
- **Direct Cross-Modal Reliability Gating**:
  $$g = \sigma(W_{\text{gate}}[f_V \parallel f_A])$$
  $$f_{\text{fused}} = \text{LayerNorm}(g \cdot f_V + (1-g) \cdot f_A)$$
  Directly feeds into the Tournament Head without any intermediate `proj_joint` bottleneck.
- **2-Level Tournament Decision Hierarchy**:
  - **Level 1**: Activity Gate Head classifies P(Feeding) vs P(None) using unified 192-dim head (`Linear(384 -> 192) + GELU + LayerNorm(192) + Linear(192 -> 1)`).
  - **Level 2**: 3 specialized pairwise subspace heads with Configurable Video Kinematics Tie-Breakers:
    - B12: Weak vs Medium (Base Head on $f_{\text{fused}}$ + Video Kinematics Referee on $f_{\text{video}}$).
    - B23: Medium vs Strong (Base Head on $f_{\text{fused}}$ + Video Kinematics Referee on $f_{\text{video}}$) [Default active].
    - B13: Weak vs Strong (Base Head on $f_{\text{fused}}$ + Video Kinematics Referee on $f_{\text{video}}$).
  - **Dynamic Referee Intervention Formulation (Sigmoid Reparameterization)**:
    $$u_{\text{tie}} = \exp(-|\text{logit}_{\text{base}}| / 2.0)$$
    $$\gamma_{\text{eff}} = \sigma(\gamma_{\text{raw}}) = \frac{1}{1 + \exp(-\gamma_{\text{raw}})} \in (0, 1)$$
    $$\text{logit} = \text{logit}_{\text{base}} + \gamma_{\text{eff}} \cdot u_{\text{tie}} \cdot \text{logit}_{\text{video}}$$
    where $\gamma_{\text{raw}}$ is a learnable scalar initialized to $0.0$, yielding an initial neutral intervention weight $\gamma_{\text{eff}} = \sigma(0.0) = 0.5$ strictly bounded within $(0, 1)$.
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
- Remote dataset repository: `manhmitcf/Results_final_video_tie_breakers`.
- Token Discovery Order:
  1. `HF_TOKEN` environment variable.
  2. Local `token.txt` (or `/marimo/token.txt`).
- **CRITICAL SECURITY RULE**: `token.txt`, `*.secret`, and private keys are listed in `.gitignore` and **MUST NEVER BE COMMITTED** to version control. Use `token.txt.template` as a reference.

---

## 5. Mandatory Verification Checklist

Before proposing or committing any code changes on branch `final/video_tie_breakers`, agents **MUST** execute and pass:

```bash
cd U_FFIA27K_multimodal
python test_tournament_architecture.py
python main.py --dry-run
```

- [x] **Parameter Budget**: Trainable parameters: Default: 5,163,057 (~5.163M); All-3: 5,312,053 (~5.312M); Baseline: 5,088,559 (~5.089M).
- [x] **Complexity Budget**: Inference FLOPs < 2.0 GFLOPs (Current: ~1.688 GFLOPs).
- [x] **Gradient Propagation**: 100% of trainable parameters receive active gradients (121/121 tensors default, 135/135 tensors all-3).
- [x] **Ablation 8 Combinations**: Full support for toggling B12, B23, B13 Video Referees ($2^3 = 8$ combinations).
- [x] **Temporal Kinematics**: Video transforms must be clip-synchronized.
- [x] **Clean Exit**: Dry-run and test suite complete with exit code 0 on both CPU and CUDA.
