# AGENTS.md — Master Architecture Specification & Operational Guidelines
# Branch: exp/ablation_audio_models_no_tie_breakers | Audio Backbones Ablation Study (~3.03M - ~4.02M Params)

This document defines the invariant architectural constraints, operational guidelines, and verification procedures for AI agents (Antigravity, Gemini, Claude, Cursor) working on the **Fish Feeding Intensity Assessment** multimodal codebase on branch `exp/ablation_audio_models_no_tie_breakers`.

---

## 1. System Architecture Overview

```text
========================================================================================
       MULTIMODAL TOURNAMENT NETWORK: AUDIO BACKBONES ABLATION STUDY (~3.03M - ~4.02M)
========================================================================================

   [Video Input: T=2 Frames]                           [Audio Input: 2.0s @ 256 kHz]
   Shape: (B, 2, 3, 224, 224)                          Shape: (B, 512000)
             │                                                    │
             ▼                                                    ▼
   [MotionKinematics7Ch]                               [TKEO-STFT Audio Frontend]
   - Spatial RGB (3 ch)                                - TKEO Adaptive Pre-Emphasis (alpha=0.99)
   - Farneback Optical Flow (u, v) (2 ch)              - cuFFT RFFT (n_fft=4096, hop=2048)
   - Velocity Magnitude |V| (1 ch)                     - Log Magnitude: log(|X| + 1e-8)
   - Fluid Vorticity omega (1 ch)                      - Spectral Aug (Cutout 24 bins & Jitter 0.02)
   Shape: (B, 2, 7, 224, 224)                          Shape: [B, 2049] (1D) or [B, T=251, 2049] (2D)
             │                                                    │
             ▼                                                    ▼
   [ConvNeXt-Nano Video Backbone]                      [Configurable Audio Backbone (Ablation)]
   - 7-Channel Stem: Conv2d(7->48, k=4, s=4)           - 'mlp': STFT-MLP (2049 -> 512 -> 224) [~1.166M]
   - 4 ConvNeXt Stages: [48, 96, 192, 384]             - 'bcresnet8': Qualcomm BC-ResNet-8 [~0.376M]
   - Stochastic Depth (DropPath [0.0 -> 0.1])          - 'bigru': CRNN-BiGRU 2-layer Sequence [~0.653M]
   Shape: f_video (B, 224) [~2.701M params]            - 'conformer': Conformer Attention-CNN [~0.794M]
             │                                         - 'bimamba': Bidirectional Audio Mamba [~0.693M]
             │                                         - 'tfmamba': Dual-Path TF-Mamba [~0.184M]
             │                                         Shape: f_audio (B, 224)
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
                     - Level 2: 3 Specialized Pairwise Subspace Heads on f_joint:
                         * B12: Weak vs Medium
                         * B23: Medium vs Strong
                         * B13: Weak vs Strong
                     - Video Tie-Breakers: All disabled for fair audio ablation
                       (enable_b12: false, enable_b23: false, enable_b13: false)
                     - Tournament Borda Voting -> Final Calibrated Probabilities
                     [~0.143M params | FLOPs: ~1.71 - ~3.20 GFLOPs]
                                       │
                                       ▼
                     [4 Feeding Intensity Predictions]
                     None (0), Strong (1), Medium (2), Weak (3)
```

### Parameter Budget Breakdown (Strict < 5.0M Limit)
- **Video Backbone (ConvNeXt-Nano 7-ch)**: `2,701,312` (~`2.701M`)
- **Video Auxiliary Head**: `900`
- **Audio Auxiliary Head**: `900`
- **Audio Frontend (TKEO-STFT LayerNorm)**: `4,098` (~`0.004M`)
- **Pairwise Tournament Fusion (Pure Baseline, No Referees)**: `142,821` (~`0.143M`)
- **Audio Backbone Options (Ablation Matrix)**:
  * `mlp`: `1,165,984` (~`1.166M`) | **Total Model**: `4,016,015` (~`4.016M`)
  * `bcresnet8`: `375,552` (~`0.376M`) | **Total Model**: `3,225,583` (~`3.226M`)
  * `bigru`: `652,864` (~`0.653M`) | **Total Model**: `3,502,895` (~`3.503M`)
  * `conformer`: `794,016` (~`0.794M`) | **Total Model**: `3,644,047` (~`3.644M`)
  * `bimamba`: `693,152` (~`0.693M`) | **Total Model**: `3,543,183` (~`3.543M`)
  * `tfmamba`: `184,288` (~`0.184M`) | **Total Model**: `3,034,319` (~`3.034M`)
- **Remaining Headroom**: `983,985` - `1,965,681` parameters below the 5.0M budget limit across all configurations.

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
  - Weight Initialization: Default PyTorch (Kaiming Uniform / He) initialization without calling _init_weights.
- **Consistent Video Transform (`ConsistentVideoTransform`)**:
  - Clip-synchronized Random Horizontal Flip (p=0.5), Random Rotation ([-15 deg, +15 deg]), Color Jitter ([0.85, 1.15]), and Random Erasing (p=0.3).

### 2.2 Audio Pipeline & Ablation Models
- **Input**: Raw 1D acoustic waveform sampled at 256,000 Hz (2.0s = 512,000 samples).
- **TKEO-STFT Frontend (`AudioFrontend`)**:
  - Teager-Kaiser Energy Operator (TKEO) Adaptive Pre-Emphasis: psi[x_n] = x_n^2 - x_{n-1}x_{n+1}.
  - Hann-windowed cuFFT Real FFT: n_fft = 4096, hop_size = 2048 -> 2049 linear frequency bins.
  - Log Magnitude: log(|X| + 1e-8).
  - Spectral Augmentation (`Spectral1DAugmentation`): Frequency Cutout (24 bins, p=0.5) and Gaussian Spectral Jitter (std=0.02) during training; identity pass-through during evaluation. Broadcasted identically across temporal dimension for 2D inputs.
  - Dynamic Output: 1D vector `[B, 2049]` for MLP (`return_2d=False`) or 2D spectrogram `[B, T=251, 2049]` for sequence/CNN/SSM models (`return_2d=True`).
- **Ablation Backbones**:
  1. **STFT-MLP (`AudioMLPBackbone`)**: 2-layer MLP projection (2049 -> 512 -> 224) with GELU, LayerNorm, and Dropout (~1.166M).
  2. **Qualcomm BC-ResNet-8 (`BCResNet8AudioBackbone`)**: Qualcomm AI Research Interspeech 2021 official architecture. Uses `nn.AdaptiveAvgPool2d((40, None))` zero-parameter adapter to feed Qualcomm's native 40 frequency bins into SubSpectralNorm and Broadcasted Residual blocks (~0.376M).
  3. **CRNN-BiGRU (`CRNNBiGRUAudioBackbone`)**: DCASE Task 4 sequence baseline. Linear projection (2049 -> 128) + 2-layer Bidirectional GRU (hidden_size=112 -> 224 output) + temporal mean pooling (~0.653M).
  4. **Conformer (`ConformerAudioBackbone`)**: Google Interspeech 2020 via `torchaudio.models.Conformer`. Linear projection (2049 -> 128) + 2 Conformer blocks (4 heads, ffn_dim=256, depthwise_conv_kernel=15) + temporal mean pooling + output projection (128 -> 224) (~0.794M).
  5. **Bidirectional Audio Mamba (`BiMambaAudioBackbone`)**: KAIST AuM (arXiv:2406.03344) & Vision Mamba (ICML 2024). Linear stem (2049 -> 128) + 3 BiMamba layers (d_state=16, dt_rank=8, expand=2) with low-rank delta, HiPPO S4D diagonal initialization (A = -exp(A_log)), inverse softplus delta bias initialization, and independent forward/backward SSM streams + temporal mean pooling (~0.693M).
  6. **Dual-Path Time-Frequency Mamba (`TFMambaAudioBackbone`)**: Interspeech 2025 (arXiv:2409.05034) & ASCMamba. 2D Conv Stem downsampling to [B, 48, 32, 32] + 2 Dual-Path TF-Mamba stages alternating between intra-frame Frequency-BiMamba and inter-frame Temporal-BiMamba + 2D global average pooling (~0.184M).
- **Initialization & Augmentation Policy**:
  - **100% Train From Scratch**: Zero pretrained weights. All weights initialized randomly using native PyTorch/Qualcomm/Mamba initializations with master seed locked to 42.
  - **Zero Extra Augmentations**: Strictly NO time masking, NO time shift. Only the existing frequency cutout (24 bins) and Gaussian jitter (std=0.02) are applied.
  - **Optimizer Parameter Grouping**: Pure PyTorch Selective Scan computes recurrence in FP32; SSM core parameters (A_log, dt_bias, D) are explicitly exempt from weight decay (weight_decay=0.0).

### 2.3 Tournament Fusion Engine (`MultimodalTournamentFusion`)
- **Reliability Gating**: alpha = sigma(W_gate[f_V || f_A]).
- **Fused & Joint Projection**: f_fused = LayerNorm(g * f_V + (1-g) * f_A), f_joint = ProjJoint(f_fused).
- **Ablation Setting**: Video tie-breakers disabled (`enable_b12: false, enable_b23: false, enable_b13: false`) to ensure fair, unconfounded comparison of acoustic feature representations.
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
- **Audio Backbone Selection**: Configurable via `"audio_backbone": "mlp" | "bcresnet8" | "bigru" | "conformer" | "bimamba" | "tfmamba"`.

---

## 4. Checkpoint & Artifact Management

### 4.1 Checkpoint Saving Hierarchy
Every run automatically exports checkpoints in `checkpoint/MultimodalSOTANet/`:
1. `best_model.pth`: Full multimodal model weights achieving peak performance.
2. `best_video_backbone.pth`: Peak weights of ConvNeXt-Nano video backbone + video aux head.
3. `best_audio_backbone.pth`: Peak weights of selected audio backbone + frontend + audio aux head.
4. `last_model.pth`: Full resumption state (model, optimizer, scheduler, epoch, metrics).

### 4.2 Logging Files
- `history.csv`: 38 columns recorded per epoch.
- `summary.csv`: Single-row consolidated metrics, latency, parameters, and GFLOPs.
- `evaluation_detailed_report.txt` and `.json`: Comprehensive classification reports.

---

## 5. Mandatory Verification Checklist

Before proposing or committing any code changes on branch `exp/ablation_audio_models_no_tie_breakers`, agents **MUST** execute and pass:

```bash
cd U_FFIA27K_multimodal
python test_tournament_architecture.py
python test_audio_backbones_ablation.py
python main.py --dry-run
```

- [x] **Parameter Budget**: Trainable parameters < 5,000,000 across all 6 audio backbones (~3.03M - ~4.02M).
- [x] **Gradient Propagation**: 100% of trainable parameters receive active gradients (no dead tensors).
- [x] **Ablation Audio Models**: All 6 backbones (`mlp`, `bcresnet8`, `bigru`, `conformer`, `bimamba`, `tfmamba`) produce identical embedding shape `[B, 224]`.
- [x] **No Tie-Breakers**: All 3 video tie-breakers verified disabled (`enable_b12: false, enable_b23: false, enable_b13: false`).
- [x] **Zero Extra Augmentations**: Strictly frequency cutout (24 bins) and Gaussian jitter (std=0.02) only.
- [x] **Zero Pretraining**: 100% trained from scratch with deterministic master seed 42.
- [x] **Clean Exit**: Pre-flight dry-run completes with exit code 0 on both CPU and CUDA.