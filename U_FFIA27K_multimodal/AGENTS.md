# AGENTS.md — Master Architecture Specification & Operational Guidelines
# Branch: exp/ablation_phyconformer_no_tie_breakers | Multimodal SOTA Tournament Network (~3.84M Params)

This document defines the invariant architectural constraints, operational guidelines, and verification procedures for AI agents (Antigravity, Gemini, Claude, Cursor) working on the **Fish Feeding Intensity Assessment** multimodal codebase.

---

## 1. System Architecture Overview

```text
========================================================================================
             MULTIMODAL SOTA TOURNAMENT NETWORK (~3.84M PARAMS)
========================================================================================

   [Video Input: T=2 Frames]                           [Audio Input: 2.0s @ 256 kHz]
   Shape: (B, 2, 3, 224, 224)                          Shape: (B, 512000)
             │                                                    │
             ▼                                                    ▼
   [MotionKinematics7Ch]                               [TKEO-STFT Audio Frontend]
   - Spatial RGB (3 ch)                                - TKEO Adaptive Pre-Emphasis (alpha=0.99)
   - Farneback Optical Flow (u, v) (2 ch)              - cuFFT RFFT (n_fft=4096, hop=2048)
   - Velocity Magnitude |V| (1 ch)                     - Log Magnitude: log(|X| + 1e-8)
   - Fluid Vorticity omega (1 ch)                      - LayerNorm across F=2049 bins
   Shape: (B, 2, 7, 224, 224)                          - Underwater HydroAcoustic Aug (Train)
             │                                         Shape: (B, 1, 251, 2049)
             ▼                                                    │
   [ConvNeXt-Nano Video Backbone]                                 ▼
   - 7-Channel Stem: Conv2d(7->48, k=4, s=4)           [Phy-Conformer Audio Backbone]
   - 4 ConvNeXt Stages: [48, 96, 192, 384]             - Tier 1: Spectral Stem (3-cell Asym DW-Conv)
   - Temporal Dynamics (Spatial + Motion + Burst)      - Tier 2: Acoustic Dual-Stream (Tonal/Transient)
   Shape: f_video (B, 224) [~2.701M params]            - Tier 3: Cadence Conformer (Macaron MHSA+Conv1D)
             │                                         Shape: f_audio (B, 224) [~0.994M params]
             │                                                    │
             └─────────────────────────┬──────────────────────────┘
                                       ▼
                     [MULTIMODAL TOURNAMENT FUSION ENGINE]
                     - Cross-Modal Reliability Gating: g = sigma(W[f_V || f_A])
                     - Fused Representation: f_fused = g * f_V + (1-g) * f_A (dim=224)
                     - Level 1: Feeding Activity Gate Head (None vs Active Feeding)
                     - Level 2: 3 Specialized Pairwise Subspace Expert Heads
                       (Ablation: All 3 Audio Tie-Breakers Disabled):
                         * B12: Weak vs Medium (Direct fused subspace)
                         * B23: Medium vs Strong (Direct fused subspace)
                         * B13: Weak vs Strong (Direct fused subspace)
                     - Tournament Borda Voting -> Final Calibrated Probabilities
                     [~0.143M params | FLOPs: 1.9925 GFLOPs]
                                       │
                                       ▼
                     [4 Feeding Intensity Predictions]
                     None (0), Strong (1), Medium (2), Weak (3)
```

### Parameter Budget Breakdown (Strict < 5.0M Limit)
- **Video Backbone (ConvNeXt-Nano 7-ch)**: `2,701,312` (~`2.701M`)
- **Audio Backbone (Phy-Conformer 256k)**: `993,792` (~`0.994M`)
- **Tournament Decision Head (Pairwise Subspace + Borda, No Tie-Breakers)**: `142,821` (~`0.143M`)
- **Auxiliary Heads + Frontend Norm**: `5,898`
- **Total Trainable Parameters**: `3,843,823` (~`3.844M`) [In full tie-breakers mode: `3,920,437` (~`3.920M`)]
- **Remaining Headroom**: `1,156,177` parameters below the 5.0M budget limit.
- **Inference Complexity**: `1.9925 GFLOPs` (profiled via native PyTorch `FlopCounterMode`).

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
  - Hann-windowed cuFFT Real FFT: n_fft = 4096, hop_size = 2048 -> 2049 linear frequency bins, T=251 frames.
  - Log Magnitude: log(|X| + 1e-8) followed by Layer Normalization across 2049 frequency bins -> [B, 1, 251, 2049].
  - 2D Hydrophone Augmentation (`UnderwaterHydroAcousticAugmenter`): Subband Frequency Masking (16 bins, p=0.5), Circular Time Shift (+/-15 frames), and Gaussian Sensor Jitter (std=0.015) during training; identity pass-through during evaluation.
- **Backbone (`PhyConformerBackbone`)**:
  - **Tier 1 (`SpectralStemBlock`)**: 3-cell hierarchical Asymmetric Depthwise-Separable Conv2D compressing $2049 \to 257 \to 65 \to 33$ bins while downsampling temporal dimension $251 \to 126 \to 63$.
  - **Tier 2 (`AcousticDualStreamBlock`)**: Orthogonal physical decomposition via Depthwise-Separable Tonal Stream ($5 \times 1$ conv for aerator/motor lines) and Transient Stream ($1 \times 5, d=2$ dilated conv for ultrasonic feeding clicks), fused with residual shortcut.
  - **Tokenizer**: Linear($1584 \to 160$) + LayerNorm(160) + learnable positional embedding `pos_emb` (std=0.02).
  - **Tier 3 (`CadenceConformerBlock`)**: Macaron Conformer Block (FFN1/2 -> MHSA 4-heads -> 1D Depthwise ConvModule $k=15$ -> FFN1/2 -> LayerNorm).
  - **Head (`DecoupledFeatureHead`)**: Decouples 5 output representations (`f_audio`, `f_frequency`, `f_rhythm`, `f_burst_a`, `tokens_audio`) normalized to 224-D with LayerNorm.

### 2.3 Tournament Fusion Engine (`MultimodalTournamentFusion`)
- **Reliability Gating**: alpha = sigma(W_gate[f_V || f_A]).
- **2-Level Tournament Decision Hierarchy**:
  - **Level 1**: Activity Gate Head classifies P(Feeding) vs P(None).
  - **Level 2**: 3 specialized pairwise subspace heads:
    - B12: Weak vs Medium.
    - B23: Medium vs Strong (audio tie-breaker toggleable, disabled in this ablation).
    - B13: Weak vs Strong (cross boundary, audio tie-breaker toggleable, disabled in this ablation).
  - **Borda Voting**: Derives calibrated multi-class distribution from tournament matchup scores.

---

## 3. Training & Optimization Policy

Configurations are defined in `config/train_config.json` and validated by `config/train_config.py`:
- **Training Strategy**: Single-Phase End-to-End simultaneously optimizing all 175 parameter tensors (Case B: no tie-breakers) or 196 tensors (Case C: all tie-breakers).
- **Optimizer**: AdamW (learning_rate = 1e-3, weight_decay = 0.05).
- **Learning Rate Schedule**: OneCycleLR (batch-level, epochs = 400, pct_start = 0.05, div_factor = 25, final_div_factor = 1000).
- **Gradient Clipping**: max_norm = 5.0.
- **Loss Function (`PairwiseTournamentLoss`)**:
  Loss = 0.5 * Loss_act + 0.5 * Loss_pairwise + 1.0 * Loss_CE + 0.3 * Loss_aux.
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
3. `best_audio_backbone.pth`: Peak weights of Phy-Conformer audio backbone + frontend + audio aux head.
4. `last_model.pth`: Full resumption state (model, optimizer, scheduler, epoch, metrics).
5. **In Dual-Track Mode (`"monitor": "both"`)**:
   - `best_model_qwk.pth`, `best_video_backbone_qwk.pth`, `best_audio_backbone_qwk.pth`: Peak validation QWK candidate checkpoints.
   - `best_model_acc.pth`, `best_video_backbone_acc.pth`, `best_audio_backbone_acc.pth`: Peak validation Accuracy candidate checkpoints.

### 4.2 Logging Files
- `history.csv`: 36 columns recorded per epoch (runtime, learning rate, train metrics, val metrics, per-class AUC/AP, and flattened 4x4 confusion matrix).
- `summary.csv`: Single-row consolidated metrics, latency, parameters, and GFLOPs.
- `evaluation_detailed_report.txt` and `.json`: Comprehensive classification reports for Fusion, Video, and Audio branches.
- `learning_curves.png` & `confusion_matrix_heatmaps.png`: High-resolution evaluation visual assets.

### 4.3 Hugging Face Integration & Security
- Remote dataset repository: `manhmitcf/Results_exp_ablation_phyconformer_no_tie_breakers`.
- Token Discovery Order:
  1. `HF_TOKEN` environment variable.
  2. Local `token.txt` (or `/marimo/token.txt`).
- **CRITICAL SECURITY RULE**: `token.txt`, `*.secret`, and private keys are listed in `.gitignore` and **MUST NEVER BE COMMITTED** to version control. Use `token.txt.template` as a reference.

---

## 5. Mandatory Verification Checklist

Before proposing or committing any code changes on branch `exp/ablation_phyconformer_no_tie_breakers`, agents **MUST** execute and pass:

```bash
cd U_FFIA27K_multimodal
python test_tournament_architecture.py
python main.py --dry-run
```

- [x] **Parameter Budget**: Trainable parameters < 5,000,000 (Current: 3,843,823 with tie-breakers off; 3,920,437 with all on).
- [x] **Complexity Budget**: Inference FLOPs < 2.0 GFLOPs (Current: 1.9925 GFLOPs).
- [x] **Gradient Propagation**: 100% of trainable parameters receive active gradients (175/175 off, 196/196 on).
- [x] **Configurable Tie-Breakers**: Full support for toggling B12, B23, B13 Audio Tie-Breakers via `train_config.json`.
- [x] **Temporal Kinematics**: Video transforms must be clip-synchronized.
- [x] **Clean Exit**: Dry-run completes with exit code 0 on both CPU and CUDA.