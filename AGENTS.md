# AGENTS.md — Multimodal Architecture & Coding Standards

> **MANDATORY INSTRUCTIONS FOR ALL AI AGENTS (Antigravity, Gemini, Claude, Cursor, Copilot)**
> 
> This document defines the non-negotiable architectural invariants, model constraints, feature extraction rules, training protocols, and repository conventions for the **Fish Feeding Intensity Assessment (FFIA)** project, specifically focusing on **`LiteFFIA-Net v2`** in `U_FFIA27K_multimodal`.
> 
> **ANY AI AGENT WORKING ON THIS REPOSITORY MUST STRICTLY ADHERE TO THIS SPECIFICATION.**
> Modifications violating parameter limits, removing core physical feature groups, or bypassing the verification suite must NOT be implemented or merged.

---

## 1. Core Architecture: `LiteFFIA-Net v2`

The network is designed for high-accuracy, edge-deployable multimodal assessment combining surface visual dynamics with underwater acoustic feeding sounds.

### 1.1 Strict Parameter & Compute Budget (Non-Negotiable)
- **Total Model Parameters**: **STRICTLY $< 5.0\text{ M}$** (Current baseline: **4,806,271 parameters ~ 4.806 M**).
  - *No change may increase total parameters to $\ge 5.0\text{ M}$.*
- **Computational Complexity**: **$< 1.5\text{ GFLOPs}$** (Current baseline: **0.911 GFLOPs** for 1 audio spec + 2 frames, $\approx 1.0\text{ GFLOPs}$ for 4 frames).
- **Inference Latency / Throughput**: Edge real-time target: **$< 25\text{ ms}$ / sample ($> 40\text{ FPS}$)** on single GPU. Current baseline: **19.80 ms (~50.5 FPS)** on RTX 3050 Laptop GPU.

### 1.2 The Three Fundamental Physical Feature Groups (Immutable Principle)
All 3 feature branches must remain intact, active, and returned in the model output dictionary:

| Feature Group | Modality & Physical Phenomenon | Backbone / Mechanism | Output Dimension | Key Tensor Key |
| :--- | :--- | :--- | :--- | :--- |
| **Group 1: Spatial Artifacts** | Surface fish clustering, water surface foam, fish body density on latest frame | **Full MobileNetV2 (19 stages, features[0..18])**, ImageNet-1K pretrained, 1280-channel bottleneck | $d = 224$ (`spatial_proj`) | `f_spatial` |
| **Group 2: Motion Artifacts** | Feeding frenzy water agitation, splashing speed, consecutive frame motion | **Motion Excitation (ME)** on Stem stage (features[:7], 32-ch, 28x28) with consecutive temporal difference $\Delta F_{\text{motion}}$ | $d = 224$ (`motion_proj`) | `f_motion` |
| **Group 3: Acoustic Artifacts** | Feeding chewing sounds (2–8 kHz) + temporal ingestion rhythm / cadence | **6-stage Inverted Residual Depthwise Conv** + **Frequency-SE (F-SE)** + **Rhythm 1D Conv** | $d = 224$ (`joint_proj`) | `f_frequency`, `f_rhythm` |

```
                                  [INPUT DATA]
                                       │
        ┌──────────────────────────────┴──────────────────────────────┐
        ▼                                                             ▼
[Video: 4 frames, 224x224]                                  [Audio: Log-Mel 100x128]
        │                                                             │
   Stem (features[:7], 32-ch)                                   6x Inverted Residual Blocks
        ├─────────────────────────────┐                               │ (1->16->...->512)
        ▼                             ▼                               ├──────────────┐
[Group 2: Motion ME]       [Group 1: Spatial Deep]                    ▼              ▼
  Consecutive Frame Diff     Stages 7..18 (1280-ch)              [Group 3a]     [Group 3b]
  ΔF = 1/(T-1) Σ |Ft - Ft-1|  Pretrained MobileNetV2             Frequency-SE   Rhythm Conv1D
  ME Gate -> motion_proj     spatial_proj (1280->224)             (2-8 kHz)     (Cadence)
        │                             │                               │              │
        └──────────────┬──────────────┘                               └───────┬──────┘
                       ▼                                                      ▼
              Visual Joint Projection                                Acoustic Joint Projection
                 (448 -> 224 dim)                                       (448 -> 224 dim)
                       │                                                      │
                       └──────────────────────┬───────────────────────────────┘
                                              ▼
                             [MULTIMODAL BMCA FUSION ENGINE]
                             - V -> A Cross-Attention (dim=224, heads=4)
                             - A -> V Cross-Attention (dim=224, heads=4)
                             - Adaptive Modality Reliability Gate (AMRG, α in [0, 1])
                             - LayerNorm(α * v_enh + (1-α) * a_enh + f_fused)
                                              │
                                              ▼
                                    [CLASSIFIER HEAD]
                                 Linear(224->96) -> GELU -> Dropout(0.2) -> Linear(96->4)
                                              │
                                              ▼
                               [Logits (4 Feeding Classes)]
```

---

## 2. Detailed Subsystem Specifications

### 2.1 Video Backbone (`models/video_backbone.py`)
- **Pretrained Weights**: Must load official ImageNet pretrained weights.
  - Compatible loader handles both modern PyTorch (`MobileNet_V2_Weights.DEFAULT`) and legacy versions (`pretrained=True`).
- **Stem Processing**: `features[:7]` maps input $(B, T, 3, 224, 224)$ into stem feature map $(B, T, 32, 28, 28)$.
- **Motion Excitation Module (`MotionExcitation`)**:
  - Consecutive multi-frame difference:
    $$\Delta F = \frac{1}{T-1} \sum_{t=1}^{T-1} |F_t - F_{t-1}|$$
  - Squeeze-and-excitation spatial-temporal gating over stem features.
  - Global average pooling + `motion_proj` $\to 224\text{ dim}$.
- **Spatial Branch**:
  - Deep stages `features[7:]` (stages 3 to 18) are evaluated on the latest frame feature map $F_{T-1}$ (or early feature pooled) to extract full 1280-channel spatial semantics without repeating heavy 1280-ch convolutions across all frames.
  - Global average pooling + `spatial_proj` (1280 $\to 224$).
- **Visual Joint Projection**: Concatenates spatial ($224$) and motion ($224$) $\to$ `joint_proj` (448 $\to 224$).

### 2.2 Audio Backbone (`models/audio_backbone.py`)
- **Input Representation**: Log-Mel spectrogram $(B, 1, 100, 128)$ (Sample rate: 64,000 Hz, window: 2048, hop: 512, mel bins: 128). Optional TKEO pre-processing.
- **Backbone**: 6 Inverted Residual Depthwise Separable convolution blocks:
  - Channels: $1 \to 16 \to 32 \to 64 \to 128 \to 256 \to 512$.
- **Frequency-SE (`FrequencySE`)**:
  - Temporal average pooling $\to (B, 512, 1, F')$.
  - Squeeze-and-excitation MLP with emphasis on feeding sound band (2–8 kHz) $\to$ `freq_proj` (512 $\to 224$).
- **Rhythm 1D Conv (`RhythmConv1D`)**:
  - Frequency average pooling $\to (B, 512, T')$.
  - 1D Temporal Convolution (kernel size 3) capturing feeding rhythm / ingestion bursts $\to$ `rhythm_proj` (256 $\to 224$).
- **Acoustic Joint Projection**: Concatenates frequency ($224$) and rhythm ($224$) $\to$ `joint_proj` (448 $\to 224$).

### 2.3 Multimodal Fusion Engine (`models/multimodal_fusion.py`)
- **Bidirectional Multi-Head Cross-Attention (BMCA)**:
  - Parameter-efficient alternative to MBT (Multimodal Bottleneck Tokens).
  - $V \to A$: Visual queries attend to Acoustic keys/values ($d = 224, \text{heads} = 4$).
  - $A \to V$: Acoustic queries attend to Visual keys/values ($d = 224, \text{heads} = 4$).
- **Adaptive Modality Reliability Gate (AMRG)**:
  - Dynamically calculates reliability factor $\alpha \in [0, 1]$:
    $$\alpha = \sigma\left(\text{Linear}(448 \to 64) \to \text{GELU} \to \text{Linear}(64 \to 1)\right)$$
  - Gated combination with normalized residual connection:
    $$e_{\text{final}} = \text{LayerNorm}\left(\alpha \cdot v_{\text{enh}} + (1 - \alpha) \cdot a_{\text{enh}} + f_{\text{fused}}\right)$$

### 2.4 Classifier Head
- `Linear(224 -> 96)` $\to$ `GELU()` $\to$ `Dropout(0.2)` $\to$ `Linear(96 -> 4)`.
- Returns dictionary containing:
  - `clipwise_output` (shape $[B, 4]$)
  - `logits` (alias to `clipwise_output`)
  - `f_spatial` (shape $[B, 224]$)
  - `f_motion` (shape $[B, 224]$)
  - `f_frequency` (shape $[B, 224]$)
  - `f_rhythm` (shape $[B, 224]$)
  - `f_fused` (shape $[B, 224]$)
  - `gating_alpha` (shape $[B, 1]$)

---

## 3. Configuration & Training Parameters

Configurations are maintained in `config/train_config.json` and mirrored in `config/train_config.py`:

```json
{
  "epochs": 400,
  "batch_size": 256,
  "learning_rate": 1e-4,
  "weight_decay": 1e-4,
  "ckpt_dir": "checkpoint/",
  "monitor": "loss",
  "early_stopping": true,
  "patience": 80,
  "delta": 0.0,
  "video_features": {
    "image_size": 224,
    "frame_policy": "end",
    "num_frames": 4
  },
  "audio_features": {
    "sample_rate": 64000,
    "window_size": 2048,
    "hop_size": 512,
    "mel_bins": 128,
    "fmin": 50,
    "fmax": 32000,
    "use_tkeo": true
  }
}
```

### Key Training Constraints:
1. **Multi-Frame Count**: `num_frames = 4` must be used across DataLoader and Video transforms.
2. **Early Stopping Patience**: `patience = 80` (epochs). Do not lower below 80.
3. **Loss Function**: `ClipCELoss` (Classification Cross-Entropy) with real validation loss tracking (never placeholder or constant zero).
4. **Per-Class Metrics**: Must report Precision, Recall, F1-Score, and Confusion Matrix across all 4 feeding intensity classes.

---

## 4. Artifact Management & Hugging Face Upload

Artifacts and training results are configured in `config/artifact_upload_config.json`:
- **Auto-Upload**: `"enabled": true` by default.
- **Hugging Face Repository**: `manhmitcf/Results_U_FFIA27K_multimodal` (Dataset repo).
- **Token Discovery Hierarchy**:
  1. Environment variable `HF_TOKEN`.
  2. Fallback: Parse `run_marimo.txt` at the repository root using regex `HF_TOKEN=([a-zA-Z0-9_]+)`.
- **Security & Git Hygiene**:
  - `run_marimo.txt` and `*.secret` are listed in `.gitignore` and **MUST NEVER BE COMMITTED**.
  - `.git`, `.zip`, `.pyc`, raw dataset directories (`raw_dataset/`, `U-FFIA/`) are strictly excluded from artifact uploads.

---

## 5. Verification & Testing Requirements

Before proposing or committing any code changes in `U_FFIA27K_multimodal`, agents **MUST** run the verification suite:

```bash
python U_FFIA27K_multimodal/test_lite_ffia.py
```

### Verification Checklist:
- [x] **Parameter Check**: Total parameters strictly $< 5,000,000$.
- [x] **FLOPs Check**: Total FLOPs $< 1.5\text{ GFLOPs}$.
- [x] **Multi-Shape Forward Check**: Passes for 4-frame, 2-frame, and single-frame inputs.
- [x] **Gradient Flow Check**: Gradients confirmed non-null in Video Stem, Motion Excitation, Audio Backbone, BMCA Attention, Modality Gate, and Classifier Head.
- [x] **Latency Benchmark**: Confirmed $< 25\text{ ms}$ on GPU (or valid CPU baseline).

---

## 6. Prohibited Anti-Patterns (What NOT to Do)

1. **NO Truncating MobileNetV2**: Do NOT revert to MobileNetV2 stage 6 (160 channels) or discard ImageNet pretrained weights.
2. **NO MBT Token Bottlenecks**: Do NOT introduce Multimodal Bottleneck Tokens (MBT); use BMCA cross-attention.
3. **NO Dropping Feature Groups**: Do NOT bypass `f_spatial`, `f_motion`, `f_frequency`, or `f_rhythm`.
4. **NO Hardcoded Credentials**: Do NOT commit Hugging Face write tokens or private keys to version control.
5. **NO Unverified Commits**: Do NOT commit changes without executing `test_lite_ffia.py`.
