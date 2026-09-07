# AGENTS.md — U_FFIA27K_multimodal Subsystem Standards

> **MANDATORY INSTRUCTIONS FOR ALL AI AGENTS OPERATING IN `U_FFIA27K_multimodal`**
>
> This directory houses **`LiteFFIA-Net v2`**, the production multimodal network for Fish Feeding Intensity Assessment.
> See the repository root [`../AGENTS.md`](../AGENTS.md) for full project invariants.

---

## Quick Reference Rules for `U_FFIA27K_multimodal`

1. **Parameter Budget**:
   - Model parameters must remain strictly **$< 5.0\text{ M}$** (Current: **4.806 M**).
   - Test command: `python test_lite_ffia.py`.

2. **3 Core Feature Groups (Do Not Remove or Alter Interfaces)**:
   - **Group 1 (Spatial)**: Full 19-stage MobileNetV2 pretrained 1280 channels (`models/video_backbone.py`).
   - **Group 2 (Motion)**: Stem Motion Excitation with consecutive differences $\Delta F = \frac{1}{T-1} \sum |F_t - F_{t-1}|$ over $T=4$ frames.
   - **Group 3 (Acoustic)**: 6 Inverted Residual blocks + Frequency-SE (2–8 kHz) + Rhythm Conv1D (`models/audio_backbone.py`).

3. **Multimodal Fusion**:
   - Bidirectional Multi-Head Cross-Attention (BMCA) with 4 heads, dimension 224 (`models/multimodal_fusion.py`).
   - Adaptive Modality Reliability Gate (AMRG): $\alpha \in [0, 1]$.
   - Bottleneck MBT tokens are strictly deprecated in favor of BMCA.

4. **Training & Configuration**:
   - Multi-frame: `video_features.num_frames = 4` in `config/train_config.json`.
   - Early stopping: `patience = 80`.
   - Real validation loss monitored (`monitor = "loss"`).

5. **Artifacts & Secrets**:
   - Hugging Face upload enabled (`config/artifact_upload_config.json`).
   - `run_marimo.txt` is git-ignored and must never be tracked.

6. **Mandatory Verification**:
   - Any agent modifying code in `U_FFIA27K_multimodal` must run:
     ```bash
     python test_lite_ffia.py
     ```
     and verify all 5 test sections pass before completing.
