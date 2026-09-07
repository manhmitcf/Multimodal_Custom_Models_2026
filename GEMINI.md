# GEMINI.md — Project Guidelines & Invariant Rules

Please refer to the comprehensive guidelines and non-negotiable architectural specifications defined in [AGENTS.md](./AGENTS.md).

All agents (Antigravity, Gemini, Claude, Cursor) working in this workspace must strictly uphold:
1. **Parameter Ceiling**: Total model parameters strictly $< 5.0\text{ M}$ (currently ~4.806 M).
2. **Three Core Feature Groups**:
   - **Group 1 (Spatial)**: Full 19-stage ImageNet pretrained MobileNetV2 (1280 channels -> 224 dim).
   - **Group 2 (Motion)**: Motion Excitation on stem stage with consecutive multi-frame differences $\Delta F_{\text{motion}} = \frac{1}{T-1} \sum |F_t - F_{t-1}|$ (32 channels -> 224 dim).
   - **Group 3 (Acoustic)**: 6-stage Depthwise Inverted Residuals + Frequency-SE (2–8 kHz) + Rhythm Conv1D (512 channels -> 224 dim).
3. **Multi-Frame Processing**: Video receives $T = 4$ frames (`num_frames = 4`).
4. **Multimodal Fusion**: Bidirectional Multi-Head Cross-Attention (BMCA) + Adaptive Modality Reliability Gate (AMRG, $\alpha \in [0, 1]$).
5. **Training Protocol**: Early stopping `patience = 80`, real validation loss tracking, ClipCELoss.
6. **Artifact Auto-Upload**: Automated upload to Hugging Face dataset `manhmitcf/Results_U_FFIA27K_multimodal`. Never commit `run_marimo.txt` or secrets.
7. **Verification Requirement**: Always pass `python U_FFIA27K_multimodal/test_lite_ffia.py` before committing.
