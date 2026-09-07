# GEMINI.md — U_FFIA27K_multimodal Invariant Rules

Please refer to [`AGENTS.md`](./AGENTS.md) and repository root [`../AGENTS.md`](../AGENTS.md) for full specifications.
All agents working in this directory must strictly comply with:
- Model: `LiteFFIA-Net v2`
- Parameter Ceiling: strictly $< 5.0\text{ M}$ (current: 4.806 M)
- Feature Groups: Group 1 (Spatial MobileNetV2 1280-ch), Group 2 (Motion Excitation $\Delta F_{\text{motion}}$), Group 3 (Frequency-SE + Rhythm Conv1D)
- Multi-frame: $T=4$ frames
- Fusion: BMCA + AMRG
- Verification: `python test_lite_ffia.py` must pass 100%.
