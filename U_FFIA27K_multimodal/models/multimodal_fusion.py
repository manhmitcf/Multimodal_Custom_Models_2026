import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SandwichBoundaryTournamentHead(nn.Module):
    """
    2-Boundary Sandwich Tournament Head with Audio STFT Tie-Breaker (~91K params).
    
    Level 1: Feeding Activity Gating
      - Distinguishes None (No feeding, quiet water) from Active Feeding (Weak, Medium, Strong).
      - Controlled by visual motion / surface activity:
        p_feeding = sigmoid(w_act^T * f_video) in (0, 1)
        p_none = 1 - p_feeding

    Level 2: 2-Boundary Sandwich (B12 & B23) + Dual Audio STFT Tie-Breaker
      - Boundary B12 (Weak vs Medium):
        Visual-Acoustic Joint Disambiguation:
        Video proposal: s_12^V = Head_12^V(f_video)
        Uncertainty: u_tie_12 = exp(-|s_12^V|)
        Audio STFT tie-breaker: s_12^A = Head_12^A(f_audio) (rhythm / flock wave splash)
        Combined logit: s_12 = s_12^V + gamma_12 * u_tie_12 * s_12^A
        P(W > M) = sigmoid(s_12)
        P(M > W) = 1 - P(W > M)

      - Boundary B23 (Medium vs Strong):
        Visual-Acoustic Joint Disambiguation:
        Video proposal: s_23^V = Head_23^V(f_video)
        Uncertainty: u_tie_23 = exp(-|s_23^V|)
        Audio STFT tie-breaker: s_23^A = Head_23^A(f_audio) (high-frequency bubble bursts)
        Combined logit: s_23 = s_23^V + gamma_23 * u_tie_23 * s_23^A
        P(M > S) = sigmoid(s_23)
        P(S > M) = 1 - P(M > S)

    Level 3: Sandwich Borda Voting (Protecting Medium from both sides!):
      - V_Weak   = P(W > M)
      - V_Medium = P(M > W) + P(M > S) = (1 - P(W > M)) + P(M > S)
      - V_Strong = P(S > M) = 1 - P(M > S)
      
      P_active = Softmax([V_Weak, V_Medium, V_Strong] * temperature)
      
    Final Output:
      P = [p_none, p_feeding * P_active] -> mapped to [None, Strong, Medium, Weak]
    """
    def __init__(self, dim: int = 224, temperature: float = 2.0) -> None:
        super().__init__()
        self.dim = dim
        self.temperature = temperature

        # Level 1: Feeding Activity Gate (None vs Active Feeding) from Video
        self.activity_head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Level 2: Dual Boundary Subspace Heads
        # B12 Video Head: Weak vs Medium (Video Stream)
        self.head_b12_v = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )
        self.head_b12 = self.head_b12_v  # Alias for backward compatibility

        # B12 Audio Tie-Breaker Head: Weak vs Medium (Audio STFT Stream)
        self.head_b12_a = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )

        # B23 Video Head: Medium vs Strong (Video Stream)
        self.head_b23_v = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )

        # B23 Audio Tie-Breaker Head: Medium vs Strong (Audio STFT Stream)
        self.head_b23_a = nn.Sequential(
            nn.Linear(dim, 112),
            nn.GELU(),
            nn.LayerNorm(112),
            nn.Linear(112, 1)
        )

        # Learnable tie-breaker influence factors (initialized to 0.5)
        self.gamma_12 = nn.Parameter(torch.tensor(0.5))
        self.gamma_23 = nn.Parameter(torch.tensor(0.5))
        self.gamma = self.gamma_23  # Alias for backward compatibility

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if f_audio is None:
            f_audio = f_video

        B = f_video.size(0)

        # 1. Level 1: Feeding Activity Gate
        logit_act = self.activity_head(f_video).squeeze(-1)    # [B]
        p_feeding = torch.sigmoid(logit_act)                   # [B] in (0, 1)
        p_none = torch.clamp(1.0 - p_feeding, min=1e-6)        # [B]

        # 2. Level 2: Dual Boundaries B12 and B23 with Audio Tie-Breakers
        # Boundary B12: Weak (Positive) vs Medium (Negative)
        logit_12_v = self.head_b12_v(f_video).squeeze(-1)      # [B] (Video proposal)
        logit_12_a = self.head_b12_a(f_audio).squeeze(-1)      # [B] (Audio tie-breaker)
        u_tie_12 = torch.exp(-torch.abs(logit_12_v))           # [B] in (0, 1]
        logit_12 = logit_12_v + self.gamma_12 * u_tie_12 * logit_12_a # [B]
        p_w_over_m = torch.sigmoid(logit_12)                  # P(W > M)
        p_m_over_w = 1.0 - p_w_over_m                         # P(M > W)

        # Boundary B23: Medium (Positive) vs Strong (Negative)
        logit_23_v = self.head_b23_v(f_video).squeeze(-1)      # [B] (Video proposal)
        logit_23_a = self.head_b23_a(f_audio).squeeze(-1)      # [B] (Audio tie-breaker)
        u_tie_23 = torch.exp(-torch.abs(logit_23_v))           # [B] in (0, 1]
        logit_23 = logit_23_v + self.gamma_23 * u_tie_23 * logit_23_a # [B]
        p_m_over_s = torch.sigmoid(logit_23)                  # P(M > S)
        p_s_over_m = 1.0 - p_m_over_s                         # P(S > M)

        # 3. Sandwich Borda Count Voting:
        # Weak is bounded on the right by B12
        v_weak = p_w_over_m                                   # [B] in [0, 1]
        # Medium is SANDWICHED from both sides: beats Weak (from left) AND beats Strong (from right)
        v_medium = p_m_over_w + p_m_over_s                    # [B] in [0, 2]
        # Strong is bounded on the left by B23
        v_strong = p_s_over_m                                 # [B] in [0, 1]

        v_voting = torch.stack([v_weak, v_medium, v_strong], dim=-1)  # [B, 3]

        # Softmax over tournament votes (Rank order: Weak=1, Med=2, Strong=3)
        p_feeding_ranks = F.softmax(v_voting * self.temperature, dim=-1)  # [B, 3]
        p_weak_given_feed = p_feeding_ranks[:, 0]
        p_med_given_feed = p_feeding_ranks[:, 1]
        p_strong_given_feed = p_feeding_ranks[:, 2]

        # 4. Final Hierarchical Combination
        p_final_none = p_none
        p_final_weak = p_feeding * p_weak_given_feed
        p_final_medium = p_feeding * p_med_given_feed
        p_final_strong = p_feeding * p_strong_given_feed

        # Map to raw dataset class indexing: [0: None, 1: Strong, 2: Medium, 3: Weak]
        p_raw = torch.stack([p_final_none, p_final_strong, p_final_medium, p_final_weak], dim=-1)
        p_raw = p_raw / torch.sum(p_raw, dim=-1, keepdim=True)
        logits_raw = torch.log(torch.clamp(p_raw, min=1e-7))

        # Expected physical intensity on 0..3 ordinal scale
        expected_intensity = (
            p_final_none * 0.0 +
            p_final_weak * 1.0 +
            p_final_medium * 2.0 +
            p_final_strong * 3.0
        ).unsqueeze(-1)

        return {
            "logits": logits_raw,
            "probabilities": p_raw,
            "expected_intensity": expected_intensity,
            "intensity_score": expected_intensity,
            # Pairwise logits & probabilities for specialized loss calculation
            "logit_act": logit_act,
            "p_feeding": p_feeding,
            "logit_12": logit_12,
            "logit_12_v": logit_12_v,
            "logit_12_a": logit_12_a,
            "u_tie_12": u_tie_12,
            "gamma_12": self.gamma_12,
            "logit_23": logit_23,
            "logit_23_v": logit_23_v,
            "logit_23_a": logit_23_a,
            "u_tie_23": u_tie_23,
            "gamma_23": self.gamma_23,
            "u_tie": u_tie_23,        # Backward compatibility alias
            "gamma": self.gamma_23,    # Backward compatibility alias
            "p_w_over_m": p_w_over_m,
            "p_m_over_s": p_m_over_s,
            "v_voting": v_voting
        }


# Backward compatibility alias
PairwiseBoundaryTournamentHead = SandwichBoundaryTournamentHead


class MultimodalTournamentFusion(nn.Module):
    """
    Multimodal Fusion with Sandwich Boundary Tournament & Dual Audio STFT Tie-Breakers (~116K params).
    - Level 1: Activity Gate (Video) -> None vs Feeding
    - Level 2: 2-Boundary Sandwich:
        * B12: Weak vs Medium (Video proposal + Audio STFT Tie-Breaker)
        * B23: Medium vs Strong (Video proposal + Audio STFT Tie-Breaker)
    - Level 3: Sandwich Borda Voting protecting Medium from both sides.
    """
    def __init__(self, dim: int = 224, temperature: float = 2.0, **kwargs) -> None:
        super().__init__()
        self.dim = dim
        # Sandwich Boundary Tournament Decision Head
        self.tournament_head = SandwichBoundaryTournamentHead(dim=dim, temperature=temperature)

    def forward(
        self,
        f_video: torch.Tensor,
        f_audio: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Sandwich Boundary Tournament with Video proposal & Dual Audio Tie-Breakers
        out = self.tournament_head(f_video=f_video, f_audio=f_audio)

        # Multi-modal diagnostics
        u_tie_12 = out["u_tie_12"].unsqueeze(-1)  # [B, 1]
        u_tie_23 = out["u_tie_23"].unsqueeze(-1)  # [B, 1]
        u_tie_mean = 0.5 * (u_tie_12 + u_tie_23)
        weight_video = 1.0 - 0.5 * u_tie_mean
        weight_audio = 0.5 * u_tie_mean
        out["gate"] = weight_video
        out["modality_weights"] = torch.cat([weight_video, weight_audio], dim=-1)
        out["f_fused"] = (f_video + f_audio) / 2.0

        # Shannon entropy uncertainty
        entropy = -torch.sum(out["probabilities"] * torch.log(torch.clamp(out["probabilities"], min=1e-7)), dim=-1, keepdim=True)
        out["uncertainty"] = entropy / 1.386294

        return out


# Backward compatibility aliases
GatedBilateralBoundaryFusion = MultimodalTournamentFusion
MultimodalBoundaryAwareFusion = MultimodalTournamentFusion
SOTAMultimodalFusion = MultimodalTournamentFusion
