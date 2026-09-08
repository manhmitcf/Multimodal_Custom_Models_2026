import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalWassersteinEvidentialLoss(nn.Module):
    """
    Ordinal Wasserstein Evidential Loss for Fish Feeding Intensity Classification.
    
    Addresses temporal transition label ambiguity between adjacent feeding stages
    (Strong -> Medium -> Weak -> None) without increasing model parameter budget:
    
    1. Physical Ordinal Mapping:
       Corrects dataset raw indexing (0: None, 1: Strong, 2: Medium, 3: Weak) into
       a physically continuous ordinal chain:
       None (0) <-> Weak (3) <-> Medium (2) <-> Strong (1).
       Permutation tensor: Pi = [0, 3, 2, 1] (an involution, Pi^-1 = Pi).
       
    2. Truncated Gaussian Soft Label Smoothing (sigma = 0.5):
       Models continuous feeding decay. Distributes ~85% mass to ground truth rank,
       ~15% to adjacent rank(s), and ~0% to far classes, eliminating gradient shock
       at ambiguous human-annotated temporal boundaries.
       
    3. Cumulative Distribution Wasserstein Distance (Squared EMD / RPS):
       Penalizes predictions proportional to ordinal distance across thresholds:
       L_EMD = (1 / (K-1)) * sum_{k=0}^{K-2} (CDF_P(k) - CDF_Q(k))^2.
       Calculated across Fused, Video, and Audio Dirichlet expected distributions.
       
    4. Trusted Evidential Dirichlet Loss (TMC):
       Optimizes Dirichlet evidence parameters alpha = e + 1 with KL annealing to
       penalize misleading evidence on non-target classes.
       
    5. Dynamic Cosine Ramp-up:
       Schedules ordinal penalty lambda_ord(t) smoothly from lambda_start (0.2) to
       lambda_end (2.0) to stabilize feature representation before sharpening ordinal boundaries.
    """
    def __init__(
        self,
        classes_num: int = 4,
        sigma: float = 0.5,
        lambda_ord_start: float = 0.2,
        lambda_ord_end: float = 2.0,
        total_epochs: int = 400,
        annealing_epochs: int = 50,
        aux_loss_weight: float = 0.5
    ) -> None:
        super().__init__()
        self.classes_num = classes_num
        self.sigma = sigma
        self.lambda_ord_start = lambda_ord_start
        self.lambda_ord_end = lambda_ord_end
        self.total_epochs = total_epochs
        self.annealing_epochs = annealing_epochs
        self.aux_loss_weight = aux_loss_weight

        # Mapping between raw dataset index [0: none, 1: strong, 2: medium, 3: weak]
        # and physical ordinal rank [0: none, 1: weak, 2: medium, 3: strong].
        # Permutation: Pi = [0, 3, 2, 1]
        self.register_buffer(
            "ordinal_perm",
            torch.tensor([0, 3, 2, 1], dtype=torch.long)
        )
        # raw_index -> ordinal_rank lookup:
        # raw 0 (none)   -> rank 0
        # raw 1 (strong) -> rank 3
        # raw 2 (medium) -> rank 2
        # raw 3 (weak)   -> rank 1
        self.register_buffer(
            "raw_to_rank",
            torch.tensor([0, 3, 2, 1], dtype=torch.long)
        )

    def compute_lambda_ord(self, epoch: int) -> float:
        """
        Calculates dynamic ordinal loss weight via cosine ramp-up schedule.
        """
        if epoch <= 1:
            return self.lambda_ord_start
        if epoch >= self.total_epochs:
            return self.lambda_ord_end
        progress = float(epoch - 1) / float(max(1, self.total_epochs - 1))
        cosine_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.lambda_ord_start + (self.lambda_ord_end - self.lambda_ord_start) * cosine_scale

    def build_gaussian_soft_target(self, target: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs Truncated Gaussian target distribution on ordinal space:
        Q_ord ~ exp(-(j - rank)^2 / (2 * sigma^2)).
        Returns:
            q_ord: [B, K] target distribution in physical ordinal order.
            q_raw: [B, K] target distribution mapped back to raw index order.
        """
        # Determine raw target class index
        if target.ndim > 1 and target.size(-1) > 1:
            raw_labels = torch.argmax(target, dim=-1)
        else:
            raw_labels = target.long().view(-1)

        # Map raw label to physical ordinal rank (0 to 3)
        target_ranks = self.raw_to_rank.to(device)[raw_labels]  # [B]

        # Ordinal rank grid [0, 1, 2, ..., K-1]
        ranks_grid = torch.arange(self.classes_num, device=device, dtype=torch.float32)  # [K]
        diff = ranks_grid.unsqueeze(0) - target_ranks.float().unsqueeze(1)  # [B, K]

        # Gaussian smoothing
        weights = torch.exp(-(diff ** 2) / (2.0 * (self.sigma ** 2)))
        q_ord = weights / weights.sum(dim=-1, keepdim=True)  # [B, K]

        # Map back to raw index space using the involution property Pi^-1 = Pi
        q_raw = q_ord[:, self.ordinal_perm.to(device)]  # [B, K]

        return q_ord, q_raw

    def compute_squared_emd(self, p_ord: torch.Tensor, q_ord: torch.Tensor) -> torch.Tensor:
        """
        Computes Wasserstein distance (Squared Earth Mover's Distance / RPS):
        W_2^2(P, Q) = (1 / (K - 1)) * sum_{k=0}^{K-2} (CDF_P(k) - CDF_Q(k))^2.
        """
        cdf_p = torch.cumsum(p_ord, dim=-1)
        cdf_q = torch.cumsum(q_ord, dim=-1)

        # Exclude the last bin since CDF_P[-1] == CDF_Q[-1] == 1.0
        diff_cdf = cdf_p[:, :-1] - cdf_q[:, :-1]
        emd_per_sample = torch.sum(diff_cdf ** 2, dim=-1) / float(self.classes_num - 1)
        return torch.mean(emd_per_sample)

    def _dirichlet_ace_loss(self, alpha: torch.Tensor, target_soft: torch.Tensor) -> torch.Tensor:
        """
        Expected Dirichlet Cross-Entropy Loss with continuous target probabilities:
        L_ace = sum_k y_k * (psi(S) - psi(alpha_k)).
        """
        S = torch.sum(alpha, dim=-1, keepdim=True)
        ace_per_sample = torch.sum(target_soft * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
        return torch.mean(ace_per_sample)

    def _dirichlet_kl_divergence(self, alpha: torch.Tensor, target_soft: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        KL divergence regularizer against uniform prior for misleading non-target evidence.
        """
        K = alpha.size(-1)
        device = alpha.device
        alpha_tilde = target_soft + (1.0 - target_soft) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)

        first_term = (
            torch.lgamma(S_tilde)
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
            - torch.lgamma(torch.tensor(float(K), device=device))
            + torch.sum(torch.lgamma(torch.ones(1, K, device=device)), dim=-1, keepdim=True)
        )
        second_term = torch.sum(
            (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=-1,
            keepdim=True
        )
        kl = torch.mean(first_term + second_term)

        # Annealing factor
        annealing_coef = min(1.0, float(epoch) / float(self.annealing_epochs))
        return annealing_coef * kl

    def forward(
        self,
        output_dict: Dict[str, Any],
        target_dict: Dict[str, Any],
        epoch: int = 1
    ) -> torch.Tensor:
        """
        Unified loss computation combining Dirichlet Evidential Loss and Ordinal Wasserstein Loss.
        """
        targets = target_dict['target']
        device = targets.device

        # 1. Build Truncated Gaussian Soft Target Distributions
        q_ord, q_raw = self.build_gaussian_soft_target(targets, device)

        # 2. Extract or derive probabilities
        if "probabilities" in output_dict:
            prob_final = output_dict["probabilities"]
        else:
            logits = output_dict.get('clipwise_output', output_dict.get('logits'))
            prob_final = F.softmax(logits, dim=-1)

        perm = self.ordinal_perm.to(device)
        prob_ord = prob_final[:, perm]

        # 3. Wasserstein Loss on Fused Prediction
        emd_fused = self.compute_squared_emd(prob_ord, q_ord)
        total_emd = emd_fused

        # 4. Optional Wasserstein auxiliary loss on unimodal evidence heads
        if "evidence_v" in output_dict and "evidence_a" in output_dict:
            alpha_v = output_dict["evidence_v"] + 1.0
            prob_v = alpha_v / torch.sum(alpha_v, dim=-1, keepdim=True)
            emd_v = self.compute_squared_emd(prob_v[:, perm], q_ord)

            alpha_a = output_dict["evidence_a"] + 1.0
            prob_a = alpha_a / torch.sum(alpha_a, dim=-1, keepdim=True)
            emd_a = self.compute_squared_emd(prob_a[:, perm], q_ord)

            total_emd = emd_fused + self.aux_loss_weight * (emd_v + emd_a)

        # 5. Evidential Dirichlet Loss (Expected Cross-Entropy + KL Regularization)
        if "alpha_final" in output_dict:
            alpha_f = output_dict["alpha_final"]
            ace_loss = self._dirichlet_ace_loss(alpha_f, q_raw)
            kl_loss = self._dirichlet_kl_divergence(alpha_f, q_raw, epoch)
            evidential_loss = ace_loss + kl_loss
        else:
            # Fallback to Soft Cross-Entropy if model outputs standard logits
            logits = output_dict.get('clipwise_output', output_dict.get('logits'))
            evidential_loss = torch.mean(-torch.sum(q_raw * F.log_softmax(logits, dim=-1), dim=-1))

        # 6. Dynamic Joint Loss Combination
        lambda_ord = self.compute_lambda_ord(epoch)
        total_loss = evidential_loss + lambda_ord * total_emd

        return total_loss

    @torch.no_grad()
    def compute_ordinal_mae(self, prob_final: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Utility metric: Computes Mean Absolute Error on the physical ordinal scale
        (None=0, Weak=1, Medium=2, Strong=3).
        """
        device = targets.device
        if targets.ndim > 1 and targets.size(-1) > 1:
            raw_target = torch.argmax(targets, dim=-1)
        else:
            raw_target = targets.long().view(-1)

        raw_pred = torch.argmax(prob_final, dim=-1)

        rank_map = self.raw_to_rank.to(device)
        pred_ranks = rank_map[raw_pred].float()
        target_ranks = rank_map[raw_target].float()

        mae = torch.mean(torch.abs(pred_ranks - target_ranks)).item()
        return float(mae)
