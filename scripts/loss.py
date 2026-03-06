"""
Hard Negative 개선을 위한 커스텀 손실 함수.

일반적인 Contrastive Loss는 easy negative에 대해서는 잘 작동하지만,
hard negative를 구분하는 데 어려움이 있습니다.

이 모듈에서는 Hard Negative에 더 높은 가중치를 부여하는
HardNegativeContrastiveLoss를 구현합니다.

구성:
1. InfoNCE Loss (in-batch negatives 활용)
2. Hard Negative에 대한 가중 margin triplet loss
3. 두 loss의 가중 합산
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeContrastiveLoss(nn.Module):
    """
    Hard Negative를 강하게 밀어내는 Contrastive Loss.

    L = L_infonce + alpha * L_hard_triplet

    - L_infonce: query와 positive 사이의 유사도를 최대화하고,
                 in-batch negative와의 유사도를 최소화
    - L_hard_triplet: hard negative와 positive 사이의 margin을 강제
    """

    def __init__(
        self,
        temperature: float = 0.05,
        margin: float = 0.3,
        hard_negative_weight: float = 3.0,
        negative_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        self.negative_weight = negative_weight

    def forward(
        self,
        query_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
        hard_negative_embeds: torch.Tensor | None = None,
        negative_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            query_embeds: (batch_size, embed_dim) - 질의 임베딩
            positive_embeds: (batch_size, embed_dim) - 정답 passage 임베딩
            hard_negative_embeds: (batch_size, num_hard_neg, embed_dim) - hard negative 임베딩 (optional)
            negative_embeds: (batch_size, num_neg, embed_dim) - easy negative 임베딩 (optional)

        Returns:
            dict with 'loss', 'infonce_loss', 'triplet_loss' keys
        """
        # Normalize embeddings
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)

        # --- 1. InfoNCE Loss with in-batch negatives ---
        # Similarity matrix: (batch_size, batch_size)
        sim_matrix = torch.matmul(query_embeds, positive_embeds.t()) / self.temperature

        # Labels: diagonal entries are positives
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        infonce_loss = F.cross_entropy(sim_matrix, labels)

        # --- 2. Hard Negative Triplet Loss ---
        triplet_loss = torch.tensor(0.0, device=query_embeds.device)
        num_triplet_terms = 0

        if hard_negative_embeds is not None:
            hard_negative_embeds = F.normalize(hard_negative_embeds, p=2, dim=-1)

            # query-positive similarity: (batch_size,)
            pos_sim = (query_embeds * positive_embeds).sum(dim=-1)

            # query-hard_negative similarity: (batch_size, num_hard_neg)
            hard_neg_sim = torch.bmm(
                hard_negative_embeds,
                query_embeds.unsqueeze(-1),
            ).squeeze(-1)

            # Margin-based triplet loss: max(0, margin - (pos_sim - hard_neg_sim))
            # Hard negative should be at least `margin` further than positive
            hard_triplet = F.relu(
                self.margin - (pos_sim.unsqueeze(-1) - hard_neg_sim)
            )
            triplet_loss = triplet_loss + self.hard_negative_weight * hard_triplet.mean()
            num_triplet_terms += 1

        if negative_embeds is not None:
            negative_embeds = F.normalize(negative_embeds, p=2, dim=-1)

            pos_sim = (query_embeds * positive_embeds).sum(dim=-1)

            neg_sim = torch.bmm(
                negative_embeds,
                query_embeds.unsqueeze(-1),
            ).squeeze(-1)

            neg_triplet = F.relu(
                self.margin - (pos_sim.unsqueeze(-1) - neg_sim)
            )
            triplet_loss = triplet_loss + self.negative_weight * neg_triplet.mean()
            num_triplet_terms += 1

        total_loss = infonce_loss + triplet_loss

        return {
            "loss": total_loss,
            "infonce_loss": infonce_loss,
            "triplet_loss": triplet_loss,
        }


class OnlineHardNegativeMiningLoss(nn.Module):
    """
    In-batch에서 가장 어려운 negative를 자동으로 찾아 학습하는 Loss.

    학습 중에 batch 내에서 가장 유사한 negative를 동적으로 선택하여
    Hard Negative 구분 능력을 강화합니다.
    """

    def __init__(self, temperature: float = 0.05, margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        query_embeds: torch.Tensor,
        positive_embeds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)
        positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)

        batch_size = query_embeds.size(0)

        # Full similarity matrix
        sim_matrix = torch.matmul(query_embeds, positive_embeds.t()) / self.temperature

        # Positive similarities (diagonal)
        pos_sim = sim_matrix.diag()

        # Mask out positives to find hardest in-batch negatives
        mask = torch.eye(batch_size, device=sim_matrix.device, dtype=torch.bool)
        neg_sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        # Hardest negative per query (most similar non-matching passage)
        hardest_neg_sim = neg_sim_matrix.max(dim=1).values

        # InfoNCE
        infonce_loss = F.cross_entropy(sim_matrix, torch.arange(batch_size, device=sim_matrix.device))

        # Hard margin loss on mined negatives
        hard_margin_loss = F.relu(self.margin - (pos_sim - hardest_neg_sim)).mean()

        total_loss = infonce_loss + hard_margin_loss

        return {
            "loss": total_loss,
            "infonce_loss": infonce_loss,
            "hard_margin_loss": hard_margin_loss,
        }
