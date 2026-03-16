"""Reranker 학습용 Dataset/Collator.

Dense embedding(dual-encoder)와 달리 reranker(cross-encoder)는
(query, passage) pair를 함께 입력으로 받아 relevance score를 예측합니다.

이 모듈은 기존 생성 데이터 포맷(data/*.json)을 사용하여
- positive pair: (normal query, 해당 scene passage)
- negative pair: (hard_negative query, 해당 scene passage) 또는 (negative query, 해당 scene passage)

를 구성합니다.

또한 pairwise 학습을 위해 한 샘플에서 (query, pos_passage, neg_passage)를
만드는 Dataset도 제공합니다.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


def metadata_to_passage(metadata: dict) -> str:
    """메타데이터를 자연어 passage로 변환합니다 (기존 dataset/evaluate와 동일한 스타일)."""
    parts: list[str] = []

    place = metadata.get("Place", "")
    time_info = metadata.get("Approximate Time", "")
    atmosphere = metadata.get("Atmosphere", "")
    if place or time_info or atmosphere:
        if all([place, time_info, atmosphere]):
            parts.append(f"{place}에서 {time_info}에 벌어지는 {atmosphere} 장면이다.")
        else:
            parts.append(f"장소: {place}. 시간: {time_info}. 분위기: {atmosphere}.")

    caption = metadata.get("caption", "")
    if caption:
        parts.append(caption)

    characters = metadata.get("Main Characters", [])
    char_descs: list[str] = []
    for char in characters:
        if isinstance(char, dict):
            char_descs.append(
                f"{char.get('name', '')}({char.get('type', '')}): {char.get('description', '')}"
            )
        elif isinstance(char, str) and char.strip():
            char_descs.append(char)
    if char_descs:
        parts.append("등장인물: " + ". ".join(char_descs) + ".")

    actions = metadata.get("Action", [])
    if actions:
        parts.append(" ".join(actions))

    keywords = metadata.get("Keywords", [])
    if keywords:
        parts.append("키워드: " + ", ".join(keywords))

    return " ".join(parts)


class ScenePairDataset(Dataset):
    """Binary classification용 (query, passage, label) pair 데이터셋.

    label: 1 (relevant), 0 (non-relevant)

    구성 방식:
    - positive: scene.query.normal + scene.metadata(passage)
    - negative: scene.query.hard_negative / scene.query.negative + 동일 passage

    주의: hard_negative/negative 쿼리는 "해당 passage에 매칭되면 안 되는" 질문이므로
    동일 passage와 묶었을 때 non-relevant(0)로 둡니다.
    """

    def __init__(
        self,
        data_path: str | Path,
        negative_source: str = "hard_negative",  # hard_negative | negative | both
        negatives_per_positive: int = 1,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.samples: list[dict] = []

        for scene in raw_data:
            passage = metadata_to_passage(scene["metadata"])
            q = scene.get("query", {})
            normal_qs = list(q.get("normal", []))
            hn_qs = list(q.get("hard_negative", []))
            neg_qs = list(q.get("negative", []))

            # positives
            for nq in normal_qs:
                self.samples.append({"query": nq, "passage": passage, "label": 1})

                # attach negatives for balance (optional)
                candidates: list[str] = []
                if negative_source in ("hard_negative", "both"):
                    candidates.extend(hn_qs)
                if negative_source in ("negative", "both"):
                    candidates.extend(neg_qs)

                if candidates and negatives_per_positive > 0:
                    for _ in range(negatives_per_positive):
                        neg_q = self.rng.choice(candidates)
                        self.samples.append({"query": neg_q, "passage": passage, "label": 0})

        # Shuffle for training stability
        self.rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class ScenePairwiseDataset(Dataset):
    """Pairwise ranking용 (query, pos_passage, neg_passage) 데이터셋.

    한 normal query에 대해
    - pos_passage: 해당 scene passage
    - neg_passage: confusable_scenes passage 또는 랜덤 다른 scene passage

    이렇게 구성하면 reranker를 "같은 query에 대해 pos score > neg score"가 되도록 학습 가능.

    이 방식은 생성 데이터의 hard_negative query를 직접 쓰지 않고,
    passage 쪽에서 hard negative를 구성합니다.
    """

    def __init__(
        self,
        data_path: str | Path,
        num_neg_passages: int = 1,
        prefer_confusable: bool = True,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.passages: list[str] = [metadata_to_passage(scene["metadata"]) for scene in raw_data]
        self.samples: list[dict] = []

        for i, scene in enumerate(raw_data):
            pos_passage = self.passages[i]
            normal_qs = list(scene.get("query", {}).get("normal", []))

            confusable = scene.get("confusable_scenes", [])
            confusable_passages = [metadata_to_passage(cs) for cs in confusable if isinstance(cs, dict)]

            for nq in normal_qs:
                for _ in range(num_neg_passages):
                    neg_passage = None
                    if prefer_confusable and confusable_passages:
                        neg_passage = self.rng.choice(confusable_passages)
                    else:
                        # random other passage
                        candidates = [p for j, p in enumerate(self.passages) if j != i]
                        if candidates:
                            neg_passage = self.rng.choice(candidates)

                    if neg_passage is None:
                        continue

                    self.samples.append(
                        {"query": nq, "pos_passage": pos_passage, "neg_passage": neg_passage}
                    )

        self.rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


@dataclass
class RerankerCollator:
    tokenizer: any
    max_length: int = 512

    def __call__(self, batch: list[dict]) -> dict:
        # classification mode
        if "label" in batch[0]:
            queries = [b["query"] for b in batch]
            passages = [b["passage"] for b in batch]
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)

            enc = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc["labels"] = labels
            return enc

        # pairwise mode
        queries = [b["query"] for b in batch]
        pos_passages = [b["pos_passage"] for b in batch]
        neg_passages = [b["neg_passage"] for b in batch]

        pos_enc = self.tokenizer(
            queries,
            pos_passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        neg_enc = self.tokenizer(
            queries,
            neg_passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pos": pos_enc,
            "neg": neg_enc,
        }
