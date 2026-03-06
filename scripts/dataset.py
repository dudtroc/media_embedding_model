"""
BGE-M3 학습용 Dataset 클래스.

학습 데이터에서 query, positive passage, hard_negative, negative를
로드하여 모델 학습에 사용합니다.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SceneTripletDataset(Dataset):
    """
    장면 검색 학습용 Triplet Dataset.

    각 샘플은 (query, positive, hard_negative_passage, negative_passage) 형태입니다.
    hard_negative_passage는 해당 query의 hard_negative 질의를 다른 장면의 passage와
    매칭한 것이 아니라, 같은 장면의 passage를 positive로, hard_negative 질의를
    negative example로 사용합니다.
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 512, num_hard_neg: int = 2, num_neg: int = 1):
        data_path = Path(data_path)
        if data_path.is_dir():
            # Load individual scene files from directory
            self.raw_data = []
            for scene_file in sorted(data_path.glob("scene_*.json")):
                with open(scene_file, "r", encoding="utf-8") as f:
                    self.raw_data.append(json.load(f))
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_neg = num_hard_neg
        self.num_neg = num_neg

        # Build index of all passages for cross-scene negatives
        self.all_passages = []
        self.samples = []

        for scene in self.raw_data:
            metadata = scene["metadata"]
            query_data = scene["query"]
            passage = self._metadata_to_passage(metadata)
            self.all_passages.append(passage)

            for normal_q in query_data["normal"]:
                self.samples.append({
                    "query": normal_q,
                    "positive": passage,
                    "hard_negative_queries": query_data.get("hard_negative", []),
                    "negative_queries": query_data.get("negative", []),
                })

    def _metadata_to_passage(self, metadata: dict) -> str:
        parts = [
            f"장소: {metadata.get('Place', '')}",
            f"시간: {metadata.get('Approximate Time', '')}",
            f"분위기: {metadata.get('Atmosphere', '')}",
            f"키워드: {', '.join(metadata.get('Keywords', []))}",
        ]
        for char in metadata.get("Main Characters", []):
            if isinstance(char, dict):
                parts.append(f"등장인물: {char.get('name', '')} ({char.get('type', '')}) - {char.get('description', '')}")
            else:
                parts.append(f"등장인물: {char}")
        parts.append(f"요약: {metadata.get('caption', '')}")
        actions = metadata.get("Action", [])
        if actions:
            parts.append(f"행동: {' '.join(actions)}")
        return " | ".join(parts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Select hard negatives (use other scene passages as hard negative passages)
        hard_neg_passages = []
        if sample["hard_negative_queries"]:
            selected = random.sample(
                sample["hard_negative_queries"],
                min(self.num_hard_neg, len(sample["hard_negative_queries"])),
            )
            # Hard negative: passage from another scene that could be confused
            # We use random other passages as the "confused" targets
            other_indices = [i for i in range(len(self.all_passages)) if self.all_passages[i] != sample["positive"]]
            for _ in selected:
                if other_indices:
                    rand_idx = random.choice(other_indices)
                    hard_neg_passages.append(self.all_passages[rand_idx])

        # Select easy negatives
        neg_passages = []
        other_indices = [i for i in range(len(self.all_passages)) if self.all_passages[i] != sample["positive"]]
        for _ in range(self.num_neg):
            if other_indices:
                rand_idx = random.choice(other_indices)
                neg_passages.append(self.all_passages[rand_idx])

        return {
            "query": sample["query"],
            "positive": sample["positive"],
            "hard_negatives": hard_neg_passages,
            "negatives": neg_passages,
        }


def collate_fn(batch: list[dict], tokenizer, max_length: int = 512) -> dict:
    """배치 데이터를 토크나이즈하여 모델 입력 형태로 변환합니다."""
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]

    # Flatten hard negatives and negatives
    hard_negatives = []
    hard_neg_counts = []
    negatives = []
    neg_counts = []

    for item in batch:
        hn = item["hard_negatives"]
        hard_negatives.extend(hn)
        hard_neg_counts.append(len(hn))

        n = item["negatives"]
        negatives.extend(n)
        neg_counts.append(len(n))

    query_enc = tokenizer(
        queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )
    positive_enc = tokenizer(
        positives, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
    )

    result = {
        "query_input_ids": query_enc["input_ids"],
        "query_attention_mask": query_enc["attention_mask"],
        "positive_input_ids": positive_enc["input_ids"],
        "positive_attention_mask": positive_enc["attention_mask"],
        "hard_neg_counts": hard_neg_counts,
        "neg_counts": neg_counts,
    }

    if hard_negatives:
        hn_enc = tokenizer(
            hard_negatives, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        result["hard_neg_input_ids"] = hn_enc["input_ids"]
        result["hard_neg_attention_mask"] = hn_enc["attention_mask"]

    if negatives:
        n_enc = tokenizer(
            negatives, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        result["neg_input_ids"] = n_enc["input_ids"]
        result["neg_attention_mask"] = n_enc["attention_mask"]

    return result
