"""
BGE-M3 학습용 Dataset 클래스.

학습 데이터에서 query, positive passage, hard_negative passage, negative passage를
로드하여 모델 학습에 사용합니다.

Hard negative passage는 confusable_scenes(혼동 유발 장면)의 passage를 사용하여
랜덤 passage가 아닌 의미적으로 유사하지만 다른 passage를 활용합니다.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SceneTripletDataset(Dataset):
    """
    장면 검색 학습용 Triplet Dataset.

    각 샘플은 (query, positive_passage, hard_negative_passages, negative_passages) 형태입니다.
    hard_negative_passages는 confusable_scenes(원본과 인물/장소는 유사하나 행동/상황이
    미묘하게 다른 변형 장면)의 passage를 사용합니다.
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 512, num_hard_neg: int = 2, num_neg: int = 1):
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_neg = num_hard_neg
        self.num_neg = num_neg

        # Build passage index and samples
        self.all_passages = []
        self.genre_passage_map: dict[str, list[str]] = {}
        self.samples = []

        for scene in self.raw_data:
            metadata = scene["metadata"]
            query_data = scene["query"]
            genre = scene.get("genre", "unknown")
            passage = self._metadata_to_passage(metadata)
            self.all_passages.append(passage)

            # Index passages by genre for same-genre fallback negatives
            if genre not in self.genre_passage_map:
                self.genre_passage_map[genre] = []
            self.genre_passage_map[genre].append(passage)

            # Build hard negative passages from confusable_scenes
            confusable_scenes = scene.get("confusable_scenes", [])
            hard_neg_passages = [self._metadata_to_passage(cs) for cs in confusable_scenes if isinstance(cs, dict)]

            for normal_q in query_data["normal"]:
                self.samples.append({
                    "query": normal_q,
                    "positive": passage,
                    "hard_negative_passages": hard_neg_passages,
                    "genre": genre,
                })

    def _metadata_to_passage(self, metadata: dict) -> str:
        """메타데이터를 자연어 passage로 변환합니다."""
        parts = []

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
        char_descs = []
        for char in characters:
            if isinstance(char, dict):
                char_descs.append(f"{char.get('name', '')}({char.get('type', '')}): {char.get('description', '')}")
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Use confusable scene passages as hard negatives
        hard_neg_passages = list(sample["hard_negative_passages"])

        # If insufficient hard negatives, supplement with same-genre passages
        if len(hard_neg_passages) < self.num_hard_neg:
            genre = sample["genre"]
            same_genre_passages = [
                p for p in self.genre_passage_map.get(genre, [])
                if p != sample["positive"]
            ]
            needed = self.num_hard_neg - len(hard_neg_passages)
            if same_genre_passages:
                hard_neg_passages.extend(
                    random.sample(same_genre_passages, min(needed, len(same_genre_passages)))
                )

        # Limit to requested number
        if len(hard_neg_passages) > self.num_hard_neg:
            hard_neg_passages = random.sample(hard_neg_passages, self.num_hard_neg)

        # Select easy negatives from random other passages
        neg_passages = []
        other_passages = [p for p in self.all_passages if p != sample["positive"]]
        for _ in range(self.num_neg):
            if other_passages:
                rand_idx = random.randint(0, len(other_passages) - 1)
                neg_passages.append(other_passages[rand_idx])

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
