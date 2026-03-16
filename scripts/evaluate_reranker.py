"""Reranker 평가 스크립트.

Dense retrieval처럼 임베딩을 미리 만들어 전체 similarity matrix를 계산하는 대신,
reranker는 (query, passage) pair를 넣어 score를 얻습니다.

여기서는 테스트 데이터에서
1) 각 scene의 passage 후보군을 구성하고
2) normal query에 대해 정답 passage가 top-k에 드는지(Recall@K, MRR)
3) hard_negative query에 대해 해당 scene passage가 top-1이 아닌지(Discrimination)

를 측정합니다.

주의: reranker는 모든 (query, passage) 조합을 점수화해야 해서 비용이 큽니다.
--num_candidates로 후보 passage 수를 제한할 수 있습니다.

사용법:
    python scripts/evaluate_reranker.py --model_path ./models/bge-reranker-v2-m3-finetuned/best
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reranker_dataset import metadata_to_passage

PROJECT_DIR = Path(__file__).resolve().parent.parent


def score_pairs(model, tokenizer, queries: list[str], passages: list[str], device, max_length: int = 512, batch_size: int = 16) -> torch.Tensor:
    """(queries[i], passages[i])를 score로 변환. return shape: (N,) logits."""
    assert len(queries) == len(passages)

    all_logits = []
    for i in range(0, len(queries), batch_size):
        q = queries[i : i + batch_size]
        p = passages[i : i + batch_size]
        enc = tokenizer(
            q,
            p,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1)
        all_logits.append(logits.detach().cpu())

    return torch.cat(all_logits, dim=0)


def build_candidate_indices(num_passages: int, gt_idx: int, num_candidates: int | None, rng: random.Random) -> list[int]:
    """정답(gt_idx)을 포함하는 후보 인덱스 리스트."""
    if num_candidates is None or num_candidates >= num_passages:
        return list(range(num_passages))

    # sample negatives
    others = [i for i in range(num_passages) if i != gt_idx]
    sampled = rng.sample(others, k=min(num_candidates - 1, len(others)))
    return [gt_idx] + sampled


def compute_ranks_for_queries(
    model,
    tokenizer,
    passages: list[str],
    queries_with_gt: list[tuple[str, int]],
    device,
    max_length: int,
    batch_size: int,
    num_candidates: int | None,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    ranks: list[int] = []

    for q_text, gt_idx in tqdm(queries_with_gt, desc="Scoring queries"):
        cand_indices = build_candidate_indices(len(passages), gt_idx, num_candidates, rng)
        cand_passages = [passages[i] for i in cand_indices]

        q_list = [q_text] * len(cand_passages)
        logits = score_pairs(model, tokenizer, q_list, cand_passages, device, max_length=max_length, batch_size=batch_size)

        # rank within candidates
        sorted_local = torch.argsort(logits, descending=True)
        # find local index of gt
        gt_local = cand_indices.index(gt_idx)
        rank_pos = (sorted_local == gt_local).nonzero(as_tuple=True)[0]
        rank = int(rank_pos[0].item()) if len(rank_pos) > 0 else len(cand_indices)
        ranks.append(rank)

    return ranks


def main():
    parser = argparse.ArgumentParser(description="Evaluate bge-reranker-v2-m3 fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned reranker model")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Do not try to reach HuggingFace Hub. Load only from local files.")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_candidates", type=int, default=None, help="Limit candidate passages per query (includes GT)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_path = Path(args.test_data) if args.test_data else (PROJECT_DIR / "data" / "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    passages: list[str] = [metadata_to_passage(scene["metadata"]) for scene in test_data]

    normal_queries: list[tuple[str, int]] = []
    hard_neg_queries: list[tuple[str, int]] = []

    for idx, scene in enumerate(test_data):
        q = scene.get("query", {})
        for nq in q.get("normal", []):
            normal_queries.append((nq, idx))
        for hq in q.get("hard_negative", []):
            hard_neg_queries.append((hq, idx))

    print(f"Passages: {len(passages)}")
    print(f"Normal queries: {len(normal_queries)}")
    print(f"Hard negative queries: {len(hard_neg_queries)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    model.to(device)
    model.eval()

    # Normal query ranks
    ranks = compute_ranks_for_queries(
        model,
        tokenizer,
        passages,
        normal_queries,
        device,
        max_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        seed=args.seed,
    )

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    recall_1 = (ranks_t < 1).float().mean().item()
    recall_5 = (ranks_t < 5).float().mean().item()
    recall_10 = (ranks_t < 10).float().mean().item()
    mrr = (1.0 / (ranks_t + 1.0)).mean().item()

    # Hard negative discrimination: should NOT rank its associated passage as #1
    hn_disc = None
    if hard_neg_queries:
        hn_ranks = compute_ranks_for_queries(
            model,
            tokenizer,
            passages,
            hard_neg_queries,
            device,
            max_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_candidates=args.num_candidates,
            seed=args.seed,
        )
        hn_ranks_t = torch.tensor(hn_ranks, dtype=torch.long)
        hn_disc = (hn_ranks_t != 0).float().mean().item()

    metrics = {
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr,
        "hard_negative_discrimination_rate": hn_disc,
        "num_candidates": args.num_candidates,
    }

    print("\n" + "=" * 60)
    print("Reranker Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if v is None:
            print(f"{k}: None")
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    out_path = Path(args.model_path) / "evaluation_reranker.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
