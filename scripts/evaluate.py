"""
BGE-M3 Fine-tuned 모델 평가 스크립트.

학습 전/후 모델의 Hard Negative 구분 성능을 비교 평가합니다.

사용법:
    python scripts/evaluate.py
    python scripts/evaluate.py --model_path ./models/bge-m3-finetuned/best
    python scripts/evaluate.py --compare  # 원본 vs fine-tuned 비교
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent


def encode_texts(model, tokenizer, texts: list[str], device, max_length: int = 512, batch_size: int = 32) -> torch.Tensor:
    """텍스트 리스트를 임베딩 벡터로 변환합니다."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            embeds = outputs.last_hidden_state[:, 0, :]
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


def metadata_to_passage(metadata: dict) -> str:
    """메타데이터를 passage 텍스트로 변환합니다."""
    parts = [
        f"장소: {metadata.get('Place', '')}",
        f"시간: {metadata.get('Approximate Time', '')}",
        f"분위기: {metadata.get('Atmosphere', '')}",
        f"키워드: {', '.join(metadata.get('Keywords', []))}",
    ]
    for char in metadata.get("Main Characters", []):
        parts.append(f"등장인물: {char.get('name', '')} ({char.get('type', '')}) - {char.get('description', '')}")
    parts.append(f"요약: {metadata.get('caption', '')}")
    actions = metadata.get("Action", [])
    if actions:
        parts.append(f"행동: {' '.join(actions)}")
    return " | ".join(parts)


def evaluate_model(model, tokenizer, test_data: list, device) -> dict:
    """모델의 검색 성능을 평가합니다."""
    # Build passages and queries
    passages = []
    normal_queries = []      # (query_text, passage_index)
    hard_neg_queries = []    # (query_text, passage_index_that_should_NOT_match)

    for idx, scene in enumerate(test_data):
        passage = metadata_to_passage(scene["metadata"])
        passages.append(passage)

        for q in scene["query"]["normal"]:
            normal_queries.append((q, idx))

        for q in scene["query"]["hard_negative"]:
            hard_neg_queries.append((q, idx))

    print(f"  Passages: {len(passages)}")
    print(f"  Normal queries: {len(normal_queries)}")
    print(f"  Hard negative queries: {len(hard_neg_queries)}")

    # Encode all passages
    passage_embeds = encode_texts(model, tokenizer, passages, device)

    # --- Normal Query Evaluation (Recall, MRR) ---
    query_texts = [q[0] for q in normal_queries]
    query_labels = [q[1] for q in normal_queries]
    query_embeds = encode_texts(model, tokenizer, query_texts, device)

    sim_matrix = torch.matmul(query_embeds, passage_embeds.t())

    ranks = []
    for i, label in enumerate(query_labels):
        sorted_idx = sim_matrix[i].argsort(descending=True)
        rank = (sorted_idx == label).nonzero(as_tuple=True)[0]
        ranks.append(rank[0].item() if len(rank) > 0 else len(passages))

    ranks_tensor = torch.tensor(ranks, dtype=torch.float)

    recall_1 = (ranks_tensor < 1).float().mean().item()
    recall_5 = (ranks_tensor < 5).float().mean().item()
    recall_10 = (ranks_tensor < 10).float().mean().item()
    mrr = (1.0 / (ranks_tensor + 1.0)).mean().item()

    # --- Hard Negative Discrimination ---
    # For each hard_negative query, check if the model correctly does NOT
    # rank the associated passage as #1
    if hard_neg_queries:
        hn_query_texts = [q[0] for q in hard_neg_queries]
        hn_scene_indices = [q[1] for q in hard_neg_queries]
        hn_embeds = encode_texts(model, tokenizer, hn_query_texts, device)

        hn_sim = torch.matmul(hn_embeds, passage_embeds.t())

        # A hard_negative query should NOT match its associated scene's passage
        hn_correct = 0
        hn_total = len(hard_neg_queries)
        hn_sim_to_wrong = []

        for i, scene_idx in enumerate(hn_scene_indices):
            top1_idx = hn_sim[i].argmax().item()
            if top1_idx != scene_idx:
                hn_correct += 1
            hn_sim_to_wrong.append(hn_sim[i, scene_idx].item())

        hn_discrimination_rate = hn_correct / hn_total if hn_total > 0 else 0.0
        avg_hn_sim = sum(hn_sim_to_wrong) / len(hn_sim_to_wrong) if hn_sim_to_wrong else 0.0
    else:
        hn_discrimination_rate = 0.0
        avg_hn_sim = 0.0

    # --- Positive vs Hard Negative Similarity Gap ---
    # Measure how well the model separates positives from hard negatives
    if normal_queries and hard_neg_queries:
        pos_sims = []
        for i, label in enumerate(query_labels):
            pos_sims.append(sim_matrix[i, label].item())
        avg_pos_sim = sum(pos_sims) / len(pos_sims)
        sim_gap = avg_pos_sim - avg_hn_sim
    else:
        avg_pos_sim = 0.0
        sim_gap = 0.0

    return {
        "recall@1": recall_1,
        "recall@5": recall_5,
        "recall@10": recall_10,
        "mrr": mrr,
        "hard_negative_discrimination_rate": hn_discrimination_rate,
        "avg_positive_similarity": avg_pos_sim,
        "avg_hard_negative_similarity": avg_hn_sim,
        "positive_hn_similarity_gap": sim_gap,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BGE-M3 model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to fine-tuned model (default: best checkpoint)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data JSON")
    parser.add_argument("--compare", action="store_true",
                        help="Compare original vs fine-tuned model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    test_path = args.test_data or (PROJECT_DIR / "data" / "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test data: {len(test_data)} scenes from {test_path}")

    if args.compare:
        # Compare original vs fine-tuned
        print("\n" + "=" * 60)
        print("Original BGE-M3")
        print("=" * 60)
        orig_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        orig_model = AutoModel.from_pretrained("BAAI/bge-m3", use_safetensors=True).to(device)
        orig_metrics = evaluate_model(orig_model, orig_tokenizer, test_data, device)
        del orig_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

        ft_path = args.model_path or str(PROJECT_DIR / "models" / "bge-m3-finetuned" / "best")
        print("\n" + "=" * 60)
        print(f"Fine-tuned BGE-M3: {ft_path}")
        print("=" * 60)
        ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
        ft_model = AutoModel.from_pretrained(ft_path, use_safetensors=True).to(device)
        ft_metrics = evaluate_model(ft_model, ft_tokenizer, test_data, device)

        # Print comparison
        print("\n" + "=" * 60)
        print("Comparison Results")
        print("=" * 60)
        print(f"{'Metric':<40} {'Original':>10} {'Fine-tuned':>10} {'Delta':>10}")
        print("-" * 70)
        for key in orig_metrics:
            orig_val = orig_metrics[key]
            ft_val = ft_metrics[key]
            delta = ft_val - orig_val
            sign = "+" if delta > 0 else ""
            print(f"{key:<40} {orig_val:>10.4f} {ft_val:>10.4f} {sign}{delta:>9.4f}")

        # Save comparison
        comparison = {"original": orig_metrics, "finetuned": ft_metrics}
        comp_path = PROJECT_DIR / "models" / "evaluation_comparison.json"
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {comp_path}")

    else:
        model_path = args.model_path or str(PROJECT_DIR / "models" / "bge-m3-finetuned" / "best")
        print(f"\nEvaluating model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, use_safetensors=True).to(device)

        metrics = evaluate_model(model, tokenizer, test_data, device)

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")

        results_path = PROJECT_DIR / "models" / "evaluation_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
