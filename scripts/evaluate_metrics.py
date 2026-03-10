"""
임베딩 모델 정량적 평가 스크립트.

Test set을 이용하여 다음 4가지 지표를 측정합니다:
1. Positive Rate: normal 쿼리 중 임계값 이상으로 매칭된 비율
2. Negative Rate: negative/hard_negative 쿼리 중 임계값 미만으로 거부된 비율
3. 분리 성공률: normal 점수 > hard_negative 점수인 비교 쌍의 비율
4. 평균 마진: normal 점수와 hard_negative 점수의 평균 차이

사용법:
    python scripts/evaluate_metrics.py
    python scripts/evaluate_metrics.py --model_path ./models/bge-m3-finetuned/best
    python scripts/evaluate_metrics.py --threshold 0.8
    python scripts/evaluate_metrics.py --sweep  # 0.5~1.0 구간 threshold 일괄 테스트
    python scripts/evaluate_metrics.py --compare  # 원본 vs fine-tuned 비교
    python scripts/evaluate_metrics.py --compare --sweep  # 비교 + sweep
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent


def encode_texts(
    model, tokenizer, texts: list[str], device,
    max_length: int = 512, batch_size: int = 32,
) -> torch.Tensor:
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


def compute_similarities(
    model, tokenizer, test_data: list, device,
) -> dict:
    """
    모델로 임베딩을 계산하고, 쿼리-passage 간 유사도를 반환합니다.
    임베딩 계산은 비용이 크므로 한 번만 수행하고,
    threshold별 지표 계산은 compute_metrics_from_sims에서 수행합니다.
    """
    passages = []
    normal_queries = []
    hard_negative_queries = []
    negative_queries = []

    for idx, scene in enumerate(test_data):
        passage = metadata_to_passage(scene["metadata"])
        passages.append(passage)

        for q in scene["query"].get("normal", []):
            normal_queries.append((q, idx))

        for q in scene["query"].get("hard_negative", []):
            hard_negative_queries.append((q, idx))

        for q in scene["query"].get("negative", []):
            negative_queries.append((q, idx))

    print(f"  Passages: {len(passages)}")
    print(f"  Normal queries: {len(normal_queries)}")
    print(f"  Hard negative queries: {len(hard_negative_queries)}")
    print(f"  Negative queries: {len(negative_queries)}")

    # --- 임베딩 계산 ---
    passage_embeds = encode_texts(model, tokenizer, passages, device)

    normal_labels = [q[1] for q in normal_queries]
    normal_sims = []
    if normal_queries:
        normal_texts = [q[0] for q in normal_queries]
        normal_embeds = encode_texts(model, tokenizer, normal_texts, device)
        for i, label in enumerate(normal_labels):
            sim = torch.dot(normal_embeds[i], passage_embeds[label]).item()
            normal_sims.append(sim)

    hn_labels = [q[1] for q in hard_negative_queries]
    hn_sims = []
    if hard_negative_queries:
        hn_texts = [q[0] for q in hard_negative_queries]
        hn_embeds = encode_texts(model, tokenizer, hn_texts, device)
        for i, label in enumerate(hn_labels):
            sim = torch.dot(hn_embeds[i], passage_embeds[label]).item()
            hn_sims.append(sim)

    neg_labels = [q[1] for q in negative_queries]
    neg_sims = []
    if negative_queries:
        neg_texts = [q[0] for q in negative_queries]
        neg_embeds = encode_texts(model, tokenizer, neg_texts, device)
        for i, label in enumerate(neg_labels):
            sim = torch.dot(neg_embeds[i], passage_embeds[label]).item()
            neg_sims.append(sim)

    return {
        "normal_sims": normal_sims,
        "normal_labels": normal_labels,
        "hn_sims": hn_sims,
        "hn_labels": hn_labels,
        "neg_sims": neg_sims,
        "neg_labels": neg_labels,
    }


def compute_metrics_from_sims(sims: dict, threshold: float) -> dict:
    """
    사전 계산된 유사도 결과로부터 threshold 기반 지표를 계산합니다.
    """
    normal_sims = sims["normal_sims"]
    normal_labels = sims["normal_labels"]
    hn_sims = sims["hn_sims"]
    hn_labels = sims["hn_labels"]
    neg_sims = sims["neg_sims"]

    # 1. Positive Rate
    normal_above = sum(1 for s in normal_sims if s >= threshold)
    positive_rate = (normal_above / len(normal_sims) * 100) if normal_sims else 0.0

    # 2. Negative Rate
    all_neg_sims = hn_sims + neg_sims
    neg_below = sum(1 for s in all_neg_sims if s < threshold)
    negative_rate = (neg_below / len(all_neg_sims) * 100) if all_neg_sims else 0.0

    hn_below = sum(1 for s in hn_sims if s < threshold)
    hn_negative_rate = (hn_below / len(hn_sims) * 100) if hn_sims else 0.0

    neg_only_below = sum(1 for s in neg_sims if s < threshold)
    neg_only_rate = (neg_only_below / len(neg_sims) * 100) if neg_sims else 0.0

    # 3. 분리 성공률 & 4. 평균 마진 (threshold 무관, 상대 비교)
    scene_normal_sims: dict[int, list[float]] = {}
    for i, label in enumerate(normal_labels):
        scene_normal_sims.setdefault(label, []).append(normal_sims[i])

    scene_hn_sims: dict[int, list[float]] = {}
    for i, label in enumerate(hn_labels):
        scene_hn_sims.setdefault(label, []).append(hn_sims[i])

    total_pairs = 0
    separation_success = 0
    margin_sum = 0.0

    for scene_idx in scene_normal_sims:
        if scene_idx not in scene_hn_sims:
            continue
        for n_sim in scene_normal_sims[scene_idx]:
            for h_sim in scene_hn_sims[scene_idx]:
                total_pairs += 1
                if n_sim > h_sim:
                    separation_success += 1
                margin_sum += (n_sim - h_sim)

    separation_rate = (separation_success / total_pairs * 100) if total_pairs > 0 else 0.0
    avg_margin = (margin_sum / total_pairs) if total_pairs > 0 else 0.0

    avg_normal_sim = sum(normal_sims) / len(normal_sims) if normal_sims else 0.0
    avg_hn_sim = sum(hn_sims) / len(hn_sims) if hn_sims else 0.0
    avg_neg_sim = sum(neg_sims) / len(neg_sims) if neg_sims else 0.0

    return {
        "threshold": threshold,
        "positive_rate (%)": round(positive_rate, 2),
        "negative_rate (%)": round(negative_rate, 2),
        "hard_negative_reject_rate (%)": round(hn_negative_rate, 2),
        "easy_negative_reject_rate (%)": round(neg_only_rate, 2),
        "separation_success_rate (%)": round(separation_rate, 2),
        "avg_margin": round(avg_margin, 4),
        "avg_normal_similarity": round(avg_normal_sim, 4),
        "avg_hard_negative_similarity": round(avg_hn_sim, 4),
        "avg_negative_similarity": round(avg_neg_sim, 4),
        "total_normal_queries": len(normal_sims),
        "total_hard_negative_queries": len(hn_sims),
        "total_negative_queries": len(neg_sims),
        "total_comparison_pairs": total_pairs,
    }


def compute_metrics(
    model, tokenizer, test_data: list, device, threshold: float = 0.85,
) -> dict:
    """단일 threshold로 평가합니다."""
    sims = compute_similarities(model, tokenizer, test_data, device)
    return compute_metrics_from_sims(sims, threshold)


def compute_sweep(
    model, tokenizer, test_data: list, device,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    여러 threshold에 대해 일괄 평가합니다.
    임베딩은 한 번만 계산하고 threshold만 변경하여 지표를 산출합니다.
    """
    if thresholds is None:
        thresholds = [round(0.5 + 0.1 * i, 1) for i in range(6)]  # 0.5 ~ 1.0

    sims = compute_similarities(model, tokenizer, test_data, device)
    results = []
    for t in thresholds:
        metrics = compute_metrics_from_sims(sims, t)
        results.append(metrics)
    return results


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """평가 결과를 보기 좋게 출력합니다."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  Threshold: {metrics['threshold']}")
    print()
    print("  [핵심 지표]")
    print(f"    1. Positive Rate (정답 매칭률):      {metrics['positive_rate (%)']:>8.2f}%")
    print(f"    2. Negative Rate (오답 거부율):       {metrics['negative_rate (%)']:>8.2f}%")
    print(f"       - Hard Negative 거부율:            {metrics['hard_negative_reject_rate (%)']:>8.2f}%")
    print(f"       - Easy Negative 거부율:            {metrics['easy_negative_reject_rate (%)']:>8.2f}%")
    print(f"    3. 분리 성공률 (normal > hard_neg):   {metrics['separation_success_rate (%)']:>8.2f}%")
    print(f"    4. 평균 마진 (normal - hard_neg):     {metrics['avg_margin']:>8.4f}")
    print()
    print("  [유사도 통계]")
    print(f"    Normal 평균 유사도:                   {metrics['avg_normal_similarity']:>8.4f}")
    print(f"    Hard Negative 평균 유사도:            {metrics['avg_hard_negative_similarity']:>8.4f}")
    print(f"    Negative 평균 유사도:                 {metrics['avg_negative_similarity']:>8.4f}")
    print()
    print("  [데이터 수]")
    print(f"    Normal 쿼리:        {metrics['total_normal_queries']:>6d}개")
    print(f"    Hard Negative 쿼리: {metrics['total_hard_negative_queries']:>6d}개")
    print(f"    Negative 쿼리:      {metrics['total_negative_queries']:>6d}개")
    print(f"    비교 쌍:            {metrics['total_comparison_pairs']:>6d}개")
    print("=" * 70)


def print_sweep_table(sweep_results: list[dict], title: str = "Threshold Sweep"):
    """여러 threshold에 대한 결과를 한눈에 볼 수 있는 테이블로 출력합니다."""
    print()
    print("=" * 100)
    print(f"  {title}")
    print("=" * 100)
    print(
        f"  {'Threshold':>9}"
        f"  {'Positive':>10}"
        f"  {'Negative':>10}"
        f"  {'HN Reject':>10}"
        f"  {'Neg Reject':>10}"
        f"  {'Separation':>10}"
        f"  {'Avg Margin':>10}"
    )
    print(
        f"  {'':>9}"
        f"  {'Rate(%)':>10}"
        f"  {'Rate(%)':>10}"
        f"  {'Rate(%)':>10}"
        f"  {'Rate(%)':>10}"
        f"  {'Rate(%)':>10}"
        f"  {'':>10}"
    )
    print("  " + "-" * 96)

    for m in sweep_results:
        print(
            f"  {m['threshold']:>9.1f}"
            f"  {m['positive_rate (%)']:>10.2f}"
            f"  {m['negative_rate (%)']:>10.2f}"
            f"  {m['hard_negative_reject_rate (%)']:>10.2f}"
            f"  {m['easy_negative_reject_rate (%)']:>10.2f}"
            f"  {m['separation_success_rate (%)']:>10.2f}"
            f"  {m['avg_margin']:>10.4f}"
        )

    print("=" * 100)
    print()
    print("  [유사도 통계] (threshold 무관)")
    m0 = sweep_results[0]
    print(f"    Normal 평균 유사도:        {m0['avg_normal_similarity']:>8.4f}")
    print(f"    Hard Negative 평균 유사도: {m0['avg_hard_negative_similarity']:>8.4f}")
    print(f"    Negative 평균 유사도:      {m0['avg_negative_similarity']:>8.4f}")
    print(f"    분리 성공률:               {m0['separation_success_rate (%)']:>8.2f}%")
    print(f"    평균 마진:                 {m0['avg_margin']:>8.4f}")
    print()
    print(f"  [데이터 수]")
    print(f"    Normal 쿼리:        {m0['total_normal_queries']:>6d}개")
    print(f"    Hard Negative 쿼리: {m0['total_hard_negative_queries']:>6d}개")
    print(f"    Negative 쿼리:      {m0['total_negative_queries']:>6d}개")
    print(f"    비교 쌍:            {m0['total_comparison_pairs']:>6d}개")
    print("=" * 100)


def print_sweep_comparison(
    orig_sweep: list[dict], ft_sweep: list[dict],
):
    """두 모델의 sweep 결과를 나란히 비교하는 테이블을 출력합니다."""
    print()
    print("=" * 115)
    print("  Threshold Sweep Comparison: Original vs Fine-tuned")
    print("=" * 115)

    # Positive Rate 비교
    print()
    print("  [Positive Rate (%)]")
    print(f"  {'Threshold':>9}  {'Original':>10}  {'Fine-tuned':>10}  {'Delta':>10}")
    print("  " + "-" * 45)
    for o, f in zip(orig_sweep, ft_sweep):
        delta = f["positive_rate (%)"] - o["positive_rate (%)"]
        sign = "+" if delta > 0 else ""
        print(f"  {o['threshold']:>9.1f}  {o['positive_rate (%)']:>10.2f}  {f['positive_rate (%)']:>10.2f}  {sign}{delta:>9.2f}")

    # Negative Rate 비교
    print()
    print("  [Negative Rate (%)]")
    print(f"  {'Threshold':>9}  {'Original':>10}  {'Fine-tuned':>10}  {'Delta':>10}")
    print("  " + "-" * 45)
    for o, f in zip(orig_sweep, ft_sweep):
        delta = f["negative_rate (%)"] - o["negative_rate (%)"]
        sign = "+" if delta > 0 else ""
        print(f"  {o['threshold']:>9.1f}  {o['negative_rate (%)']:>10.2f}  {f['negative_rate (%)']:>10.2f}  {sign}{delta:>9.2f}")

    # Hard Negative Reject Rate 비교
    print()
    print("  [Hard Negative Reject Rate (%)]")
    print(f"  {'Threshold':>9}  {'Original':>10}  {'Fine-tuned':>10}  {'Delta':>10}")
    print("  " + "-" * 45)
    for o, f in zip(orig_sweep, ft_sweep):
        delta = f["hard_negative_reject_rate (%)"] - o["hard_negative_reject_rate (%)"]
        sign = "+" if delta > 0 else ""
        print(f"  {o['threshold']:>9.1f}  {o['hard_negative_reject_rate (%)']:>10.2f}  {f['hard_negative_reject_rate (%)']:>10.2f}  {sign}{delta:>9.2f}")

    # 분리 성공률 & 평균 마진 (threshold 무관)
    print()
    print("  [Threshold 무관 지표]")
    print(f"  {'Metric':<35}  {'Original':>10}  {'Fine-tuned':>10}  {'Delta':>10}")
    print("  " + "-" * 70)
    for key, name in [
        ("separation_success_rate (%)", "분리 성공률 (%)"),
        ("avg_margin", "평균 마진"),
        ("avg_normal_similarity", "Normal 평균 유사도"),
        ("avg_hard_negative_similarity", "Hard Neg 평균 유사도"),
        ("avg_negative_similarity", "Negative 평균 유사도"),
    ]:
        o_val = orig_sweep[0][key]
        f_val = ft_sweep[0][key]
        delta = f_val - o_val
        sign = "+" if delta > 0 else ""
        print(f"  {name:<35}  {o_val:>10.4f}  {f_val:>10.4f}  {sign}{delta:>9.4f}")

    print("=" * 115)


def print_comparison(orig_metrics: dict, ft_metrics: dict):
    """원본 vs fine-tuned 모델의 비교 결과를 출력합니다."""
    print()
    print("=" * 78)
    print("  Comparison: Original vs Fine-tuned")
    print("=" * 78)
    print(f"  {'Metric':<38} {'Original':>10} {'Fine-tuned':>10} {'Delta':>10}")
    print("  " + "-" * 74)

    keys = [
        ("positive_rate (%)", "Positive Rate (%)"),
        ("negative_rate (%)", "Negative Rate (%)"),
        ("hard_negative_reject_rate (%)", "Hard Neg Reject Rate (%)"),
        ("easy_negative_reject_rate (%)", "Easy Neg Reject Rate (%)"),
        ("separation_success_rate (%)", "Separation Success Rate (%)"),
        ("avg_margin", "Avg Margin"),
        ("avg_normal_similarity", "Avg Normal Similarity"),
        ("avg_hard_negative_similarity", "Avg Hard Neg Similarity"),
        ("avg_negative_similarity", "Avg Negative Similarity"),
    ]

    for key, display_name in keys:
        orig_val = orig_metrics[key]
        ft_val = ft_metrics[key]
        delta = ft_val - orig_val
        sign = "+" if delta > 0 else ""
        print(f"  {display_name:<38} {orig_val:>10.4f} {ft_val:>10.4f} {sign}{delta:>9.4f}")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="임베딩 모델 정량적 평가")
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="평가할 모델 경로 (default: best checkpoint)",
    )
    parser.add_argument(
        "--test_data", type=str, default=None,
        help="테스트 데이터 JSON 경로",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="유사도 임계값 (default: 0.85)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="threshold 0.5~1.0 구간을 0.1 간격으로 일괄 테스트",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="원본 vs fine-tuned 모델 비교",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test data
    test_path = args.test_data or (PROJECT_DIR / "data" / "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test data: {len(test_data)} scenes from {test_path}")

    sweep_thresholds = [round(0.5 + 0.1 * i, 1) for i in range(6)]  # 0.5 ~ 1.0

    if args.compare:
        # --- 원본 모델 ---
        print("\n" + "=" * 70)
        print("  Loading: Original BGE-M3")
        print("=" * 70)
        orig_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        orig_model = AutoModel.from_pretrained("BAAI/bge-m3", use_safetensors=True).to(device)

        if args.sweep:
            orig_sweep = compute_sweep(
                orig_model, orig_tokenizer, test_data, device, sweep_thresholds,
            )
            print_sweep_table(orig_sweep, "Original BGE-M3 - Threshold Sweep")
        else:
            orig_metrics = compute_metrics(
                orig_model, orig_tokenizer, test_data, device, args.threshold,
            )
            print_metrics(orig_metrics, "Original BGE-M3")

        del orig_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --- Fine-tuned 모델 ---
        ft_path = args.model_path or str(
            PROJECT_DIR / "models" / "bge-m3-finetuned" / "best"
        )
        print("\n" + "=" * 70)
        print(f"  Loading: Fine-tuned BGE-M3 ({ft_path})")
        print("=" * 70)
        ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
        ft_model = AutoModel.from_pretrained(ft_path, use_safetensors=True).to(device)

        if args.sweep:
            ft_sweep = compute_sweep(
                ft_model, ft_tokenizer, test_data, device, sweep_thresholds,
            )
            print_sweep_table(ft_sweep, f"Fine-tuned BGE-M3 - Threshold Sweep")
            print_sweep_comparison(orig_sweep, ft_sweep)

            output_path = args.output or str(
                PROJECT_DIR / "models" / "evaluation_sweep_comparison.json"
            )
            save_data = {
                "thresholds": sweep_thresholds,
                "original": orig_sweep,
                "finetuned": ft_sweep,
            }
        else:
            ft_metrics = compute_metrics(
                ft_model, ft_tokenizer, test_data, device, args.threshold,
            )
            print_metrics(ft_metrics, f"Fine-tuned BGE-M3 ({ft_path})")
            print_comparison(orig_metrics, ft_metrics)

            output_path = args.output or str(
                PROJECT_DIR / "models" / "evaluation_metrics_comparison.json"
            )
            save_data = {
                "threshold": args.threshold,
                "original": orig_metrics,
                "finetuned": ft_metrics,
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")

    else:
        # --- 단일 모델 평가 ---
        model_path = args.model_path or str(
            PROJECT_DIR / "models" / "bge-m3-finetuned" / "best"
        )
        print(f"\nLoading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, use_safetensors=True).to(device)

        if args.sweep:
            sweep_results = compute_sweep(
                model, tokenizer, test_data, device, sweep_thresholds,
            )
            print_sweep_table(sweep_results, f"Model: {model_path}")

            output_path = args.output or str(
                PROJECT_DIR / "models" / "evaluation_sweep.json"
            )
            save_data = {
                "model_path": model_path,
                "thresholds": sweep_thresholds,
                "results": sweep_results,
            }
        else:
            metrics = compute_metrics(model, tokenizer, test_data, device, args.threshold)
            print_metrics(metrics, f"Model: {model_path}")

            output_path = args.output or str(
                PROJECT_DIR / "models" / "evaluation_metrics.json"
            )
            save_data = metrics

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
