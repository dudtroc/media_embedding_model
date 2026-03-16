"""
다양한 임베딩 모델 비교 평가 스크립트.

BGE-M3 (원본/파인튜닝)와 외부 모델(Qwen3-Embedding 등)을 동일 테스트셋으로 비교합니다.
evaluate_metrics.py와 동일한 지표를 사용하되, 모델별 임베딩 방식 차이를 처리합니다.

지표:
1. Positive Rate: normal 쿼리 중 임계값 이상으로 매칭된 비율
2. Negative Rate: negative/hard_negative 쿼리 중 임계값 미만으로 거부된 비율
3. 분리 성공률: normal 점수 > hard_negative 점수인 비교 쌍의 비율
4. 평균 마진: normal 점수와 hard_negative 점수의 평균 차이

사용법:
    # 기본 3개 모델 전체 비교 (sweep)
    python scripts/evaluate_compare_models.py --sweep

    # Qwen3 전체 3종 비교
    python scripts/evaluate_compare_models.py --models qwen3 qwen3-4b qwen3-8b --sweep

    # 특정 모델만 평가
    python scripts/evaluate_compare_models.py --models qwen3-4b

    # 단일 threshold
    python scripts/evaluate_compare_models.py --threshold 0.85

    # 파인튜닝 모델 경로 지정
    python scripts/evaluate_compare_models.py --finetuned_path ./models/bge-m3-finetuned_20000/best --sweep
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent

# ─── 모델별 인코딩 전략 ───


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Qwen3-Embedding용 last token pooling."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def encode_bge(
    model, tokenizer, texts: list[str], device,
    max_length: int = 512, batch_size: int = 32,
) -> torch.Tensor:
    """BGE-M3 인코딩: CLS token pooling."""
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


def encode_qwen3(
    model, tokenizer, texts: list[str], device,
    max_length: int = 512, batch_size: int = 32,
) -> torch.Tensor:
    """Qwen3-Embedding 인코딩: last token pooling, left padding."""
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
            embeds = _last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"],
            )
            embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)


# ─── 모델 정의 레지스트리 ───

QWEN3_TASK_INSTRUCTION = (
    "Given a media scene search query, retrieve the relevant scene description"
)


def _qwen3_format_query(query: str) -> str:
    """Qwen3-Embedding 쿼리에 instruction prefix 추가."""
    return f"Instruct: {QWEN3_TASK_INSTRUCTION}\nQuery:{query}"


MODEL_REGISTRY = {
    "bge-m3": {
        "name": "BGE-M3 (Original)",
        "model_id": "BAAI/bge-m3",
        "encode_fn": encode_bge,
        "format_query": None,  # 그대로 사용
        "tokenizer_kwargs": {},
        "batch_size": 32,
    },
    "bge-m3-finetuned": {
        "name": "BGE-M3 (Fine-tuned)",
        "model_id": None,  # --finetuned_path로 지정
        "encode_fn": encode_bge,
        "format_query": None,
        "tokenizer_kwargs": {},
        "batch_size": 32,
    },
    "qwen3": {
        "name": "Qwen3-Embedding-0.6B",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "encode_fn": encode_qwen3,
        "format_query": _qwen3_format_query,
        "tokenizer_kwargs": {"padding_side": "left"},
        "batch_size": 32,
    },
    "qwen3-4b": {
        "name": "Qwen3-Embedding-4B",
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "encode_fn": encode_qwen3,
        "format_query": _qwen3_format_query,
        "tokenizer_kwargs": {"padding_side": "left"},
        "batch_size": 8,
    },
    "qwen3-8b": {
        "name": "Qwen3-Embedding-8B",
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "encode_fn": encode_qwen3,
        "format_query": _qwen3_format_query,
        "tokenizer_kwargs": {"padding_side": "left"},
        "batch_size": 4,
    },
}

# ─── 공통 유틸리티 (evaluate_metrics.py와 동일) ───


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
    encode_fn, model, tokenizer, test_data: list, device,
    format_query=None, batch_size: int = 32,
) -> dict:
    """모델로 임베딩을 계산하고 쿼리-passage 간 유사도를 반환합니다."""
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
    print(f"  Batch size: {batch_size}")

    # passage 임베딩 (instruction 없이)
    passage_embeds = encode_fn(model, tokenizer, passages, device, batch_size=batch_size)

    def _compute_query_sims(query_pairs):
        if not query_pairs:
            return [], []
        texts = [q[0] for q in query_pairs]
        labels = [q[1] for q in query_pairs]
        if format_query:
            texts = [format_query(t) for t in texts]
        embeds = encode_fn(model, tokenizer, texts, device, batch_size=batch_size)
        sims = []
        for i, label in enumerate(labels):
            sim = torch.dot(embeds[i], passage_embeds[label]).item()
            sims.append(sim)
        return sims, labels

    normal_sims, normal_labels = _compute_query_sims(normal_queries)
    hn_sims, hn_labels = _compute_query_sims(hard_negative_queries)
    neg_sims, neg_labels = _compute_query_sims(negative_queries)

    return {
        "normal_sims": normal_sims,
        "normal_labels": normal_labels,
        "hn_sims": hn_sims,
        "hn_labels": hn_labels,
        "neg_sims": neg_sims,
        "neg_labels": neg_labels,
    }


def compute_metrics_from_sims(sims: dict, threshold: float) -> dict:
    """사전 계산된 유사도 결과로부터 threshold 기반 지표를 계산합니다."""
    normal_sims = sims["normal_sims"]
    normal_labels = sims["normal_labels"]
    hn_sims = sims["hn_sims"]
    hn_labels = sims["hn_labels"]
    neg_sims = sims["neg_sims"]

    normal_above = sum(1 for s in normal_sims if s >= threshold)
    positive_rate = (normal_above / len(normal_sims) * 100) if normal_sims else 0.0

    all_neg_sims = hn_sims + neg_sims
    neg_below = sum(1 for s in all_neg_sims if s < threshold)
    negative_rate = (neg_below / len(all_neg_sims) * 100) if all_neg_sims else 0.0

    hn_below = sum(1 for s in hn_sims if s < threshold)
    hn_negative_rate = (hn_below / len(hn_sims) * 100) if hn_sims else 0.0

    neg_only_below = sum(1 for s in neg_sims if s < threshold)
    neg_only_rate = (neg_only_below / len(neg_sims) * 100) if neg_sims else 0.0

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
                margin_sum += n_sim - h_sim

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


# ─── 출력 함수 ───


def print_multi_model_sweep(all_results: dict[str, list[dict]]):
    """여러 모델의 sweep 결과를 한눈에 비교하는 테이블을 출력합니다."""
    model_keys = list(all_results.keys())
    thresholds = [r["threshold"] for r in all_results[model_keys[0]]]

    metrics_to_show = [
        ("positive_rate (%)", "Positive Rate (%)"),
        ("negative_rate (%)", "Negative Rate (%)"),
        ("hard_negative_reject_rate (%)", "Hard Neg Reject Rate (%)"),
        ("easy_negative_reject_rate (%)", "Easy Neg Reject Rate (%)"),
    ]

    col_width = max(len(name) for _, name in MODEL_REGISTRY.items()
                    if _ in model_keys) if model_keys else 12
    col_width = max(col_width, 12)

    for metric_key, metric_name in metrics_to_show:
        print()
        print("=" * (12 + (col_width + 3) * len(model_keys)))
        print(f"  [{metric_name}]")
        print("=" * (12 + (col_width + 3) * len(model_keys)))

        header = f"  {'Threshold':>9}"
        for mk in model_keys:
            name = MODEL_REGISTRY[mk]["name"]
            header += f"  {name:>{col_width}}"
        print(header)
        print("  " + "-" * (9 + (col_width + 2) * len(model_keys)))

        for t_idx, t in enumerate(thresholds):
            row = f"  {t:>9.1f}"
            for mk in model_keys:
                val = all_results[mk][t_idx][metric_key]
                row += f"  {val:>{col_width}.2f}"
            print(row)

        print()

    # threshold 무관 지표
    print()
    print("=" * (38 + (col_width + 3) * len(model_keys)))
    print("  [Threshold 무관 지표]")
    print("=" * (38 + (col_width + 3) * len(model_keys)))

    header = f"  {'Metric':<35}"
    for mk in model_keys:
        name = MODEL_REGISTRY[mk]["name"]
        header += f"  {name:>{col_width}}"
    print(header)
    print("  " + "-" * (35 + (col_width + 2) * len(model_keys)))

    for key, name in [
        ("separation_success_rate (%)", "분리 성공률 (%)"),
        ("avg_margin", "평균 마진"),
        ("avg_normal_similarity", "Normal 평균 유사도"),
        ("avg_hard_negative_similarity", "Hard Neg 평균 유사도"),
        ("avg_negative_similarity", "Negative 평균 유사도"),
    ]:
        row = f"  {name:<35}"
        for mk in model_keys:
            val = all_results[mk][0][key]
            row += f"  {val:>{col_width}.4f}"
        print(row)

    print("=" * (38 + (col_width + 3) * len(model_keys)))


def print_multi_model_single(all_results: dict[str, dict]):
    """여러 모델의 단일 threshold 결과를 비교합니다."""
    model_keys = list(all_results.keys())
    col_width = max(
        len(MODEL_REGISTRY[mk]["name"]) for mk in model_keys
    )
    col_width = max(col_width, 12)

    print()
    print("=" * (38 + (col_width + 3) * len(model_keys)))
    print(f"  Multi-Model Comparison (threshold={all_results[model_keys[0]]['threshold']})")
    print("=" * (38 + (col_width + 3) * len(model_keys)))

    header = f"  {'Metric':<35}"
    for mk in model_keys:
        header += f"  {MODEL_REGISTRY[mk]['name']:>{col_width}}"
    print(header)
    print("  " + "-" * (35 + (col_width + 2) * len(model_keys)))

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
        row = f"  {display_name:<35}"
        for mk in model_keys:
            val = all_results[mk][key]
            row += f"  {val:>{col_width}.4f}"
        print(row)

    print("=" * (38 + (col_width + 3) * len(model_keys)))


def load_model(model_key: str, finetuned_path: str | None, device: torch.device):
    """모델과 토크나이저를 로드합니다."""
    config = MODEL_REGISTRY[model_key]
    model_id = config["model_id"]

    if model_key == "bge-m3-finetuned":
        model_id = finetuned_path or str(
            PROJECT_DIR / "models" / "bge-m3-finetuned_20000" / "best"
        )

    # 로컬 경로인 경우 존재 여부 확인
    model_path = Path(model_id)
    if model_path.is_absolute() or model_id.startswith("."):
        if not model_path.exists():
            raise FileNotFoundError(
                f"로컬 모델 경로를 찾을 수 없습니다: {model_id}\n"
                f"  --finetuned_path 옵션으로 올바른 경로를 지정하세요.\n"
                f"  예: --finetuned_path ./models/bge-m3-finetuned_20000/best"
            )

    print(f"\n  Loading: {config['name']} ({model_id})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, **config["tokenizer_kwargs"],
    )
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="다양한 임베딩 모델 비교 평가",
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["bge-m3", "bge-m3-finetuned", "qwen3"],
        choices=list(MODEL_REGISTRY.keys()),
        help="평가할 모델 목록 (default: bge-m3, bge-m3-finetuned, qwen3)",
    )
    parser.add_argument(
        "--finetuned_path", type=str, default=None,
        help="Fine-tuned BGE-M3 모델 경로",
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
        "--output", type=str, default=None,
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_path = args.test_data or (PROJECT_DIR / "data" / "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Test data: {len(test_data)} scenes from {test_path}")

    sweep_thresholds = [round(0.5 + 0.1 * i, 1) for i in range(6)]

    all_sweep_results = {}
    all_single_results = {}

    total_models = len(args.models)
    for model_idx, model_key in enumerate(args.models, start=1):
        config = MODEL_REGISTRY[model_key]
        print("\n" + "=" * 70)
        print(f"  [{model_idx}/{total_models}] Evaluating: {config['name']}")
        print("=" * 70)

        model, tokenizer = load_model(model_key, args.finetuned_path, device)

        sims = compute_similarities(
            config["encode_fn"], model, tokenizer, test_data, device,
            format_query=config["format_query"],
            batch_size=config["batch_size"],
        )

        if args.sweep:
            sweep_results = []
            for t in sweep_thresholds:
                metrics = compute_metrics_from_sims(sims, t)
                sweep_results.append(metrics)
            all_sweep_results[model_key] = sweep_results

            # 모델별 즉시 결과 출력
            print(f"\n  --- {config['name']} 결과 ---")
            print(f"  {'Threshold':>9}  {'Pos Rate':>10}  {'Neg Rate':>10}  {'HN Reject':>10}  {'Sep Rate':>10}  {'Avg Margin':>11}")
            print("  " + "-" * 65)
            for r in sweep_results:
                print(
                    f"  {r['threshold']:>9.1f}  "
                    f"{r['positive_rate (%)']:>10.2f}  "
                    f"{r['negative_rate (%)']:>10.2f}  "
                    f"{r['hard_negative_reject_rate (%)']:>10.2f}  "
                    f"{r['separation_success_rate (%)']:>10.2f}  "
                    f"{r['avg_margin']:>11.4f}"
                )
        else:
            metrics = compute_metrics_from_sims(sims, args.threshold)
            all_single_results[model_key] = metrics

            # 모델별 즉시 결과 출력
            print(f"\n  --- {config['name']} 결과 (threshold={args.threshold}) ---")
            for k, v in metrics.items():
                if k != "threshold":
                    print(f"  {k:<40} {v}")

        # 메모리 해제
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"\n  [{model_idx}/{total_models}] {config['name']} 완료. 다음 모델로 이동...")

    # 전체 모델 비교 테이블 출력
    print("\n\n" + "=" * 70)
    print("  전체 모델 비교 결과")
    print("=" * 70)
    if args.sweep:
        print_multi_model_sweep(all_sweep_results)
        save_data = {
            "thresholds": sweep_thresholds,
            "models": {
                mk: {
                    "name": MODEL_REGISTRY[mk]["name"],
                    "results": results,
                }
                for mk, results in all_sweep_results.items()
            },
        }
    else:
        print_multi_model_single(all_single_results)
        save_data = {
            "threshold": args.threshold,
            "models": {
                mk: {
                    "name": MODEL_REGISTRY[mk]["name"],
                    "results": results,
                }
                for mk, results in all_single_results.items()
            },
        }

    output_path = args.output or str(
        PROJECT_DIR / "models" / "evaluation_multi_model_comparison.json"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
