"""
임베딩 모델 벤치마크: 처리 속도 및 GPU 메모리 사용량 측정.

측정 항목:
  - 모델 로드 시간 (초)
  - GPU VRAM 사용량 (모델 로드 후 / 인코딩 중 최대)
  - 인코딩 처리 속도 (texts/sec)
  - 배치 크기별 throughput 및 latency
  - 파라미터 수 (M/B)

사용법:
    # 기본 실행 (전체 모델)
    python scripts/benchmark_models.py

    # 특정 모델만
    python scripts/benchmark_models.py --models bge-m3 qwen3

    # 배치 크기 및 반복 횟수 조정
    python scripts/benchmark_models.py --batch_sizes 8 16 32 --num_texts 200 --warmup 2 --runs 5

    # 파인튜닝 모델 경로 지정
    python scripts/benchmark_models.py --finetuned_path ./models/bge-m3-finetuned_20000/best

    # CPU만 사용
    python scripts/benchmark_models.py --device cpu
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ─── 더미 텍스트 생성 ───

_SAMPLE_SENTENCES = [
    "병원 응급실에서 새벽 2시에 벌어지는 긴박한 장면이다. 의사와 간호사가 중증 환자를 치료하고 있다.",
    "서울 도심 카페에서 오후 3시에 벌어지는 따뜻한 장면이다. 두 남녀가 커피를 마시며 대화를 나눈다.",
    "학교 운동장에서 점심시간에 벌어지는 활기찬 장면이다. 아이들이 뛰어놀며 즐거운 시간을 보낸다.",
    "경찰서 취조실에서 밤에 벌어지는 긴장된 장면이다. 형사가 용의자를 심문하고 있다.",
    "산속 오두막에서 새벽에 벌어지는 고요한 장면이다. 홀로 앉아 책을 읽는 노인의 모습이 담겨 있다.",
]


def make_dummy_texts(n: int) -> list[str]:
    """n개의 더미 텍스트 생성 (반복)."""
    return [_SAMPLE_SENTENCES[i % len(_SAMPLE_SENTENCES)] for i in range(n)]


# ─── 모델별 인코딩 전략 ───


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_len = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), seq_len
        ]


def encode_bge(model, tokenizer, texts: list[str], device, batch_size: int = 32, max_length: int = 512):
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=-1)
            all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)


def encode_qwen3(model, tokenizer, texts: list[str], device, batch_size: int = 32, max_length: int = 512):
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            emb = _last_token_pool(out.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(emb, p=2, dim=-1)
            all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)


# ─── 모델 레지스트리 ───

QWEN3_INSTRUCTION = "Given a media scene search query, retrieve the relevant scene description"

MODEL_REGISTRY = {
    "bge-m3": {
        "name": "BGE-M3 (Original)",
        "model_id": "BAAI/bge-m3",
        "encode_fn": encode_bge,
        "tokenizer_kwargs": {},
        "default_batch_size": 32,
    },
    "bge-m3-finetuned": {
        "name": "BGE-M3 (Fine-tuned)",
        "model_id": None,
        "encode_fn": encode_bge,
        "tokenizer_kwargs": {},
        "default_batch_size": 32,
    },
    "qwen3": {
        "name": "Qwen3-Embedding-0.6B",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "encode_fn": encode_qwen3,
        "tokenizer_kwargs": {"padding_side": "left"},
        "default_batch_size": 32,
    },
    "qwen3-4b": {
        "name": "Qwen3-Embedding-4B",
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "encode_fn": encode_qwen3,
        "tokenizer_kwargs": {"padding_side": "left"},
        "default_batch_size": 8,
    },
    "qwen3-8b": {
        "name": "Qwen3-Embedding-8B",
        "model_id": "Qwen/Qwen3-Embedding-8B",
        "encode_fn": encode_qwen3,
        "tokenizer_kwargs": {"padding_side": "left"},
        "default_batch_size": 4,
    },
}


# ─── GPU 메모리 유틸리티 ───

def get_gpu_memory_mb(device: torch.device) -> dict[str, float]:
    """현재 GPU 메모리 상태를 MB 단위로 반환."""
    if device.type != "cuda":
        return {"allocated": 0.0, "reserved": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated(device) / 1024 ** 2,
        "reserved": torch.cuda.memory_reserved(device) / 1024 ** 2,
    }


def reset_gpu_memory_stats(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory_mb(device: torch.device) -> float:
    """peak allocated VRAM (MB)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ─── 벤치마크 실행 ───

def benchmark_model(
    model_key: str,
    finetuned_path: str | None,
    device: torch.device,
    batch_sizes: list[int],
    num_texts: int,
    warmup_runs: int,
    measure_runs: int,
) -> dict:
    cfg = MODEL_REGISTRY[model_key]
    model_id = cfg["model_id"]

    if model_key == "bge-m3-finetuned":
        model_id = finetuned_path or str(
            PROJECT_DIR / "models" / "bge-m3-finetuned_20000" / "best"
        )

    model_path = Path(model_id)
    if model_path.is_absolute() or model_id.startswith("."):
        if not model_path.exists():
            return {
                "model_key": model_key,
                "name": cfg["name"],
                "error": f"경로 없음: {model_id}",
            }

    print(f"\n{'=' * 70}")
    print(f"  {cfg['name']}")
    print(f"  model_id: {model_id}")
    print(f"{'=' * 70}")

    # ── 메모리 초기화 ──
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    baseline_mem = get_gpu_memory_mb(device)["allocated"]

    # ── 모델 로드 시간 측정 ──
    print("  [1/3] 모델 로딩 중...", end="", flush=True)
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, **cfg["tokenizer_kwargs"])
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_time = time.perf_counter() - load_start
    print(f" {load_time:.2f}s")

    # ── 모델 로드 후 VRAM ──
    after_load_mem = get_gpu_memory_mb(device)["allocated"]
    model_vram_mb = after_load_mem - baseline_mem
    n_params = count_parameters(model)
    embed_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else None

    print(f"  파라미터: {n_params / 1e6:.1f}M  |  VRAM 점유: {model_vram_mb:.0f} MB  |  임베딩 차원: {embed_dim}")

    encode_fn = cfg["encode_fn"]
    texts = make_dummy_texts(num_texts)

    batch_results = []

    print(f"  [2/3] 워밍업 ({warmup_runs}회)...", end="", flush=True)
    default_bs = cfg["default_batch_size"]
    for _ in range(warmup_runs):
        encode_fn(model, tokenizer, texts[:default_bs], device, batch_size=default_bs)
    print(" 완료")

    print(f"  [3/3] 배치 크기별 속도 측정 ({measure_runs}회 평균)")
    for bs in batch_sizes:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        latencies = []
        errors = []
        for run in range(measure_runs):
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                encode_fn(model, tokenizer, texts, device, batch_size=bs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies.append(time.perf_counter() - t0)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    errors.append("OOM")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                raise

        peak_vram = get_peak_gpu_memory_mb(device)

        if errors:
            result = {
                "batch_size": bs,
                "status": "OOM",
                "throughput_texts_per_sec": None,
                "latency_per_text_ms": None,
                "peak_vram_mb": None,
                "inference_vram_mb": None,
            }
            print(f"    batch={bs:>3}  → OOM")
        else:
            avg_latency = sum(latencies) / len(latencies)
            throughput = num_texts / avg_latency
            inference_only_vram = peak_vram - model_vram_mb
            result = {
                "batch_size": bs,
                "status": "ok",
                "throughput_texts_per_sec": round(throughput, 2),
                "latency_per_text_ms": round(avg_latency / num_texts * 1000, 3),
                "peak_vram_mb": round(peak_vram, 1),
                "inference_vram_mb": round(max(inference_only_vram, 0), 1),
            }
            print(
                f"    batch={bs:>3}  "
                f"throughput={throughput:>7.1f} texts/s  "
                f"latency={avg_latency / num_texts * 1000:>6.3f} ms/text  "
                f"peak VRAM={peak_vram:>7.0f} MB"
            )

        batch_results.append(result)

    # ── 정리 ──
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "model_key": model_key,
        "name": cfg["name"],
        "model_id": model_id,
        "load_time_sec": round(load_time, 3),
        "num_parameters_M": round(n_params / 1e6, 1),
        "embedding_dim": embed_dim,
        "model_vram_mb": round(model_vram_mb, 1),
        "num_texts": num_texts,
        "warmup_runs": warmup_runs,
        "measure_runs": measure_runs,
        "batch_results": batch_results,
    }


# ─── 결과 요약 출력 ───

def print_summary(results: list[dict], batch_sizes: list[int]):
    ok_results = [r for r in results if "error" not in r]
    err_results = [r for r in results if "error" in r]

    if not ok_results:
        print("\n측정된 결과가 없습니다.")
        return

    # 모델 기본 스펙 표
    print("\n\n" + "=" * 90)
    print("  모델 기본 스펙")
    print("=" * 90)
    header = f"  {'모델':<30}  {'파라미터':>10}  {'임베딩 차원':>10}  {'로드 시간':>10}  {'모델 VRAM':>10}"
    print(header)
    print("  " + "-" * 76)
    for r in ok_results:
        print(
            f"  {r['name']:<30}  "
            f"{r['num_parameters_M']:>9.1f}M  "
            f"{str(r['embedding_dim'] or '-'):>10}  "
            f"{r['load_time_sec']:>9.2f}s  "
            f"{r['model_vram_mb']:>8.0f} MB"
        )
    print("=" * 90)

    # 배치 크기별 throughput 표
    print("\n\n" + "=" * 90)
    print("  처리 속도 (texts/sec) — 높을수록 좋음")
    print("=" * 90)
    col_w = 12
    header = f"  {'모델':<30}" + "".join(f"  {'batch='+str(bs):>{col_w}}" for bs in batch_sizes)
    print(header)
    print("  " + "-" * (30 + (col_w + 2) * len(batch_sizes)))
    for r in ok_results:
        row = f"  {r['name']:<30}"
        for bs in batch_sizes:
            br = next((b for b in r["batch_results"] if b["batch_size"] == bs), None)
            if br is None or br["status"] == "OOM":
                cell = "OOM"
            else:
                cell = f"{br['throughput_texts_per_sec']:.1f}"
            row += f"  {cell:>{col_w}}"
        print(row)
    print("=" * 90)

    # 배치 크기별 peak VRAM 표
    print("\n\n" + "=" * 90)
    print("  최대 VRAM 사용량 (MB) — peak allocated during encoding")
    print("=" * 90)
    header = f"  {'모델':<30}" + "".join(f"  {'batch='+str(bs):>{col_w}}" for bs in batch_sizes)
    print(header)
    print("  " + "-" * (30 + (col_w + 2) * len(batch_sizes)))
    for r in ok_results:
        row = f"  {r['name']:<30}"
        for bs in batch_sizes:
            br = next((b for b in r["batch_results"] if b["batch_size"] == bs), None)
            if br is None or br["status"] == "OOM":
                cell = "OOM"
            elif br["peak_vram_mb"] is None:
                cell = "N/A"
            else:
                cell = f"{br['peak_vram_mb']:.0f} MB"
            row += f"  {cell:>{col_w}}"
        print(row)
    print("=" * 90)

    if err_results:
        print("\n  [로드 실패 모델]")
        for r in err_results:
            print(f"  - {r['name']}: {r['error']}")


# ─── main ───

def main():
    parser = argparse.ArgumentParser(description="임베딩 모델 처리 속도 & VRAM 벤치마크")
    parser.add_argument(
        "--models", nargs="+",
        default=["bge-m3", "bge-m3-finetuned", "qwen3"],
        choices=list(MODEL_REGISTRY.keys()),
        help="벤치마크할 모델 목록 (default: bge-m3 bge-m3-finetuned qwen3)",
    )
    parser.add_argument(
        "--finetuned_path", type=str, default=None,
        help="파인튜닝 BGE-M3 경로 (default: models/bge-m3-finetuned_20000/best)",
    )
    parser.add_argument(
        "--batch_sizes", nargs="+", type=int, default=[1, 8, 16, 32],
        help="테스트할 배치 크기 목록 (default: 1 8 16 32)",
    )
    parser.add_argument(
        "--num_texts", type=int, default=100,
        help="인코딩할 텍스트 수 (default: 100)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="워밍업 반복 횟수 (default: 2)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="측정 반복 횟수 (평균 계산용, default: 3)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="사용할 device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="결과 저장 JSON 경로 (default: models/benchmark_results.json)",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n  Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_vram = props.total_memory / 1024 ** 2
        print(f"  GPU    : {props.name}  (VRAM {total_vram:.0f} MB)")
    print(f"  Models : {args.models}")
    print(f"  Batches: {args.batch_sizes}")
    print(f"  Texts  : {args.num_texts}  |  Warmup: {args.warmup}  |  Runs: {args.runs}")

    all_results = []
    for model_key in args.models:
        result = benchmark_model(
            model_key=model_key,
            finetuned_path=args.finetuned_path,
            device=device,
            batch_sizes=args.batch_sizes,
            num_texts=args.num_texts,
            warmup_runs=args.warmup,
            measure_runs=args.runs,
        )
        all_results.append(result)

    print_summary(all_results, args.batch_sizes)

    output_path = args.output or str(PROJECT_DIR / "models" / "benchmark_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {output_path}\n")


if __name__ == "__main__":
    main()
