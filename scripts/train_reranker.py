"""bge-reranker-v2-m3 Fine-tuning 스크립트.

Dense embedding(dual-encoder) 대신 cross-encoder reranker를 학습합니다.
입력은 (query, passage) pair이며, 모델은 relevance score(로짓)를 출력합니다.

학습 모드:
1) classification: BCEWithLogitsLoss로 label(1/0) 학습
2) pairwise: 같은 query에 대해 pos score > neg score가 되도록 margin ranking loss 학습

기본은 classification 모드입니다.

사용법:
    python scripts/train_reranker.py
    python scripts/train_reranker.py --config configs/training_config.yaml --mode classification
    python scripts/train_reranker.py --mode pairwise
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from reranker_dataset import RerankerCollator, ScenePairDataset, ScenePairwiseDataset

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = PROJECT_DIR / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch_classification(model, dataloader, optimizer, scheduler, device, epoch, grad_accum: int = 1):
    model.train()
    total_loss = 0.0
    num_steps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [classification]")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        batch = _move_to_device(batch, device)

        # AutoModelForSequenceClassification expects labels shape (bs,) for regression/binary.
        # We pass float labels and compute BCEWithLogits manually for clarity.
        labels = batch.pop("labels")

        outputs = model(**batch)
        logits = outputs.logits.squeeze(-1)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        num_steps += 1
        pbar.set_postfix({"loss": f"{total_loss / max(num_steps, 1):.4f}"})

    return {"loss": total_loss / max(num_steps, 1)}


def train_one_epoch_pairwise(model, dataloader, optimizer, scheduler, device, epoch, margin: float = 1.0, grad_accum: int = 1):
    model.train()
    total_loss = 0.0
    num_steps = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [pairwise]")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        pos = _move_to_device(batch["pos"], device)
        neg = _move_to_device(batch["neg"], device)

        pos_logits = model(**pos).logits.squeeze(-1)
        neg_logits = model(**neg).logits.squeeze(-1)

        # want pos_logits > neg_logits
        loss = torch.nn.functional.relu(margin - (pos_logits - neg_logits)).mean()
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        num_steps += 1
        pbar.set_postfix({"loss": f"{total_loss / max(num_steps, 1):.4f}"})

    return {"loss": total_loss / max(num_steps, 1)}


@torch.no_grad()
def evaluate_classification(model, dataloader, device, threshold: float = 0.0) -> dict:
    """Binary classification 관점의 간단 평가.

    threshold는 logit 기준 (0이면 sigmoid 0.5).
    """
    model.eval()
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating [classification]"):
        batch = _move_to_device(batch, device)
        labels = batch.pop("labels")
        logits = model(**batch).logits.squeeze(-1)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    preds = (logits >= threshold).float()
    acc = (preds == labels).float().mean().item()

    # AUC (optional)
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labels.numpy(), logits.numpy())
    except Exception:
        auc = None

    return {"accuracy": acc, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune bge-reranker-v2-m3")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="classification", choices=["classification", "pairwise"])

    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--val_data", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output_dir", type=str, default="./models/bge-reranker-v2-m3-finetuned")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    # classification dataset options
    parser.add_argument("--negative_source", type=str, default="hard_negative", choices=["hard_negative", "negative", "both"])
    parser.add_argument("--negatives_per_positive", type=int, default=1)

    # pairwise dataset options
    parser.add_argument("--num_neg_passages", type=int, default=1)
    parser.add_argument("--prefer_confusable", action="store_true")
    parser.add_argument("--pairwise_margin", type=float, default=1.0)

    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})

    # hyperparams fallback: reuse existing training config keys when possible
    epochs = args.epochs or int(train_cfg.get("epochs", 3))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 8))
    lr = args.learning_rate or float(train_cfg.get("learning_rate", 2e-5))
    max_len = args.max_seq_length or int(train_cfg.get("max_seq_length", 512))
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else float(train_cfg.get("warmup_ratio", 0.1))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_cfg.get("weight_decay", 0.01))
    grad_accum = args.gradient_accumulation_steps or int(train_cfg.get("gradient_accumulation_steps", 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model/tokenizer
    print(f"Loading reranker model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # num_labels=1 => regression/binary logit
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)

    # Data
    train_path = Path(args.train_data) if args.train_data else (PROJECT_DIR / "data" / "train.json")
    val_path = Path(args.val_data) if args.val_data else (PROJECT_DIR / "data" / "val.json")

    if args.mode == "classification":
        train_ds = ScenePairDataset(
            train_path,
            negative_source=args.negative_source,
            negatives_per_positive=args.negatives_per_positive,
        )
        val_ds = ScenePairDataset(
            val_path,
            negative_source=args.negative_source,
            negatives_per_positive=args.negatives_per_positive,
        )
        collator = RerankerCollator(tokenizer=tokenizer, max_length=max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator)

    else:
        train_ds = ScenePairwiseDataset(
            train_path,
            num_neg_passages=args.num_neg_passages,
            prefer_confusable=args.prefer_confusable,
        )
        val_ds = ScenePairwiseDataset(
            val_path,
            num_neg_passages=args.num_neg_passages,
            prefer_confusable=args.prefer_confusable,
        )
        collator = RerankerCollator(tokenizer=tokenizer, max_length=max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collator)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Optimizer/scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = (len(train_loader) * epochs) // max(grad_accum, 1)
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    training_log = []

    for epoch in range(epochs):
        if args.mode == "classification":
            train_metrics = train_one_epoch_classification(
                model, train_loader, optimizer, scheduler, device, epoch, grad_accum=grad_accum
            )
            val_metrics = evaluate_classification(model, val_loader, device)
            metric_for_best = val_metrics.get("auc") if val_metrics.get("auc") is not None else val_metrics.get("accuracy")
        else:
            train_metrics = train_one_epoch_pairwise(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                epoch,
                margin=args.pairwise_margin,
                grad_accum=grad_accum,
            )
            # pairwise는 간단히 loss만 기록 (별도 retrieval 평가 스크립트 제공)
            val_metrics = {"note": "pairwise mode: use evaluate_reranker.py for rerank metrics"}
            metric_for_best = -(train_metrics["loss"])  # lower is better

        log_entry = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        training_log.append(log_entry)

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        # Save best
        if metric_for_best is not None and metric_for_best > best_metric:
            best_metric = metric_for_best
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"=> New best ({best_metric}). Saved to {best_dir}")

        # checkpoint per epoch
        ckpt_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # final
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    print(f"\nTraining complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
