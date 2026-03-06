"""
BGE-M3 Fine-tuning 스크립트 - Hard Negative 개선.

사용법:
    python scripts/train.py
    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from dataset import SceneTripletDataset, collate_fn
from loss import HardNegativeContrastiveLoss, OnlineHardNegativeMiningLoss

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = PROJECT_DIR / "configs" / "training_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def encode(model, input_ids, attention_mask):
    """BGE-M3 모델로부터 [CLS] 토큰 임베딩을 추출합니다."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # BGE-M3 uses CLS token embedding
    return outputs.last_hidden_state[:, 0, :]


def reshape_negatives(flat_embeds: torch.Tensor, counts: list[int]) -> torch.Tensor:
    """
    Flatten된 negative 임베딩을 (batch_size, num_neg, embed_dim)으로 재구성합니다.
    각 샘플의 negative 수가 다를 수 있으므로 padding 처리합니다.
    """
    batch_size = len(counts)
    max_count = max(counts) if counts else 0
    if max_count == 0:
        return None

    embed_dim = flat_embeds.size(-1)
    result = torch.zeros(batch_size, max_count, embed_dim, device=flat_embeds.device)

    offset = 0
    for i, count in enumerate(counts):
        if count > 0:
            result[i, :count] = flat_embeds[offset : offset + count]
            offset += count

    return result


def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, epoch, config):
    model.train()
    total_loss = 0.0
    total_infonce = 0.0
    total_triplet = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch in pbar:
        # Move to device
        query_ids = batch["query_input_ids"].to(device)
        query_mask = batch["query_attention_mask"].to(device)
        pos_ids = batch["positive_input_ids"].to(device)
        pos_mask = batch["positive_attention_mask"].to(device)

        # Encode query and positive
        query_embeds = encode(model, query_ids, query_mask)
        positive_embeds = encode(model, pos_ids, pos_mask)

        # Encode hard negatives if present
        hard_neg_embeds = None
        if "hard_neg_input_ids" in batch:
            hn_ids = batch["hard_neg_input_ids"].to(device)
            hn_mask = batch["hard_neg_attention_mask"].to(device)
            hn_flat = encode(model, hn_ids, hn_mask)
            hard_neg_embeds = reshape_negatives(hn_flat, batch["hard_neg_counts"])

        # Encode negatives if present
        neg_embeds = None
        if "neg_input_ids" in batch:
            n_ids = batch["neg_input_ids"].to(device)
            n_mask = batch["neg_attention_mask"].to(device)
            n_flat = encode(model, n_ids, n_mask)
            neg_embeds = reshape_negatives(n_flat, batch["neg_counts"])

        # Compute loss
        if isinstance(loss_fn, HardNegativeContrastiveLoss):
            loss_dict = loss_fn(query_embeds, positive_embeds, hard_neg_embeds, neg_embeds)
        else:
            loss_dict = loss_fn(query_embeds, positive_embeds)

        loss = loss_dict["loss"]

        # Gradient accumulation
        grad_accum = config["training"].get("gradient_accumulation_steps", 1)
        loss = loss / grad_accum

        loss.backward()

        if (num_batches + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss_dict["loss"].item()
        total_infonce += loss_dict.get("infonce_loss", torch.tensor(0.0)).item()
        total_triplet += loss_dict.get("triplet_loss", loss_dict.get("hard_margin_loss", torch.tensor(0.0))).item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "infonce": f"{total_infonce / num_batches:.4f}",
            "triplet": f"{total_triplet / num_batches:.4f}",
        })

    return {
        "loss": total_loss / max(num_batches, 1),
        "infonce_loss": total_infonce / max(num_batches, 1),
        "triplet_loss": total_triplet / max(num_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Collect all embeddings for retrieval metrics
    all_query_embeds = []
    all_positive_embeds = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        query_ids = batch["query_input_ids"].to(device)
        query_mask = batch["query_attention_mask"].to(device)
        pos_ids = batch["positive_input_ids"].to(device)
        pos_mask = batch["positive_attention_mask"].to(device)

        query_embeds = encode(model, query_ids, query_mask)
        positive_embeds = encode(model, pos_ids, pos_mask)

        hard_neg_embeds = None
        if "hard_neg_input_ids" in batch:
            hn_ids = batch["hard_neg_input_ids"].to(device)
            hn_mask = batch["hard_neg_attention_mask"].to(device)
            hn_flat = encode(model, hn_ids, hn_mask)
            hard_neg_embeds = reshape_negatives(hn_flat, batch["hard_neg_counts"])

        neg_embeds = None
        if "neg_input_ids" in batch:
            n_ids = batch["neg_input_ids"].to(device)
            n_mask = batch["neg_attention_mask"].to(device)
            n_flat = encode(model, n_ids, n_mask)
            neg_embeds = reshape_negatives(n_flat, batch["neg_counts"])

        if isinstance(loss_fn, HardNegativeContrastiveLoss):
            loss_dict = loss_fn(query_embeds, positive_embeds, hard_neg_embeds, neg_embeds)
        else:
            loss_dict = loss_fn(query_embeds, positive_embeds)

        total_loss += loss_dict["loss"].item()
        num_batches += 1

        all_query_embeds.append(query_embeds.cpu())
        all_positive_embeds.append(positive_embeds.cpu())

    avg_loss = total_loss / max(num_batches, 1)

    # Compute retrieval metrics
    all_query_embeds = torch.cat(all_query_embeds, dim=0)
    all_positive_embeds = torch.cat(all_positive_embeds, dim=0)

    metrics = compute_retrieval_metrics(all_query_embeds, all_positive_embeds)
    metrics["val_loss"] = avg_loss

    return metrics


def compute_retrieval_metrics(query_embeds: torch.Tensor, positive_embeds: torch.Tensor) -> dict:
    """Recall@K, MRR, NDCG 등 검색 성능 지표를 계산합니다."""
    import torch.nn.functional as F

    query_embeds = F.normalize(query_embeds, p=2, dim=-1)
    positive_embeds = F.normalize(positive_embeds, p=2, dim=-1)

    # Similarity matrix: (num_queries, num_passages)
    sim_matrix = torch.matmul(query_embeds, positive_embeds.t())
    num_queries = sim_matrix.size(0)

    # Ground truth: query i matches passage i
    labels = torch.arange(num_queries)

    # Rankings
    sorted_indices = sim_matrix.argsort(dim=1, descending=True)
    ranks = torch.zeros(num_queries, dtype=torch.long)
    for i in range(num_queries):
        rank_pos = (sorted_indices[i] == labels[i]).nonzero(as_tuple=True)[0]
        ranks[i] = rank_pos[0] if len(rank_pos) > 0 else num_queries

    # Recall@K
    recall_at_1 = (ranks < 1).float().mean().item()
    recall_at_5 = (ranks < 5).float().mean().item()
    recall_at_10 = (ranks < 10).float().mean().item()

    # MRR
    mrr = (1.0 / (ranks.float() + 1.0)).mean().item()

    # NDCG@10
    ndcg_10 = compute_ndcg(ranks, k=10)

    # Hard Negative Discrimination Rate
    # Percentage of queries where the positive is ranked higher than
    # the most similar non-positive
    hn_disc_rate = (ranks == 0).float().mean().item()

    return {
        "recall@1": recall_at_1,
        "recall@5": recall_at_5,
        "recall@10": recall_at_10,
        "mrr": mrr,
        "ndcg@10": ndcg_10,
        "hard_negative_discrimination_rate": hn_disc_rate,
    }


def compute_ndcg(ranks: torch.Tensor, k: int = 10) -> float:
    """NDCG@K를 계산합니다."""
    import math
    dcg = 0.0
    for rank in ranks:
        if rank < k:
            dcg += 1.0 / math.log2(rank.item() + 2)
    ideal_dcg = 1.0 / math.log2(2)  # Only one relevant doc per query
    ndcg = dcg / (len(ranks) * ideal_dcg) if ideal_dcg > 0 else 0.0
    return ndcg


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 Fine-tuning for Hard Negative improvement")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train_config = config["training"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = train_config["base_model"]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    # Load datasets
    data_dir = PROJECT_DIR / "data"
    train_dataset = SceneTripletDataset(
        data_dir / "train.json", tokenizer, max_length=train_config["max_seq_length"],
    )
    val_dataset = SceneTripletDataset(
        data_dir / "val.json", tokenizer, max_length=train_config["max_seq_length"],
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    collate = partial(collate_fn, tokenizer=tokenizer, max_length=train_config["max_seq_length"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )

    # Loss function
    loss_config = train_config["loss"]
    if loss_config["type"] == "hard_negative_contrastive":
        loss_fn = HardNegativeContrastiveLoss(
            temperature=loss_config["temperature"],
            margin=loss_config["margin"],
            hard_negative_weight=loss_config["hard_negative_weight"],
            negative_weight=loss_config["negative_weight"],
        )
    elif loss_config["type"] == "online_hard_negative":
        loss_fn = OnlineHardNegativeMiningLoss(
            temperature=loss_config["temperature"],
            margin=loss_config["margin"],
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")

    print(f"Loss function: {loss_config['type']}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    total_steps = len(train_loader) * train_config["epochs"] // train_config.get("gradient_accumulation_steps", 1)
    warmup_steps = int(total_steps * train_config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = None
    if train_config.get("fp16", False) and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # Output directory
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_mrr = 0.0
    training_log = []

    print(f"\nStarting training for {train_config['epochs']} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print()

    for epoch in range(train_config["epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device, epoch, config,
        )

        val_metrics = evaluate(model, val_loader, loss_fn, device)

        log_entry = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        training_log.append(log_entry)

        print(f"\n--- Epoch {epoch + 1} ---")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"  Recall@1:   {val_metrics['recall@1']:.4f}")
        print(f"  Recall@5:   {val_metrics['recall@5']:.4f}")
        print(f"  MRR:        {val_metrics['mrr']:.4f}")
        print(f"  NDCG@10:    {val_metrics['ndcg@10']:.4f}")
        print(f"  HN Disc:    {val_metrics['hard_negative_discrimination_rate']:.4f}")

        # Save best model
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            print(f"  => New best MRR! Saving model...")
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # Save final model
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete! Best MRR: {best_mrr:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
