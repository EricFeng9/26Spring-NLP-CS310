# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Script for pretraining a small GPT-2 124M parameter model
on Chinese Wikipedia text data.

Before running this script, make sure you:
1. Extracted and preprocessed the text data
2. Trained a BPE tokenizer on the text data
"""

import argparse
import os
import time
from pathlib import Path

from tokenizers import Tokenizer
import torch
from utils import (
    GPTModel,
    calc_loss_batch,
    create_dataloader_v1,
    evaluate_model,
    plot_losses,
    read_data_from_path,
)


def create_dataloaders(
    text_data, tokenizer, train_ratio, batch_size, max_length, stride, num_workers=0
):
    """Create training and validation dataloaders from text data."""
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def convert_time(seconds):
    """Convert seconds to hours, minutes, seconds."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def train_model_simple(
    model,
    optimizer,
    device,
    n_epochs,
    eval_freq,
    eval_iter,
    output_dir,
    save_ckpt_freq,
    tokenizer,
    data_path,
    batch_size=1024,
    train_ratio=0.90,
):
    """
    Simple training loop for GPT model.
    
    Args:
        model: The GPT model to train
        optimizer: The optimizer
        device: Device to train on
        n_epochs: Number of epochs to train
        eval_freq: Evaluate every N steps
        eval_iter: Number of iterations for evaluation
        output_dir: Directory to save checkpoints
        save_ckpt_freq: Save checkpoint every N steps
        tokenizer: Tokenizer for encoding text
        data_path: Path to the training data file or directory
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training (rest for validation)
        
    Returns:
        Tuple of (train_losses, val_losses, track_tokens_seen)
    """
    ### START YOUR CODE ###
    # Initialize tracking variables
    import csv
    import json

    train_losses, val_losses, track_tokens_seen = [], [], []
    eval_records = []
    tokens_seen = 0
    global_step = 0
    start_time = time.time()

    #读取训练文本
    text_data = read_data_from_path(data_path)

    # 确保语料末尾有结束符，便于样本边界学习
    if not text_data.endswith("<|endoftext|>"):
        text_data = text_data.rstrip() + " <|endoftext|>"

    # 使用模型上下文长度构建 dataloader，stride 取上下文长度减少重叠
    max_length = model.pos_emb.num_embeddings
    train_loader, val_loader = create_dataloaders(
        text_data=text_data,
        tokenizer=tokenizer,
        train_ratio=train_ratio,
        batch_size=batch_size,
        max_length=max_length,
        stride=max_length,
        num_workers=0,
    )

    #打印关键信息
    total_batches = len(train_loader)
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else model.tok_emb.num_embeddings
    print("\n===== Training Data Summary =====")
    print(f"Data path: {data_path}")
    print(f"Data size (characters): {len(text_data):,}")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Context length: {max_length}")
    print(f"Train batches/epoch: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    print(f"Train ratio: {train_ratio:.2f}")
    print("=================================\n")

    #写入运行配置，方便后续报告直接用
    run_summary = {
        "data_path": str(data_path),
        "data_size_characters": len(text_data),
        "vocab_size": int(vocab_size),
        "context_length": int(max_length),
        "batch_size": int(batch_size),
        "train_ratio": float(train_ratio),
        "n_epochs": int(n_epochs),
        "eval_freq": int(eval_freq),
        "eval_iter": int(eval_iter),
        "save_ckpt_freq": int(save_ckpt_freq),
        "total_train_batches_per_epoch": int(total_batches),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    #前向、反向、优化，并按步数做评估与存档
    try:
        for epoch in range(n_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()

                global_step += 1
                tokens_seen += input_batch.numel()

                if eval_freq > 0 and global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        eval_iter=eval_iter,
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)

                    elapsed = time.time() - start_time
                    eval_records.append({
                        "global_step": int(global_step),
                        "epoch": int(epoch + 1),
                        "tokens_seen": int(tokens_seen),
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "elapsed_seconds": float(elapsed),
                    })
                    print(
                        f"[eval] step={global_step:,} "
                        f"tokens={tokens_seen:,} "
                        f"train_loss={train_loss:.4f} "
                        f"val_loss={val_loss:.4f}"
                    )

                if save_ckpt_freq > 0 and global_step % save_ckpt_freq == 0:
                    ckpt_path = output_dir / f"model_step_{global_step}.pth"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[ckpt] Saved checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        #中断时保存权重，避免训练进度丢失
        interrupted_ckpt = output_dir / f"model_interrupted_step_{global_step}.pth"
        torch.save(model.state_dict(), interrupted_ckpt)
        print(f"\nTraining interrupted. Saved checkpoint to: {interrupted_ckpt}")
    finally:
        #将评估记录保存为 CSV 与 JSON，便于作图和报告引用
        metrics_csv_path = output_dir / "training_metrics.csv"
        with open(metrics_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "global_step",
                    "epoch",
                    "tokens_seen",
                    "train_loss",
                    "val_loss",
                    "elapsed_seconds",
                ],
            )
            writer.writeheader()
            writer.writerows(eval_records)

        metrics_json_path = output_dir / "training_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_records, f, ensure_ascii=False, indent=2)

        # 保存一个结束时 checkpoint，便于恢复训练
        final_ckpt = output_dir / "model_last_checkpoint.pth"
        torch.save(model.state_dict(), final_ckpt)
        print(f"[ckpt] Saved final checkpoint: {final_ckpt}")
    
    ### END YOUR CODE ###
    
    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="GPT Model Training Configuration",
    )

    parser.add_argument(
        "--data_file", "--data",
        type=str,
        required=True,
        help="Path to the training data file or directory containing .txt files (e.g., data/wiki_zh_2019.txt or data/wiki_zh_2019/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_checkpoints",
        help="Directory where the model checkpoints will be saved",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the tokenizer JSON file (e.g., tokenizer/wikizh_tokenizer_whitespace.json)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=100,
        help="Frequency of evaluations during training (in steps)",
    )
    parser.add_argument(
        "--save_ckpt_freq",
        type=int,
        default=100_000,
        help="Frequency of saving model checkpoints during training (in steps)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.90, help="Ratio of data for training (rest for validation)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=52000, help="Vocabulary size (should match tokenizer)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Uses a very small model for debugging purposes",
    )

    args = parser.parse_args()

    # Set model configuration
    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,
            "context_length": 10,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    else:
        GPT_CONFIG_124M = {
            "vocab_size": args.vocab_size,  # Should match tokenizer vocab size
            "context_length": 1024,  # Context length
            "emb_dim": 768,  # Embedding dimension
            "n_heads": 12,  # Number of attention heads
            "n_layers": 12,  # Number of layers
            "drop_rate": 0.1,  # Dropout rate
            "qkv_bias": False,  # Query-key-value bias
        }

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    # Verify vocab size matches
    actual_vocab_size = tokenizer.get_vocab_size()
    if actual_vocab_size != args.vocab_size:
        print(f"Warning: Tokenizer vocab size ({actual_vocab_size}) doesn't match --vocab_size ({args.vocab_size})")
        print(f"Updating model config to use vocab size: {actual_vocab_size}")
        GPT_CONFIG_124M["vocab_size"] = actual_vocab_size

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {output_dir.absolute()}")

    # Train model
    print("\nStarting training...")
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        optimizer=optimizer,
        device=device,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        tokenizer=tokenizer,
        data_path=args.data_file,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
    )

    ### START YOUR CODE ###
    # TODO: Plot losses if train_losses is not empty
    # Hint: Use torch.linspace to create epochs_tensor and call plot_losses()
    import json

    if train_losses:
        # 将评估点映射到 epoch 轴，便于可视化训练曲线
        epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    else:
        print("No evaluation records were collected; skipping loss plot.")

    # TODO: Save the final model to output_dir / "model_final.pth"
    final_model_path = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    #保存最终摘要，方便报告直接用
    final_summary = {
        "data_file": args.data_file,
        "tokenizer_path": args.tokenizer,
        "output_dir": str(output_dir),
        "n_epochs": int(args.n_epochs),
        "batch_size": int(args.batch_size),
        "train_ratio": float(args.train_ratio),
        "learning_rate": float(args.lr),
        "eval_freq": int(args.eval_freq),
        "save_ckpt_freq": int(args.save_ckpt_freq),
        "vocab_size_arg": int(args.vocab_size),
        "model_vocab_size": int(GPT_CONFIG_124M["vocab_size"]),
        "num_eval_points": int(len(train_losses)),
        "last_train_loss": float(train_losses[-1]) if train_losses else None,
        "last_val_loss": float(val_losses[-1]) if val_losses else None,
        "last_tokens_seen": int(tokens_seen[-1]) if tokens_seen else None,
    }
    with open(output_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    ### END YOUR CODE ###
    
    # Print GPU memory usage if CUDA is available
    if torch.cuda.is_available():
        print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("Training completed!")
