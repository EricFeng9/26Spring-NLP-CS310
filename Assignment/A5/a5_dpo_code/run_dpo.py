"""
Direct Preference Optimization (DPO) training script.
Based on the dpo-from-scratch.ipynb notebook.
"""

import json
import os
import time
import argparse
from functools import partial

import matplotlib.pyplot as plt
import tiktoken
import torch
import torch.nn.functional as F
from utils import GPTModel, generate, text_to_token_ids, token_ids_to_text
from torch.utils.data import DataLoader, Dataset

#####################################
# Dataset utilities
#####################################


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def format_response(response):
    # 将回答统一包成 SFT/DPO 使用的响应段，保证 chosen 和 rejected 的条件上下文完全一致。
    return f"### Response:\n{response}"


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            # prompt 只包含指令和可选输入；回答部分单独拼接，便于后面 mask 掉 prompt token。
            prompt = format_input(entry)
            chosen = format_response(entry["chosen"])
            rejected = format_response(entry["rejected"])

            chosen_full_text = prompt + "\n\n" + chosen
            rejected_full_text = prompt + "\n\n" + rejected

            # DPO 训练需要比较完整序列概率，prompt_tokens 只用于定位 loss 的起点。
            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append(
                {
                    "prompt": prompt_tokens,
                    "chosen": chosen_full_tokens,
                    "rejected": rejected_full_tokens,
                    "index": len(self.encoded_texts),
                }
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu",
):
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": [],
        "index": [],
    }

    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)
        batch_data["index"].append(item["index"])

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()
            mask[len(sequence) :] = False

            if mask_prompt_tokens:
                mask[: prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]
        batch_data[key] = tensor_stack.to(device)

    batch_data["index"] = torch.tensor(batch_data["index"], dtype=torch.long)

    return batch_data


def init_data_loaders(data, tokenizer, batch_size, collate_fn):
    # Split data into train_data, test_data, val_data
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = (
        len(data) - train_portion - test_portion
    )  # Remaining 5% for validation

    # 按作业要求固定划分数据，避免验证集混入训练过程。
    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    # Dataset 在初始化时完成 tokenization，训练时 DataLoader 只负责 batch padding 和 mask。
    train_dataset = PreferenceDataset(train_data, tokenizer)
    test_dataset = PreferenceDataset(test_data, tokenizer)
    val_dataset = PreferenceDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset


#####################################
# DPO Loss
#####################################


def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
):
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_logprobs(logits, labels, selection_mask=None):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(
        input=log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()
        selected_log_probs = selected_log_probs * mask
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
        return avg_log_prob
    else:
        return selected_log_probs.mean(-1)


def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"],
    )
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"],
    )

    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"],
        )
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"],
        )

    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta,
    )
    return loss, chosen_rewards, rejected_rewards


def compute_reference_logprobs_batch(batch, reference_model):
    # reference model 固定不训练，只需要在无梯度模式下给出 chosen/rejected 的基准 logprob。
    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"],
        )
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"],
        )

    return ref_chosen_log_probas, ref_rejected_log_probas


def precompute_reference_logprobs(dataset, model, collate_fn, batch_size, device):
    # reference model 等于训练前的 policy；预计算每条样本的 logprob 后即可释放第二个模型。
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    chosen_logprobs = torch.empty(len(dataset), dtype=torch.float32)
    rejected_logprobs = torch.empty(len(dataset), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            chosen = compute_logprobs(
                logits=model(batch["chosen"]),
                labels=batch["chosen"],
                selection_mask=batch["chosen_mask"],
            )
            rejected = compute_logprobs(
                logits=model(batch["rejected"]),
                labels=batch["rejected"],
                selection_mask=batch["rejected_mask"],
            )
            indices = batch["index"]
            chosen_logprobs[indices] = chosen.detach().cpu()
            rejected_logprobs[indices] = rejected.detach().cpu()
            print(
                f"Precomputed reference batch {batch_idx + 1}/{len(loader)}",
                flush=True,
            )

    return {
        "chosen": chosen_logprobs.to(device),
        "rejected": rejected_logprobs.to(device),
    }


def compute_dpo_loss_batch_with_cached_reference(batch, policy_model, reference_cache, beta):
    # 从缓存读取 reference logprob，训练时只保留一个可训练模型，显著降低内存峰值。
    indices = batch["index"].to(policy_model.tok_emb.weight.device)
    ref_chosen_log_probas = reference_cache["chosen"][indices]
    ref_rejected_log_probas = reference_cache["rejected"][indices]

    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"],
    )
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"],
    )

    return compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta,
    )


#####################################
# Training
#####################################


def compute_dpo_metrics_loader(data_loader, policy_model, reference_cache, beta, eval_iter):
    # 在不更新参数的情况下统计若干个 batch 的平均 loss 和 reward，用于画训练曲线。
    policy_model.eval()

    total_loss = 0.0
    total_chosen_reward = 0.0
    total_rejected_reward = 0.0
    num_batches = min(eval_iter, len(data_loader))

    if num_batches == 0:
        raise ValueError("DataLoader is empty; cannot compute DPO metrics.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            loss, chosen_reward, rejected_reward = compute_dpo_loss_batch_with_cached_reference(
                batch=batch,
                policy_model=policy_model,
                reference_cache=reference_cache,
                beta=beta,
            )
            total_loss += loss.item()
            total_chosen_reward += chosen_reward.item()
            total_rejected_reward += rejected_reward.item()

    return (
        total_loss / num_batches,
        total_chosen_reward / num_batches,
        total_rejected_reward / num_batches,
    )


def train_model_dpo_simple(
    policy_model,
    reference_model,
    train_loader,
    val_loader,
    train_dataset,
    val_dataset,
    collate_fn,
    optimizer,
    num_epochs,
    beta,
    eval_freq,
    eval_iter,
    tokenizer,
    grad_accum_steps,
    max_steps=None,
    checkpoint_freq=None,
    output_prefix="gpt2-medium355M-dpo",
):
    # 作业要求训练主函数同时接收 policy_model 和 reference_model：
    # policy_model 参与梯度更新；reference_model 固定在 eval 模式，只提供 DPO 的基准 logprob。
    policy_model.train()
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False

    # reference_model 不训练，预先缓存每条样本的 logprob 与每步实时前向等价，同时降低训练阶段显存峰值。
    reference_device = policy_model.tok_emb.weight.device
    train_reference_cache = precompute_reference_logprobs(
        train_dataset,
        reference_model,
        collate_fn,
        batch_size=1,
        device=reference_device,
    )
    val_reference_cache = precompute_reference_logprobs(
        val_dataset,
        reference_model,
        collate_fn,
        batch_size=1,
        device=reference_device,
    )
    # 缓存已经包含 reference_model 的全部训练所需信息，后续训练阶段只保留 policy_model 在 GPU 上。
    reference_model.to("cpu")
    if reference_device.type == "cuda":
        torch.cuda.empty_cache()

    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "train_reward_margins": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "val_reward_margins": [],
        "tokens_seen": [],
        "global_steps": [],
    }
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        policy_model.train()

        optimizer.zero_grad()
        for batch in train_loader:
            loss, chosen_reward, rejected_reward = compute_dpo_loss_batch_with_cached_reference(
                batch=batch,
                policy_model=policy_model,
                reference_cache=train_reference_cache,
                beta=beta,
            )
            # 使用 micro-batch 累积梯度，降低单次前向/反向的激活显存峰值。
            (loss / grad_accum_steps).backward()

            tokens_seen += batch["chosen"].numel() + batch["rejected"].numel()
            global_step += 1

            if (global_step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 周期性评估 train/val，记录 DPO loss 和 reward margin 的真实走势。
            if global_step % eval_freq == 0:
                train_loss, train_chosen, train_rejected = compute_dpo_metrics_loader(
                    train_loader, policy_model, train_reference_cache, beta, eval_iter
                )
                val_loss, val_chosen, val_rejected = compute_dpo_metrics_loader(
                    val_loader, policy_model, val_reference_cache, beta, eval_iter
                )

                tracking["train_losses"].append(train_loss)
                tracking["train_chosen_rewards"].append(train_chosen)
                tracking["train_rejected_rewards"].append(train_rejected)
                tracking["train_reward_margins"].append(train_chosen - train_rejected)
                tracking["val_losses"].append(val_loss)
                tracking["val_chosen_rewards"].append(val_chosen)
                tracking["val_rejected_rewards"].append(val_rejected)
                tracking["val_reward_margins"].append(val_chosen - val_rejected)
                tracking["tokens_seen"].append(tokens_seen)
                tracking["global_steps"].append(global_step)

                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                    f"Train margin {train_chosen - train_rejected:.3f}, "
                    f"Val margin {val_chosen - val_rejected:.3f}",
                    flush=True,
                )

                policy_model.train()

            # 周期性保存早停候选模型，后续可用评测子集筛选最佳 checkpoint。
            if checkpoint_freq is not None and global_step > 0 and global_step % checkpoint_freq == 0:
                checkpoint_path = f"{output_prefix}-step{global_step}.pth"
                torch.save(policy_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}", flush=True)

            if max_steps is not None and global_step >= max_steps:
                return tracking

        # epoch 结束时处理不足 grad_accum_steps 的尾部梯度，保证所有样本都参与更新。
        if (global_step + 1) % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    return tracking


def plot_training_curves(tracking, output_prefix="dpo"):
    # 保存 loss 曲线，报告会直接引用该图片。
    steps = tracking["global_steps"]
    if not steps:
        raise ValueError("No tracked metrics found; cannot plot curves.")

    plt.figure(figsize=(7, 4))
    plt.plot(steps, tracking["train_losses"], label="Train loss")
    plt.plot(steps, tracking["val_losses"], label="Validation loss")
    plt.xlabel("Training step")
    plt.ylabel("DPO loss")
    plt.title("DPO Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_loss_curve.png", dpi=200)
    plt.savefig(f"{output_prefix}_loss_curve.pdf")
    if output_prefix != "dpo":
        plt.savefig("dpo_loss_curve.png", dpi=200)
        plt.savefig("dpo_loss_curve.pdf")
    plt.close()

    # 保存 reward margin 曲线，margin 越大说明 policy 相对 reference 更偏好 chosen。
    plt.figure(figsize=(7, 4))
    plt.plot(steps, tracking["train_reward_margins"], label="Train reward margin")
    plt.plot(steps, tracking["val_reward_margins"], label="Validation reward margin")
    plt.xlabel("Training step")
    plt.ylabel("Chosen reward - rejected reward")
    plt.title("DPO Reward Margin Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_reward_margin_curve.png", dpi=200)
    plt.savefig(f"{output_prefix}_reward_margin_curve.pdf")
    if output_prefix != "dpo":
        plt.savefig("dpo_reward_margin_curve.png", dpi=200)
        plt.savefig("dpo_reward_margin_curve.pdf")
    plt.close()


#####################################
# Main
#####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=40)
    parser.add_argument("--eval_iter", type=int, default=2)
    parser.add_argument("--checkpoint_freq", type=int, default=None)
    parser.add_argument("--output_prefix", type=str, default="gpt2-medium355M-dpo")
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # Load dataset
    file_path = "instruction-data-with-preference.json"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print("Number of entries:", len(data))

    # Tokenizer and data loaders
    tokenizer = tiktoken.get_encoding("gpt2")

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        mask_prompt_tokens=True,
        allowed_max_length=1024,
    )
    # DPO 每个 batch 需要 chosen/rejected 两条带梯度前向；用 micro-batch 避免激活内存峰值过高。
    batch_size = 1
    grad_accum_steps = 8
    torch.manual_seed(123)

    # Initialize data loaders
    train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset = init_data_loaders(
        data=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
    )
    print(
        f"DataLoader sizes: train={len(train_loader)}, "
        f"test={len(test_loader)}, val={len(val_loader)}"
    )

    # Configure the model
    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }
    model_configs = {
        "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    }
    BASE_CONFIG.update(model_configs["355M"])

    model_path = "/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Lab/week11/gpt2-355M-sft.pth"

    # Load policy model and frozen reference model from the same SFT checkpoint.
    print("Initializing policy and reference models...", flush=True)
    policy_model = GPTModel(BASE_CONFIG)
    reference_model = GPTModel(BASE_CONFIG)
    print("Loading SFT checkpoint...", flush=True)
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    print("Loading checkpoint into policy and reference models...", flush=True)
    policy_model.load_state_dict(state_dict)
    reference_model.load_state_dict(state_dict)
    # 两个模型加载完后立即释放临时权重字典，避免后台训练占用多余主存。
    del state_dict
    print("Moving models to device...", flush=True)
    policy_model.to(device)
    reference_model.to(device)
    policy_model.train()
    reference_model.eval()
    print("Pretrained policy and reference models loaded.")

    # Training
    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.01)
    num_epochs = 1

    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=customized_collate_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        beta=args.beta,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        tokenizer=tokenizer,
        grad_accum_steps=grad_accum_steps,
        max_steps=args.max_steps,
        checkpoint_freq=args.checkpoint_freq,
        output_prefix=args.output_prefix,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save the policy model
    final_model_path = f"{args.output_prefix}.pth"
    tracking_path = f"{args.output_prefix}-tracking.json"
    torch.save(policy_model.state_dict(), final_model_path)
    with open(tracking_path, "w", encoding="utf-8") as file:
        json.dump(tracking, file, indent=2)
    # 同时保留作业默认文件名，方便原始提交路径不变。
    if final_model_path != "gpt2-medium355M-dpo.pth":
        torch.save(policy_model.state_dict(), "gpt2-medium355M-dpo.pth")
        with open("dpo_tracking.json", "w", encoding="utf-8") as file:
            json.dump(tracking, file, indent=2)
    print(f"Saved policy model to {final_model_path}")

    # Plot the loss and reward margin curves
    plot_training_curves(tracking, output_prefix=args.output_prefix)
    print("Saved DPO curves to dpo_loss_curve.* and dpo_reward_margin_curve.*")
