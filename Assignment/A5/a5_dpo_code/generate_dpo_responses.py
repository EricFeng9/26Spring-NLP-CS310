import argparse
import json
import os

import tiktoken
import torch
import torch.nn.functional as F
import tqdm

from utils import (
    GPTModel,
    generate,
    generate_with_kv_cache,
    text_to_token_ids,
    token_ids_to_text,
)

# Model config
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
}

CHOOSE_MODEL = "355M"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])


def load_model(model_path):
    # 严格使用命令行传入的模型路径，保证生成结果来自 DPO 训练后的权重。
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    )
    model.eval()

    return model


def load_data(file_path):
    # AlpacaEval 输入是 JSON 列表，读取后逐条生成模型回答。
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    # 生成时显式给出回答段标记，使输入格式和 DPO/SFT 训练格式一致。
    return instruction_text + "\n\n### Response:\n"


def format_input_without_response_marker(entry):
    # 保留作业原始脚本的 prompt 格式，用于和参考输出的生成方式对齐。
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )


def generate_with_repetition_penalty(
    model,
    idx,
    max_new_tokens,
    context_size,
    repetition_penalty,
    eos_id,
):
    # 贪心解码加入重复惩罚，直接压制 GPT-2 在长生成中常见的循环复读。
    generated = idx.clone()
    for _ in range(max_new_tokens):
        idx_cond = generated[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        if next_id.item() == eos_id:
            break
        generated = torch.cat([generated, next_id], dim=1)

    return generated


def clean_response(text):
    # 生成出现新的分隔符时，后面通常是在复读 prompt；截断能保留第一段回答。
    stop_markers = ["\n\n### Instruction:", "\n### Instruction:", "\n\nBelow is an instruction"]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.replace("### Response:", "").strip()


def main(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    data = load_data(args.input)

    tokenizer = tiktoken.get_encoding("gpt2")

    # 加载指定模型并切换到评估模式，避免 dropout 等训练行为影响生成。
    model = load_model(args.model)
    model.to(device)
    model_name = os.path.basename(args.model)

    outputs = []
    for entry in tqdm.tqdm(data):
        if args.prompt_style == "with_response":
            input_text = format_input(entry)
        else:
            input_text = format_input_without_response_marker(entry)

        input_ids = text_to_token_ids(input_text, tokenizer).to(device)
        if args.repetition_penalty == 1.0:
            token_ids = generate_with_kv_cache(
                model=model,
                idx=input_ids,
                max_new_tokens=args.max_new_tokens,
                context_size=BASE_CONFIG["context_length"],
                eos_id=50256,
            )
        else:
            token_ids = generate_with_repetition_penalty(
                model=model,
                idx=input_ids,
                max_new_tokens=args.max_new_tokens,
                context_size=BASE_CONFIG["context_length"],
                repetition_penalty=args.repetition_penalty,
                eos_id=50256,
            )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = clean_response(generated_text[len(input_text) :])
        outputs.append(
            {
                "dataset": entry["dataset"],
                "instruction": entry["instruction"],
                "output": response_text,
                "generator": model_name,
            }
        )
    # 输出字段与 reference_outputs.json 保持一致，供 alpaca_eval 直接读取。
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint",
        default="gpt2-medium355M-sft.pth",
    )
    parser.add_argument(
        "--prompt_style",
        choices=["without_response", "with_response"],
        default="without_response",
        help="Generation prompt format.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
