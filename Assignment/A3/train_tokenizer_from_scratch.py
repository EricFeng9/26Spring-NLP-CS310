import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def train_tokenizer(
    input_file: Path,
    vocab_size: int,
    pre_tokenizer_name: str,
    min_freq: int,
    output_tokenizer: Path,
    output_report: Path,
):
    """训练 BPE tokenizer，并输出可写入报告的统计信息。"""
    if not input_file.exists() or not input_file.is_file():
        raise FileNotFoundError(f"--input must be a text file, got: {input_file}")

    tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))
    if pre_tokenizer_name == "Whitespace":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        min_frequency=min_freq,
    )

    print(f"Training tokenizer from: {input_file}")
    tokenizer.train([str(input_file)], trainer)

    output_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_tokenizer))
    print(f"Saved tokenizer to: {output_tokenizer}")

    # 统计语料 token 总量，便于 Requirement 2(c) 直接写报告
    total_tokens = 0
    total_lines = 0
    with input_file.open("r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            total_tokens += len(tokenizer.encode(line).ids)

    report = {
        "input_file": str(input_file),
        "output_tokenizer": str(output_tokenizer),
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(),
        "pre_tokenizer": pre_tokenizer_name,
        "min_frequency": min_freq,
        "total_nonempty_lines": total_lines,
        "total_tokens_in_corpus": total_tokens,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with output_report.open("w", encoding="utf-8") as wf:
        json.dump(report, wf, ensure_ascii=False, indent=2)
    print(f"Saved report to: {output_report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on Chinese text data from scratch.")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed text file, e.g., ./wikizh.txt")
    parser.add_argument("--vocab_size", type=int, default=52000, help="Vocabulary size.")
    parser.add_argument(
        "--pre_tokenizer",
        type=str,
        choices=["Whitespace", "ByteLevel"],
        default="Whitespace",
        help="Pre-tokenizer to use.",
    )
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum frequency for vocabulary entries.")
    parser.add_argument("--output", type=str, default="wikizh_tokenizer_whitespace.json", help="Tokenizer output path.")
    parser.add_argument(
        "--report",
        type=str,
        default="tokenizer_report.json",
        help="Path to save training statistics JSON.",
    )
    args = parser.parse_args()

    train_tokenizer(
        input_file=Path(args.input),
        vocab_size=args.vocab_size,
        pre_tokenizer_name=args.pre_tokenizer,
        min_freq=args.min_freq,
        output_tokenizer=Path(args.output),
        output_report=Path(args.report),
    )


if __name__ == "__main__":
    main()
