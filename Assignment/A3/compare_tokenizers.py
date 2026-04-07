import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer


def compare(tokenizer_path: Path, text: str, output_path: Path):
    """对比自训 tokenizer 与原始 GPT-2 tokenizer 的分词效果。"""
    tok_custom = Tokenizer.from_file(str(tokenizer_path))
    tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")

    custom_ids = tok_custom.encode(text).ids
    gpt2_ids = tok_gpt2.encode(text)

    custom_pieces = [tok_custom.decode([tid]) for tid in custom_ids]
    gpt2_pieces = [tok_gpt2.decode([tid]) for tid in gpt2_ids]

    result = {
        "text": text,
        "custom_tokenizer_path": str(tokenizer_path),
        "custom_token_ids": custom_ids,
        "custom_pieces": custom_pieces,
        "gpt2_token_ids": gpt2_ids,
        "gpt2_pieces": gpt2_pieces,
        "custom_token_count": len(custom_ids),
        "gpt2_token_count": len(gpt2_ids),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as wf:
        json.dump(result, wf, ensure_ascii=False, indent=2)

    print("Trained tokenizer:", custom_ids, custom_pieces)
    print("Original GPT-2 tokenizer:", gpt2_ids, gpt2_pieces)
    print(f"Saved compare result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare custom tokenizer with original GPT-2 tokenizer.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to custom tokenizer JSON file.")
    parser.add_argument("--text", type=str, default="太阳照常升起。", help="Sample text to compare.")
    parser.add_argument("--output", type=str, default="compare_tokenizers_result.json", help="Output comparison JSON.")
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    compare(tokenizer_path=tokenizer_path, text=args.text, output_path=Path(args.output))


if __name__ == "__main__":
    main()
