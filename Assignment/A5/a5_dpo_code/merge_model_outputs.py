import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", default="generation_shards")
    parser.add_argument("--output", default="model_outputs.json")
    args = parser.parse_args()

    # 按原始 alpaca_eval.json 的 instruction 顺序合并分片输出，确保评测输入顺序稳定。
    with open(ROOT / "alpaca_eval.json", "r", encoding="utf-8") as file:
        original_data = json.load(file)

    outputs_by_instruction = {}
    shard_dir = Path(args.shard_dir)
    if not shard_dir.is_absolute():
        shard_dir = ROOT / shard_dir
    for shard_path in sorted(shard_dir.glob("model_outputs_shard_*.json")):
        with open(shard_path, "r", encoding="utf-8") as file:
            for entry in json.load(file):
                outputs_by_instruction[entry["instruction"]] = entry

    merged = [outputs_by_instruction[entry["instruction"]] for entry in original_data]
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(merged, file, indent=2)

    print(f"Merged {len(merged)} outputs to {output_path}")


if __name__ == "__main__":
    main()
