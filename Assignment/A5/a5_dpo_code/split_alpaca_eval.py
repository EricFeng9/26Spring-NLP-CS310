import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="alpaca_eval.json")
    parser.add_argument("--output_dir", default="generation_shards")
    parser.add_argument("--num_shards", type=int, default=8)
    args = parser.parse_args()

    # 将 AlpacaEval 输入平均切成多份，方便多个后台推理进程并行生成。
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    shard_dir = Path(args.output_dir)
    if not shard_dir.is_absolute():
        shard_dir = ROOT / shard_dir
    shard_dir.mkdir(exist_ok=True)

    for shard_id in range(args.num_shards):
        shard = data[shard_id::args.num_shards]
        with open(shard_dir / f"alpaca_eval_shard_{shard_id}.json", "w", encoding="utf-8") as file:
            json.dump(shard, file, indent=2)


if __name__ == "__main__":
    main()
