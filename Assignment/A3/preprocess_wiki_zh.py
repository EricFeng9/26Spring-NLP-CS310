import argparse
import json
from pathlib import Path


def iter_source_files(input_path: Path):
    """遍历输入目录中的语料分片文件。"""
    if input_path.is_file():
        yield input_path
        return

    for p in sorted(input_path.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix in {".txt", ""} or p.name.startswith("wiki_"):
            yield p


def preprocess(input_path: Path, output_text: Path, output_stats: Path, add_eot: bool):
    """将每行 JSON 的 title + text 提取为纯文本语料。"""
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_stats.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    doc_count = 0
    invalid_lines = 0
    total_chars = 0

    with output_text.open("w", encoding="utf-8") as wf:
        for src_file in iter_source_files(input_path):
            file_count += 1
            with src_file.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_lines += 1
                        continue

                    title = str(obj.get("title", "")).strip()
                    text = str(obj.get("text", "")).strip()
                    merged = f"{title}\n{text}".strip()
                    if not merged:
                        continue

                    wf.write(merged)
                    wf.write("\n")
                    if add_eot:
                        wf.write("<|endoftext|>\n")
                    doc_count += 1
                    total_chars += len(merged)

    stats = {
        "input_path": str(input_path),
        "output_text": str(output_text),
        "files_processed": file_count,
        "documents_extracted": doc_count,
        "invalid_json_lines": invalid_lines,
        "total_characters": total_chars,
        "add_endoftext": add_eot,
    }
    with output_stats.open("w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print("Preprocess finished.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Extract Chinese Wiki JSON lines into plain text corpus.")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory (e.g., ./wiki_zh).")
    parser.add_argument("--output_text", type=str, default="wikizh.txt", help="Output plain text corpus file.")
    parser.add_argument("--output_stats", type=str, default="preprocess_stats.json", help="Output stats JSON file.")
    parser.add_argument("--no_eot", action="store_true", help="Do not append <|endoftext|> between samples.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    preprocess(
        input_path=input_path,
        output_text=Path(args.output_text),
        output_stats=Path(args.output_stats),
        add_eot=not args.no_eot,
    )


if __name__ == "__main__":
    main()
