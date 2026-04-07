# A3 完整运行流程

## 0. 进入目录并安装依赖
```bash
cd /Users/ericfeng/Documents/Sustech/26Spring-NLP/Assignment/A3
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

---

## 1. Requirement 1：预处理 wiki_zh 原始分片

输入是 `wiki_zh` 目录（每行 JSON，含 `title` 与 `text`）。  
下面命令会生成：
- `wikizh.txt`：训练 tokenizer / 预训练模型使用的纯文本语料
- `preprocess_stats.json`：可直接写进报告的统计信息

```bash
python preprocess_wiki_zh.py \
  --input ./wiki_zh \
  --output_text ./wikizh.txt \
  --output_stats ./preprocess_stats.json
```

---

## 2. Requirement 2：训练 tokenizer

下面命令会生成：
- `wikizh_tokenizer_whitespace.json`：训练好的 tokenizer
- `tokenizer_report.json`：包含词表大小、语料总 token 数等报告信息

```bash
python train_tokenizer_from_scratch.py \
  --input ./wikizh.txt \
  --vocab_size 52000 \
  --pre_tokenizer Whitespace \
  --min_freq 2 \
  --output ./wikizh_tokenizer_whitespace.json \
  --report ./tokenizer_report.json
```

如果你想用 ByteLevel：
```bash
python train_tokenizer_from_scratch.py \
  --input ./wikizh.txt \
  --vocab_size 52000 \
  --pre_tokenizer ByteLevel \
  --output ./wikizh_tokenizer_bytelevel.json \
  --report ./tokenizer_report_bytelevel.json
```

---

## 3. Requirement 2：对比分词效果（截图用）

会生成 `compare_tokenizers_result.json`，同时在终端打印结果，方便截图放报告。

```bash
python compare_tokenizers.py \
  --tokenizer ./wikizh_tokenizer_whitespace.json \
  --text "太阳照常升起。" \
  --output ./compare_tokenizers_result.json
```

---

## 4. Requirement 3/4：运行预训练

```bash
python run_pretrain.py \
  --data_file ./wikizh.txt \
  --tokenizer ./wikizh_tokenizer_whitespace.json \
  --output_dir ./model_checkpoints \
  --n_epochs 1 \
  --batch_size 4 \
  --train_ratio 0.9 \
  --eval_freq 100 \
  --save_ckpt_freq 1000 \
  --lr 1e-4 \
  --vocab_size 52000
```

仅调试流程（小模型）：
```bash
python run_pretrain.py \
  --data_file ./wikizh.txt \
  --tokenizer ./wikizh_tokenizer_whitespace.json \
  --output_dir ./model_checkpoints_debug \
  --debug
```

---

## 5. 报告可直接使用的输出文件

### Requirement 1
- `preprocess_stats.json`

### Requirement 2
- `tokenizer_report.json`
- `compare_tokenizers_result.json`
- 对比脚本终端截图

### Requirement 3/4（`--output_dir` 下）
- `model_final.pth`
- `model_last_checkpoint.pth`
- `model_step_*.pth`
- `training_metrics.csv`
- `training_metrics.json`
- `run_summary.json`
- `final_summary.json`

另外：若有评估记录，当前目录会生成 `loss.pdf`（loss 曲线）。
