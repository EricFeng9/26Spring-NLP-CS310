# A3 运行说明（简要）

## 1. 安装依赖
```bash
cd /Users/ericfeng/Documents/Sustech/26Spring-NLP/Assignment/A3
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## 2. 训练 tokenizer（Requirement 2）
```bash
python train_tokenizer_from_scratch.py \
  --input ./wiki_zh \
  --vocab_size 52000 \
  --pre_tokenizer Whitespace \
  --output ./wikizh_tokenizer.json
```

说明：运行后会在默认输出路径下生成 tokenizer 的 `.json` 文件（如 `wikizh_tokenizer.json`）。

## 3. 对比 tokenizer（Requirement 2）
```bash
python compare_tokenizers.py
```

说明：把终端输出截图用于报告。

## 4. 运行预训练（Requirement 3/4）
```bash
python run_pretrain.py \
  --data_file ./wiki_zh \
  --tokenizer ./wikizh_tokenizer.json \
  --output_dir ./model_checkpoints \
  --n_epochs 1 \
  --batch_size 4 \
  --train_ratio 0.9 \
  --eval_freq 100 \
  --save_ckpt_freq 1000 \
  --lr 1e-4 \
  --vocab_size 52000
```

如果只是先验证流程是否正确，可加 `--debug` 先跑小模型：
```bash
python run_pretrain.py \
  --data_file ./wiki_zh \
  --tokenizer ./wikizh_tokenizer.json \
  --output_dir ./model_checkpoints_debug \
  --debug
```

## 5. 训练输出（写报告会用到）

`--output_dir` 目录下会自动生成：
- `model_final.pth`：最终模型
- `model_last_checkpoint.pth`：最后一次 checkpoint
- `model_step_*.pth`：周期保存的 checkpoint
- `training_metrics.csv`：每次评估的 loss / tokens_seen / step
- `training_metrics.json`：同上（JSON）
- `run_summary.json`：数据规模和训练配置
- `final_summary.json`：最终关键结果摘要

另外，若有评估记录，会在当前工作目录保存 `loss.pdf`（loss 曲线图）。
