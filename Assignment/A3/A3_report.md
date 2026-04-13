# CS310 NLP Assignment 3 Report: Pretraining a GPT-2 Model on Chinese Wikipedia

## 0. 实验环境与目录
- 项目目录：`/Users/ericfeng/Documents/Sustech/26Spring-NLP/Assignment/A3`
- 主要代码文件：
  - `preprocess_wiki_zh.py`
  - `train_tokenizer_from_scratch.py`
  - `compare_tokenizers.py`
  - `run_pretrain.py`
  - `utils.py`

---

## 1. Requirement 1: 数据抽取与预处理（5 分）

### 1.1 方法说明
- 原始语料来自 `wiki_zh` 目录下的分片文件（每行一条 JSON）。
- 预处理脚本读取每行 JSON，提取并拼接 `title + text`，并在样本之间追加 `<|endoftext|>`。
- 输出纯文本语料文件用于后续 tokenizer 训练与预训练。

### 1.2 结果统计
- 处理文件数：**1274**
- 抽取文档数：**1,043,224**
- 无效 JSON 行数：**0**
- 文本总字符数：**452,493,892**
- 预处理后语料输出：`wikizh.txt`

### 1.3 证据来源
- 证据文件：`preprocess_stats.json`
- 生成脚本：`preprocess_wiki_zh.py`

---

## 2. Requirement 2: 从零训练 BPE Tokenizer（5 分）

### 2.1 训练配置
- 输入语料：`wikizh.txt`
- 词表大小（请求）：**52,000**
- 实际词表大小：**52,000**
- 预分词器：**Whitespace**
- 最小词频：**2**

### 2.2 训练结果
- 输出 tokenizer 文件：`wikizh_tokenizer_whitespace.json`
- 语料非空行数：**7,424,610**
- 语料总 token 数（使用新 tokenizer 统计）：**257,656,189**

### 2.3 新 tokenizer 与原始 GPT-2 tokenizer 对比
- 对比样例文本：`太阳照常升起。`
- 新 tokenizer 分词结果（5 个 token）：`["太阳", "照", "常", "升起", "。"]`
- 原始 GPT-2 tokenizer 分词结果（16 个 token）：多数为乱码分片 `�`

结论：新训练的中文 tokenizer 在中文文本上分词显著更合理。

### 2.4 证据来源
- 训练统计证据文件：`tokenizer_report.json`
- 对比结果证据文件：`compare_tokenizers_result.json`
- 生成脚本：
  - `tokenizer_report.json` 由 `train_tokenizer_from_scratch.py` 生成
  - `compare_tokenizers_result.json` 由 `compare_tokenizers.py` 生成
  - tokenizer 文件 `wikizh_tokenizer_whitespace.json` 由 `train_tokenizer_from_scratch.py` 生成

---

## 3. Requirement 3: 补全预训练脚本（10 分）

### 3.1 已实现功能
- 读取训练数据并构建 train/val dataloader
- 训练循环中维护 `global_step` 与 `tokens_seen`
- 按 `eval_freq` 周期评估 train/val loss 并记录
- 按 `save_ckpt_freq` 周期保存 checkpoint
- 训练中断（KeyboardInterrupt）时保存中断 checkpoint
- 训练结束保存最终 checkpoint 与最终模型
- 自动输出报告所需的统计文件（JSON/CSV）

### 3.2 最终模型超参数（与本次运行一致）
- `vocab_size=52000`
- `context_length=1024`
- `emb_dim=768`
- `n_heads=12`
- `n_layers=12`
- `drop_rate=0.1`
- `qkv_bias=False`

### 3.3 证据来源
- 代码实现文件：`run_pretrain.py`（训练主流程），`utils.py`（模型与loss等工具函数）
- 训练配置证据文件：`model_checkpoints/run_summary.json`
- 最终配置与结果摘要：`model_checkpoints/final_summary.json`
- 生成脚本：`run_pretrain.py`

---

## 4. Requirement 4: 从零预训练与结果汇报（10 分）

### 4.1 训练参数
- 训练数据：`./wikizh.txt`
- tokenizer：`./wikizh_tokenizer_whitespace.json`
- 训练轮数：`n_epochs=1`
- batch size：`8`
- train/val 划分：`0.9 / 0.1`
- 初始学习率：`1e-4`
- `eval_freq=100`
- `save_ckpt_freq=1000`
- 每个 epoch 训练 batch 数：`28369`

### 4.2 训练过程与损失曲线
- 评估点数量：**283**
- 首次评估（step=100）：`train_loss=8.6856`, `val_loss=8.6628`
- 最后一次评估（step=28300）：`train_loss=4.2398`, `val_loss=5.1266`
- 最优验证集损失：`val_loss=5.1037`（step=28000）
- 已见 token 总数（最后评估点）：`231,833,600`
- 训练时长（按最后评估点计）：约 `7.75` 小时

损失曲线文件：`loss.pdf`

### 4.3 模型与 checkpoint 产物
- 最终模型：`model_checkpoints/model_final.pth`
- 最后 checkpoint：`model_checkpoints/model_last_checkpoint.pth`
- 周期 checkpoint 示例：`model_checkpoints/model_step_28000.pth`

### 4.4 证据来源
- 训练日志证据文件：`model_checkpoints/training_metrics.csv`、`model_checkpoints/training_metrics.json`
- 训练参数证据文件：`model_checkpoints/run_summary.json`
- 最终结果证据文件：`model_checkpoints/final_summary.json`
- 损失曲线证据文件：`loss.pdf`
- 生成脚本：`run_pretrain.py`（调用 `utils.py` 中的 `plot_losses` 生成 `loss.pdf`）

---

## 5. 作业提交清单（与题目要求对应）
- 预训练脚本：`run_pretrain.py`
- 新训练 tokenizer：`wikizh_tokenizer_whitespace.json`
- 最终模型权重：`model_checkpoints/model_final.pth`
- 结果报告：`A3_report.md`（可转为 PDF 提交）

---

## 6. 说明
- 本地日志与曲线已覆盖 Requirement 1~4 的主要证据项。
- 题目提到的 held-out test perplexity 为助教评分环节计算，本地未提供独立测试集与计算脚本，因此本报告不包含该项最终分数。
