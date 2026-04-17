# CS310 NLP Assignment 3 Report: Pretraining a GPT-2 Model on Chinese Wikipedia
> 冯俊铭 12311031

## 0. 实验环境
- 主要代码文件：
  - `preprocess_wiki_zh.py`
  - `train_tokenizer_from_scratch.py`
  - `compare_tokenizers.py`
  - `run_pretrain.py`
  - `utils.py`

---

## 1. Requirement 1: 数据抽取与预处理

### 1.1 方法说明
- 原始语料来自 `wiki_zh` 目录分片文件（每行 1 条 JSON）。
- 预处理时提取并拼接 `title + text`。
- 文档间追加 `<|endoftext|>`，输出到单一语料文件 `wikizh.txt`，用于 tokenizer 训练与预训练。

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

## 2. Requirement 2: 从零训练 BPE Tokenizer

### 2.1 训练配置
- 输入语料：`wikizh.txt`
- 词表大小（请求）：**52,000**
- 实际词表大小：**52,000**
- 预分词器：**ByteLevel**
- 最小词频：**2**

### 2.2 训练结果
- 输出 tokenizer 文件：`wikizh_tokenizer_bytelevel.json`
- 语料非空行数：**7,424,610**
- 语料总 token 数（新 tokenizer 统计）：**258,905,826**

### 2.3 新 tokenizer 与原始 GPT-2 tokenizer 对比
- 对比样例文本：`太阳照常升起。`
- 新 tokenizer token 数：**5**
- 原始 GPT-2 tokenizer token 数：**16**
- ByteLevel 下单 token 解码显示为字节级片段（如 `Ġå¤ªéĺ³`），属于正常现象；对比结果显示新 tokenizer 在中文上的切分长度明显更紧凑。

### 2.4 数据来源
- 训练统计文件：`tokenizer_report_bytelevel.json`
- 对比结果文件：`compare_tokenizers_result_bytelevel.json`
- 生成脚本：
  - `tokenizer_report_bytelevel.json` 由 `train_tokenizer_from_scratch.py` 生成
  - `compare_tokenizers_result_bytelevel.json` 由 `compare_tokenizers.py` 生成
  - tokenizer 文件 `wikizh_tokenizer_bytelevel.json` 由 `train_tokenizer_from_scratch.py` 生成

---

## 3. Requirement 3: 补全预训练脚本

### 3.1 已实现功能
- 支持读取单文件或目录数据并构建 train/val dataloader
- 使用 `global_step` 计数并持续跟踪 `tokens_seen`
- 按 `eval_freq` 周期评估并记录 train/val loss
- 按 `save_ckpt_freq` 周期保存 checkpoint
- 训练结束保存 `model_last_checkpoint.pth` 与 `model_final.pth`
- 自动输出 `run_summary.json`、`final_summary.json`、`training_metrics.csv/json`

### 3.2 最终模型超参数（与本次运行一致）
- `vocab_size=52000`（与 tokenizer 一致）
- `context_length=1024`
- `emb_dim=768`
- `n_heads=12`
- `n_layers=12`
- `drop_rate=0.1`
- `qkv_bias=False`

### 3.3 数据来源
- 代码实现文件：`run_pretrain.py`（训练主流程）、`utils.py`（模型与loss工具函数）
- 训练配置文件：`model_checkpoints_seed10086_wd001_bytelevel/run_summary.json`
- 最终配置与结果摘要：`model_checkpoints_seed10086_wd001_bytelevel/final_summary.json`

---

## 4. Requirement 4: 从零预训练与结果汇报

### 4.1 训练参数
- 训练数据：`./wikizh.txt`
- tokenizer：`./wikizh_tokenizer_bytelevel.json`
- 训练轮数：`n_epochs=1`
- batch size：`8`
- train/val 划分：`0.9 / 0.1`
- 初始学习率：`1e-4`
- weight decay：`0.01`
- seed：`10086`
- `eval_freq=100`, `eval_iter=10`, `save_ckpt_freq=1000`
- 每个 epoch 训练 batch 数：`29674`
- 实际完成训练 step：`29600`

### 4.2 训练过程与损失曲线
- 评估点数量：**296**
- 首次评估（step=100）：`train_loss=9.9028`, `val_loss=9.9659`
- `~100M tokens`（step=12300, tokens=100,761,600）：`train_loss=5.3320`, `val_loss=5.5873`
- 最后一次评估（step=29600）：`train_loss=4.7342`, `val_loss=4.9129`
- 最优验证集损失：`val_loss=4.9129`（step=29600）
- 已见 token 总数（最后评估点）：`242,483,200`
- 训练时长（按最后评估点计）：约 `6.76` 小时（24329.59 秒）

损失曲线文件：`loss.pdf`、`loss.png`

![Training and Validation Loss Curve](./loss.png)


### 4.3 数据来源
- 训练日志文件：
  - `model_checkpoints_seed10086_wd001_bytelevel/training_metrics.csv`
  - `model_checkpoints_seed10086_wd001_bytelevel/training_metrics.json`
- 训练参数文件：`model_checkpoints_seed10086_wd001_bytelevel/run_summary.json`
- 最终结果文件：`model_checkpoints_seed10086_wd001_bytelevel/final_summary.json`
- 损失曲线文件：`loss.pdf`、`loss.png`

---

## 5. 作业提交清单
- 预训练脚本：`run_pretrain.py`
- 新训练 tokenizer：`wikizh_tokenizer_bytelevel.json`
- 最终模型权重：`model_checkpoints_seed10086_wd001_bytelevel/model_final.pth`
- 结果报告：`A3_report.pdf`
