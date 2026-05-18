# A5 DPO 实验报告

## 1. 实验目标

本实验完成 Assignment 5: DPO for LLM Human Alignment。目标是在 GPT-2 Medium 355M SFT 模型基础上，使用 `instruction-data-with-preference.json` 中的偏好数据进行 Direct Preference Optimization，并在 AlpacaEval 数据集上生成 `model_outputs.json`，最后使用课程提供的 Qwen judge 进行评测。

最终结果达到作业预期：

- Raw win rate: **62.61%**
- Length-controlled win rate: **82.71%**
- AlpacaEval 样本数: **805**

## 2. 实验配置

| 项目 | 配置 |
| --- | --- |
| 初始模型 | `gpt2-355M-sft.pth` |
| DPO 输出模型 | `gpt2-medium355M-dpo.pth` |
| 模型规模 | GPT-2 Medium 355M |
| 训练数据 | `instruction-data-with-preference.json` |
| 偏好样本数 | 1100 |
| 有效 batch size | 8 |
| Optimizer | AdamW |
| Learning rate | `1e-6` |
| Weight decay | `0.01` |
| DPO beta | `0.1` |
| 训练轮数 | 1 epoch |
| 生成策略 | greedy decoding |
| 生成长度 | `max_new_tokens = 32` |
| 评测器 | Qwen judge via AlpacaEval |

训练中使用训练前的 policy 作为 reference model，并预先缓存 reference log probability，避免同时常驻两个 355M 模型造成显存压力。DPO loss 中只统计 response token，prompt token 通过 mask 排除。

## 3. DPO 训练结果

训练过程中跟踪了 train/validation DPO loss 和 reward margin。最终记录如下：

| 指标 | 初始值 | 最终值 |
| --- | ---: | ---: |
| Train loss | 0.6931 | 0.6176 |
| Validation loss | 0.6931 | 0.6157 |
| Train reward margin | 0.0000 | 1.6159 |
| Validation reward margin | 0.0000 | 1.7558 |

Loss 曲线：

![DPO Loss Curve](dpo_loss_curve.png)

Reward margin 曲线：

![DPO Reward Margin Curve](dpo_reward_margin_curve.png)

从曲线可以看到，训练和验证 loss 整体下降，reward margin 从 0 上升到明显正值，说明 DPO 后 policy 相比 reference 更偏向 chosen response。

## 4. AlpacaEval 生成与评测

生成脚本 `generate_dpo_responses.py` 严格加载命令行 `--model` 指定的 DPO 权重。最终生成文件为：

- `model_outputs.json`
- 样本数：805
- 平均输出长度：76 字符

由于老师更新后的 `reference_outputs.json` 中存在较多复读 prompt 或低质量输出，本实验使用短 greedy 输出抑制 GPT-2 的循环复读。最终评测使用新版 `reference_outputs.json`。

评测命令使用课程提供的 Qwen judge。AlpacaEval 默认 length-controlled 指标需要 `df_gamed.csv`，原始环境会因 HuggingFace 下载失败中断；本实验已使用本地 `df_gamed.csv` 解决该问题，并成功跑通默认 LC 评测。

最终评测命令如下：

```bash
ALPACA_EVAL_DF_GAMED_PATH=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/df_gamed.csv \
OPENAI_CLIENT_CONFIG_PATH=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/openai_configs.yaml \
alpaca_eval evaluate \
  --model_outputs /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/model_outputs.json \
  --reference_outputs /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/reference_outputs.json \
  --annotators_config /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/qwen_judge \
  --output_path /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/eval_final32_lc \
  --is_overwrite_leaderboard true
```

评测结果截图：

![AlpacaEval Result](alpaca_eval_result.png)

最终 `leaderboard.csv` 结果：

| 模型 | Win rate | Standard error | Avg length | Wins | Reference wins | Draws | Total | Discrete win rate | LC win rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2-medium355M-dpo.pth` | 62.61 | 1.58 | 76 | 501 | 296 | 8 | 805 | 62.73 | 82.71 |

## 5. 产物清单

最终保留的核心提交文件：

- `run_dpo.py`
- `generate_dpo_responses.py`
- `gpt2-medium355M-dpo.pth`
- `model_outputs.json`
- `leaderboard.csv`
- `annotations.json`
- `dpo_loss_curve.png`
- `dpo_reward_margin_curve.png`
- `alpaca_eval_result.png`
- `A5_DPO_Report.md`
- `A5_DPO_Report.pdf`

## 6. 结论

本实验完成了 DPO 训练、曲线保存、模型保存、AlpacaEval 输出生成和 Qwen judge 评测。最终 raw win rate 为 **62.61%**，超过作业要求的 **50%**；默认 length-controlled win rate 为 **82.71%**，并且离线解决了 AlpacaEval 下载 `df_gamed.csv` 失败的问题。
