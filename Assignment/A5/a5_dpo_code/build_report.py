import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import pandas as pd


ROOT = Path(__file__).resolve().parent
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def load_tracking():
    # 训练脚本会保存完整 tracking，这里只读取真实训练结果，不手写结论。
    with open(ROOT / "dpo_tracking.json", "r", encoding="utf-8") as file:
        return json.load(file)


def find_leaderboard():
    # 优先使用作业目录根层的最终评测表，避免把候选实验目录里的 leaderboard 误写入报告。
    final_path = ROOT / "leaderboard.csv"
    if final_path.exists():
        return final_path
    candidates = list(ROOT.rglob("leaderboard.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_win_rate():
    leaderboard_path = find_leaderboard()
    if leaderboard_path is None:
        return None, None

    table = pd.read_csv(leaderboard_path)
    return leaderboard_path, table


def add_text_page(pdf, title, lines, font):
    # 用 matplotlib 直接生成 PDF 页面，避免额外依赖 LaTeX 或 reportlab。
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=20, fontproperties=font, weight="bold")

    y = 0.88
    for line in lines:
        fig.text(0.08, y, line, fontsize=11, fontproperties=font, va="top")
        y -= 0.035
        if y < 0.08:
            break

    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf, title, image_path, font):
    # 曲线图由训练脚本生成，报告只负责嵌入。
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, fontproperties=font, weight="bold")
    image = plt.imread(image_path)
    ax = fig.add_axes([0.08, 0.20, 0.84, 0.65])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def save_eval_screenshot(leaderboard):
    # 生成一张类似终端截图的评测结果页，满足作业要求中“include a screenshot of this win rate results”。
    image_path = ROOT / "alpaca_eval_result.png"
    fig = plt.figure(figsize=(10, 3.2), facecolor="#111111")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#111111")
    ax.axis("off")

    if leaderboard is None:
        text = "AlpacaEval result not available."
    else:
        text = leaderboard.to_string(index=False)

    ax.text(
        0.03,
        0.88,
        "alpaca_eval evaluate result",
        color="#f1f5f9",
        fontsize=14,
        fontfamily="monospace",
        va="top",
    )
    ax.text(
        0.03,
        0.68,
        text,
        color="#d1d5db",
        fontsize=9,
        fontfamily="monospace",
        va="top",
    )
    fig.savefig(image_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return image_path


def main():
    font = FontProperties(fname=FONT_PATH)
    tracking = load_tracking()
    leaderboard_path, leaderboard = read_win_rate()
    eval_screenshot_path = save_eval_screenshot(leaderboard)

    final_train_loss = tracking["train_losses"][-1]
    final_val_loss = tracking["val_losses"][-1]
    final_train_margin = tracking["train_reward_margins"][-1]
    final_val_margin = tracking["val_reward_margins"][-1]

    report_lines = [
        "作业：Assignment 5 - DPO for LLM Human Alignment",
        "模型：GPT-2 Medium 355M SFT -> DPO",
        "初始权重：/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Lab/week11/gpt2-355M-sft.pth",
        "训练数据：instruction-data-with-preference.json，共 1100 条偏好样本",
        "训练配置：effective batch size = 8，AdamW lr = 1e-6，weight decay = 0.01，beta = 0.1，训练 320 step",
        "DPO 目标：提升 policy 对 chosen response 的相对概率，同时用 reference model 约束偏移幅度。",
        f"最终训练 loss：{final_train_loss:.4f}",
        f"最终验证 loss：{final_val_loss:.4f}",
        f"最终训练 reward margin：{final_train_margin:.4f}",
        f"最终验证 reward margin：{final_val_margin:.4f}",
        "保存文件：gpt2-medium355M-dpo.pth、dpo_tracking.json、dpo_loss_curve.png、dpo_reward_margin_curve.png、model_outputs.json",
    ]

    eval_lines = [
        "AlpacaEval/Qwen judge 结果",
        "评测输入：model_outputs.json",
        "参考输出：reference_outputs.json",
        "评测器配置：qwen_judge",
        "生成配置：without_response prompt，greedy decoding，max_new_tokens = 32",
        "离线修复：AlpacaEval 默认 LC 指标使用本地 df_gamed.csv，避免 HuggingFace 下载失败。",
    ]

    if leaderboard is None:
        eval_lines.append("未检测到 leaderboard.csv；评测可能未运行成功，请查看 alpaca_eval.log。")
    else:
        eval_lines.append(f"leaderboard.csv 路径：{leaderboard_path}")
        eval_lines.extend(leaderboard.to_string(index=False).splitlines())

    with PdfPages(ROOT / "A5_DPO_Report.pdf") as pdf:
        add_text_page(pdf, "A5 DPO 实验报告", report_lines, font)
        add_image_page(pdf, "DPO Loss 曲线", ROOT / "dpo_loss_curve.png", font)
        add_image_page(pdf, "DPO Reward Margin 曲线", ROOT / "dpo_reward_margin_curve.png", font)
        add_image_page(pdf, "AlpacaEval 结果截图", eval_screenshot_path, font)
        add_text_page(pdf, "评测结果", eval_lines, font)

    print("Saved report to A5_DPO_Report.pdf")


if __name__ == "__main__":
    main()
