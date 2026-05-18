from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "a5_dpo_code"
OUTPUT = ROOT / "Report.pdf"


def paragraph(text, style):
    # 将报告中的等宽标记转换为 PDF 可读的 HTML 片段。
    return Paragraph(text.replace("`", ""), style)


def add_heading(story, text, style):
    # 每个功能模块使用明确标题，方便阅卷时快速定位作业要求。
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 0.12 * inch))


def add_table(story, rows, widths):
    # 统一表格样式，保证训练配置和结果在 PDF 中对齐展示。
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))


def add_image(story, image_name, width=6.6 * inch):
    # 图片文件来自真实训练和评测产物，报告只负责嵌入。
    image_path = CODE_DIR / image_name
    img = Image(str(image_path))
    ratio = width / img.imageWidth
    img.drawWidth = width
    img.drawHeight = img.imageHeight * ratio
    story.append(KeepTogether([img, Spacer(1, 0.18 * inch)]))


def build_report():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyTight",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=14,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=7.2,
            leading=9,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=8,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="A5 DPO Experiment Report",
    )

    story = []
    story.append(Paragraph("A5 DPO Experiment Report", styles["Title"]))
    story.append(Paragraph("Junming Feng - 12311031", styles["BodyTight"]))
    story.append(Spacer(1, 0.18 * inch))

    add_heading(story, "1. Objective", styles["Heading2"])
    story.append(
        paragraph(
            "This assignment implements Direct Preference Optimization (DPO) for human alignment "
            "on top of the GPT-2 Medium 355M SFT checkpoint. The policy model is trained with "
            "instruction-data-with-preference.json, then used to generate model_outputs.json on "
            "AlpacaEval. The generated outputs are evaluated with the course-provided Qwen judge.",
            styles["BodyTight"],
        )
    )
    story.append(
        paragraph(
            "Final results: raw win rate 62.61%, length-controlled win rate 82.71%, "
            "and 805 AlpacaEval examples.",
            styles["BodyTight"],
        )
    )

    add_heading(story, "2. Experimental Setup", styles["Heading2"])
    setup_rows = [
        ["Item", "Configuration"],
        ["Initial model", "gpt2-355M-sft.pth"],
        ["DPO model", "gpt2-medium355M-dpo.pth"],
        ["Model size", "GPT-2 Medium 355M"],
        ["Training data", "instruction-data-with-preference.json"],
        ["Preference examples", "1100"],
        ["Effective batch size", "8"],
        ["Optimizer", "AdamW"],
        ["Learning rate", "1e-6"],
        ["Weight decay", "0.01"],
        ["DPO beta", "0.1"],
        ["Training epochs", "1 epoch"],
        ["Generation strategy", "Greedy decoding, max_new_tokens = 32"],
        ["Evaluator", "Qwen judge via AlpacaEval"],
    ]
    add_table(story, setup_rows, [2.2 * inch, 4.5 * inch])
    story.append(
        paragraph(
            "The reference model is initialized from the same SFT checkpoint as the policy model "
            "and kept frozen in evaluation mode. Its chosen and rejected response log probabilities "
            "are precomputed before policy updates. The DPO loss is computed only on response tokens; "
            "prompt tokens are excluded by the response mask.",
            styles["BodyTight"],
        )
    )

    add_heading(story, "3. DPO Training Results", styles["Heading2"])
    result_rows = [
        ["Metric", "Initial value", "Final value"],
        ["Train loss", "0.6931", "0.6176"],
        ["Validation loss", "0.6931", "0.6157"],
        ["Train reward margin", "0.0000", "1.6159"],
        ["Validation reward margin", "0.0000", "1.7558"],
    ]
    add_table(story, result_rows, [2.8 * inch, 1.8 * inch, 1.8 * inch])
    story.append(
        paragraph(
            "The loss curves decrease during training, while the reward margins increase from 0 "
            "to clearly positive values. This shows that the DPO-trained policy assigns higher "
            "relative probability to chosen responses than rejected responses compared with the "
            "frozen reference model.",
            styles["BodyTight"],
        )
    )
    add_image(story, "dpo_loss_curve.png")
    add_image(story, "dpo_reward_margin_curve.png")

    story.append(PageBreak())
    add_heading(story, "4. AlpacaEval Generation and Evaluation", styles["Heading2"])
    story.append(
        paragraph(
            "The generation script generate_dpo_responses.py strictly loads the DPO checkpoint "
            "specified by --model. The final model_outputs.json contains 805 examples with an "
            "average output length of 76 characters.",
            styles["BodyTight"],
        )
    )
    story.append(
        paragraph(
            "The evaluation uses the course-provided Qwen judge configuration. AlpacaEval's "
            "length-controlled metric requires df_gamed.csv, so the local copy is used to avoid "
            "a HuggingFace download failure in the evaluation environment.",
            styles["BodyTight"],
        )
    )
    command = """ALPACA_EVAL_DF_GAMED_PATH=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/df_gamed.csv \\
OPENAI_CLIENT_CONFIG_PATH=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/openai_configs.yaml \\
alpaca_eval evaluate \\
  --model_outputs /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/model_outputs.json \\
  --reference_outputs /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/reference_outputs.json \\
  --annotators_config /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/qwen_judge \\
  --output_path /data/student/Fengjunming/Temp/26Spring-NLP-CS310/Assignment/A5/a5_dpo_code/eval_final32_lc \\
  --is_overwrite_leaderboard true"""
    story.append(Paragraph(command.replace("\n", "<br/>"), styles["SmallCode"]))
    add_image(story, "alpaca_eval_result.png")
    leaderboard_rows = [
        [
            "Model",
            "Win rate",
            "Std err",
            "Avg len",
            "Wins",
            "Ref wins",
            "Draws",
            "Total",
            "Discrete",
            "LC win",
        ],
        ["gpt2-medium355M-dpo.pth", "62.61", "1.58", "76", "501", "296", "8", "805", "62.73", "82.71"],
    ]
    add_table(
        story,
        leaderboard_rows,
        [1.65 * inch, 0.58 * inch, 0.55 * inch, 0.5 * inch, 0.43 * inch, 0.55 * inch, 0.45 * inch, 0.43 * inch, 0.58 * inch, 0.58 * inch],
    )

    add_heading(story, "5. Deliverables", styles["Heading2"])
    story.append(
        paragraph(
            "The submission package contains run_dpo.py, generate_dpo_responses.py, "
            "model_outputs.json, leaderboard.csv, annotations.json, dpo_tracking.json, the loss "
            "and reward-margin curves, alpaca_eval_result.png, A5_DPO_Report.md, "
            "and Report.pdf.",
            styles["BodyTight"],
        )
    )
    story.append(
        paragraph(
            "The DPO checkpoint gpt2-medium355M-dpo.pth is saved locally as the resulting model. "
            "It is excluded from the submission zip because the assignment submit list requires "
            "the training script, generated model outputs, and PDF report.",
            styles["BodyTight"],
        )
    )

    add_heading(story, "6. Conclusion", styles["Heading2"])
    story.append(
        paragraph(
            "This experiment completes DPO training, curve saving, model output generation, and "
            "Qwen-judge AlpacaEval evaluation. The final raw win rate is 62.61%, which is above "
            "the required 50% threshold. The length-controlled win rate is 82.71%.",
            styles["BodyTight"],
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build_report()
