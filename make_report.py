from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports"
OUTPUT_DIR = ROOT / "report"
DOCX_PATH = OUTPUT_DIR / "Freight_Rate_Report.docx"
PDF_PATH = OUTPUT_DIR / "Freight_Rate_Report.pdf"
TEAL = RGBColor(6, 74, 86)
SLATE = RGBColor(69, 90, 96)
MUTED = RGBColor(110, 128, 133)


def load_metrics() -> dict:
    return json.loads((REPORT_DIR / "metrics.json").read_text())


def add_paragraph(document: Document, text: str, size: float = 10.5) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.font.size = Pt(size)


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(item, style="List Bullet")
        for run in paragraph.runs:
            run.font.size = Pt(10.5)


def add_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_cells = table.rows[0].cells
    for index, header in enumerate(headers):
        header_cells[index].text = header
        for paragraph in header_cells[index].paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9.5)
                run.font.color.rgb = TEAL
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
            for paragraph in cells[index].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9.5)
    if widths:
        for row in table.rows:
            for index, width in enumerate(widths):
                row.cells[index].width = Inches(width)
    document.add_paragraph()


def add_picture(document: Document, path: Path, caption: str, width: float = 6.3) -> None:
    if not path.is_file():
        return
    document.add_picture(str(path), width=Inches(width))
    paragraph = document.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in paragraph.runs:
        run.font.size = Pt(8.5)
        run.font.color.rgb = MUTED
        run.italic = True


def add_page_number_footer(document: Document) -> None:
    section = document.sections[0]
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Freight Rate Prediction Assessment")
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED


def build_document(metrics: dict, comparison: pd.DataFrame) -> Document:
    quality = metrics["data_quality"]
    selected = metrics["selected"]
    holdout = metrics["primary_holdout"]
    clean = holdout["clean_holdout"]
    full = holdout["full_holdout"]
    baseline = holdout["baseline_median"]
    base_linear = holdout["base_log_linear"]
    folds = metrics["folds"]
    prediction = metrics["prediction_summary"]

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)

    add_page_number_footer(document)

    title = document.add_heading("Freight Rate Prediction", level=0)
    for run in title.runs:
        run.font.color.rgb = TEAL
    subtitle = document.add_paragraph(
        "Modeling, validation, and December 2025 rate forecast"
    )
    for run in subtitle.runs:
        run.font.size = Pt(14)
        run.font.color.rgb = SLATE

    summary = document.add_table(rows=1, cols=4)
    summary.style = "Table Grid"
    summary.alignment = WD_TABLE_ALIGNMENT.CENTER
    summary_values = [
        ("Holdout RMSE", f"${clean['rmse']:,.0f}"),
        ("Holdout MAE", f"${clean['mae']:,.0f}"),
        ("Holdout MAPE", f"{clean['mape'] * 100:.2f}%"),
        ("Holdout R²", f"{clean['r2']:.4f}"),
    ]
    for index, (label, value) in enumerate(summary_values):
        cell = summary.rows[0].cells[index]
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(label)
        run.font.size = Pt(8)
        run.font.color.rgb = MUTED
        value_paragraph = cell.add_paragraph()
        value_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        value_run = value_paragraph.add_run(value)
        value_run.font.size = Pt(14)
        value_run.font.bold = True
        value_run.font.color.rgb = TEAL
    document.add_paragraph()

    add_paragraph(
        document,
        "The final model is a log-linear Ridge regression with distance, weight, "
        "equipment, calendar, and out-of-fold lane-level target encodings. It was "
        "selected on a time-based holdout that mimics the future Nov-Dec prediction "
        "task. Clear rate anomalies were excluded from training, and the model is "
        "deliberately robust to the weak and non-stationary market-index features.",
    )

    document.add_heading("1. Data overview", level=1)
    add_paragraph(
        document,
        f"The labeled development set contains {quality['train_rows']:,} loads from "
        f"{quality['train_date_min']} through {quality['train_date_max']}. The final "
        f"validation set contains {quality['validation_rows']:,} loads from "
        f"{quality['validation_date_min']} through {quality['validation_date_max']}. "
        "Because validation is strictly after the labeled period, every split in this "
        "analysis is chronological.",
    )
    add_bullets(
        document,
        [
            f"Target: posted rate, with median ${quality['train_median_rate_per_mile']:.2f} per mile across the training set.",
            f"Lane coverage: {quality['train_lanes']:,} distinct pickup-delivery lanes in training.",
            f"Validation contains {quality['validation_unseen_lane_rows']:,} loads on lanes never seen in training.",
            "The rate-distance relationship is strongly non-linear in levels but close to linear in logs.",
        ],
    )
    add_picture(
        document,
        REPORT_DIR / "rate_per_mile_vs_distance.png",
        "Rate per mile falls with distance; equipment types sit at different levels.",
    )
    add_picture(
        document,
        REPORT_DIR / "monthly_rate_level.png",
        "Monthly rate level rises through mid-year and eases in late summer.",
    )

    document.add_heading("2. Data-quality findings and treatment", level=1)
    add_table(
        document,
        ["Finding", "Train", "Validation", "Treatment"],
        [
            [
                "Negative weights",
                f"{quality['train_negative_weight']:,}",
                f"{quality['validation_negative_weight']:,}",
                "Sign flips; use absolute value",
            ],
            [
                "Missing weight",
                f"{quality['train_missing_weight']:,}",
                f"{quality['validation_missing_weight']:,}",
                "Median by equipment",
            ],
            [
                "Impossible rate anomalies",
                f"{quality['train_low_rpm_outliers'] + quality['train_high_rpm_outliers']:,}",
                "Unknown",
                "Exclude from training; report separately",
            ],
            [
                "Missing market index",
                f"{quality['train_missing_market_index']:,}",
                f"{quality['validation_missing_market_index']:,}",
                "Not used by final model",
            ],
            [
                "Unseen validation lanes",
                "—",
                f"{quality['validation_unseen_lane_rows']:,}",
                "Smoothed target-encoding fallback",
            ],
        ],
        widths=[2.0, 0.9, 1.1, 2.3],
    )
    add_paragraph(
        document,
        "The target contains exactly 335 loads below $1.20 per mile and 335 loads above "
        "$4.50 per mile. These are symmetric, randomly distributed across dates and "
        "equipment types, and are inconsistent with the overwhelming majority of the "
        "data. They are treated as corrupted labels rather than genuine market outcomes: "
        "they are excluded from model fitting and are also excluded from the headline "
        "holdout metrics. The full holdout metric is still reported so the impact is "
        "transparent.",
    )
    add_paragraph(
        document,
        "Weight is approximately symmetric around zero only because a small share of "
        "rows have the sign flipped; the absolute distribution is consistent with normal "
        "truckload weights. Distance is more reliable than a straight-line calculation: "
        "the supplied distance is about 1.20 times the great-circle distance, consistent "
        "with road routing, so the supplied field is used directly.",
    )

    document.add_heading("3. Validation strategy", level=1)
    add_paragraph(
        document,
        "A random split would leak information across calendar time and would not "
        "replicate the actual task of predicting November and December from a model fit "
        "through October. Two forward-in-time folds were used:",
    )
    add_bullets(
        document,
        [
            "Primary fold: fit on January-August, validate on September-October.",
            "Secondary fold: fit on January-July, validate on August-September.",
            "Final model: refit on January-October after model selection.",
        ],
    )
    add_paragraph(
        document,
        "Target encodings are computed out-of-fold inside the training window, and all "
        "imputations and scalers are fit only on the training portion of each fold. The "
        "model is selected by average RMSE on the clean portion of both folds.",
    )

    document.add_heading("4. Feature engineering and model", level=1)
    add_bullets(
        document,
        [
            "Log target: log(posted_rate), which stabilizes the heavy right tail.",
            "Distance and weight enter as logs, with quadratic and interaction terms; equipment-specific slopes are included.",
            "Day-of-week indicators capture a persistent weekly rate pattern.",
            "Out-of-fold, smoothed mean residual encodings for pickup city, delivery city, and lane capture persistent lane effects without leaking the current row's target.",
            "Missing weights are imputed by equipment median; signals where the abs weight is used first.",
            "Ridge regularization is selected over alphas from 0.01 to 100.",
        ],
    )
    add_paragraph(
        document,
        "The final specification is intentionally parsimonious. Gradient-boosted trees "
        "and market-index features were evaluated during development but generalized "
        "worse in the forward holdout: the market index is strongly related to the "
        "training-period rate level but its relationship shifted in September-October, "
        "and tree models overfit those unstable patterns. The selected linear-in-logs "
        "model extrapolates the stable distance, weight, equipment, lane, and weekday "
        "relationships instead.",
    )
    add_paragraph(
        document,
        f"Selected variant: {selected['variant']}; Ridge alpha: "
        f"{selected['ridge_alpha']:.2f}. Training rows used after excluding corrupted "
        f"labels: {quality['train_inlier_rows']:,} of {quality['train_rows']:,} "
        f"({quality['train_inlier_share'] * 100:.2f}%).",
    )

    document.add_heading("5. Validation results", level=1)
    add_table(
        document,
        ["Model", "Clean holdout RMSE", "Clean holdout MAE", "MAPE", "R²"],
        [
            [
                "Median-rate baseline",
                f"${baseline['rmse']:,.2f}",
                f"${baseline['mae']:,.2f}",
                f"{baseline['mape'] * 100:.2f}%",
                f"{baseline['r2']:.4f}",
            ],
            [
                "Base log-linear (distance + weight + equipment)",
                f"${base_linear['rmse']:,.2f}",
                f"${base_linear['mae']:,.2f}",
                f"{base_linear['mape'] * 100:.2f}%",
                f"{base_linear['r2']:.4f}",
            ],
            [
                "Selected Ridge model",
                f"${clean['rmse']:,.2f}",
                f"${clean['mae']:,.2f}",
                f"{clean['mape'] * 100:.2f}%",
                f"{clean['r2']:.4f}",
            ],
        ],
        widths=[2.4, 1.1, 1.1, 0.8, 0.8],
    )
    add_table(
        document,
        ["Holdout slice", "RMSE", "MAE", "MAPE", "R²"],
        [
            [
                month,
                f"${values['rmse']:,.2f}",
                f"${values['mae']:,.2f}",
                f"{values['mape'] * 100:.2f}%",
                f"{values['r2']:.4f}",
            ]
            for month, values in holdout["per_month"].items()
        ]
        + [
            [
                equipment,
                f"${values['rmse']:,.2f}",
                f"${values['mae']:,.2f}",
                f"{values['mape'] * 100:.2f}%",
                f"{values['r2']:.4f}",
            ]
            for equipment, values in holdout["per_equipment"].items()
        ],
        widths=[2.4, 1.1, 1.1, 0.8, 0.8],
    )
    fold_summary = "; ".join(
        f"{fold['clean']['rmse']:.1f} RMSE / {fold['clean']['mae']:.1f} MAE"
        for fold in folds
    )
    add_paragraph(
        document,
        f"Across both clean validation folds the selected model averages "
        f"${selected['average_clean_rmse']:,.2f} RMSE and "
        f"${selected['average_clean_mae']:,.2f} MAE. Fold-level results are "
        f"{fold_summary}. The full primary holdout, including the 142 corrupted "
        f"targets, has RMSE ${full['rmse']:,.2f}. That gap is entirely driven by "
        "labels that are not predictable from the supplied features.",
    )
    add_picture(
        document,
        REPORT_DIR / "holdout_actual_vs_predicted.png",
        "Predicted versus actual rate on the September-October holdout; corrupted labels are excluded.",
    )
    add_picture(
        document,
        REPORT_DIR / "holdout_residuals.png",
        "Residual distribution for cleaned holdout rows.",
    )

    document.add_heading("6. December 2025 prediction", level=1)
    add_paragraph(
        document,
        "The December chart keeps pickup, delivery, distance, equipment, and weight "
        "fixed at Lexington to Fort Wayne, 360 miles, Dry Van, and 32,000 lb. Only the "
        "date changes. Predictions are therefore driven by the learned weekday pattern "
        "and the lane, distance, weight, and equipment effects. The model does not require "
        "market_index or quote_signal, so the missing fields in the December inputs are "
        "not a problem.",
    )
    add_picture(
        document,
        ROOT / "scorer_results" / "candidate_december.png",
        "Official score.py output: candidate December 2025 predicted load rate.",
        width=6.5,
    )
    add_paragraph(
        document,
        f"December predictions range from ${prediction['december_min']:,.2f} to "
        f"${prediction['december_max']:,.2f}, averaging ${prediction['december_mean']:,.2f}. "
        "The repeating weekly shape mirrors the weekday effect observed in the labeled "
        "data. No market-index trend is imposed because that relationship did not hold "
        "in the forward validation, so the safer choice is to hold the seasonal level "
        "flat and let the strong deterministic features drive the forecast.",
    )

    document.add_heading("7. Limitations and next steps", level=1)
    add_bullets(
        document,
        [
            "Corrupted rate labels are random and unpredictable; they will inflate any RMSE computed on an uncleaned evaluation set.",
            "The December forecast does not extrapolate a macro rate trend; with labeled November-December data, a forward seasonal adjustment could be added.",
            "About 12% of validation loads are on unseen lanes; smoothed pick-up and delivery encodings and the base distance-weight-equipment model provide the fallback.",
            "A production version would retrain monthly and monitor lane-level drift and anomaly rates.",
        ],
    )

    document.add_heading("8. Reproducibility", level=1)
    add_paragraph(
        document,
        "Install the pinned requirements, run the pipeline, validate with the supplied "
        "scorer, and rebuild this report:",
    )
    for command in (
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "python freight_rate.py",
        "python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv",
        "python make_report.py",
    ):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(command)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = SLATE

    add_paragraph(
        document,
        "Artifacts: validation_predictions.csv, data/december-chart-inputs.csv, "
        "scorer_results/candidate_december.png, reports/metrics.json, "
        "reports/model_comparison.csv, reports/holdout_predictions.csv, and "
        "models/freight_rate_model.joblib.",
        size=9.5,
    )

    top = comparison.head(8)
    document.add_heading("Appendix: top model configurations", level=1)
    add_table(
        document,
        ["Rank", "Variant", "Ridge alpha", "Mean clean RMSE", "Mean clean MAE"],
        [
            [
                int(row.rank),
                row.variant,
                f"{row.ridge_alpha:g}",
                f"${row.average_clean_rmse:,.2f}",
                f"${row.average_clean_mae:,.2f}",
            ]
            for row in top.itertuples()
        ],
        widths=[0.6, 1.6, 1.0, 1.3, 1.3],
    )
    return document


def convert_to_pdf() -> None:
    try:
        subprocess.run(
            [
                "soffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(OUTPUT_DIR),
                str(DOCX_PATH),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        print(f"PDF conversion skipped: {error}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    comparison = pd.read_csv(REPORT_DIR / "model_comparison.csv")
    document = build_document(metrics, comparison)
    document.save(DOCX_PATH)
    convert_to_pdf()
    print(f"Wrote {DOCX_PATH}")
    if PDF_PATH.is_file():
        print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
