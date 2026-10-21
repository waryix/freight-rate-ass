from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

RPM_LOW = 1.2
RPM_HIGH = 4.5
TE_SMOOTHING = 25.0
TE_COLUMNS = ("pickup", "delivery", "lane")
TE_FEATURES = ["pickup_te", "delivery_te", "lane_te"]

BASE_FEATURES = ["log_distance", "log_weight", "is_reefer", "is_flatbed"]
POLY_FEATURES = [
    "log_distance_sq",
    "log_weight_sq",
    "log_distance_x_log_weight",
    "log_distance_x_reefer",
    "log_distance_x_flatbed",
    "log_weight_x_reefer",
    "log_weight_x_flatbed",
]
DOW_FEATURES = ["dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]

MODEL_VARIANTS = {
    "base": BASE_FEATURES,
    "base_poly_dow": BASE_FEATURES + POLY_FEATURES + DOW_FEATURES,
    "full": BASE_FEATURES + POLY_FEATURES + DOW_FEATURES + TE_FEATURES,
    "full_no_dow": BASE_FEATURES + POLY_FEATURES + TE_FEATURES,
}
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class DataContext:
    weight_medians: dict[str, float]
    global_weight_median: float
    counts: dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class ResidualEncoder:
    smoothing: float = TE_SMOOTHING
    prior: float = 0.0
    maps: dict[str, pd.Series] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, residual: np.ndarray) -> "ResidualEncoder":
        self.prior = float(np.mean(residual))
        working = pd.DataFrame({"residual": residual})
        for column in TE_COLUMNS:
            working[column] = frame[column].to_numpy()
        for column in TE_COLUMNS:
            grouped = working.groupby(column)["residual"].agg(["sum", "count"])
            self.maps[column] = (grouped["sum"] + self.prior * self.smoothing) / (
                grouped["count"] + self.smoothing
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for column in TE_COLUMNS:
            values = frame[column].map(self.maps[column])
            output[f"{column}_te"] = values.fillna(self.prior).astype(float)
        return output


@dataclass
class RateModel:
    base_model: LinearRegression
    encoder: ResidualEncoder
    scaler: StandardScaler
    model: Ridge
    feature_names: list[str]
    context: DataContext
    metadata: dict[str, Any]

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        base_pred = predict_base(self.base_model, frame)
        te_frame = self.encoder.transform(frame)
        model_frame = build_model_frame(frame, te_frame)
        model_frame["base_pred"] = base_pred
        design = model_frame[self.feature_names].to_numpy(dtype=float)
        return self.model.predict(self.scaler.transform(design))

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        rate = np.exp(np.clip(self.predict_log(frame), 0.0, 12.0))
        distance = frame["distance"].to_numpy(dtype=float)
        return np.clip(rate, RPM_LOW * distance, RPM_HIGH * distance)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    validation = pd.read_csv(VALIDATION_PATH, parse_dates=["date"])
    december = pd.read_csv(DECEMBER_PATH, parse_dates=["date"])
    template = pd.read_csv(TEMPLATE_PATH)
    return train, validation, december, template


def build_context(train: pd.DataFrame) -> DataContext:
    weight_abs = train["weight"].abs()
    weight_medians = weight_abs.groupby(train["equipment"]).median().dropna()
    lane = train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)
    return DataContext(
        weight_medians={key: float(value) for key, value in weight_medians.items()},
        global_weight_median=float(weight_abs.median()),
        counts={
            "pickup_count": train["pickup"].value_counts(),
            "delivery_count": train["delivery"].value_counts(),
            "lane_count": lane.value_counts(),
        },
    )


def prepare_frame(frame: pd.DataFrame, context: DataContext) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["weight_missing"] = data["weight"].isna().astype(float)
    weight = data["weight"].abs()
    equipment_median = data["equipment"].map(context.weight_medians)
    data["weight_abs"] = (
        weight.fillna(equipment_median)
        .fillna(context.global_weight_median)
        .astype(float)
    )
    data["log_weight"] = np.log1p(data["weight_abs"])
    data["distance"] = data["distance"].astype(float)
    data["log_distance"] = np.log(data["distance"])
    data["is_reefer"] = (data["equipment"] == "Reefer").astype(float)
    data["is_flatbed"] = (data["equipment"] == "Flatbed").astype(float)
    data["dow"] = data["date"].dt.dayofweek.astype(float)
    data["is_weekend"] = (data["dow"] >= 5).astype(float)
    data["lane"] = data["pickup"].astype(str) + " -> " + data["delivery"].astype(str)
    data["log_distance_sq"] = data["log_distance"] ** 2
    data["log_weight_sq"] = data["log_weight"] ** 2
    data["log_distance_x_log_weight"] = data["log_distance"] * data["log_weight"]
    data["log_distance_x_reefer"] = data["log_distance"] * data["is_reefer"]
    data["log_distance_x_flatbed"] = data["log_distance"] * data["is_flatbed"]
    data["log_weight_x_reefer"] = data["log_weight"] * data["is_reefer"]
    data["log_weight_x_flatbed"] = data["log_weight"] * data["is_flatbed"]
    for column in range(1, 7):
        data[f"dow_{column}"] = (data["dow"] == column).astype(float)
    return data


def build_model_frame(frame: pd.DataFrame, te_frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in TE_FEATURES:
        working[column] = te_frame[column].to_numpy(dtype=float)
    return working


def fit_base_model(frame: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(
        frame[BASE_FEATURES].to_numpy(dtype=float),
        np.log(frame["posted_rate"].to_numpy(dtype=float)),
    )
    return model


def predict_base(model: LinearRegression, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame[BASE_FEATURES].to_numpy(dtype=float))


def target_inlier_mask(frame: pd.DataFrame) -> np.ndarray:
    rate_per_mile = frame["posted_rate"].to_numpy(dtype=float) / frame[
        "distance"
    ].to_numpy(dtype=float)
    return (rate_per_mile >= RPM_LOW) & (rate_per_mile <= RPM_HIGH)


def oof_encode(
    frame: pd.DataFrame, residual: np.ndarray, inlier_mask: np.ndarray
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index, columns=TE_FEATURES, dtype=float)
    positions = np.flatnonzero(inlier_mask)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_positions, apply_positions in splitter.split(positions):
        fit_index = positions[fit_positions]
        apply_index = positions[apply_positions]
        encoder = ResidualEncoder().fit(frame.iloc[fit_index], residual[fit_index])
        result.iloc[apply_index, :] = encoder.transform(frame.iloc[apply_index]).to_numpy()
    full_encoder = ResidualEncoder().fit(frame.iloc[positions], residual[positions])
    outside = np.flatnonzero(~inlier_mask)
    if len(outside):
        result.iloc[outside, :] = full_encoder.transform(frame.iloc[outside]).to_numpy()
    return result


def fit_encoder(frame: pd.DataFrame, residual: np.ndarray) -> ResidualEncoder:
    return ResidualEncoder().fit(frame, residual)


def metric_bundle(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = actual != 0
    return {
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape": float(
            np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero]))
        ),
        "r2": float(r2_score(actual, predicted)),
        "n": int(len(actual)),
    }


@dataclass
class FoldData:
    name: str
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    inlier_mask: np.ndarray
    model_frame: pd.DataFrame
    target_log: np.ndarray
    test_model_frame: pd.DataFrame
    test_actual: np.ndarray
    test_inlier_mask: np.ndarray
    test_distance: np.ndarray
    base_test_log: np.ndarray


def build_fold_data(
    name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> FoldData:
    inlier = target_inlier_mask(train_frame)
    cleaned = train_frame.loc[inlier].copy()
    base_model = fit_base_model(cleaned)
    train_base_log = predict_base(base_model, train_frame)
    test_base_log = predict_base(base_model, test_frame)
    residual = np.log(train_frame["posted_rate"].to_numpy(dtype=float)) - train_base_log
    te_train = oof_encode(train_frame, residual, inlier)
    encoder = fit_encoder(cleaned, residual[inlier])
    te_test = encoder.transform(test_frame)
    return FoldData(
        name=name,
        train_frame=train_frame,
        test_frame=test_frame,
        inlier_mask=inlier,
        model_frame=build_model_frame(train_frame, te_train),
        target_log=np.log(train_frame["posted_rate"].to_numpy(dtype=float)),
        test_model_frame=build_model_frame(test_frame, te_test),
        test_actual=test_frame["posted_rate"].to_numpy(dtype=float),
        test_inlier_mask=target_inlier_mask(test_frame),
        test_distance=test_frame["distance"].to_numpy(dtype=float),
        base_test_log=test_base_log,
    )


def fit_ridge(
    fold: FoldData, feature_names: list[str], alpha: float
) -> tuple[Ridge, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    design = scaler.fit_transform(
        fold.model_frame.loc[fold.inlier_mask, feature_names].to_numpy(dtype=float)
    )
    model = Ridge(alpha=alpha)
    model.fit(design, fold.target_log[fold.inlier_mask])
    test_design = scaler.transform(
        fold.test_model_frame[feature_names].to_numpy(dtype=float)
    )
    return model, scaler, model.predict(test_design)


def predictions_from_log(
    log_prediction: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    rate = np.exp(np.clip(np.asarray(log_prediction, dtype=float), 0.0, 12.0))
    return np.clip(rate, RPM_LOW * distance, RPM_HIGH * distance)


def fold_metric_record(fold: FoldData, predicted: np.ndarray) -> dict[str, Any]:
    inlier = fold.test_inlier_mask
    return {
        "full": metric_bundle(fold.test_actual, predicted),
        "clean": metric_bundle(fold.test_actual[inlier], predicted[inlier]),
        "clean_share": float(inlier.mean()),
    }


def evaluate_configurations(folds: list[FoldData]) -> dict[str, Any]:
    results = []
    for variant_name, feature_names in MODEL_VARIANTS.items():
        for alpha in RIDGE_ALPHAS:
            records = []
            for fold in folds:
                _, _, test_log = fit_ridge(fold, feature_names, alpha)
                predicted = predictions_from_log(test_log, fold.test_distance)
                records.append(fold_metric_record(fold, predicted))
            average_rmse = float(np.mean([record["clean"]["rmse"] for record in records]))
            average_mae = float(np.mean([record["clean"]["mae"] for record in records]))
            results.append(
                {
                    "variant": variant_name,
                    "features": feature_names,
                    "ridge_alpha": alpha,
                    "average_clean_rmse": average_rmse,
                    "average_clean_mae": average_mae,
                    "folds": records,
                }
            )
    results.sort(key=lambda item: (item["average_clean_rmse"], item["average_clean_mae"]))
    return {"best": results[0], "all": results}


def build_report_plots(train: pd.DataFrame, fold: FoldData, predicted: np.ndarray) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_frame = pd.DataFrame(
        {
            "actual": fold.test_actual,
            "predicted": predicted,
            "clean": fold.test_inlier_mask,
            "month": fold.test_frame["date"].dt.strftime("%Y-%m"),
            "equipment": fold.test_frame["equipment"],
        }
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    clean_plot = plot_frame[plot_frame["clean"]]
    axis.scatter(
        clean_plot["actual"],
        clean_plot["predicted"],
        s=10,
        alpha=0.45,
        color="#064A56",
        linewidths=0,
    )
    upper = float(
        np.percentile(
            np.maximum(clean_plot["actual"], clean_plot["predicted"]), 99.9
        )
    )
    axis.plot([0, upper], [0, upper], color="#455A60", linewidth=1.0, linestyle="--")
    axis.set_xlim(0, upper)
    axis.set_ylim(0, upper)
    axis.set_xlabel("Actual posted rate ($)")
    axis.set_ylabel("Predicted posted rate ($)")
    axis.set_title(
        "Time holdout: actual vs predicted (cleaned rows)",
        loc="left",
        fontweight="bold",
    )
    axis.grid(color="#D9E2E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "holdout_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    residuals = (plot_frame["predicted"] - plot_frame["actual"])[plot_frame["clean"]]
    limit = float(np.percentile(np.abs(residuals), 99.5))
    fig, axis = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    axis.hist(
        residuals.clip(-limit, limit),
        bins=60,
        color="#064A56",
        alpha=0.85,
    )
    axis.axvline(0, color="#455A60", linewidth=1.0)
    axis.set_xlabel("Prediction error ($), central 99.5% shown")
    axis.set_ylabel("Loads")
    axis.set_title("Holdout residuals (cleaned rows)", loc="left", fontweight="bold")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "holdout_residuals.png", bbox_inches="tight")
    plt.close(fig)

    train = train.copy()
    train["rpm"] = train["posted_rate"] / train["distance"]
    fig, axis = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for equipment, color in (
        ("Dry Van", "#064A56"),
        ("Reefer", "#C48A2A"),
        ("Flatbed", "#7A5AA6"),
    ):
        subset = train[train["equipment"] == equipment]
        axis.scatter(
            subset["distance"],
            subset["rpm"],
            s=4,
            alpha=0.25,
            color=color,
            linewidths=0,
            label=equipment,
        )
    axis.set_xlabel("Distance (miles)")
    axis.set_ylabel("Rate per mile ($)")
    axis.set_title("Rate per mile declines with distance", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(color="#D9E2E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "rate_per_mile_vs_distance.png", bbox_inches="tight")
    plt.close(fig)

    monthly = (
        train.groupby([train["date"].dt.to_period("M"), "equipment"])["rpm"]
        .mean()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()
    fig, axis = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    for equipment, color in (
        ("Dry Van", "#064A56"),
        ("Reefer", "#C48A2A"),
        ("Flatbed", "#7A5AA6"),
    ):
        subset = monthly[monthly["equipment"] == equipment]
        axis.plot(
            subset["date"],
            subset["rpm"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=equipment,
        )
    axis.set_ylabel("Mean rate per mile ($)")
    axis.set_title("Monthly rate level by equipment", loc="left", fontweight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "monthly_rate_level.png", bbox_inches="tight")
    plt.close(fig)


def fit_final_model(
    train: pd.DataFrame,
    context: DataContext,
    selection: dict[str, Any],
) -> RateModel:
    inlier = target_inlier_mask(train)
    cleaned = train.loc[inlier].copy()
    base_model = fit_base_model(cleaned)
    base_log = predict_base(base_model, train)
    residual = np.log(train["posted_rate"].to_numpy(dtype=float)) - base_log
    te_train = oof_encode(train, residual, inlier)
    encoder = fit_encoder(cleaned, residual[inlier])
    model_frame = build_model_frame(train, te_train)
    feature_names = selection["features"]
    scaler = StandardScaler()
    design = scaler.fit_transform(
        model_frame.loc[inlier, feature_names].to_numpy(dtype=float)
    )
    ridge = Ridge(alpha=float(selection["ridge_alpha"]))
    ridge.fit(design, np.log(train["posted_rate"].to_numpy(dtype=float))[inlier])
    metadata = {
        "variant": selection["variant"],
        "ridge_alpha": float(selection["ridge_alpha"]),
        "features": feature_names,
        "training_rows": int(len(cleaned)),
        "training_rows_total": int(len(train)),
    }
    return RateModel(
        base_model=base_model,
        encoder=encoder,
        scaler=scaler,
        model=ridge,
        feature_names=feature_names,
        context=context,
        metadata=metadata,
    )


def save_rate_model(model: RateModel, path: Path) -> None:
    artifact = {
        "base_model": model.base_model,
        "encoder": {
            "smoothing": model.encoder.smoothing,
            "prior": model.encoder.prior,
            "maps": model.encoder.maps,
        },
        "scaler": model.scaler,
        "model": model.model,
        "feature_names": model.feature_names,
        "metadata": model.metadata,
        "context": {
            "weight_medians": model.context.weight_medians,
            "global_weight_median": model.context.global_weight_median,
            "counts": model.context.counts,
        },
    }
    joblib.dump(artifact, path)


def load_rate_model(path: Path) -> RateModel:
    artifact = joblib.load(path)
    encoder = ResidualEncoder(
        smoothing=float(artifact["encoder"]["smoothing"]),
        prior=float(artifact["encoder"]["prior"]),
    )
    encoder.maps = artifact["encoder"]["maps"]
    context = DataContext(
        weight_medians=artifact["context"]["weight_medians"],
        global_weight_median=float(artifact["context"]["global_weight_median"]),
        counts=artifact["context"]["counts"],
    )
    return RateModel(
        base_model=artifact["base_model"],
        encoder=encoder,
        scaler=artifact["scaler"],
        model=artifact["model"],
        feature_names=artifact["feature_names"],
        context=context,
        metadata=artifact["metadata"],
    )


def write_predictions(
    model: RateModel,
    validation_prepared: pd.DataFrame,
    december_prepared: pd.DataFrame,
    december_original: pd.DataFrame,
    template: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_rate = model.predict_rate(validation_prepared)
    validation_output = pd.DataFrame(
        {
            "load_id": validation_prepared["load_id"].to_numpy(),
            "predicted_rate": np.round(validation_rate, 2),
        }
    )
    submission = template[["load_id"]].merge(
        validation_output, on="load_id", how="left", validate="one_to_one"
    )
    if submission["predicted_rate"].isna().any():
        raise ValueError("validation predictions are incomplete")
    submission.to_csv(PREDICTIONS_PATH, index=False)

    december_rate = model.predict_rate(december_prepared)
    december_output = december_original.copy()
    december_output["predicted_rate"] = np.round(december_rate, 2)
    december_output = december_output[december_original.columns]
    december_output.to_csv(DECEMBER_PATH, index=False)
    return submission, december_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, validate and score the freight rate model."
    )
    parser.add_argument("--skip-report-plots", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, validation_raw, december_raw, template = load_inputs()
    context = build_context(train_raw)
    train = prepare_frame(train_raw, context)
    validation = prepare_frame(validation_raw, context)
    december = prepare_frame(december_raw, context)

    fold_definitions = [
        ("Sep-Oct holdout", "2025-09-01", "2025-11-01"),
        ("Aug-Sep holdout", "2025-08-01", "2025-10-01"),
    ]
    folds = []
    for name, start, end in fold_definitions:
        train_mask = train["date"] < pd.Timestamp(start)
        test_mask = (train["date"] >= pd.Timestamp(start)) & (
            train["date"] < pd.Timestamp(end)
        )
        folds.append(
            build_fold_data(
                name,
                train.loc[train_mask].copy(),
                train.loc[test_mask].copy(),
            )
        )

    evaluation = evaluate_configurations(folds)
    best = evaluation["best"]
    comparison = pd.DataFrame(
        [
            {
                "rank": rank + 1,
                "variant": item["variant"],
                "ridge_alpha": item["ridge_alpha"],
                "average_clean_rmse": item["average_clean_rmse"],
                "average_clean_mae": item["average_clean_mae"],
            }
            for rank, item in enumerate(evaluation["all"])
        ]
    )
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    primary = folds[0]
    _, _, primary_test_log = fit_ridge(
        primary, best["features"], float(best["ridge_alpha"])
    )
    primary_predicted = predictions_from_log(primary_test_log, primary.test_distance)
    if not args.skip_report_plots:
        build_report_plots(train, primary, primary_predicted)

    primary_frame = primary.test_frame.reset_index(drop=True)
    primary_prediction_frame = pd.DataFrame(
        {
            "load_id": primary_frame["load_id"],
            "date": primary_frame["date"],
            "equipment": primary_frame["equipment"],
            "distance": primary_frame["distance"],
            "actual": primary.test_actual,
            "predicted": primary_predicted,
            "clean": primary.test_inlier_mask,
        }
    )
    primary_prediction_frame["error"] = (
        primary_prediction_frame["predicted"] - primary_prediction_frame["actual"]
    )
    primary_prediction_frame.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    per_month = {}
    for month, subset in primary_prediction_frame.groupby(
        primary_prediction_frame["date"].dt.strftime("%Y-%m")
    ):
        clean_subset = subset[subset["clean"]]
        per_month[month] = metric_bundle(
            clean_subset["actual"].to_numpy(), clean_subset["predicted"].to_numpy()
        )
    per_equipment = {}
    for equipment, subset in primary_prediction_frame.groupby("equipment"):
        clean_subset = subset[subset["clean"]]
        per_equipment[equipment] = metric_bundle(
            clean_subset["actual"].to_numpy(), clean_subset["predicted"].to_numpy()
        )

    baseline_median = float(
        train.loc[target_inlier_mask(train), "posted_rate"].median()
    )
    baseline_test = np.full(len(primary.test_actual), baseline_median)
    baseline_metrics = metric_bundle(
        primary.test_actual[primary.test_inlier_mask],
        baseline_test[primary.test_inlier_mask],
    )
    base_only = predictions_from_log(primary.base_test_log, primary.test_distance)
    base_only_metrics = metric_bundle(
        primary.test_actual[primary.test_inlier_mask],
        base_only[primary.test_inlier_mask],
    )

    final_model = fit_final_model(train, context, best)
    submission, december_output = write_predictions(
        final_model,
        validation,
        december,
        december_raw.copy(),
        template,
    )
    save_rate_model(final_model, MODEL_DIR / "freight_rate_model.joblib")

    data_quality = {
        "train_rows": int(len(train_raw)),
        "validation_rows": int(len(validation_raw)),
        "december_rows": int(len(december_raw)),
        "train_date_min": str(train_raw["date"].min().date()),
        "train_date_max": str(train_raw["date"].max().date()),
        "validation_date_min": str(validation_raw["date"].min().date()),
        "validation_date_max": str(validation_raw["date"].max().date()),
        "train_missing_weight": int(train_raw["weight"].isna().sum()),
        "validation_missing_weight": int(validation_raw["weight"].isna().sum()),
        "train_missing_market_index": int(train_raw["market_index"].isna().sum()),
        "validation_missing_market_index": int(
            validation_raw["market_index"].isna().sum()
        ),
        "train_negative_weight": int((train_raw["weight"] < 0).sum()),
        "validation_negative_weight": int((validation_raw["weight"] < 0).sum()),
        "train_low_rpm_outliers": int(
            (train_raw["posted_rate"] / train_raw["distance"] < RPM_LOW).sum()
        ),
        "train_high_rpm_outliers": int(
            (train_raw["posted_rate"] / train_raw["distance"] > RPM_HIGH).sum()
        ),
        "train_inlier_rows": int(target_inlier_mask(train).sum()),
        "train_inlier_share": float(target_inlier_mask(train).mean()),
        "validation_unseen_pickup_rows": int(
            (~validation_raw["pickup"].isin(train_raw["pickup"])).sum()
        ),
        "validation_unseen_delivery_rows": int(
            (~validation_raw["delivery"].isin(train_raw["delivery"])).sum()
        ),
        "validation_unseen_lane_rows": int(
            (
                ~(
                    validation_raw["pickup"].astype(str)
                    + " -> "
                    + validation_raw["delivery"].astype(str)
                ).isin(
                    train_raw["pickup"].astype(str)
                    + " -> "
                    + train_raw["delivery"].astype(str)
                )
            ).sum()
        ),
        "train_lanes": int(
            (train_raw["pickup"].astype(str) + " -> " + train_raw["delivery"].astype(str)).nunique()
        ),
        "train_median_rate_per_mile": float(
            (train_raw["posted_rate"] / train_raw["distance"]).median()
        ),
    }
    metrics = {
        "data_quality": data_quality,
        "selected": {
            "variant": best["variant"],
            "features": best["features"],
            "ridge_alpha": float(best["ridge_alpha"]),
            "average_clean_rmse": float(best["average_clean_rmse"]),
            "average_clean_mae": float(best["average_clean_mae"]),
        },
        "folds": best["folds"],
        "primary_holdout": {
            "per_month": per_month,
            "per_equipment": per_equipment,
            "baseline_median": baseline_metrics,
            "base_log_linear": base_only_metrics,
            "full_holdout": metric_bundle(primary.test_actual, primary_predicted),
            "clean_holdout": metric_bundle(
                primary.test_actual[primary.test_inlier_mask],
                primary_predicted[primary.test_inlier_mask],
            ),
        },
        "prediction_summary": {
            "validation_min": float(submission["predicted_rate"].min()),
            "validation_max": float(submission["predicted_rate"].max()),
            "validation_mean": float(submission["predicted_rate"].mean()),
            "december_min": float(december_output["predicted_rate"].min()),
            "december_max": float(december_output["predicted_rate"].max()),
            "december_mean": float(december_output["predicted_rate"].mean()),
        },
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics["selected"], indent=2))
    print(json.dumps(metrics["primary_holdout"]["clean_holdout"], indent=2))
    print(f"Wrote {PREDICTIONS_PATH}")
    print(f"Wrote {DECEMBER_PATH}")


if __name__ == "__main__":
    main()
