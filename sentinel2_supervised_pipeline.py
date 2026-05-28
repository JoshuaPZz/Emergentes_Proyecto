#!/usr/bin/env python3
"""
Pipeline profesional para clasificacion supervisada de Sentinel-2.

Incluye:
- Extraccion robusta de pixeles por poligono desde raster multibanda.
- Limpieza de dataset y exportacion a TSV.
- Split estratificado train/test.
- Entrenamiento multi-modelo con opcion de GridSearchCV.
- Evaluacion completa con metricas y matrices de confusion.
- Seleccion automatica del mejor modelo por F1 macro y Kappa.
- Exportacion de modelo, metricas, ranking, graficos y reporte Markdown.

Compatible con Python 3.12+.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import joblib
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from rasterio.mask import mask
from shapely.geometry.base import BaseGeometry
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# Variante comun cuando no se incluye B8 en el stack multibanda.
DEFAULT_BAND_NAMES_WITHOUT_B8 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
EXPECTED_CLASSES = {"Agua", "Playa_sal", "Arena"}


@dataclass
class RasterInfo:
    width: int
    height: int
    count: int
    crs: Any
    nodata: float | int | None
    dtype: str


class PipelineError(Exception):
    """Error controlado para el pipeline."""


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sentinel2_pipeline")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de clasificacion supervisada para Sentinel-2 usando ML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raster", type=Path, default=Path("convertido.tif"), help="Raster multibanda")
    parser.add_argument("--training", type=Path, default=Path("training.gpkg"), help="GeoPackage de entrenamiento")
    parser.add_argument("--class-column", type=str, default="class_name", help="Campo de clase")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directorio de salida")
    parser.add_argument("--training-tsv", type=str, default="training.tsv", help="Nombre del dataset TSV")
    parser.add_argument("--test-size", type=float, default=0.30, help="Proporcion para test")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--all-touched", action="store_true", help="Incluir pixeles tocados por borde")
    parser.add_argument("--tune", action="store_true", help="Activar GridSearchCV para ajuste")
    parser.add_argument("--cv-folds", type=int, default=5, help="Numero de folds para GridSearchCV")
    parser.add_argument("--max-polygons", type=int, default=None, help="Limitar cantidad de poligonos para pruebas")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximo de muestras por clase para entrenar (subsampling estratificado).",
    )
    parser.add_argument("--disable-progress", action="store_true", help="Deshabilitar barras de progreso")
    parser.add_argument("--disable-rf", action="store_true", help="Deshabilitar Random Forest")
    return parser.parse_args()


def validate_inputs(raster_path: Path, training_path: Path, output_dir: Path) -> None:
    if not raster_path.exists():
        raise PipelineError(f"No existe raster: {raster_path}")
    if not training_path.exists():
        raise PipelineError(f"No existe vector de entrenamiento: {training_path}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _safe_fix_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None:
        return None
    if geom.is_empty:
        return None
    if geom.is_valid:
        return geom

    try:
        fixed = geom.buffer(0)
        if fixed is None or fixed.is_empty:
            return None
        return fixed if fixed.is_valid else None
    except Exception:
        return None


def load_and_prepare_training(
    training_path: Path,
    class_column: str,
    raster_crs: Any,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(training_path)
    except Exception as exc:
        raise PipelineError(f"Error leyendo {training_path}: {exc}") from exc

    if gdf.empty:
        raise PipelineError("La capa de entrenamiento esta vacia")
    if class_column not in gdf.columns:
        raise PipelineError(f"No se encontro columna de clase: {class_column}")
    if "geometry" not in gdf.columns:
        raise PipelineError("No se encontro columna geometry en la capa de entrenamiento")

    initial_count = len(gdf)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf["geometry"] = gdf["geometry"].apply(_safe_fix_geometry)
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise PipelineError("Todas las geometrias quedaron invalidas o vacias tras limpieza")

    dropped = initial_count - len(gdf)
    if dropped > 0:
        logger.warning("Se descartaron %d geometrias invalidas/vacias", dropped)

    gdf[class_column] = gdf[class_column].astype(str).str.strip()
    gdf = gdf[gdf[class_column] != ""].copy()
    if gdf.empty:
        raise PipelineError("No hay etiquetas validas en la columna de clase")

    if gdf.crs is None:
        raise PipelineError("El GeoPackage no tiene CRS definido")

    if gdf.crs != raster_crs:
        logger.warning("CRS diferente detectado. Reproyectando entrenamiento a CRS del raster")
        gdf = gdf.to_crs(raster_crs)

    present_classes = set(gdf[class_column].unique())
    missing_expected = EXPECTED_CLASSES - present_classes
    if missing_expected:
        logger.warning("Faltan clases esperadas en entrenamiento: %s", sorted(missing_expected))

    return gdf


def open_raster_info(raster_path: Path) -> RasterInfo:
    try:
        with rasterio.open(raster_path) as src:
            return RasterInfo(
                width=src.width,
                height=src.height,
                count=src.count,
                crs=src.crs,
                nodata=src.nodata,
                dtype=src.dtypes[0] if src.dtypes else "unknown",
            )
    except Exception as exc:
        raise PipelineError(f"No se pudo abrir raster {raster_path}: {exc}") from exc


def resolve_band_names(raster_count: int) -> list[str]:
    """Resuelve nombres de bandas de forma robusta.

    - 10 bandas: esquema Sentinel-2 esperado con B8 y B8A.
    - 9 bandas: asume faltante B8 (caso solicitado por usuario).
    - Otro numero: nombres genericos band_1..band_n.
    """
    if raster_count == len(DEFAULT_BAND_NAMES):
        return DEFAULT_BAND_NAMES.copy()
    if raster_count == len(DEFAULT_BAND_NAMES_WITHOUT_B8):
        return DEFAULT_BAND_NAMES_WITHOUT_B8.copy()
    return [f"band_{i+1}" for i in range(raster_count)]


def log_band_schema_warning(band_names: list[str], logger: logging.Logger) -> None:
    if band_names == DEFAULT_BAND_NAMES:
        logger.info("Esquema de bandas detectado: Sentinel-2 completo (incluye B8)")
        return
    if band_names == DEFAULT_BAND_NAMES_WITHOUT_B8:
        logger.warning(
            "Esquema de bandas detectado sin B8. El pipeline continua con 9 bandas: %s",
            band_names,
        )
        return
    logger.warning(
        "Cantidad/esquema de bandas no estandar (%d). Se usaran nombres genericos: %s",
        len(band_names),
        band_names,
    )


def extract_pixels_from_polygons(
    raster_path: Path,
    gdf: gpd.GeoDataFrame,
    class_column: str,
    logger: logging.Logger,
    all_touched: bool = False,
    max_polygons: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    pixel_rows: list[pd.DataFrame] = []
    pixel_counts: dict[str, int] = {}

    if max_polygons is not None:
        gdf = cast(gpd.GeoDataFrame, gdf.iloc[:max_polygons].copy())
        logger.warning("Modo limitado: usando solo %d poligonos", len(gdf))

    try:
        with rasterio.open(raster_path) as src:
            band_names = resolve_band_names(src.count)
            log_band_schema_warning(band_names, logger)
            records = list(gdf[[class_column, "geometry"]].to_dict("records"))
            iterator: Any = records

            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(records),
                    desc="Extrayendo pixeles",
                    unit="poligono",
                )

            skipped_no_overlap = 0
            skipped_empty_pixels = 0

            for row in iterator:
                class_value = str(row.get(class_column, "")).strip()
                geom = row.get("geometry")
                if geom is None or geom.is_empty:
                    continue

                try:
                    masked, _ = mask(
                        src,
                        [geom],
                        crop=True,
                        all_touched=all_touched,
                        filled=False,
                    )
                except ValueError:
                    skipped_no_overlap += 1
                    continue
                except Exception as exc:
                    logger.warning("Fallo extraccion en poligono (%s): %s", class_value, exc)
                    continue

                flattened = np.ma.transpose(masked, (1, 2, 0)).reshape(-1, src.count)
                mask_arr = np.ma.getmaskarray(flattened)
                valid = ~np.any(mask_arr, axis=1)

                if not np.any(valid):
                    skipped_empty_pixels += 1
                    continue

                pixels = np.asarray(flattened[valid], dtype=np.float32)

                if src.nodata is not None:
                    nodata_mask = np.any(pixels == src.nodata, axis=1)
                    pixels = pixels[~nodata_mask]

                finite_mask = np.all(np.isfinite(pixels), axis=1)
                pixels = pixels[finite_mask]

                if pixels.size == 0:
                    skipped_empty_pixels += 1
                    continue

                df_chunk = pd.DataFrame(pixels, columns=band_names)
                df_chunk[class_column] = class_value
                pixel_rows.append(df_chunk)
                pixel_counts[class_value] = pixel_counts.get(class_value, 0) + len(df_chunk)

            if skipped_no_overlap > 0:
                logger.warning("Poligonos sin interseccion con raster: %d", skipped_no_overlap)
            if skipped_empty_pixels > 0:
                logger.warning("Poligonos sin pixeles utiles tras limpieza: %d", skipped_empty_pixels)

    except Exception as exc:
        raise PipelineError(f"Error durante extraccion de pixeles: {exc}") from exc

    if not pixel_rows:
        raise PipelineError("No se extrajeron pixeles validos. Revisar geometrias, CRS o nodata")

    dataset = pd.concat(pixel_rows, ignore_index=True)
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    if dataset.empty:
        raise PipelineError("El dataset final quedo vacio tras limpieza")

    return dataset, pixel_counts, band_names


def log_dataset_statistics(
    dataset: pd.DataFrame,
    class_column: str,
    raster_info: RasterInfo,
    logger: logging.Logger,
) -> pd.DataFrame:
    class_counts = dataset[class_column].value_counts().sort_values(ascending=False)
    balance = (class_counts / class_counts.sum() * 100.0).round(2)

    logger.info("Resumen raster: width=%d height=%d bandas=%d dtype=%s nodata=%s", raster_info.width, raster_info.height, raster_info.count, raster_info.dtype, raster_info.nodata)
    logger.info("Total pixeles en dataset: %d", len(dataset))

    logger.info("Pixeles por clase:")
    for cls, count in class_counts.items():
        logger.info("  - %s: %d pixeles (%.2f%%)", cls, count, balance.loc[cls])

    balance_df = pd.DataFrame(
        {
            "class_name": class_counts.index,
            "pixel_count": class_counts.values,
            "percent": [balance.loc[c] for c in class_counts.index],
        }
    )
    return balance_df


def build_models(random_state: int, include_rf: bool = True) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
        "DecisionTree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "scale": False,
            "params": {
                "clf__max_depth": [None, 5, 10, 20],
                "clf__min_samples_split": [2, 5, 10],
            },
        },
        "SVM": {
            "estimator": SVC(kernel="rbf", probability=False, random_state=random_state),
            "scale": True,
            "params": {
                "clf__C": [1.0, 10.0, 100.0],
                "clf__gamma": ["scale", "auto"],
            },
        },
        "KNN": {
            "estimator": KNeighborsClassifier(),
            "scale": True,
            "params": {
                "clf__n_neighbors": [3, 5, 9, 15],
                "clf__weights": ["uniform", "distance"],
            },
        },
        "GaussianNB": {
            "estimator": GaussianNB(),
            "scale": True,
            "params": {
                "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        },
        "MLPClassifier": {
            "estimator": MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=300,
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=15,
            ),
            "scale": True,
            "params": {
                "clf__hidden_layer_sizes": [(128, 64), (128, 128), (256, 128)],
                "clf__alpha": [0.0001, 0.001, 0.01],
                "clf__learning_rate_init": [0.001, 0.0005],
            },
        },
    }

    if include_rf:
        models["RandomForest"] = {
            "estimator": RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            "scale": False,
            "params": {
                "clf__n_estimators": [200, 300, 500],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
            },
        }

    return models


def build_pipeline(estimator: BaseEstimator, use_scaler: bool) -> Pipeline:
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def train_one_model(
    model_name: str,
    model_cfg: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool,
    cv_folds: int,
    random_state: int,
    logger: logging.Logger,
) -> BaseEstimator:
    pipeline = build_pipeline(model_cfg["estimator"], model_cfg["scale"])

    if not tune:
        pipeline.fit(X_train, y_train)
        return pipeline

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=model_cfg["params"],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    gs.fit(X_train, y_train)
    logger.info("%s | Mejor score CV (f1_macro): %.4f", model_name, gs.best_score_)
    logger.info("%s | Mejores hiperparametros: %s", model_name, gs.best_params_)
    return gs.best_estimator_


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any], np.ndarray]:
    y_pred = cast(Any, model).predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = cast(dict[str, Any], classification_report(y_test, y_pred, output_dict=True, zero_division=0))
    return metrics, cm, report, y_pred


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: Path,
) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    out_path = output_dir / f"confusion_matrix_{model_name}.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_comparative_plots(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    sns.set_theme(style="whitegrid")
    saved_paths: list[Path] = []

    metrics_to_plot = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "kappa"]

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
        plot_df = metrics_df.sort_values(metric, ascending=False)
        sns.barplot(data=plot_df, x="model", y=metric, palette="viridis", ax=ax)
        ax.set_title(f"Comparative {metric}", fontsize=12)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)

        for i, v in enumerate(plot_df[metric].values):
            ax.text(i, min(v + 0.02, 0.995), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        out_path = output_dir / f"metric_{metric}.png"
        fig.savefig(str(out_path), bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def export_feature_importance(
    model: BaseEstimator,
    model_name: str,
    feature_names: list[str],
    output_dir: Path,
) -> Path | None:
    try:
        clf = model.named_steps.get("clf")  # type: ignore[attr-defined]
    except Exception:
        return None

    if clf is None or not hasattr(clf, "feature_importances_"):
        return None

    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df.sort_values("importance", ascending=False, inplace=True)

    csv_path = output_dir / f"feature_importance_{model_name}.csv"
    imp_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    sns.barplot(data=imp_df, x="importance", y="feature", palette="mako", ax=ax)
    ax.set_title(f"Feature Importance - {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(str(output_dir / f"feature_importance_{model_name}.png"), bbox_inches="tight")
    plt.close(fig)

    return csv_path


def make_markdown_report(
    output_path: Path,
    raster_info: RasterInfo,
    balance_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    best_model_name: str,
    class_names: list[str],
    input_raster: Path,
    input_training: Path,
    training_tsv: Path,
    tune: bool,
) -> None:
    lines = []
    lines.append("# Sentinel-2 Supervised Classification Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Raster: {input_raster}")
    lines.append(f"- Training vector: {input_training}")
    lines.append(f"- Training dataset TSV: {training_tsv}")
    lines.append("")
    lines.append("## Raster Summary")
    lines.append(f"- Width: {raster_info.width}")
    lines.append(f"- Height: {raster_info.height}")
    lines.append(f"- Bands: {raster_info.count}")
    lines.append(f"- CRS: {raster_info.crs}")
    lines.append(f"- Dtype: {raster_info.dtype}")
    lines.append(f"- Nodata: {raster_info.nodata}")
    lines.append("")
    lines.append("## Classes")
    lines.append(f"- Classes found: {', '.join(class_names)}")
    lines.append("")
    lines.append("## Extraction Statistics")
    lines.append(balance_df.to_markdown(index=False) if hasattr(balance_df, "to_markdown") else balance_df.to_string(index=False))
    lines.append("")
    lines.append("## Model Metrics")
    lines.append(metrics_df.to_markdown(index=False) if hasattr(metrics_df, "to_markdown") else metrics_df.to_string(index=False))
    lines.append("")
    lines.append("## Ranking")
    lines.append(ranking_df.to_markdown(index=False) if hasattr(ranking_df, "to_markdown") else ranking_df.to_string(index=False))
    lines.append("")
    lines.append("## Best Model")
    lines.append(f"- Selected model: {best_model_name}")
    lines.append(f"- Selection criteria: max F1 macro, then max Kappa")
    lines.append(f"- Grid search enabled: {tune}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.output_dir)

    try:
        logger.info("Iniciando pipeline de clasificacion Sentinel-2")
        validate_inputs(args.raster, args.training, args.output_dir)

        raster_info = open_raster_info(args.raster)
        logger.info(
            "Raster detectado: %s | width=%d height=%d bandas=%d",
            args.raster,
            raster_info.width,
            raster_info.height,
            raster_info.count,
        )

        gdf = load_and_prepare_training(args.training, args.class_column, raster_info.crs, logger)
        logger.info("Poligonos validos para extraccion: %d", len(gdf))

        dataset, pixel_counts, feature_names = extract_pixels_from_polygons(
            raster_path=args.raster,
            gdf=gdf,
            class_column=args.class_column,
            logger=logger,
            all_touched=args.all_touched,
            max_polygons=args.max_polygons,
            show_progress=not args.disable_progress,
        )

        balance_df = log_dataset_statistics(dataset, args.class_column, raster_info, logger)

        # Guardamos el dataset completo y, opcionalmente, uno muestreado por clase
        full_tsv_path = args.output_dir / (args.training_tsv.replace('.tsv','') + "_full.tsv")
        dataset.to_csv(full_tsv_path, sep="\t", index=False)
        logger.info("Dataset completo exportado: %s", full_tsv_path)

        sampled_dataset = dataset
        if args.max_samples_per_class is not None and args.max_samples_per_class > 0:
            logger.info("Subsampling estratificado: max %d muestras por clase", args.max_samples_per_class)
            sampled = (
                dataset.groupby(args.class_column, group_keys=False)
                .apply(
                    lambda x: x.sample(n=min(len(x), args.max_samples_per_class), random_state=args.random_state)
                )
                .reset_index(drop=True)
            )
            sampled_dataset = sampled
            logger.info("Dataset muestreado contiene %d filas", len(sampled_dataset))

        tsv_path = args.output_dir / args.training_tsv
        sampled_dataset.to_csv(tsv_path, sep="\t", index=False)
        logger.info("Dataset (usado para entrenamiento) exportado: %s", tsv_path)

        X = dataset[feature_names].to_numpy(dtype=np.float32)
        y_raw = dataset[args.class_column].astype(str).to_numpy()

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = list(label_encoder.classes_)

        logger.info("Clases codificadas: %s", class_names)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        logger.info(
            "Split estratificado completado | train=%d test=%d",
            len(y_train),
            len(y_test),
        )

        models_cfg = build_models(args.random_state, include_rf=not args.disable_rf)
        logger.info("Modelos a entrenar: %s", list(models_cfg.keys()))

        model_results: list[dict[str, Any]] = []
        trained_models: dict[str, BaseEstimator] = {}
        reports_dict: dict[str, dict[str, Any]] = {}

        model_iterator = models_cfg.items()
        if not args.disable_progress:
            model_iterator = tqdm(model_iterator, total=len(models_cfg), desc="Entrenando modelos", unit="modelo")

        confusion_dir = args.output_dir / "confusion_matrices"
        confusion_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_cfg in model_iterator:
            logger.info("Entrenando modelo: %s", model_name)
            fitted = train_one_model(
                model_name=model_name,
                model_cfg=model_cfg,
                X_train=X_train,
                y_train=y_train,
                tune=args.tune,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                logger=logger,
            )
            trained_models[model_name] = fitted

            metrics, cm, report, _ = evaluate_model(fitted, X_test, y_test)
            reports_dict[model_name] = report

            cm_path = save_confusion_matrix_plot(cm, class_names, model_name, confusion_dir)
            logger.info("Matriz de confusion guardada: %s", cm_path)

            row = {"model": model_name, **metrics}
            model_results.append(row)

        metrics_df = pd.DataFrame(model_results)
        if metrics_df.empty:
            raise PipelineError("No se obtuvieron metricas de modelos")

        ranking_df = metrics_df.sort_values(["f1_macro", "kappa"], ascending=[False, False]).reset_index(drop=True)
        best_model_name = str(ranking_df.loc[0, "model"])
        best_model = trained_models[best_model_name]

        logger.info("Mejor modelo seleccionado: %s", best_model_name)
        logger.info(
            "Score mejor modelo | f1_macro=%.4f kappa=%.4f",
            ranking_df.loc[0, "f1_macro"],
            ranking_df.loc[0, "kappa"],
        )

        metrics_csv = args.output_dir / "metrics_models.csv"
        ranking_csv = args.output_dir / "ranking_models.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        ranking_df.to_csv(ranking_csv, index=False)

        balance_csv = args.output_dir / "dataset_balance.csv"
        balance_df.to_csv(balance_csv, index=False)

        reports_json = args.output_dir / "classification_reports.joblib"
        joblib.dump(reports_dict, reports_json)

        comparative_dir = args.output_dir / "comparative_plots"
        comparative_dir.mkdir(parents=True, exist_ok=True)
        comparative_plots = save_comparative_plots(metrics_df, comparative_dir)
        logger.info("Graficos comparativos generados: %d", len(comparative_plots))

        fi_path = export_feature_importance(best_model, best_model_name, feature_names, args.output_dir)
        if fi_path:
            logger.info("Feature importance exportado: %s", fi_path)

        model_artifact = {
            "model": best_model,
            "model_name": best_model_name,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "class_names": class_names,
            "raster_bands": resolve_band_names(raster_info.count),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_raster": str(args.raster),
            "training_vector": str(args.training),
            "pixel_counts": pixel_counts,
        }
        model_path = args.output_dir / f"best_model_{best_model_name}.joblib"
        joblib.dump(model_artifact, model_path)
        logger.info("Modelo entrenado exportado: %s", model_path)

        report_md = args.output_dir / "final_report.md"
        make_markdown_report(
            output_path=report_md,
            raster_info=raster_info,
            balance_df=balance_df,
            metrics_df=metrics_df,
            ranking_df=ranking_df,
            best_model_name=best_model_name,
            class_names=class_names,
            input_raster=args.raster,
            input_training=args.training,
            training_tsv=tsv_path,
            tune=args.tune,
        )
        logger.info("Reporte final generado: %s", report_md)

        logger.info("Pipeline finalizado con exito")
        logger.info("Archivos clave:")
        logger.info("  - Dataset TSV: %s", tsv_path)
        logger.info("  - Metricas: %s", metrics_csv)
        logger.info("  - Ranking: %s", ranking_csv)
        logger.info("  - Modelo: %s", model_path)
        logger.info("  - Reporte: %s", report_md)
        return 0

    except PipelineError as exc:
        logger.error("Error de pipeline: %s", exc)
        return 2
    except Exception as exc:
        logger.error("Error no controlado: %s", exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
