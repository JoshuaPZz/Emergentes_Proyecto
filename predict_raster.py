#!/usr/bin/env python3
"""
Aplicacion de modelo supervisado sobre raster completo.

Carga:
- Modelo exportado por sentinel2_supervised_pipeline.py
- Raster multibanda original

Genera:
- GeoTIFF clasificado de una sola banda

Compatible con Python 3.12+
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import rasterio
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clasificacion de raster completo usando modelo entrenado."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Archivo best_model_*.joblib",
    )

    parser.add_argument(
        "--raster",
        type=Path,
        required=True,
        help="Raster multibanda original",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/classified_map.tif"),
        help="GeoTIFF de salida",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Tamaño de bloque para procesar el raster",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("Cargando modelo...")

    artifact = joblib.load(args.model)

    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    feature_names = artifact["feature_names"]

    print(f"Modelo: {artifact['model_name']}")
    print(f"Clases: {list(label_encoder.classes_)}")

    with rasterio.open(args.raster) as src:

        if src.count != len(feature_names):
            raise ValueError(
                f"El raster tiene {src.count} bandas pero el modelo espera "
                f"{len(feature_names)}"
            )

        profile = src.profile.copy()

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw",
            nodata=255,
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(args.output, "w", **profile) as dst:

            windows = list(src.block_windows(1))

            for _, window in tqdm(
                windows,
                desc="Clasificando raster",
                unit="bloque",
            ):

                data = src.read(window=window)

                bands, rows, cols = data.shape

                pixels = (
                    data.reshape(bands, rows * cols)
                    .transpose()
                    .astype(np.float32)
                )

                valid_mask = np.all(np.isfinite(pixels), axis=1)

                predictions = np.full(
                    rows * cols,
                    255,
                    dtype=np.uint8,
                )

                if np.any(valid_mask):

                    pred = model.predict(pixels[valid_mask])

                    predictions[valid_mask] = pred.astype(np.uint8)

                classified = predictions.reshape(rows, cols)

                dst.write(classified, 1, window=window)

    print()
    print("Clasificacion finalizada")
    print(f"Salida: {args.output}")
    print()
    print("Codificacion de clases:")

    for idx, cls in enumerate(label_encoder.classes_):
        print(f"{idx} -> {cls}")


if __name__ == "__main__":
    main()