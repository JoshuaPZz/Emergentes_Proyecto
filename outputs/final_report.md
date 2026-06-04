# Sentinel-2 Supervised Classification Report

Generated: 2026-06-03T23:35:35

## Inputs
- Raster: raster.tif
- Training vector: training.gpkg
- Training dataset TSV: outputs\training.tsv

## Raster Summary
- Width: 5490
- Height: 5490
- Bands: 9
- CRS: EPSG:32640
- Dtype: uint16
- Nodata: None

## Classes
- Classes found: Agua, Arena, Playa_sal

## Extraction Statistics
| class_name   |   pixel_count |   percent |
|:-------------|--------------:|----------:|
| Playa_sal    |         27941 |     34.52 |
| Agua         |         27112 |     33.5  |
| Arena        |         25889 |     31.98 |

## Model Metrics
| model         |   accuracy |   precision_macro |   recall_macro |   f1_macro |    kappa |
|:--------------|-----------:|------------------:|---------------:|-----------:|---------:|
| DecisionTree  |   0.999918 |          0.999917 |       0.999917 |   0.999917 | 0.999876 |
| SVM           |   1        |          1        |       1        |   1        | 1        |
| KNN           |   1        |          1        |       1        |   1        | 1        |
| GaussianNB    |   0.986328 |          0.986706 |       0.986072 |   0.986324 | 0.979477 |
| MLPClassifier |   1        |          1        |       1        |   1        | 1        |
| RandomForest  |   1        |          1        |       1        |   1        | 1        |

## Ranking
| model         |   accuracy |   precision_macro |   recall_macro |   f1_macro |    kappa |
|:--------------|-----------:|------------------:|---------------:|-----------:|---------:|
| SVM           |   1        |          1        |       1        |   1        | 1        |
| KNN           |   1        |          1        |       1        |   1        | 1        |
| MLPClassifier |   1        |          1        |       1        |   1        | 1        |
| RandomForest  |   1        |          1        |       1        |   1        | 1        |
| DecisionTree  |   0.999918 |          0.999917 |       0.999917 |   0.999917 | 0.999876 |
| GaussianNB    |   0.986328 |          0.986706 |       0.986072 |   0.986324 | 0.979477 |

## Best Model
- Selected model: SVM
- Selection criteria: max F1 macro, then max Kappa
- Grid search enabled: True