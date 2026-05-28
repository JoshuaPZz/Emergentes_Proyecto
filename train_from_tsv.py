#!/usr/bin/env python3
"""
Entrenador rápido desde TSV para completar la fase de entrenamiento.
- Carga outputs/training.tsv
- Muestreo estratificado por clase (parámetro)
- Train/test split estratificado
- Entrena modelos (DecisionTree, SVM, KNN, GaussianNB, MLP, RandomForest)
- Guarda métricas, matrices y modelo final en outputs/

Diseñado para ejecutarse dentro de .venv del proyecto.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('train_from_tsv')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(ch)
    fh = logging.FileHandler(output_dir / 'train_from_tsv.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(fh)
    return logger


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tsv', type=Path, default=Path('outputs/training.tsv'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs'))
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--sample-per-class', type=int, default=100000, help='Max samples por clase (0 = usar todo)')
    p.add_argument('--test-size', type=float, default=0.3)
    p.add_argument('--tune', action='store_true')
    return p.parse_args()


def build_models(random_state:int=42):
    models = {
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'SVM': SVC(kernel='rbf', probability=False, random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=random_state, early_stopping=True),
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1),
    }
    return models


def evaluate_and_save(model, X_test, y_test, le, model_name, out_dir, logger):
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test,y_pred),
        'precision_macro': precision_score(y_test,y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test,y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test,y_pred, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_test,y_pred)
    }
    logger.info('%s metrics: %s', model_name, metrics)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion {model_name}')
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    fig.savefig(out_dir / f'confusion_{model_name}.png', bbox_inches='tight')
    plt.close(fig)
    return metrics


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir)
    logger.info('Cargando TSV: %s', args.tsv)
    if not args.tsv.exists():
        logger.error('TSV no encontrado: %s', args.tsv)
        return 2
    df = pd.read_csv(args.tsv, sep='\t')
    if df.empty:
        logger.error('TSV vacio')
        return 2
    # detectar columna de clase
    class_cols = [c for c in df.columns if c.lower() in ('class','class_name','label')]
    if not class_cols:
        logger.error('No se encontro columna de clase en TSV')
        return 2
    class_col = class_cols[0]
    feature_cols = [c for c in df.columns if c != class_col]
    logger.info('Clase: %s  features: %d', class_col, len(feature_cols))

    # muestreo estratificado por clase
    if args.sample_per_class > 0:
        parts = []
        for cls, g in df.groupby(class_col):
            n = min(len(g), args.sample_per_class)
            parts.append(g.sample(n=n, random_state=args.random_state))
        df_sample = pd.concat(parts).sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
        logger.info('Muestreo realizado: total %d (max %d por clase)', len(df_sample), args.sample_per_class)
    else:
        df_sample = df
        logger.info('Usando dataset completo: %d registros', len(df_sample))

    X = df_sample[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df_sample[class_col].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    logger.info('Clases detectadas: %s', class_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)
    logger.info('Split: train=%d test=%d', len(y_train), len(y_test))

    models = build_models(args.random_state)
    results = []
    trained = {}
    for name, clf in models.items():
        logger.info('Entrenando %s', name)
        # pipeline con escalado para los que lo necesitan
        if name in ('SVM','KNN','GaussianNB','MLPClassifier'):
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        else:
            pipe = Pipeline([('clf', clf)])
        pipe.fit(X_train, y_train)
        metrics = evaluate_and_save(pipe, X_test, y_test, le, name, args.output_dir, logger)
        results.append({'model': name, **metrics})
        trained[name] = pipe
        # guardado intermedio
        joblib.dump(pipe, args.output_dir / f'model_{name}.joblib')

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(args.output_dir / 'metrics_from_tsv.csv', index=False)
    # seleccionar mejor por f1_macro y kappa
    ranking = metrics_df.sort_values(['f1_macro','kappa'], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(args.output_dir / 'ranking_from_tsv.csv', index=False)
    best_name = ranking.loc[0,'model']
    best_model = trained[best_name]
    logger.info('Mejor modelo: %s', best_name)
    joblib.dump({'model': best_model, 'label_encoder': le, 'feature_cols': feature_cols, 'class_names': class_names, 'created_at': datetime.now().isoformat()}, args.output_dir / f'best_model_{best_name}.joblib')
    logger.info('Entrenamiento completado. Artefactos en %s', args.output_dir)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
