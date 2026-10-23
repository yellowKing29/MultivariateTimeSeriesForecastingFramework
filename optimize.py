"""
Script de Optimización de Hiperparámetros con Optuna.
Realiza búsqueda paramétrica independiente para cada modelo evaluando 
exclusivamente sobre el conjunto de validación para prevenir Data Leakage.
"""

import os
import gc
import yaml
import time
import numpy as np
import torch
import pandas as pd
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from typing import Dict, Any, Callable

# Importaciones del Framework Modular
from src.data.datasets import TimeSeriesDatasetImpl
from src.data.preprocessors import PreprocessorImpl
from src.data.adapters import AdapterImpl
from src.evaluation.evaluators import EvaluatorImpl
from src.models.dl_models import iTransformerWrapper, DLinearWrapper
from src.models.ml_models import XGBoostWrapper, RandomForestWrapper
from src.models.baseline import MultiSARIMAXWrapper

def get_objective(
    model_name: str,
    adapter: AdapterImpl,
    evaluator: EvaluatorImpl,
    train_scaled: pd.DataFrame,
    val_scaled: pd.DataFrame,
    target_columns: list,
    window_size: int,
    horizon: int,
    device_name: str
) -> Callable[[optuna.Trial], float]:
    """Generador de la Función Objetivo para Optuna."""
    
    def objective(trial: optuna.Trial) -> float:
        try:
            # 1. Definición del Espacio de Búsqueda Paramétrico
            hyperparams = {}
            if model_name == "itransformer":
                hyperparams = {
                    "lookback_len": window_size,
                    "pred_len": horizon,
                    "target_indices": [train_scaled.columns.get_loc(col) for col in target_columns],
                    "num_variates": len(target_columns),
                    "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
                    "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                    "e_layers": trial.suggest_int("e_layers", 1, 3),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "epochs": 30,
                    "patience": 5
                }
            elif model_name == "dlinear":
                hyperparams = {
                    "look_back": window_size,
                    "horizon": horizon,
                    "target_indices": [train_scaled.columns.get_loc(col) for col in target_columns],
                    "moving_avg_window": trial.suggest_int("moving_avg_window", 15, 35, step=2),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "epochs": 30,
                    "patience": 5
                }
            elif model_name == "xgboost":
                hyperparams = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "n_jobs": -1
                }
            elif model_name == "randomForest":
                hyperparams = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "n_jobs": -1
                }
            elif model_name == "sarimax":
                hyperparams = {
                    "order": (
                        trial.suggest_int("p", 0, 2),
                        trial.suggest_int("d", 0, 1),
                        trial.suggest_int("q", 0, 2)
                    ),
                    "seasonal_order": (0, 0, 0, 0),
                    "enforce_stationarity": False,
                    "enforce_invertibility": False
                }

            # 2. Extracción de Datos (Adecuación Dimensional)
            if model_name in ["itransformer", "dlinear"]:
                X_train, y_train = adapter.get_3d_tensors(*adapter.create_sliding_windows(train_scaled, target_columns, window_size, horizon))
                X_val, y_val = adapter.get_3d_tensors(*adapter.create_sliding_windows(val_scaled, target_columns, window_size, horizon))
                model_class = iTransformerWrapper if model_name == "itransformer" else DLinearWrapper
                
            elif model_name in ["xgboost", "randomForest"]:
                X_train, y_train = adapter.get_2d_arrays(*adapter.create_sliding_windows(train_scaled, target_columns, window_size, horizon))
                X_val, y_val = adapter.get_2d_arrays(*adapter.create_sliding_windows(val_scaled, target_columns, window_size, horizon))
                model_class = XGBoostWrapper if model_name == "xgboost" else RandomForestWrapper
                
            elif model_name == "sarimax":
                X_train_df, _ = adapter.get_statsmodels_format(train_scaled, target_columns)
                X_val_df, _ = adapter.get_statsmodels_format(val_scaled, target_columns)
                X_train, y_train = X_train_df.values, None
                X_val, y_val = X_val_df.values, X_val_df.values
                model_class = MultiSARIMAXWrapper

            # 3. Entrenamiento y Evaluación
            model = model_class(**hyperparams)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, optuna_trial=trial)
            
            predictions = model.predict(X_val)
            
            # Detecta explosión de gradientes en Deep Learning y descarta el trial
            if np.isnan(predictions).any():
                print(f"[Aviso] Trial fallido por inestabilidad numérica (Predicciones NaN).")
                return float('inf')
            # ----------------------------------------------
            
            # Evaluación estructural (Comparación directa en escala normalizada para simplificar la función de pérdida)
            if model_name != "sarimax":
                predictions_flat = predictions.reshape(-1, len(target_columns))
                y_val_flat = y_val.reshape(-1, len(target_columns)) if isinstance(y_val, np.ndarray) else y_val.cpu().numpy().reshape(-1, len(target_columns))
            else:
                predictions_flat = predictions
                y_val_flat = y_val
                
            metrics = evaluator.calculate_metrics(y_val_flat, predictions_flat)
            return metrics["RMSE"]

        except optuna.exceptions.TrialPruned:
            raise
        except (RuntimeError, ValueError) as e:
            # Captura excepciones de VRAM, convergencia o inestabilidad matemática (NaNs)
            if "out of memory" in str(e).lower() or "linalg" in str(e).lower() or "nan" in str(e).lower():
                print(f"[Aviso] Trial fallido por recursos o inestabilidad: {e}")
                return float('inf')
            raise
        finally:
            # Liberación estricta de memoria para GPU (esencial en Mallas Grandes)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return objective

def main():
    print("="*60)
    print(" INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS (OPTUNA)")
    print("="*60)

    # 1. Carga de Configuración Base y Módulos
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    global_params = config["global"]
    dataset_path = global_params["dataset_path"]
    window_size = global_params["look_back"]
    horizon = global_params["horizon"]
    
    dataset = TimeSeriesDatasetImpl()
    data = dataset.load_data(dataset_path)
    
    target_columns_config = global_params.get("target_columns", None)
    if isinstance(target_columns_config, str):
        target_columns_config = [target_columns_config]
    target_columns = target_columns_config if target_columns_config else data.columns.tolist()

    # Extraemos solo Train y Val (Evitamos Test para prevenir Leakage)
    train, val, _ = dataset.get_partitions(data, train_ratio=0.7, val_ratio=0.15)
    
    preprocessor = PreprocessorImpl()
    train_imputed = preprocessor.impute_missing(train)
    val_imputed = preprocessor.impute_missing(val)
    preprocessor.fit(train_imputed)
    train_scaled = preprocessor.transform(train_imputed)
    val_scaled = preprocessor.transform(val_imputed)

    adapter = AdapterImpl()
    evaluator = EvaluatorImpl()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_optimize = config["orchestration"]["models_to_run"]
    best_configurations = {}

    # 2. Bucle de Optimización Independiente por Modelo
    for model_name in models_to_optimize:
        print(f"\n[{model_name.upper()}] Iniciando Estudio Optuna...")
        
        study = optuna.create_study(
            direction="minimize",
            pruner=SuccessiveHalvingPruner()
        )
        
        objective = get_objective(
            model_name, adapter, evaluator, train_scaled, val_scaled, 
            target_columns, window_size, horizon, device
        )
        
        # Ejecutar optimización (15 trials para DL por tiempo, 30 para ML)
        n_trials = 15 if model_name in ["itransformer", "dlinear"] else 30
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
        
        print(f"[{model_name.upper()}] Mejor RMSE de Validación: {study.best_value:.4f}")
        print(f"[{model_name.upper()}] Mejores Parámetros: {study.best_params}")
        
        # Almacenar la mejor configuración
        best_configurations[model_name] = study.best_params
        
        # Estabilización de VRAM entre modelos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 3. Exportación Consolidada
    export_data = {"models": best_configurations}
    with open("best_config.yaml", "w") as f:
        yaml.dump(export_data, f, default_flow_style=False)
        
    print("\n" + "="*60)
    print(" OPTIMIZACIÓN FINALIZADA. Resultados guardados en 'best_config.yaml'")
    print("="*60)

if __name__ == "__main__":
    main()
