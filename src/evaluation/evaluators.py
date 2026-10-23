"""
Módulo evaluation.evaluators
Implementación de la evaluación rigurosa, cálculo de métricas, 
tests estadísticos (Diebold-Mariano) y registro de experimentos (MLflow).
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import scipy.stats
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
import mlflow

# Asumiendo que BaseEvaluator está en src.core.interfaces
# from src.core.interfaces import BaseEvaluator
# from src.data.preprocessors import PreprocessorImpl


class EvaluatorImpl: # Hereda de BaseEvaluator
    """
    Implementación concreta para la evaluación del modelo.
    Garantiza que todas las métricas se calculen en la escala física real 
    del problema y proporciona significancia estadística para comparar modelos.
    """

    def denormalize(
        self, y_true: np.ndarray, y_pred: np.ndarray, preprocessor: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Revierte el escalado de los datos a su magnitud física original utilizando 
        el preprocesador ajustado en el conjunto de entrenamiento.

        Args:
            y_true (np.ndarray): Valores reales normalizados.
            y_pred (np.ndarray): Predicciones normalizadas.
            preprocessor (Any): Instancia ajustada de PreprocessorImpl.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Valores reales y predicciones en escala original.
        """
        y_true_denorm = preprocessor.inverse_transform(y_true)
        y_pred_denorm = preprocessor.inverse_transform(y_pred)
        
        return y_true_denorm, y_pred_denorm

    def calculate_metrics(
        self, 
        y_true_norm: Union[np.ndarray, Any], 
        y_pred_norm: Union[np.ndarray, Any],
        y_true_denorm: Optional[Union[np.ndarray, Any]] = None, 
        y_pred_denorm: Optional[Union[np.ndarray, Any]] = None
    ) -> Dict[str, float]:
        """
        Calcula las métricas de rendimiento. Si no se proveen los arreglos desnormalizados,
        calcula únicamente las métricas en el espacio normalizado (retrocompatible con Optuna).
        """
        # Validaciones de dimensionalidad en espacio normalizado
        if y_true_norm.ndim > 2:
            y_true_norm = y_true_norm.reshape(y_true_norm.shape[0], -1)
            y_pred_norm = y_pred_norm.reshape(y_pred_norm.shape[0], -1)

        # 1. Espacio Normalizado (Siempre se calcula)
        norm_mse = float(mean_squared_error(y_true_norm, y_pred_norm))
        norm_mae = float(mean_absolute_error(y_true_norm, y_pred_norm))
        norm_rmse = float(np.sqrt(norm_mse))

        metrics = {
            "norm_MSE": norm_mse,
            "norm_MAE": norm_mae,
            "norm_RMSE": norm_rmse,
            # Llaves de respaldo para retrocompatibilidad directa con optimize.py
            "MSE": norm_mse,
            "MAE": norm_mae,
            "RMSE": norm_rmse
        }

        # 2. Espacio Desnormalizado (Se ejecuta en main.py cuando se pasan los 4 arreglos)
        if y_true_denorm is not None and y_pred_denorm is not None:
            if y_true_denorm.ndim > 2:
                y_true_denorm = y_true_denorm.reshape(y_true_denorm.shape[0], -1)
                y_pred_denorm = y_pred_denorm.reshape(y_pred_denorm.shape[0], -1)

            denorm_mse = mean_squared_error(y_true_denorm, y_pred_denorm)
            denorm_rmse = float(np.sqrt(denorm_mse))
            denorm_mae = float(mean_absolute_error(y_true_denorm, y_pred_denorm))
            denorm_mape = float(mean_absolute_percentage_error(y_true_denorm, y_pred_denorm))

            # En main.py limpiamos los alias genéricos para no duplicar columnas en MLflow
            metrics = {
                "norm_MSE": norm_mse,
                "norm_MAE": norm_mae,
                "denorm_RMSE": denorm_rmse,
                "denorm_MAE": denorm_mae,
                "denorm_MAPE": denorm_mape
            }

        return metrics      
    def diebold_mariano_test(
        self, 
        y_true: np.ndarray, 
        y_pred_1: np.ndarray, 
        y_pred_2: np.ndarray, 
        horizon: int = 1
    ) -> Dict[str, float]:
        """
        Implementación matemática rigurosa del Test de Diebold-Mariano.
        Evalúa si la diferencia en la precisión predictiva entre dos modelos (p. ej., 
        Baseline VARMAX vs iTransformer) es estadísticamente significativa.

        Args:
            y_true (np.ndarray): Valores reales observados.
            y_pred_1 (np.ndarray): Predicciones del Modelo 1.
            y_pred_2 (np.ndarray): Predicciones del Modelo 2.
            horizon (int): Horizonte de predicción para calcular la autocovarianza (lags).

        Returns:
            Dict[str, float]: El estadístico DM y el valor p (p-value).
        """
        # Aplanar para comparar la secuencia completa de errores
        e1 = (y_true - y_pred_1).flatten()
        e2 = (y_true - y_pred_2).flatten()

        # Diferencia de la función de pérdida (Error Cuadrático)
        d = (e1 ** 2) - (e2 ** 2)
        d_mean = np.mean(d)
        N = len(d)

        # Cálculo de la autocovarianza hasta el rezago: horizon - 1
        gamma = []
        for k in range(horizon):
            if k == 0:
                gamma.append(np.mean((d - d_mean) ** 2))
            else:
                gamma.append(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))

        # Varianza asintótica de la media muestral (Long-run variance)
        V_d = gamma[0] + 2 * sum(gamma[1:])

        # Evitar división por cero en series idénticas
        if V_d <= 0:
            return {"DM_Statistic": 0.0, "p_value": 1.0}

        # Estadístico y p-value usando distribución Normal estándar
        dm_stat = d_mean / np.sqrt(V_d / N)
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(dm_stat)))

        return {
            "DM_Statistic": float(dm_stat),
            "p_value": float(p_value)
        }

    def log_experiment(
        self, 
        run_name: str, 
        hiperparametros: Dict[str, Any], 
        metricas: Dict[str, float], 
        tiempos: Dict[str, float],
        tracking_uri: str = "http://127.0.0.1:5000"
    ) -> None:
        """
        Orquesta el registro del experimento en una sesión local de MLflow.
        Cumple con el contrato `log_to_mlflow` de BaseEvaluator integrando perfilado.

        Args:
            run_name (str): Identificador único para la ejecución en MLflow.
            hiperparametros (Dict[str, Any]): Diccionario de parámetros inyectados al modelo.
            metricas (Dict[str, float]): Resultados devueltos por calculate_metrics().
            tiempos (Dict[str, float]): Tiempos de latencia devueltos por el wrapper del modelo.
            tracking_uri (str): URI del servidor local de MLflow.
        """
        # Configurar conexión al servidor local
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Tesis_Multivariate_Forecasting")

        # Iniciar contexto seguro de MLflow
        with mlflow.start_run(run_name=run_name):
            # 1. Registro en bloque de hiperparámetros
            mlflow.log_params(hiperparametros)
            
            # 2. Registro de métricas de precisión
            mlflow.log_metrics(metricas)
            
            # 3. Registro de tiempos de perfilado computacional
            mlflow.log_metrics({
                "training_time_seconds": tiempos.get("training_time", 0.0),
                "inference_time_seconds": tiempos.get("inference_time", 0.0)
            })
            
            print(f"[{run_name}] Experimento registrado exitosamente en MLflow.")
