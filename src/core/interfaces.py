"""
Módulo core.interfaces
Define las Clases Base Abstractas (ABCs) para el framework de pronóstico multivariado.
Garantiza el Principio de Responsabilidad Única (SRP) y el flujo estricto de tipos de datos.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List

import pandas as pd
import numpy as np
import torch


class BaseTimeSeriesDataset(ABC):
    """
    Contrato para la ingesta y partición de datos de series de tiempo multivariadas.
    Responsabilidad: Leer la fuente de datos, facilitar el EDA y dividir en conjuntos.
    """

    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos crudos desde la fuente.

        Args:
            file_path (str): Ruta al archivo de datos.

        Returns:
            pd.DataFrame: DataFrame con índice temporal y múltiples variables.
        """
        raise NotImplementedError

    @abstractmethod
    def run_basic_eda(self, data: pd.DataFrame) -> None:
        """
        Ejecuta rutinas de Análisis Exploratorio de Datos básico (estadísticas, nulos).

        Args:
            data (pd.DataFrame): Datos a analizar.
        """
        raise NotImplementedError

    @abstractmethod
    def get_partitions(
        self, data: pd.DataFrame, train_ratio: float, val_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide la serie de tiempo respetando el orden cronológico.

        Args:
            data (pd.DataFrame): DataFrame completo.
            train_ratio (float): Proporción para entrenamiento (ej. 0.7).
            val_ratio (float): Proporción para validación (ej. 0.15).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Conjuntos Train, Val y Test.
        """
        raise NotImplementedError


class BasePreprocessor(ABC):
    """
    Contrato para la limpieza, imputación y escalado de datos.
    Responsabilidad: Transformar pd.DataFrame crudo a pd.DataFrame procesado y manejar desnormalización.
    """

    @abstractmethod
    def impute_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores nulos en el conjunto de datos.

        Args:
            data (pd.DataFrame): Datos con posibles valores nulos.

        Returns:
            pd.DataFrame: Datos sin valores nulos.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BasePreprocessor':
        """
        Calcula y almacena los parámetros de escalado (ej. media y varianza) basado en Train.

        Args:
            data (pd.DataFrame): Datos de entrenamiento.

        Returns:
            BasePreprocessor: La instancia actual ajustada (permite encadenamiento).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el escalado a los datos utilizando los parámetros ajustados.

        Args:
            data (pd.DataFrame): Datos a escalar.

        Returns:
            pd.DataFrame: Datos escalados.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Desnormaliza las predicciones para devolverlas a su escala física original.

        Args:
            data (np.ndarray): Predicciones normalizadas.

        Returns:
            np.ndarray: Predicciones en la escala original.
        """
        raise NotImplementedError


class BaseAdapter(ABC):
    """
    Contrato para la generación de ventanas temporales e inyección a estructuras de datos específicas.
    Responsabilidad: Convertir DataFrames tabulares secuenciales en tensores/matrices listos para ML/DL.
    """

    @abstractmethod
    def create_sliding_windows(
        self, data: pd.DataFrame, target_columns: List[str], window_size: int, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera ventanas deslizantes (X) y los objetivos de pronóstico (y).

        Args:
            data (pd.DataFrame): Datos multivariados preprocesados.
            target_columns (List[str]): Nombres de las variables a predecir.
            window_size (int): Pasos de tiempo históricos a utilizar (look-back).
            horizon (int): Pasos de tiempo a predecir hacia el futuro.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tupla temporal (X_array_3d, y_array).
        """
        raise NotImplementedError

    @abstractmethod
    def to_2d_numpy(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplana la estructura temporal 3D a 2D, ideal para Scikit-Learn o XGBoost.

        Args:
            X (np.ndarray): Matriz de características (samples, time_steps, features).
            y (np.ndarray): Matriz de objetivos.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Matrices 2D (samples, time_steps * features).
        """
        raise NotImplementedError

    @abstractmethod
    def to_3d_tensor(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convierte arreglos a tensores de PyTorch preservando la dimensión temporal.

        Args:
            X (np.ndarray): Matriz de características.
            y (np.ndarray): Matriz de objetivos.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensores de PyTorch.
        """
        raise NotImplementedError
    
    @abstractmethod
    def to_temporal_blocks(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, Any]:
        """
        Prepara los datos para modelos estadísticos como VARMAX (Statsmodels).
        
        Args:
            X (np.ndarray): Matriz de características exógenas (si aplica).
            y (np.ndarray): Matriz de variables endógenas.
            
        Returns:
            Tuple[Any, Any]: Estructuras requeridas por la API de Statsmodels.
        """
        raise NotImplementedError


class BaseModelWrapper(ABC):
    """
    Contrato envoltorio para estandarizar la interacción con diversos algoritmos (XGB, LSTM, VARMAX).
    Responsabilidad: Inicialización por hiperparámetros (**kwargs), entrenamiento y predicción.
    Nota Arquitectónica: Las implementaciones de esta clase deben incluir decoradores
    de perfilado computacional en sus métodos fit() y predict().
    """

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """
        Constructor que acepta hiperparámetros dinámicos desde archivos de configuración.

        Args:
            **kwargs: Diccionario desempaquetado de hiperparámetros.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any], 
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'BaseModelWrapper':
        """
        Entrena el modelo subyacente. Debe registrar el tiempo de ejecución (profiling).

        Args:
            X (Union[np.ndarray, torch.Tensor, Any]): Características adaptadas.
            y (Union[np.ndarray, torch.Tensor, Any]): Objetivos adaptados.

        Returns:
            BaseModelWrapper: La instancia del modelo entrenado.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Union[np.ndarray, torch.Tensor, Any]) -> np.ndarray:
        """
        Genera predicciones sobre nuevos datos. Las predicciones retornan normalizadas y como numpy array.
        Debe registrar el tiempo de inferencia (profiling).

        Args:
            X (Union[np.ndarray, torch.Tensor, Any]): Características para predecir.

        Returns:
            np.ndarray: Predicciones normalizadas (samples, horizon * n_targets).
        """
        raise NotImplementedError


class BaseEvaluator(ABC):
    """
    Contrato para la evaluación del modelo estadísticamente rigurosa y registro de artefactos.
    Responsabilidad: Cálculo de métricas, tests de significancia estadística y trazabilidad.
    """

    @abstractmethod
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de error (ej. RMSE, MAE, R2). Ambas entradas deben estar desnormalizadas.

        Args:
            y_true (np.ndarray): Valores reales en escala original.
            y_pred (np.ndarray): Predicciones en escala original.

        Returns:
            Dict[str, float]: Diccionario con las métricas calculadas.
        """
        raise NotImplementedError

    @abstractmethod
    def diebold_mariano_test(
        self, y_true: np.ndarray, y_pred_1: np.ndarray, y_pred_2: np.ndarray
    ) -> Dict[str, float]:
        """
        Ejecuta el Test de Diebold-Mariano para evaluar si la diferencia en la precisión
        predictiva de dos modelos es estadísticamente significativa.

        Args:
            y_true (np.ndarray): Valores reales.
            y_pred_1 (np.ndarray): Predicciones del Modelo 1 (ej. Baseline).
            y_pred_2 (np.ndarray): Predicciones del Modelo 2 (ej. Modelo Propuesto).

        Returns:
            Dict[str, float]: Diccionario con el estadístico DM y el p-value.
        """
        raise NotImplementedError

    @abstractmethod
    def log_to_mlflow(
        self, 
        run_name: str, 
        parameters: Dict[str, Any], 
        metrics: Dict[str, float], 
        artifacts_path: Optional[str] = None
    ) -> None:
        """
        Registra el experimento de forma estandarizada en un servidor local de MLflow.

        Args:
            run_name (str): Nombre identificador de la ejecución.
            parameters (Dict[str, Any]): Diccionario de hiperparámetros.
            metrics (Dict[str, float]): Diccionario de métricas evaluadas.
            artifacts_path (Optional[str]): Ruta temporal a gráficos/modelos exportados para registro.
        """
        raise NotImplementedError
