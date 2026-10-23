"""
Módulo data.adapters
Implementación concreta para la adaptación dimensional de series de tiempo multivariadas.
Incluye soporte para A/B Testing de sesgo inductivo espacial:
Channel Dependence (CD) vs. Channel Independence (CI).
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch

# Asumiendo que BaseAdapter está en src.core.interfaces
# from src.core.interfaces import BaseAdapter


class AdapterImpl:  # Hereda de BaseAdapter
    """
    Implementación concreta para la generación de ventanas temporales e inyección 
    a estructuras de datos específicas (2D CD/CI para ML, 3D para DL, continuo para Estadísticos).
    """

    def create_sliding_windows(
        self, data: pd.DataFrame, target_columns: List[str], window_size: int, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera ventanas deslizantes superpuestas para características (X) y objetivos (y).
        
        Args:
            data (pd.DataFrame): Datos multivariados preprocesados.
            target_columns (List[str]): Nombres de las variables a predecir.
            window_size (int): Longitud del contexto histórico (look-back).
            horizon (int): Pasos de tiempo a predecir hacia el futuro.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - X: Arreglo 3D de dimensiones (Muestras, window_size, Caracteristicas).
                - y: Arreglo 3D de dimensiones (Muestras, horizon, Objetivos).
        """
        target_indices = [data.columns.get_loc(col) for col in target_columns]
        data_values = data.values
        
        X, y = [], []
        n_samples = len(data) - window_size - horizon + 1
        
        for i in range(n_samples):
            X_window = data_values[i : i + window_size, :]
            y_window = data_values[i + window_size : i + window_size + horizon, target_indices]
            
            X.append(X_window)
            y.append(y_window)
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def get_2d_arrays(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplana la estructura temporal 3D a 2D (Equivalente a estrategia CD).
        Mantiene retrocompatibilidad con modelos estándar como RandomForest.
        
        Args:
            X (np.ndarray): Matriz 3D (Muestras, Longitud_Contexto, Caracteristicas).
            y (np.ndarray): Matriz 3D (Muestras, Horizonte, Objetivos).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - X_2d: (Muestras, Longitud_Contexto * Caracteristicas)
                - y_2d: (Muestras, Horizonte * Objetivos)
        """
        muestras_x = X.shape[0]
        muestras_y = y.shape[0]
        
        X_2d = X.reshape(muestras_x, -1)
        y_2d = y.reshape(muestras_y, -1)
        
        return X_2d, y_2d

    # =========================================================================
    # MÉTODOS ESTÁTICOS PARA A/B TESTING: CHANNEL DEPENDENCE (CD) vs INDEPENDENCE (CI)
    # =========================================================================

    @staticmethod
    def to_cd_2d(tensor_3d: np.ndarray) -> np.ndarray:
        """
        Estrategia CD (Aplanamiento Espacial):
        Transforma un tensor 3D (Batch, Time, Channels) en 2D (Batch, Time * Channels).
        Mezcla todas las variables en un solo vector por muestra para capturar correlaciones cruzadas.
        """
        if tensor_3d.ndim != 3:
            raise ValueError(f"[Adapter CD] Se esperaba tensor 3D (B, T, C), se recibió {tensor_3d.ndim}D.")
        b, t, c = tensor_3d.shape
        return tensor_3d.reshape(b, t * c)

    @staticmethod
    def inverse_cd_3d(matrix_2d: np.ndarray, batch_size: int, pred_len: int, channels: int) -> np.ndarray:
        """
        Inversa de Estrategia CD:
        Reconstruye la salida 2D de XGBoost (Batch, Pred_Len * Channels) 
        al estándar 3D del evaluador (Batch, Pred_Len, Channels).
        """
        if matrix_2d.ndim == 1:
            matrix_2d = matrix_2d.reshape(batch_size, -1)
        return matrix_2d.reshape(batch_size, pred_len, channels)

    @staticmethod
    def to_ci_2d(tensor_3d: np.ndarray) -> np.ndarray:
        """
        Estrategia CI (Apilamiento de Canales):
        Transforma un tensor 3D (Batch, Time, Channels) en 2D (Batch * Channels, Time).
        Permuta a (Batch, Channels, Time) antes de remodelar para garantizar que cada fila 
        sea la serie temporal pura de un único canal independiente.
        """
        if tensor_3d.ndim != 3:
            raise ValueError(f"[Adapter CI] Se esperaba tensor 3D (B, T, C), se recibió {tensor_3d.ndim}D.")
        b, t, c = tensor_3d.shape
        # Transponer (0, 2, 1) -> (Batch, Channels, Time) en memoria contigua
        transposed = np.ascontiguousarray(np.transpose(tensor_3d, (0, 2, 1)))
        return transposed.reshape(b * c, t)

    @staticmethod
    def inverse_ci_3d(matrix_2d: np.ndarray, batch_size: int, pred_len: int, channels: int) -> np.ndarray:
        """
        Inversa de Estrategia CI:
        Reconstruye la salida 2D de XGBoost (Batch * Channels, Pred_Len)
        al estándar 3D del evaluador (Batch, Pred_Len, Channels).
        """
        if matrix_2d.ndim == 1:
            matrix_2d = matrix_2d.reshape(batch_size * channels, -1)
        # 1. Desapilar a (Batch, Channels, Pred_Len)
        tensor_bcp = matrix_2d.reshape(batch_size, channels, pred_len)
        # 2. Restaurar orden de ejes estándar (Batch, Pred_Len, Channels)
        return np.ascontiguousarray(np.transpose(tensor_bcp, (0, 2, 1)))

    # =========================================================================

    def get_3d_tensors(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convierte los arreglos NumPy a tensores de PyTorch para modelos Deep Learning.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        return X_tensor, y_tensor

    def get_statsmodels_format(
        self, data: pd.DataFrame, target_columns: List[str]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Prepara los datos en formato tabular continuo y sin superposición para VARMAX/SARIMAX.
        """
        endog = data[target_columns].copy()
        exog_columns = [col for col in data.columns if col not in target_columns]
        
        if exog_columns:
            exog = data[exog_columns].copy()
        else:
            exog = None
            
        return endog, exog
