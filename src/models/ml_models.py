"""
Módulo models.implementations
Implementación de los wrappers para los modelos de Machine Learning y Deep Learning.
Gestiona el ciclo de vida, hiperparámetros y el perfilado de tiempos de ejecución.
"""

import time
from typing import Any, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import gc
from sklearn.ensemble import RandomForestRegressor

# Asumiendo que BaseModelWrapper está en src.core.interfaces
from src.core.interfaces import BaseModelWrapper
from src.data.adapters import AdapterImpl


class RandomForestWrapper: # Hereda de BaseModelWrapper
    """
    Wrapper arquitectónico para el modelo Random Forest Regressor de Scikit-Learn.
    
    Diseñado para procesar datos tabulares (2D aplanados). Gracias a su soporte 
    nativo para regresión multisalida, este modelo puede ingerir una matriz objetivo 'y' 
    con dimensionalidad (Muestras, Horizonte_Prediccion * Caracteristicas) y optimizar
    sus árboles de decisión para predecir todo el horizonte multivariado en un solo paso.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Extrae los hiperparámetros del ensamble e inicializa el regresor.
        Inyecta n_jobs=-1 por defecto para garantizar paralelización en CPU.
        """
        self.n_estimators = kwargs.get('n_estimators', 100)
        self.max_depth = kwargs.get('max_depth', None)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        self.random_state = kwargs.get('random_state', 42)
        
        # Parámetro clave para rendimiento en datasets masivos
        self.n_jobs = kwargs.get('n_jobs', -1)
        
        # Instancia nativa del modelo
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        # Atributos de perfilado computacional
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any], 
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'RandomForestWrapper':
        """
        Entrena el bosque aleatorio sobre los arreglos bidimensionales aplanados.
        
        Args:
            X (Union[np.ndarray, Any]): Características (Muestras, Lookback * Vars).
            y (Union[np.ndarray, Any]): Objetivo (Muestras, Horizonte * Vars).

        Returns:
            RandomForestWrapper: Instancia del wrapper con el modelo ajustado.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError(
                f"RandomForestWrapper requiere arreglos 2D aplanados. "
                f"Se recibió X con {X.ndim}D e y con {y.ndim}D."
            )

        start_time = time.time()
        
        # Aplanar a vector 1D si es estrictamente univariado para evitar el DataConversionWarning
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        
        # Entrenamiento nativo (construcción de árboles optimizada en C++)
        self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Ejecuta la inferencia multisalida sobre el conjunto de prueba.
        
        Args:
            X (Union[np.ndarray, Any]): Características de prueba (Muestras, Lookback * Vars).
            
        Returns:
            np.ndarray: Predicciones aplanadas (Muestras, Horizonte * Vars).
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        start_time = time.time()
        
        # Inferencia nativa
        predictions = self.model.predict(X)
        
        self.inference_time = time.time() - start_time
        
        # Retorno en el formato exacto requerido por el Evaluador y el Desnormalizador
        return predictions




class _BaseXGBoostGPU:  # Hereda de BaseModelWrapper
    """
    Clase base interna para configurar aceleración GPU (hist + cuda), 
    árboles vectoriales (multi_output_tree) con control estricto de VRAM y poda con Optuna.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.seq_len: Optional[int] = kwargs.pop('lookback_len', kwargs.pop('look_back', None))
        self.pred_len: Optional[int] = kwargs.pop('pred_len', kwargs.pop('horizon', None))
        self.target_indices: Optional[list] = kwargs.pop('target_indices', None)

        self.n_estimators: int = kwargs.pop('n_estimators', 100)
        self.max_depth: int = kwargs.pop('max_depth', 6)
        self.learning_rate: float = kwargs.pop('learning_rate', 0.05)
        self.subsample: float = kwargs.pop('subsample', 0.8)
        self.colsample_bytree: float = kwargs.pop('colsample_bytree', 0.8)
        self.colsample_bylevel: float = kwargs.pop('colsample_bylevel', 1.0)
        self.max_bin: int = kwargs.pop('max_bin', 64)
        self.max_cached_hist_node: int = kwargs.pop('max_cached_hist_node', 65536)
        self.max_train_samples: Optional[int] = kwargs.pop('max_train_samples', None)
        self.random_state: int = kwargs.pop('random_state', 42)

        # Limpiar hiperparámetros exclusivos de Deep Learning si vienen en kwargs
        for dl_key in [
            'epochs', 'batch_size', 'patience', 'd_model', 
            'n_heads', 'e_layers', 'dropout', 'use_amp', 'accumulation_steps'
        ]:
            kwargs.pop(dl_key, None)

        self.xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'max_bin': self.max_bin,
            'max_cached_hist_node': self.max_cached_hist_node,
            'random_state': self.random_state,
            'tree_method': 'hist',
            'device': 'cuda',
            'multi_strategy': 'multi_output_tree',
            'eval_metric': 'rmse',
            **kwargs
        }

        self.model = xgb.XGBRegressor(**self.xgb_params)
        self._fitted_pred_len: int = 0
        self._fitted_out_channels: int = 0
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def _free_gpu_memory(self) -> None:
        """Libera fragmentos huérfanos de VRAM antes de alocar histogramas en CUDA."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_numpy_3d(self, data: Any) -> np.ndarray:
        """Convierte tensores PyTorch a NumPy 3D si es necesario."""
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(
                f"Los nuevos wrappers de XGBoost requieren tensores 3D (Batch, Time, Channels) "
                f"directamente de create_sliding_windows(). Se recibió forma {data.shape}."
            )
        return data.astype(np.float32, copy=False)

    def _setup_pruning(self, optuna_trial: Optional[Any], has_val: bool) -> None:
        if optuna_trial is not None and has_val:
            from optuna_integration.xgboost import XGBoostPruningCallback
            self.model.set_params(callbacks=[XGBoostPruningCallback(optuna_trial, "validation_0-rmse")])
        else:
            self.model.set_params(callbacks=None)

class XGBoostCDWrapper(_BaseXGBoostGPU):
    """
    XGBoost con estrategia Channel Dependence (CD).
    Transforma X: (B, L, C_in) -> (B, L * C_in) e y: (B, P, C_out) -> (B, P * C_out).
    Incluye control matemático del tensor de histogramas CUDA para GPUs de 12GB.
    """

    def __init__(self, **kwargs: Any) -> None:
        # En multi_output_tree en GPU, la VRAM escala con: 2^max_depth * max_bin * Cols_X * Cols_Y
        kwargs.setdefault('max_depth', 3)
        kwargs.setdefault('max_bin', 16)
        kwargs.setdefault('colsample_bytree', 0.2)
        kwargs.setdefault('max_cached_hist_node', 8)
        kwargs.setdefault('max_train_samples', 8000)
        self.input_lag_stride: int = kwargs.pop('input_lag_stride', 2)
        super().__init__(**kwargs)

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any],
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'XGBoostCDWrapper':
        self._free_gpu_memory()

        X_3d = self._to_numpy_3d(X)
        y_3d = self._to_numpy_3d(y)

        _, self._fitted_pred_len, self._fitted_out_channels = y_3d.shape
        total_outputs = self._fitted_pred_len * self._fitted_out_channels

        # Protección estricta contra OOM en device_vector.cu cuando Salidas > 500
        if total_outputs >= 500:
            self.model.set_params(
                max_depth=min(self.max_depth, 3),
                max_bin=min(self.max_bin, 16),
                max_cached_hist_node=8
            )
            # Submuestrear el eje temporal de entrada (X) por factor input_lag_stride (ej. cada 2 pasos)
            # Reduce a la mitad las columnas de entrada sin alterar la forma de salida (P, C_out)
            if self.input_lag_stride > 1 and X_3d.shape[1] >= 4:
                X_3d = X_3d[:, ::self.input_lag_stride, :]

        # Submuestreo uniforme de ventanas de entrenamiento para liberar VRAM de Gradientes/Hessianos
        if self.max_train_samples is not None and X_3d.shape[0] > self.max_train_samples:
            indices = np.linspace(0, X_3d.shape[0] - 1, self.max_train_samples, dtype=int)
            X_3d = X_3d[indices]
            y_3d = y_3d[indices]

        X_cd = AdapterImpl.to_cd_2d(X_3d)
        y_cd = AdapterImpl.to_cd_2d(y_3d)

        del X_3d, y_3d
        self._free_gpu_memory()

        fit_params = {}
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_3d = self._to_numpy_3d(X_val)
            y_val_3d = self._to_numpy_3d(y_val)
            if total_outputs >= 500 and self.input_lag_stride > 1 and X_val_3d.shape[1] >= 4:
                X_val_3d = X_val_3d[:, ::self.input_lag_stride, :]
            if self.max_train_samples is not None and X_val_3d.shape[0] > (self.max_train_samples // 4):
                val_idx = np.linspace(0, X_val_3d.shape[0] - 1, self.max_train_samples // 4, dtype=int)
                X_val_3d = X_val_3d[val_idx]
                y_val_3d = y_val_3d[val_idx]
            fit_params['eval_set'] = [(AdapterImpl.to_cd_2d(X_val_3d), AdapterImpl.to_cd_2d(y_val_3d))]
            fit_params['verbose'] = False

        self._setup_pruning(optuna_trial, has_val)

        start_time = time.time()
        self.model.fit(X_cd, y_cd, **fit_params)
        self.training_time = time.time() - start_time

        del X_cd, y_cd
        self._free_gpu_memory()
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        X_3d = self._to_numpy_3d(X)
        batch_size = X_3d.shape[0]
        total_outputs = self._fitted_pred_len * self._fitted_out_channels

        # Aplicar exactamente el mismo stride temporal en la entrada X durante la inferencia
        if total_outputs >= 500 and self.input_lag_stride > 1 and X_3d.shape[1] >= 4:
            X_3d = X_3d[:, ::self.input_lag_stride, :]

        X_cd = AdapterImpl.to_cd_2d(X_3d)
        
        dmatrix_eval = xgb.DMatrix(X_cd)

        start_time = time.time()
        preds_2d = self.model.get_booster().predict(dmatrix_eval)
        self.inference_time = time.time() - start_time

        return AdapterImpl.inverse_cd_3d(
            matrix_2d=preds_2d,
            batch_size=batch_size,
            pred_len=self._fitted_pred_len,
            channels=self._fitted_out_channels
        )

class XGBoostCIWrapper(_BaseXGBoostGPU):
    """
    XGBoost con estrategia Channel Independence (CI).
    Transforma X: (B, L, C) -> (B * C, L) e y: (B, P, C) -> (B * C, P).
    """

    def _align_channels(self, X_3d: np.ndarray, out_channels: int) -> np.ndarray:
        """
        Si create_sliding_windows generó X con todas las columnas (ej. 21) pero 'y' solo 
        tiene las columnas objetivo (ej. 1), filtra X para mantener la relación 1-a-1 de CI.
        """
        if X_3d.shape[2] != out_channels:
            if self.target_indices is not None and len(self.target_indices) == out_channels:
                return X_3d[:, :, self.target_indices]
            raise ValueError(
                f"[XGBoostCIWrapper] Discrepancia de canales entre X ({X_3d.shape[2]}) e y ({out_channels}). "
                f"Pasa 'target_indices' en hyperparams para alinear los canales."
            )
        return X_3d

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any],
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'XGBoostCIWrapper':
        X_3d = self._to_numpy_3d(X)
        y_3d = self._to_numpy_3d(y)

        _, self._fitted_pred_len, self._fitted_out_channels = y_3d.shape
        X_3d = self._align_channels(X_3d, self._fitted_out_channels)

        X_ci = AdapterImpl.to_ci_2d(X_3d)
        y_ci = AdapterImpl.to_ci_2d(y_3d)

        fit_params = {}
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_3d = self._align_channels(self._to_numpy_3d(X_val), self._fitted_out_channels)
            y_val_3d = self._to_numpy_3d(y_val)
            fit_params['eval_set'] = [(AdapterImpl.to_ci_2d(X_val_3d), AdapterImpl.to_ci_2d(y_val_3d))]
            fit_params['verbose'] = False

        self._setup_pruning(optuna_trial, has_val)

        start_time = time.time()
        self.model.fit(X_ci, y_ci, **fit_params)
        self.training_time = time.time() - start_time
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        X_3d = self._to_numpy_3d(X)
        X_3d = self._align_channels(X_3d, self._fitted_out_channels)
        batch_size = X_3d.shape[0]

        X_ci = AdapterImpl.to_ci_2d(X_3d)

        start_time = time.time()
        preds_2d = self.model.predict(X_ci)
        self.inference_time = time.time() - start_time

        return AdapterImpl.inverse_ci_3d(
            matrix_2d=preds_2d,
            batch_size=batch_size,
            pred_len=self._fitted_pred_len,
            channels=self._fitted_out_channels
        )
