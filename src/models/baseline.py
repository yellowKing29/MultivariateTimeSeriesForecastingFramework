"""
Módulo models.baseline.sarimax
Implementación del Wrapper para el modelo clásico SARIMAX adaptado para 
pronóstico multivariado mediante la estrategia de Independent Univariate Forecasts.
"""

import time
import warnings
from typing import Any, List, Union, Optional
import numpy as np
import pmdarima as pm

# Se requiere tener statsmodels instalado: pip install statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Asumiendo que BaseModelWrapper está en src.core.interfaces
from src.core.interfaces import BaseModelWrapper


class MultiSARIMAXWrapper: # Hereda de BaseModelWrapper
    """
    Wrapper arquitectónico para orquestar múltiples modelos SARIMAX univariados
    de forma concurrente. Diseñado para recibir datos continuos bidimensionales 
    y ensamblar pronósticos multivariados independientes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Inicializa la estructura para N modelos y extrae los hiperparámetros estadísticos.
        """
        self.order = kwargs.get('order', (1, 0, 0))
        self.seasonal_order = kwargs.get('seasonal_order', (0, 0, 0, 0))
        self.enforce_stationarity = kwargs.get('enforce_stationarity', False)
        self.enforce_invertibility = kwargs.get('enforce_invertibility', False)
        
        # Almacenamiento dinámico para los N modelos ajustados
        self.models: List[Any] = []
        
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any], 
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'MultiSARIMAXWrapper':
        """
        Ajusta N modelos SARIMAX independientes, iterando sobre las características 
        del conjunto de entrenamiento continuo bidimensional.
        
        Args:
            X (Union[np.ndarray, Any]): Matriz de características (Pasos_Temporales, Caracteristicas).
            y (Optional): Por consistencia de interfaz, aunque SARIMAX es autorregresivo sobre X.

        Returns:
            MultiSARIMAXWrapper: La instancia actual con los N modelos ajustados.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            
        if X.ndim != 2:
            raise ValueError(f"MultiSARIMAXWrapper requiere un arreglo 2D continuo. Se recibió: {X.ndim}D")

        n_features = X.shape[1]
        self.models = []

        start_time = time.time()

        for i in range(n_features):
            endog_series = X[:, i]
            
            try:
                # Construcción del modelo univariado
                model = SARIMAX(
                    endog=endog_series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility
                )
                
                # Gestión de advertencias para el optimizador L-BFGS-B
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                    
                    fitted_model = model.fit(disp=False)
                    self.models.append(fitted_model)
                    
            except np.linalg.LinAlgError as e:
                raise RuntimeError(
                    f"Error de Álgebra Lineal al ajustar la característica {i}. "
                    f"Esto suele ocurrir si la serie es constante o los hiperparámetros son inválidos. Detalle: {str(e)}"
                )
            except Exception as e:
                raise RuntimeError(f"Fallo inesperado al ajustar SARIMAX en la característica {i}: {str(e)}")

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Genera pronósticos independientes iterando sobre los modelos guardados y 
        apilando los resultados para retornar una matriz multivariada.

        Args:
            X (Union[np.ndarray, Any]): Conjunto de prueba continuo (Pasos_Temporales, Caracteristicas).
                                        Se utiliza estrictamente para deducir el horizonte.

        Returns:
            np.ndarray: Matriz de predicciones bidimensional (Horizonte_Prediccion, Caracteristicas).
        """
        if not self.models:
            raise RuntimeError("Los modelos no han sido ajustados. Llame a fit() primero.")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # En enfoques clásicos, el horizonte está determinado por la longitud del conjunto de prueba a predecir
        horizon = X.shape[0]
        n_features = len(self.models)
        
        predictions = np.zeros((horizon, n_features))

        start_time = time.time()

        for i, fitted_model in enumerate(self.models):
            try:
                # forecast() proyecta 'horizon' pasos hacia el futuro desde el final de los datos de entrenamiento
                pred_series = fitted_model.forecast(steps=horizon)
                predictions[:, i] = pred_series
            except Exception as e:
                raise RuntimeError(f"Fallo durante la inferencia en la característica {i}: {str(e)}")

        self.inference_time = time.time() - start_time
        
        return predictions
        
"""
Módulo models.baseline.auto_arima
Implementación del Wrapper para el modelo estadístico Auto-ARIMA (pmdarima).
Diseñado para modelar explícitamente la estacionalidad ($m$) y prevenir el colapso
hacia la media en horizontes de predicción a largo plazo, utilizando la estrategia
de Independent Univariate Forecasts.
"""




class AutoARIMAWrapper:
    """
    Wrapper arquitectónico para Auto-ARIMA univariado múltiple.
    Optimizada para alta dimensionalidad estacional mediante subsampling temporal 
    y acotamiento algorítmico.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Extrae los parámetros estacionales y los límites de rendimiento.
        """
        self.seasonal_period = kwargs.get('seasonal_period', None)
        if self.seasonal_period is None:
            raise ValueError("El hiperparámetro 'seasonal_period' es obligatorio.")
            
        # Parámetros de optimización de rendimiento
        # max_train_size: Límite de observaciones recientes usadas para ajustar los parámetros. 
        # 2016 equivale a 14 días completos de datos a intervalos de 10 minutos (144 * 14)
        self.max_train_size = kwargs.get('max_train_size', 2016)
        
        self.models: List[Any] = []
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Optional[Union[np.ndarray, Any]] = None,
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'AutoARIMAWrapper':
        
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        n_features = X.shape[1]
        self.models = []

        start_time = time.time()

        for i in range(n_features):
            endog_series = X[:, i]
            
            # 1. Truncamiento de Historial (Data Subsampling)
            if self.max_train_size and len(endog_series) > self.max_train_size:
                endog_series = endog_series[-self.max_train_size:]
            
            try:
                # 2. Búsqueda Acotada y Agresiva (Fast Search)
                model = pm.auto_arima(
                    y=endog_series,
                    seasonal=True,
                    m=self.seasonal_period,
                    stepwise=True,       # Algoritmo heurístico (evita Grid Search completo)
                    max_p=2, max_q=2,    # Limita la complejidad autorregresiva estándar
                    max_P=1, max_Q=1,    # Limita la complejidad estacional (crucial para m=144)
                    max_d=1, max_D=1,    # Evita sobre-diferenciación
                    max_order=4,         # Suma máxima de p+q+P+Q
                    n_jobs=1,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                self.models.append(model)
                
            except Exception as e:
                raise RuntimeError(f"Fallo al ajustar Auto-ARIMA en la característica {i}: {str(e)}")

        self.training_time = time.time() - start_time
        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Los modelos Auto-ARIMA no han sido ajustados.")
            
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        horizon = X.shape[0]
        n_features = len(self.models)
        predictions = np.zeros((horizon, n_features))

        start_time = time.time()

        for i, fitted_model in enumerate(self.models):
            # Inferencia dinámica hacia el futuro
            pred_series = fitted_model.predict(n_periods=horizon)
            predictions[:, i] = pred_series

        self.inference_time = time.time() - start_time
        return predictions
