from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class PreprocessorImpl: # Hereda de BasePreprocessor
    """
    Implementación concreta para el manejo de nulos y escalado estandarizado (Z-score).
    Diseñado para prevenir estrictamente el Data Leakage temporal.
    """

    def __init__(self) -> None:
        """
        Inicializa el preprocesador delegando el escalado a StandardScaler de scikit-learn.
        """
        self.scaler = StandardScaler()
        self._is_fitted = False

    def impute_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores nulos utilizando interpolación temporal lineal, seguida de 
        un ffill (forward fill) y bfill (backward fill) para cubrir los extremos.
        """
        # Se asume que el índice es de tipo datetime para una correcta interpolación por tiempo.
        # Si no lo es, pandas recurrirá a interpolación lineal por defecto.
        imputed_data = data.copy()
        imputed_data = imputed_data.interpolate(method='time')
        imputed_data = imputed_data.ffill().bfill()
        return imputed_data

    def fit(self, data: pd.DataFrame) -> 'PreprocessorImpl':
        """
        Ajusta el escalador EXCLUSIVAMENTE con el conjunto de entrenamiento.
        Calcula la media y la desviación estándar empírica.
        """
        self.scaler.fit(data)
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el escalado estandarizado utilizando los parámetros aprendidos en fit().
        Mantiene la estructura (índices y columnas) del DataFrame original.
        """
        if not self._is_fitted:
            raise RuntimeError("El preprocesador debe ser ajustado con fit() antes de llamar a transform().")

        scaled_array = self.scaler.transform(data)
        
        # Reconstruir el DataFrame para preservar el flujo de Pandas antes del Adapter
        scaled_data = pd.DataFrame(
            scaled_array, 
            index=data.index, 
            columns=data.columns
        )
        return scaled_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Revierte el escalado para devolver las predicciones a su magnitud física original.
        Es vital para el cálculo de métricas de negocio (como el RMSE).
        """
        if not self._is_fitted:
            raise RuntimeError("El preprocesador no ha sido ajustado.")
            
        return self.scaler.inverse_transform(data)
