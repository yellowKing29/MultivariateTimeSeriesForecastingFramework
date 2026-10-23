"""
Módulo data.implementations
Implementación concreta de las clases de ingesta y preprocesamiento de datos.
"""

from typing import Tuple
import pandas as pd
import numpy as np

# Asumiendo que BaseTimeSeriesDataset y BasePreprocessor están en src.core.interfaces
from src.core.interfaces import BaseTimeSeriesDataset, BasePreprocessor

class TimeSeriesDatasetImpl: # Hereda de BaseTimeSeriesDataset
    """
    Implementación concreta para la ingesta y particionado estático temporal de datos.
    """

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga los datos crudos desde un CSV asumiendo que la primera columna 
        es un índice temporal.
        """
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # Asegurar frecuencia inferida si es posible y ordenar cronológicamente
            data = data.sort_index()
            return data
        except Exception as e:
            raise RuntimeError(f"Error al cargar el archivo de datos en {file_path}: {str(e)}")

    def run_basic_eda(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta un Análisis Exploratorio básico y retorna la matriz de correlación cruzada.
        Nota: Se amplía el contrato base (None) para devolver la matriz según requerimiento.
        """
        print("--- Resumen Estadístico Básico ---")
        print(data.describe())
        print("\n--- Conteo de Valores Nulos ---")
        print(data.isnull().sum())
        
        # Calcular correlación cruzada de Pearson
        correlation_matrix = data.corr(method='pearson')
        return correlation_matrix

    def get_partitions(
        self, data: pd.DataFrame, train_ratio: float, val_ratio: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide el conjunto de datos de forma estática respetando el orden temporal.
        """
        if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio >= 1.0):
            raise ValueError("Las proporciones de train y val deben ser positivas y sumar menos de 1.0")

        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        # Partición estricta sin mezcla (shuffle=False implícito por indexación)
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()

        return train_data, val_data, test_data



