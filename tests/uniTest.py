import pytest
import numpy as np
import pandas as pd
import torch

# Asumiendo que AdapterImpl está en src.data.adapters
# from src.data.adapters import AdapterImpl

@pytest.fixture
def dummy_data() -> pd.DataFrame:
    """
    Fixture de pytest: Genera un DataFrame multivariado simulado.
    100 pasos de tiempo, 3 variables con índice de fechas temporal.
    """
    fechas = pd.date_range(start="2023-01-01", periods=100, freq="D")
    # Semilla fija para reproducibilidad en pruebas
    np.random.seed(42)
    datos = np.random.rand(100, 3)
    return pd.DataFrame(datos, index=fechas, columns=["feature_1", "feature_2", "target_1"])

@pytest.fixture
def adapter() -> 'AdapterImpl':
    """Fixture de pytest: Instancia limpia de AdapterImpl para cada prueba."""
    return AdapterImpl()

def test_get_2d_arrays_dimensionality(dummy_data: pd.DataFrame, adapter: 'AdapterImpl') -> None:
    """
    Valida la extracción a 2D para algoritmos basados en árboles (XGBoost).
    Verifica que las dimensiones tras el aplanamiento sean matemáticamente exactas.
    """
    window_size = 10
    horizon = 1
    # Usaremos las 3 columnas como objetivos para que 'y' tenga 3 características
    target_columns = ["feature_1", "feature_2", "target_1"]
    
    # 1. Generar ventanas 3D base
    X_3d, y_3d = adapter.create_sliding_windows(
        dummy_data, target_columns, window_size, horizon
    )
    
    # 2. Transformar a 2D
    X_2d, y_2d = adapter.get_2d_arrays(X_3d, y_3d)
    
    # Cálculo esperado de muestras: 100 (total) - 10 (contexto) - 1 (horizonte) + 1 = 90 muestras
    # Características aplanadas de X: 10 (pasos de tiempo) * 3 (variables) = 30
    # Características aplanadas de y: 1 (horizonte) * 3 (variables) = 3
    
    assert isinstance(X_2d, np.ndarray), "X_2d debe ser un numpy.ndarray"
    assert isinstance(y_2d, np.ndarray), "y_2d debe ser un numpy.ndarray"
    assert X_2d.shape == (90, 30), f"Error de dimensionalidad en X_2d. Se obtuvo: {X_2d.shape}"
    assert y_2d.shape == (90, 3), f"Error de dimensionalidad en y_2d. Se obtuvo: {y_2d.shape}"

def test_get_3d_tensors_dimensionality(dummy_data: pd.DataFrame, adapter: 'AdapterImpl') -> None:
    """
    Valida la conversión a tensores 3D de PyTorch para Deep Learning.
    Verifica preservación estricta de la dimensión temporal.
    """
    window_size = 10
    horizon = 1
    target_columns = ["feature_1", "feature_2", "target_1"]
    
    # 1. Generar ventanas 3D base
    X_3d, y_3d = adapter.create_sliding_windows(
        dummy_data, target_columns, window_size, horizon
    )
    
    # 2. Transformar a Tensores
    X_tensor, y_tensor = adapter.get_3d_tensors(X_3d, y_3d)
    
    assert isinstance(X_tensor, torch.Tensor), "X_tensor debe ser un torch.Tensor"
    assert isinstance(y_tensor, torch.Tensor), "y_tensor debe ser un torch.Tensor"
    
    # Las dimensiones en DL deben mantenerse como (Batch, TimeSteps, Features)
    # y_tensor tendrá el horizonte explícito como TimeSteps
    assert X_tensor.shape == (90, 10, 3), f"Error de dimensionalidad en X_tensor. Se obtuvo: {X_tensor.shape}"
    assert y_tensor.shape == (90, 1, 3), f"Error de dimensionalidad en y_tensor. Se obtuvo: {y_tensor.shape}"

def test_sliding_windows_edge_case_insufficient_data(adapter: 'AdapterImpl') -> None:
    """
    Caso Límite: El DataFrame de entrada es más corto que la ventana solicitada.
    Valida que el sistema maneje la falta de datos sin fallar en memoria, retornando arreglos vacíos.
    """
    # DataFrame extremadamente corto (5 filas)
    short_data = pd.DataFrame(np.random.rand(5, 3), columns=["A", "B", "C"])
    window_size = 10
    horizon = 1
    target_columns = ["A", "B", "C"]
    
    X_3d, y_3d = adapter.create_sliding_windows(
        short_data, target_columns, window_size, horizon
    )
    
    # Al no haber suficientes datos para una sola ventana, debe retornar 0 muestras
    assert len(X_3d) == 0, "X_3d debe estar vacío al recibir un DataFrame inservible por tamaño."
    assert len(y_3d) == 0, "y_3d debe estar vacío al recibir un DataFrame inservible por tamaño."
