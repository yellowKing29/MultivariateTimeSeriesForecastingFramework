"""
main.py
Script orquestador central del framework de pronóstico multivariado.
Conecta la configuración, el pipeline de datos, el entrenamiento y el registro de experimentos.
"""

import yaml
import torch
import numpy as np

# Importaciones de los módulos implementados previamente
# Se asume la estructura del proyecto definida en la Fase 1
from src.data.datasets import TimeSeriesDatasetImpl
from src.data.preprocessors import PreprocessorImpl
from src.data.adapters import AdapterImpl
from src.models.ml_models import XGBoostCIWrapper
from src.models.ml_models import XGBoostCDWrapper
from src.models.ml_models import RandomForestWrapper
from src.models.dl_models import iTransformerWrapper
from src.models.dl_models import DLinearWrapper
from src.models.baseline import MultiSARIMAXWrapper
from src.models.baseline import AutoARIMAWrapper
from src.evaluation.evaluators import EvaluatorImpl
from src.visualization.plotters import plot_forecast_comparisons
from src.visualization.plotters import plot_forecast_with_zoom
import os


def load_configuration(config_path: str = "config.yaml") -> dict:
    """1. Carga de Configuración"""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    # 1. Leer Configuración
    config = load_configuration()
    global_params = config["global"]
    
    dataset_path = global_params["dataset_path"]
    window_size = global_params["look_back"]
    horizon = global_params["horizon"]
    
    # 2. Pipeline de Datos
    dataset = TimeSeriesDatasetImpl()
    
    # Simulación de carga (en producción: data = dataset.load_data(dataset_path))
    print(f"Cargando datos desde: {dataset_path}")
    data = dataset.load_data(dataset_path)

    # Partición temporal estática (70% train, 15% val, 15% test)
    train, val, test = dataset.get_partitions(data, train_ratio=0.7, val_ratio=0.15)
    
    preprocessor = PreprocessorImpl()
    
    # Imputación por bloque aislado
    train_imputed = preprocessor.impute_missing(train)
    val_imputed = preprocessor.impute_missing(val)
    test_imputed = preprocessor.impute_missing(test)
    
    # Escalado sin Data Leakage (fit solo en train)
    preprocessor.fit(train_imputed)
    train_scaled = preprocessor.transform(train_imputed)
    test_scaled = preprocessor.transform(test_imputed)
    # Nota: validation no se utiliza en ML clásico para fit(), pero estaría disponible aquí.
    
    adapter = AdapterImpl()
    
    # --- REEMPLAZA LA ASIGNACIÓN DE target_columns ---
    target_columns_config = global_params.get("target_columns", None)
    
    # Si viene como un string puro desde el YAML, lo convertimos a lista
    if isinstance(target_columns_config, str):
        target_columns_config = [target_columns_config]
    # ------------------------------
    
    if target_columns_config and len(target_columns_config) > 0:
        target_columns = target_columns_config
        print(f"Objetivos específicos configurados: {target_columns}")
    else:
        target_columns = data.columns.tolist()
        print(f"Objetivos no especificados. Prediciendo todas las {len(target_columns)} variables.")

    # Extraer los índices numéricos de las columnas objetivo
    target_indices_list = [train.columns.get_loc(col) for col in target_columns]

    
    # Ventanas temporales base 3D
    X_train_base, y_train_base = adapter.create_sliding_windows(
        train_scaled, target_columns, window_size, horizon
    )
    X_test_base, y_test_base = adapter.create_sliding_windows(
        test_scaled, target_columns, window_size, horizon
    )
    
    evaluator = EvaluatorImpl()
    models_to_run = config["orchestration"]["models_to_run"]
    experiment_name = config["orchestration"]["experiment_name"]
    
    # Bucle de Experimentos
    model_predictions = {}
    y_true_reference = None

    # 3. Bucle de Experimentos
    for model_name in models_to_run:
        print(f"\nIniciando pipeline para modelo: {model_name.upper()}")
        hyperparams = config["models"].get(model_name, {}).copy()
        
        if model_name == "sarimax1":
            # 1. Adaptación de Datos: SARIMAX requiere la serie 2D continua, no ventanas 3D
            X_train_df, _ = adapter.get_statsmodels_format(train_scaled, target_columns)
            X_test_df, _ = adapter.get_statsmodels_format(test_scaled, target_columns)
            
            # Convertimos a numpy arrays (X_train actúa como variable endógena a ajustar)
            X_train = X_train_df.values
            y_train = None  # No requiere 'y' explícito para el ajuste
            
            # X_test se usa para extraer la longitud del horizonte
            X_test = X_test_df.values
            y_test = X_test.copy()  # El ground truth para evaluar es la misma serie futura
            
            # 2. Conversión YAML a Tupla: statsmodels requiere tuplas estrictas
            if "order" in hyperparams and isinstance(hyperparams["order"], list):
                hyperparams["order"] = tuple(hyperparams["order"])
            if "seasonal_order" in hyperparams and isinstance(hyperparams["seasonal_order"], list):
                hyperparams["seasonal_order"] = tuple(hyperparams["seasonal_order"])
                
            model = MultiSARIMAXWrapper(**hyperparams)
            
        elif model_name == "sarimax":
            # 1. Adaptación de Datos: Auto-ARIMA requiere la serie 2D continua, no ventanas 3D
            X_train_df, _ = adapter.get_statsmodels_format(train_scaled, target_columns)
            X_test_df, _ = adapter.get_statsmodels_format(test_scaled, target_columns)
            
            # Convertimos a numpy arrays (X_train actúa como variable endógena a ajustar)
            X_train = X_train_df.values
            y_train = None  # No requiere 'y' explícito para el ajuste
            
            # X_test se usa para extraer la longitud del horizonte
            X_test = X_test_df.values
            y_test = X_test.copy()  # El ground truth para evaluar es la misma serie futura
            
            # 2. Instanciamos el nuevo wrapper (la búsqueda p,d,q y P,D,Q es automática)
            model = AutoARIMAWrapper(**hyperparams)
            
            
        
        elif model_name == "randomForest":
            # Extracción aplanada nativa (2D) exclusiva para Random Forest
            X_train, y_train = adapter.get_2d_arrays(X_train_base, y_train_base)
            X_test, y_test = adapter.get_2d_arrays(X_test_base, y_test_base)
            model = RandomForestWrapper(**hyperparams)

        elif model_name in ["xgboost_cd", "xgboost_ci"]:
            # Los nuevos wrappers de XGBoost reciben tensores 3D directamente (Batch, Time, Channels)
            # y ejecutan la transformación CD o CI (y su inversa) de forma encapsulada
            X_train, y_train = X_train_base, y_train_base
            X_test, y_test = X_test_base, y_test_base
            
            # Inyectar índices objetivo para alinear canales en caso de predicción parcial (ej. 1 sola variable)
            hyperparams["target_indices"] = [train.columns.get_loc(col) for col in target_columns]
            
            if model_name == "xgboost_cd":
                model = XGBoostCDWrapper(**hyperparams)
            else:
                model = XGBoostCIWrapper(**hyperparams)
                
        elif model_name == "itransformer":
            X_train, y_train = adapter.get_3d_tensors(X_train_base, y_train_base)
            X_test, y_test = adapter.get_3d_tensors(X_test_base, y_test_base)
            
            hyperparams["num_variates"] = X_train_base.shape[-1]
            hyperparams["lookback_len"] = window_size
            hyperparams["pred_len"] = horizon
            hyperparams["target_indices"] = target_indices_list  # Inyección de índices
            model = iTransformerWrapper(**hyperparams)
            
        elif model_name == "dlinear":
            X_train, y_train = adapter.get_3d_tensors(X_train_base, y_train_base)
            X_test, y_test = adapter.get_3d_tensors(X_test_base, y_test_base)
            
            hyperparams["look_back"] = window_size
            hyperparams["horizon"] = horizon
            hyperparams["target_indices"] = target_indices_list  # Inyección de índices
            model = DLinearWrapper(**hyperparams)
        
        else:
            print(f"Modelo {model_name} no soportado en este script.")
            continue

        # 4. Entrenamiento y Evaluación
        print("Entrenando modelo...")
        model.fit(X_train, y_train)
        
        print("Generando predicciones sobre test...")
        predictions = model.predict(X_test)
        
        # Homologar y_test para evaluación
        if isinstance(y_test, torch.Tensor):
            y_test_eval = y_test.cpu().numpy()
        else:
            y_test_eval = y_test

        # 5. Métricas (Desnormalización y cálculo)
        y_test_flat = y_test_eval.reshape(-1, len(target_columns))
        predictions_flat = predictions.reshape(-1, len(target_columns))

        # Reconstrucción dimensional para evitar el error de broadcast del StandardScaler
        n_features_total = train.shape[1]
        dummy_y_true = np.zeros((y_test_flat.shape[0], n_features_total))
        dummy_y_pred = np.zeros((predictions_flat.shape[0], n_features_total))

        # Encontrar los índices exactos de las columnas objetivo en el dataset original
        target_indices = [train.columns.get_loc(col) for col in target_columns]

        # Inyectar los valores predichos y reales en su posición correspondiente
        for i, idx in enumerate(target_indices):
            dummy_y_true[:, idx] = y_test_flat[:, i]
            dummy_y_pred[:, idx] = predictions_flat[:, i]

        # Desnormalizar la matriz completa (21 columnas)
        y_true_denorm_full, y_pred_denorm_full = evaluator.denormalize(
            dummy_y_true, dummy_y_pred, preprocessor
        )

        # Extraer exclusivamente las columnas objetivo ya devueltas a su escala física
        # (Tu código previo de desnormalización)
        # Extraer exclusivamente las columnas objetivo ya devueltas a su escala física
        y_true_denorm = y_true_denorm_full[:, target_indices]
        y_pred_denorm = y_pred_denorm_full[:, target_indices]

        # =====================================================================
        # EXTRACCIÓN DE SERIE CONTINUA (Para Diebold-Mariano y Gráficas)
        # =====================================================================
        if model_name != "sarimax":
            # Obtenemos la cantidad de ventanas generadas en el conjunto de prueba
            num_windows = X_test.shape[0]
            n_targets = len(target_columns)
            
            # Calculamos el horizonte dinámicamente: Total_Filas_Aplanadas / Num_Ventanas
            horizonte_pred = y_pred_denorm.shape[0] // num_windows
            
            # Reconstruimos el tensor 3D (Ventanas, Horizonte, Variables) y tomamos el último paso [:, -1, :]
            # Esto funciona tanto para DL (que venía en 3D) como para ML (que venía aplanado en 2D)
            y_pred_plot = y_pred_denorm.reshape(num_windows, horizonte_pred, n_targets)[:, -1, :]
            y_true_plot = y_true_denorm.reshape(num_windows, horizonte_pred, n_targets)[:, -1, :]
        else:
            # SARIMAX / Auto-ARIMA no usa ventanas deslizantes, ya es una serie continua 2D
            y_pred_plot = y_pred_denorm
            y_true_plot = y_true_denorm

        # Almacenar la serie continua (~7,714 puntos) para Diebold-Mariano y Matplotlib
        model_predictions[model_name] = y_pred_plot
        if y_true_reference is None:
            y_true_reference = y_true_plot
        # =====================================================================        
        # Las métricas globales se siguen calculando con la matriz aplanada masiva 
        # para reportar el desempeño formal completo en MLflow
        metricas = evaluator.calculate_metrics(
            y_true_norm=y_test_flat,        
            y_pred_norm=predictions_flat,   
            y_true_denorm=y_true_denorm,    
            y_pred_denorm=y_pred_denorm     
        )

     
        print("Resultados de la Evaluación:")
        for metrica, valor in metricas.items():
            print(f"  - {metrica}: {valor:.4f}")

        # 6. Cierre y Trazabilidad (MLflow)
        tiempos = {
            "training_time": model.training_time,
            "inference_time": model.inference_time
        }
        
        run_name = f"{experiment_name}_{model_name}"
        evaluator.log_experiment(run_name, hyperparams, metricas, tiempos)
        
# 7. Análisis de Significancia Estadística (Diebold-Mariano)
    print("\n" + "="*60)
    print(" ANÁLISIS DE SIGNIFICANCIA ESTADÍSTICA (DIEBOLD-MARIANO)")
    print("="*60)
    
    import itertools
    
    model_names = list(model_predictions.keys())
    
    if len(model_names) > 1:
        # --- ALINEACIÓN TEMPORAL DINÁMICA ---
        # 1. Encontrar la longitud mínima entre todas las predicciones
        min_len = min(len(preds) for preds in model_predictions.values())
        
        # 2. Alinear todas las predicciones recortando los valores más antiguos
        # Esto asegura que SARIMAX y ML/DL se evalúen exactamente en el mismo bloque de tiempo final
        for name in model_names:
            model_predictions[name] = model_predictions[name][-min_len:]
            
        # 3. Alinear el ground truth de referencia
        y_true_aligned = y_true_reference[-min_len:]
        # ------------------------------------

        # Generar todas las combinaciones posibles de pares de modelos
        for model_a, model_b in itertools.combinations(model_names, 2):
            print(f"\nComparando: {model_a.upper()} vs {model_b.upper()}")
            
            dm_results = evaluator.diebold_mariano_test(
                y_true=y_true_aligned,
                y_pred_1=model_predictions[model_a],
                y_pred_2=model_predictions[model_b],
                horizon=horizon
            )
            
            dm_stat = dm_results["DM_Statistic"]
            p_value = dm_results["p_value"]
            
            print(f"  - Estadístico DM: {dm_stat:.4f}")
            print(f"  - P-value: {p_value:.4e}")
            
            # Interpretación académica (nivel de significancia alfa = 0.05)
            if p_value < 0.05:
                # Un estadístico DM negativo indica que la pérdida del Modelo 1 (A) es menor
                if dm_stat < 0:
                    ganador = model_a.upper()
                else:
                    ganador = model_b.upper()
                print(f"  -> Conclusión: La diferencia es ESTADÍSTICAMENTE SIGNIFICATIVA.")
                print(f"  -> {ganador} es significativamente superior predictivamente.")
            else:
                print(f"  -> Conclusión: NO HAY DIFERENCIA estadísticamente significativa.")
    else:
        print("Se requiere ejecutar al menos dos modelos para el Test de Diebold-Mariano.")
        
    # =====================================================================
    # 8. Visualización Científica (Módulo 6)
    # =====================================================================
    print("\n" + "="*60)
    print(" GENERACIÓN DE GRÁFICAS DE GRADO CIENTÍFICO")
    print("="*60)
    
    if len(model_predictions) == 0:
        print("No hay predicciones generadas para graficar.")
    else:
        # 1. Extraer min_len de forma segura e independiente
        min_len = len(list(model_predictions.values())[0])
        
        # 2. Alinear estrictamente el Eje X (Fechas) y el Eje Y (Ground Truth) a min_len
        time_index_aligned = test.index[-min_len:]
        y_true_aligned = y_true_reference[-min_len:]
        
        # Crear directorio para almacenar las figuras generadas
        figures_dir = "results/figures"
        os.makedirs(figures_dir, exist_ok=True)
        
        # Iterar sobre cada variable objetivo para soportar predicciones multivariadas simultáneas
        for i, feature_name in enumerate(target_columns):
            print(f"Generando gráfica comparativa para la variable: {feature_name}...")
            
            # Formatear el nombre del archivo
            safe_feature_name = feature_name.replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(figures_dir, f"comparativa_{safe_feature_name}")
            
            plot_forecast_comparisons(
                y_true=y_true_aligned,
                predictions_dict=model_predictions,
                time_index=time_index_aligned,
                feature_idx=i,
                feature_name=feature_name,
                save_path=save_path
            )
            
            #print(f"Generando gráfica comparativa con zoom para: {feature_name}...")
            #save_path_zoom = os.path.join(figures_dir, f"comparativa_zoom_{safe_feature_name}")
            
            # Parámetros de la ventana de zoom (ej. las últimas 200 observaciones)
            #total_len = len(time_index_aligned)
            #zoom_start = total_len - 200
            #zoom_end = total_len - 1
            
            #plot_forecast_with_zoom(
            #    y_true=y_true_aligned,
            #    predictions_dict=model_predictions,
            #    time_index=time_index_aligned,
            #    feature_idx=i,
            #    feature_name=feature_name,
            #    zoom_start_idx=zoom_start,
            #    zoom_end_idx=zoom_end,
            #    save_path=save_path_zoom,
            #   inset_bounds=(0.02, 0.55, 0.35, 0.4) 
            #)
            
    print("\n✅ Pipeline de experimentación completado exitosamente.")
    
if __name__ == "__main__":
    main()
