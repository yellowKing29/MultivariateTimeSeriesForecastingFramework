import yaml

# Diccionario de configuración con hiperparámetros
config_data = {
    "global": {
        "dataset_path": "data/raw/weather.csv",
        "look_back": 96,          # Aumentado para dar contexto suficiente a los modelos profundos
        "horizon": 96,            # Ajustado para evaluar predicción a múltiples pasos (1 día)
        "target_columns": None #"T (degC)"
    },
    "models": {
        "sarimax1": {
            "order": [2, 0, 1],             
            "seasonal_order": [1, 1, 1, 24], 
            "enforce_stationarity": False,
            "enforce_invertibility": False
        },
        "sarimax": {
            "seasonal_period": 24,    # Corregido para muestreo por hora (1 día = 24 horas)
            "max_train_size": 720     # Reducido para evitar tiempos de cómputo inasumibles en pmdarima
        },
        "randomForest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 10,
            "n_jobs": -1
        },
        "xgboost_cd": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.2,
            "max_bin": 32,
            "max_cached_hist_node": 16,
            "max_train_samples": 12000
        },
        "xgboost_ci": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_bin":256
        },
        "itransformer": {
            "d_model": 512,           # Estándar del paper: máxima riqueza en el token invertido
            "n_heads": 8,             # Perfecto matemático: 512 / 8 = 64 dimensiones por cabezal
            "e_layers": 2,            # Mantenemos 2 capas para proteger la memoria VRAM (3 si sobra memoria)
            "dropout": 0.1,           # Regularización estándar
    
    # Hiperparámetros de Entrenamiento
            "batch_size": 16,         # Reducción CRÍTICA (de 32 a 16) para compensar el tamaño del modelo
            "epochs": 20,             # Reducción drástica (de 100 a 20); los Transformers convergen rápido
            "learning_rate": 0.0001,  # Reducido (de 5e-4 a 1e-4) para estabilizar un modelo mucho más ancho
            "patience": 3,             # No pierdas tiempo esperando si no mejora en 3 épocas
            "use_amp": True             # Reduce a la mitad el tamaño de los tensores flotantes en VRAM
        },
        "dlinear": {
            "moving_avg_window": 25,  # Se mantiene (estándar del paper original para ETTh2)
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": 5
        }
    },
    "orchestration": {
        "models_to_run": ["randomForest"],
        "experiment_name": "experimentoWeatherRF96"
    }
}

#"sarimax", "randomForest", "xgboost_ci","xgboost_cd", "dlinear","itransformer",

def generate_yaml_config(filename='config.yaml'):
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Archivo {filename} generado exitosamente.")

if __name__ == "__main__":
    generate_yaml_config()
