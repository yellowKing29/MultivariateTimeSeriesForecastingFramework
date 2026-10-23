import yaml

# Diccionario de configuración con hiperparámetros
config_data = {
    "global": {
        "dataset_path": "data/raw/traffic.csv",
        "look_back": 96,          # Aumentado para dar contexto suficiente a los modelos profundos
        "horizon": 24,            # Ajustado para evaluar predicción a múltiples pasos (1 día)
        "target_columns": None
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
            "n_estimators": 50,
            "max_depth": 8,
            "max_features": "sqrt",
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "n_jobs": -1
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "tree_method": "hist",    # CRÍTICO: Construye histogramas aproximados, usa 1/8 de la memoria
            "device": "cuda",
            "subsample": 0.3,         # Añadido para prevenir sobreajuste
            "colsample_bytree": 0.3   # Añadido para prevenir sobreajuste
        },
        "itransformer": {
            "d_model": 128,           # Reducido drásticamente para evitar OOM y sobreajuste
            "n_heads": 8,
            "epochs": 100,
            "batch_size": 32,
            "e_layers": 2,
            "dropout": 0.1,
            "learning_rate": 0.0005,  # Parámetro vital para convergencia en Transformers
            "patience": 5
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
        "models_to_run": ["dlinear"],
        "experiment_name": "experimentoTrafficDli24"
    }
}

#"sarimax", "randomForest",  "xgboost", "itransformer","dlinear"

def generate_yaml_config(filename='config.yaml'):
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"Archivo {filename} generado exitosamente.")

if __name__ == "__main__":
    generate_yaml_config()
