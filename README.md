# Multivariate Time-Series Forecasting Benchmark Framework (MTS-Bench)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA_Enabled-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-GPU_Hist-red.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-ASHA_Pruning-blueviolet.svg)](https://optuna.org/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2.svg?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Resumen / Abstract

Este repositorio contiene el código fuente, canalizaciones de preprocesamiento y entornos de evaluación experimental desarrollados para la tesis de **Maestría en Ciencias en Computación**. El proyecto implementa un *framework* modular y reproducible diseñado para la evaluación comparativa y rigurosa de métodos de pronóstico en series de tiempo multivariadas bajo el paradigma **Multivariate-to-Multivariate (M)**.

El objetivo central es contrastar frontalmente modelos estadísticos autorregresivos clásicos, algoritmos de aprendizaje supervisado tabular basados en árboles de decisión y arquitecturas modernas de aprendizaje profundo (Deep Learning). El sistema evalúa la fidelidad del ajuste empírico reportando **MSE** y **MAE** (tanto en espacio normalizado como desnormalizado a escala original), analizando simultáneamente la huella computacional mediante el perfilado de tiempos de entrenamiento, latencia de inferencia y consumo de memoria VRAM/RAM.

---

## 2. Arquitectura del Sistema

El diseño del *framework* sigue principios de alta cohesión y bajo acoplamiento mediante interfaces abstractas y adaptadores estructurados:

```
┌────────────────────────────────────────────────────────┐
│             Data Preprocessing Pipeline                │
│  - Time Synchronization & Imputation                   │
│  - Chronological Split (Train / Val / Test)            │
│  - Scalers (StandardScaler / Target-Aware Rescaler)   │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌───────────────────┐       ┌───────────────────┐
   │ 2D Tabular/Series │       │  3D Long Formats  │
   │    Data Bundle    │       │ (unique_id, ds, y)│
   └─────────┬─────────┘       └─────────┬─────────┘
             │                           │
             ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│   Tabular / Classic     │ │      Deep Learning      │
│        Wrappers         │ │         Wrappers        │
│ ─────────────────────── │ │ ─────────────────────── │
│ - Auto-ARIMA            │ │ - DLinear               │
│ - Random Forest         │ │ - iTransformer          │
│ - XGBoost               │ │                         │
│   (CD vs. CI modes)     │ │ (AMP + GradAccum)       │
└────────────┬────────────┘ └────────────┬────────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│             Benchmarking & Evaluation Engine           │
│  - Metrics: MSE, MAE (Normalized & Unscaled)           │
│  - Computational Profiler (Execution Time & VRAM Peak) │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│             MLflow Tracking & Optuna HPO               │
│  - Artifact Logging, Metrics Logging & Model Registry  │
│  - Distributed Hyperband / ASHA Pruning Optimization   │
└────────────────────────────────────────────────────────┘
```

* **Dataset / Preprocesamiento:** Ingesta estandarizada, validación de integridad temporal, división cronológica estricta sin fuga de información (*lookahead bias*) y ajuste de escaladores de distribución.
* **Adaptadores de Datos:** Transformación matricial fluida entre arreglos tabulares bidimensionales $(N, D)$ con retardos (*lags*) y tensores tridimensionales en formato largo estandarizado (`unique_id`, `ds`, `y`).
* **Wrappers de Modelos (`BaseModelWrapper`):** Encapsulación unificada que desacopla la semántica de entrenamiento e inferencia multi-paso de las librerías base.
* **Orquestador y Motor de Evaluación:** Módulo encargado de gestionar la ejecución por lotes, asegurar la coherencia de dimensiones en las ventanas deslizantes y calcular métricas de error.
* **Capa de Registro y Optimización:** Integración con MLflow para el almacenamiento reproducible de artefactos y Optuna para el barrido eficiente del espacio de búsqueda de hiperparámetros.

---

## 3. Modelos Implementados

| Familia | Modelo | Características de Implementación | Estrategia Multivariada |
| :--- | :--- | :--- | :--- |
| **Estadística Clásica** | **Auto-ARIMA** | Optimización de órdenes $(p, d, q) \times (P, D, Q)_s$ mediante `pmdarima` con estacionalidad forzada. | Canales independientes ajustados en paralelo por variable. |
| **Supervisado Tabular** | **Random Forest** | Ensambles de regresión multi-salida con descomposición adaptativa de dependencias cruzadas. | Prueba A/B: *Channel Dependence* (MIMO) vs. *Channel Independence* (`multi_output_tree`). |
| **Supervisado Tabular** | **XGBoost** | Árboles con gradiente potenciado optimizados para cómputo masivo en coma flotante. | Prueba A/B: Descomposición univariada paralela vs. árboles vectoriales acoplados. |
| **Deep Learning** | **DLinear** | Descomposición de series en tendencia y estacionalidad acoplada a proyecciones lineales unívocas directas. | *Channel-Independent / Linear Projection*. |
| **Deep Learning** | **iTransformer** | Inversión del mecanismo de autoatención: las series temporales completas se proyectan como *tokens* individuales (*Variate-as-Token*). | Modelado de dependencias inter-variadas sobre espacios latentes temporales globales. |

---

## 4. Optimizaciones Técnicas Destacadas

### Mitigación de Fallos Out-Of-Memory (OOM) en Deep Learning
El entrenamiento de modelos basados en transformadores sobre secuencias multivariadas masivas (e.g., *Traffic*, *Electricity*) induce un sobrecosto severo en la memoria de la GPU ($O(V^2)$ o $O(T^2)$). Para solventarlo sin degradar la convergencia numérica:
* **Automatic Mixed Precision (AMP):** Implementación de `torch.cuda.amp.autocast()` para calcular la atención y activaciones en `float16` / `bfloat16`, preservando pesos maestros en `float32`. Reduce la huella de VRAM hasta en un **45%**.
* **Gradient Accumulation:** Desacopla el tamaño de lote computacional del tamaño de lote estadístico efectivo, permitiendo procesar ventanas con micro-lotes en tarjetas de consumo (e.g., NVIDIA RTX de 12 GB) sin desbordar memoria.

### Aceleración por GPU en Ensambles Tabulares
* **XGBoost `tree_method="hist"` con CUDA:** Los algoritmos tabulares evitan el cuello de botella del ordenamiento recursivo binario en CPU discretizando las variables en histogramas procesados paralelamente en GPU. Esto confiere aceleraciones de hasta **12x** en datasets de gran dimensión ($N > 10^5$).

### Optimización Bayesiana Eficiente (HPO)
* **ASHA Pruning:** Implementación del *Asynchronous Successive Halving Algorithm* a través de Optuna, deteniendo tempranamente los ensayos de hiperparámetros que divergen o muestran tasas de pérdida subóptimas durante las primeras épocas de entrenamiento.

---

## 5. Datasets Soportados (Benchmarks Estándar)

El repositorio incluye cargadores integrados para los conjuntos de datos canónicos del estado del arte:

1. **ETTh2 (Electricity Transformer Temperature):** Registro bianual horario/15-minutal de temperaturas de aceite y variables de carga eléctrica en transformadores.
2. **Weather:** 21 variables meteorológicas registradas con una frecuencia de 10 minutos por el Instituto Max Planck de Biogeoquímica.
3. **Traffic:** Mediciones horarias del Departamento de Transporte de California que describen tasas de ocupación de carriles en autopistas de la Bahía de San Francisco.
4. **Electricity:** Consumo horario de electricidad en kilovatios (kW) de 321 clientes recopilado entre 2012 y 2014.

---

## 6. Stack Tecnológico

* **Lenguaje:** Python 3.10+
* **Deep Learning Framework:** PyTorch & PyTorch Lightning
* **Librerías de Series de Tiempo:** NeuralForecast, pmdarima
* **Modelos Tabulares & ML:** Scikit-Learn, XGBoost (CUDA Hist)
* **Procesamiento de Datos:** NumPy, Pandas, SciPy
* **Optimización de Hiperparámetros:** Optuna (c/ ASHA Pruner)
* **Seguimiento Experimental:** MLflow

---

## 7. Instalación

Las siguientes instrucciones están optimizadas para sistemas Linux basados en Debian/Ubuntu con soporte de GPU NVIDIA (controladores CUDA instalados).

### Clonar el Repositorio
```bash
git clone https://github.com/TU_USUARIO/time-series-eval-framework.git
cd time-series-eval-framework
```

### Configuración del Entorno Virtual (Conda recomendado)
```bash
# Crear el entorno con Python 3.10
conda create -n ts-bench python=3.10 -y
conda activate ts-bench

# Instalar PyTorch con aceleración CUDA compatible (ejemplo para CUDA 11.8/12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalar requerimientos del sistema
pip install -r requirements.txt
```

---

## 8. Guía de Uso (Usage)

### Configuración Centralizada (`config.yaml`)
Todos los parámetros operacionales, horizontes de pronóstico, estrategias multivariadas y rutas de registro se definen en `config.yaml`:

```yaml
experiment:
  name: "MTS_Benchmark_MasterThesis"
  tracking_uri: "http://127.0.0.1:5000"

dataset:
  name: "Weather"
  target_strategy: "multivariate" # M (Multivariate-to-Multivariate)
  seq_len: 96                     # Lookback window (input_size)
  pred_len: 24                    # Forecast horizon (h)
  train_ratio: 0.7
  val_ratio: 0.1
  test_ratio: 0.2

models:
  xgboost:
    tree_method: "hist"
    device: "cuda"
    architecture: "channel_independence" # o "channel_dependence"
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6
  itransformer:
    d_model: 128
    n_heads: 8
    e_layers: 2
    batch_size: 32
    accumulate_grad_batches: 2
    precision: "16-mixed"
    max_epochs: 50
    patience: 5
```

### Ejecución del Pipeline Principal de Entrenamiento
Para entrenar y evaluar los modelos especificados sobre el dataset configurado:

```bash
python main.py --config config.yaml --model itransformer --dataset Weather
```

### Búsqueda de Hiperparámetros con Optuna y ASHA
Para inicializar el barrido bayesiano con poda anticipada de modelos ineficientes:

```bash
python optimize.py --config config.yaml --model xgboost --n_trials 50
```

### Visualización y Análisis en MLflow
Levanta el servidor local de seguimiento para examinar curvas de convergencia, métricas desnormalizadas y artefactos generados:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Navega a `http://localhost:5000` en tu navegador para interactuar con el panel de control.

---

## 9. Estructura del Proyecto

```text
time-series-eval-framework/
├── config/
│   ├── config.yaml               # Archivo maestro de configuración
│   └── search_spaces.yaml        # Espacios de búsqueda para Optuna
├── data/
│   ├── raw/                      # Datasets canónicos (ETTh2, Weather, etc.)
│   └── processed/                # Series estructuradas y normalizadas
├── src/
│   ├── data/
│   │   ├── dataset_loader.py     # Ingestores de series de tiempo
│   │   └── preprocessor.py       # Escalado, segmentación y validación temporal
│   ├── models/
│   │   ├── base.py               # Clase base abstracta BaseModelWrapper
│   │   ├── classic.py            # Adaptador Auto-ARIMA (pmdarima)
│   │   ├── tabular.py            # Random Forest & XGBoost (CD/CI architectures)
│   │   └── deep_learning.py      # DLinear & iTransformer (AMP, grad_acc)
│   ├── evaluation/
│   │   ├── metrics.py            # MSE/MAE normalizados y desnormalizados
│   │   └── profiler.py           # Perfilador de latencia y uso de memoria VRAM
│   └── utils/
│       ├── logger.py             # Configuración centralizada de logging
│       └── plotting.py           # Gráficas de horizonte y comparación
├── tests/                        # Pruebas unitarias de dimensionalidad y tensores
├── main.py                       # Orquestador del pipeline de entrenamiento y prueba
├── optimize.py                   # Motor de optimización bayesiana (Optuna + ASHA)
├── .gitignore                    # Reglas de exclusión para Git
├── requirements.txt              # Dependencias del ecosistema Python
└── README.md                     # Documentación técnica formal
```

---

## 10. Referencias Académicas

1. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). **iTransformer: Inverted Transformers Are Effective for Time Series Forecasting**. *International Conference on Learning Representations (ICLR)*.
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). **Are Transformers Effective for Time Series?** *AAAI Conference on Artificial Intelligence*, 37(9), 11121-11128.
3. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System**. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*.
