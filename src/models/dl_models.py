"""
Módulo models.dl_models
Implementación de los wrappers para los modelos Deep Learning.
Gestiona el ciclo de vida, hiperparámetros y el perfilado de tiempos de ejecución.


Implementación formal del modelo DLinear (Zeng et al., 2023) y su Wrapper para el framework.
Garantiza independencia de canales (Channel Independence) y descomposición de series de tiempo.
"""
import time
from typing import Any, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset, random_split


# Asumiendo que BaseModelWrapper está en src.core.interfaces
from src.core.interfaces import BaseModelWrapper

class iTransformerArchitecture(nn.Module):
    """
    Arquitectura iTransformer formal (Variate-as-Token).
    Incrusta las series completas (look-back) como tokens independientes y utiliza 
    Atención Multivariada para aprender las dependencias cruzadas entre variables.
    """

    def __init__(
        self, 
        seq_len: int, 
        pred_len: int, 
        d_model: int = 512, 
        n_heads: int = 8, 
        e_layers: int = 2, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Inversión de Tokens (Variate-as-Token)
        # Proyecta la historia completa de cada variable hacia el espacio latente d_model
        self.embedding = nn.Linear(seq_len, d_model)
        
        # 2. Atención Multivariada y Redes Feed-Forward (Transformer Encoder)
        # Se aplica LayerNorm y conexiones residuales por defecto en TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # 3. Feed-Forward Temporal (Proyector)
        # Decodifica las representaciones ocultas directamente hacia el horizonte de predicción
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante del modelo iTransformer.
        
        Args:
            x (torch.Tensor): Tensor de entrada dimensionalidad (Batch, Seq_Len, Variates).
            
        Returns:
            torch.Tensor: Tensor de predicción dimensionalidad (Batch, Pred_Len, Variates).
        """
        # Transponer para que las variables (Variates) actúen como la secuencia de tokens
        # Forma resultante: (Batch, Variates, Seq_Len)
        x_inv = x.permute(0, 2, 1)
        
        # Incrustación (Embedding): (Batch, Variates, d_model)
        x_emb = self.embedding(x_inv)
        
        # Atención cruzada entre variables: (Batch, Variates, d_model)
        # El encoder busca correlaciones espaciales, no temporales.
        enc_out = self.encoder(x_emb)
        
        # Proyección temporal: (Batch, Variates, Pred_Len)
        out = self.projector(enc_out)
        
        # Restaurar a la dimensionalidad estándar del framework: (Batch, Pred_Len, Variates)
        return out.permute(0, 2, 1)


class iTransformerWrapper: # Hereda de BaseModelWrapper
    """
    Wrapper arquitectónico para gestionar el ciclo de vida de iTransformer.
    Encapsula el manejo dinámico de hiperparámetros, optimización con AdamW, 
    Early Stopping, filtrado de Channel Independence, AMP, Acumulación de Gradientes 
    y perfilado de tiempos.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Extrae los hiperparámetros topológicos, de entrenamiento y de optimización de VRAM.
        """
        # Topología del modelo
        # --- CORRECCIÓN AQUÍ: Capturar las llaves exactas que inyecta main.py ---
        self.seq_len = kwargs.get('lookback_len', kwargs.get('look_back', 10))
        self.pred_len = kwargs.get('pred_len', kwargs.get('horizon', 1))
        # -------------------------------------------------------------------------    
        self.d_model = kwargs.get('d_model', 512)
        self.n_heads = kwargs.get('n_heads', 8)
        self.e_layers = kwargs.get('e_layers', 2)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Hiperparámetros de entrenamiento
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.patience = kwargs.get('patience', 5)
        
        # Parámetros de optimización de memoria (VRAM)
        self.accumulation_steps = kwargs.get('accumulation_steps', 1)
        self.use_amp = kwargs.get('use_amp', False)
        
        # Índices objetivo para Independencia de Canales dinámica
        self.target_indices = kwargs.get('target_indices', None)
        
        # Infraestructura PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = iTransformerArchitecture(
            seq_len=self.seq_len, 
            pred_len=self.pred_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        # AdamW es matemáticamente superior para Transformers debido a su manejo del Weight Decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Métricas de perfilado
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any], 
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'iTransformerWrapper':
        """
        Entrena el modelo iTransformer implementando validación interna para Early Stopping,
        Precisión Mixta Automática (AMP) y Acumulación de Gradientes.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        
        # Partición dinámica para Early Stopping (90% Train / 10% Val)
        total_size = len(dataset)
        val_size = max(int(0.1 * total_size), 1) if total_size > 10 else 0
        train_size = total_size - val_size
        
        if val_size > 0:
            train_data, val_data = random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        else:
            train_data = dataset
            val_loader = None
            
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        best_loss = float('inf')
        patience_counter = 0

        # Inicialización del escalador de gradientes para Precisión Mixta (AMP)
        scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad()
            
            num_batches = len(train_loader)
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)
                
                # Inferencia y pérdida en Precisión Mixta Automática
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(batch_X)
                    
                    # --- Filtro de Independencia de Canales (Channel Independence) ---
                    if self.target_indices is not None:
                        outputs = outputs[:, :, self.target_indices]
                    
                    raw_loss = self.criterion(outputs, batch_y)
                    loss = raw_loss / self.accumulation_steps
                
                # Backward pass escalado
                scaler.scale(loss).backward()
                
                # Paso de optimización condicional por Acumulación de Gradientes
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    scaler.unscale_(self.optimizer)
                    # Gradient Clipping: Práctica estándar en Transformers para evitar explosión de gradientes
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                
                # Acumular el valor real de la pérdida
                train_loss += raw_loss.item() * batch_X.size(0)
                
            train_loss /= len(train_data)
            
            # Liberación explícita de VRAM antes de la fase de validación
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Evaluación de Early Stopping y Pruning
            current_loss = train_loss
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device, non_blocking=True), batch_y.to(self.device, non_blocking=True)
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            outputs = self.model(batch_X)
                            
                            if getattr(self, 'target_indices', None) is not None:
                                outputs = outputs[:, :, self.target_indices]
                                
                            loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                
                val_loss /= val_size
                current_loss = val_loss
                
                # Integración con Optuna (ASHA Pruning)
                if optuna_trial is not None:
                    import optuna
                    optuna_trial.report(val_loss, epoch)
                    if optuna_trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            
            # Verificación de paciencia
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[iTransformer] Early stopping activado en la época {epoch + 1}")
                    break
                    
        self.training_time = time.time() - start_time
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Ejecuta inferencia vectorizada sobre el conjunto de prueba de forma segura,
        pre-alocando memoria en CPU mediante np.empty para prevenir errores OOM (Killed)
        en datasets masivos de alta dimensionalidad (ej. Traffic con 862 variables).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        start_time = time.time()
        
        # DataLoader predictivo
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        
        total_samples = len(X)
        
        # 1. Inferir dimensiones exactas de la salida usando un mini-lote de prueba
        sample_batch = next(iter(loader))[0].to(self.device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
            sample_out = self.model(sample_batch)
            if self.target_indices is not None:
                sample_out = sample_out[:, :, self.target_indices]
                
        pred_len = sample_out.shape[1]
        out_channels = sample_out.shape[2]
        
        # 2. Pre-alocar el arreglo final en la RAM del sistema (CPU) para cero fragmentación
        final_predictions = np.empty((total_samples, pred_len, out_channels), dtype=np.float32)
        
        current_idx = 0
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(batch_X)
                    
                    # --- Filtro de Independencia de Canales ---
                    if self.target_indices is not None:
                        outputs = outputs[:, :, self.target_indices]
                
                batch_size_current = batch_X.shape[0]
                # Inyección directa por rebanado de memoria (evita listas dinámicas y np.concatenate)
                final_predictions[current_idx : current_idx + batch_size_current] = outputs.float().cpu().numpy()
                current_idx += batch_size_current
                
        self.inference_time = time.time() - start_time
        return final_predictions

class MovingAverageBlock(nn.Module):
    """
    Bloque de promedio móvil unidimensional para extraer la tendencia temporal.
    Aplica padding dinámico para preservar la longitud de la secuencia original.
    """

    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor de dimensiones (Batch, Seq_Len, Features).
        Returns:
            torch.Tensor: Tensor suavizado de dimensiones (Batch, Seq_Len, Features).
        """
        # Calcular padding para mantener la longitud de la secuencia
        padding_front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        padding_back = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        
        # Ajuste para ventanas pares
        if self.kernel_size % 2 == 0:
            padding_back = torch.cat([x[:, -1:, :], padding_back], dim=1)
            
        x_padded = torch.cat([padding_front, x, padding_back], dim=1)
        
        # PyTorch AvgPool1d espera (Batch, Channels, Seq_Len)
        x_transposed = x_padded.permute(0, 2, 1)
        out = self.avg_pool(x_transposed)
        
        # Retornar a la dimensionalidad original (Batch, Seq_Len, Features)
        return out.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """
    Bloque de descomposición estructural de la serie de tiempo.
    Separa la señal en su componente de tendencia (Trend) y residual (Remainder).
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverageBlock(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Tensor de entrada (Batch, Seq_Len, Features).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Residual y Tendencia.
        """
        trend = self.moving_avg(x)
        remainder = x - trend
        return remainder, trend


class DLinearArchitecture(nn.Module):
    """
    Arquitectura DLinear formal. Emplea Channel Independence, procesando 
    las múltiples variables de manera concurrente pero independiente mediante proyecciones 
    lineales compartidas a lo largo de la dimensión temporal.
    """

    def __init__(self, seq_len: int, pred_len: int, moving_avg_window: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.decomposition = SeriesDecomposition(kernel_size=moving_avg_window)
        
        # Proyecciones lineales para mapear de Seq_Len a Pred_Len
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_remainder = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pase hacia adelante del modelo.
        
        Args:
            x (torch.Tensor): Tensor 3D (Batch, Seq_Len, Features).
            
        Returns:
            torch.Tensor: Tensor 3D de predicciones (Batch, Pred_Len, Features).
        """
        remainder, trend = self.decomposition(x)
        
        # Transponer para aplicar la capa lineal sobre la dimensión temporal: (Batch, Features, Seq_Len)
        trend_transposed = trend.permute(0, 2, 1)
        remainder_transposed = remainder.permute(0, 2, 1)
        
        # Proyección y retorno a la dimensionalidad original: (Batch, Pred_Len, Features)
        trend_forecast = self.linear_trend(trend_transposed).permute(0, 2, 1)
        remainder_forecast = self.linear_remainder(remainder_transposed).permute(0, 2, 1)
        
        return trend_forecast + remainder_forecast


class DLinearWrapper: # Hereda de BaseModelWrapper
    """
    Wrapper arquitectónico para gestionar el ciclo de vida de DLinear.
    Encapsula hiperparámetros, dataloaders, optimización, early stopping y perfilado.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Extrae hiperparámetros de arquitectura y entrenamiento inyectados dinámicamente.
        """
        # Hiperparámetros de Arquitectura
        self.seq_len = kwargs.get('look_back', 96)
        self.pred_len = kwargs.get('horizon', 24)
        self.moving_avg_window = kwargs.get('moving_avg_window', 25)
        # Agregar junto a la extracción de hiperparámetros
        self.target_indices = kwargs.get('target_indices', None)
        
        # Hiperparámetros de Entrenamiento
        self.epochs = kwargs.get('epochs', 50)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.patience = kwargs.get('patience', 5)
        
        # Infraestructura y Trazabilidad
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DLinearArchitecture(
            seq_len=self.seq_len, 
            pred_len=self.pred_len, 
            moving_avg_window=self.moving_avg_window
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def fit(
        self, 
        X: Union[np.ndarray, Any], 
        y: Union[np.ndarray, Any], 
        X_val: Optional[Union[np.ndarray, Any]] = None,
        y_val: Optional[Union[np.ndarray, Any]] = None,
        optuna_trial: Optional[Any] = None
    ) -> 'DLinearWrapper':
        """
        Entrena el modelo DLinear aplicando partición interna para Early Stopping.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        
        # Crear partición interna 90/10 para Early Stopping
        total_size = len(dataset)
        val_size = max(int(0.1 * total_size), 1) if total_size > 10 else 0
        train_size = total_size - val_size
        
        if val_size > 0:
            train_data, val_data = random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        else:
            train_data = dataset
            val_loader = None
            
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0

        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # FILTRO DE INDEPENDENCIA DE CANALES
                if self.target_indices is not None:
                    outputs = outputs[:, :, self.target_indices]
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_data)
            
            # Lógica de Early Stopping
            current_loss = train_loss
            # Evaluación de Early Stopping y Pruning
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        
                        if getattr(self, 'target_indices', None) is not None:
                            outputs = outputs[:, :, self.target_indices]
                            
                        loss = self.criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                
                val_loss /= val_size
                current_loss = val_loss
                
                # Integración con Optuna (ASHA Pruning)
                if optuna_trial is not None:
                    import optuna
                    optuna_trial.report(val_loss, epoch)
                    if optuna_trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                # Se podría guardar el state_dict óptimo aquí (torch.save)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping activado en la época {epoch + 1}")
                    break
                    
        self.training_time = time.time() - start_time
        return self

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Ejecuta inferencia vectorizada sobre nuevos datos en evaluación.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        start_time = time.time()
        
        # Procesamiento en bloque si cabe en memoria, o DataLoader para batching predictivo
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                # FILTRO DE INDEPENDENCIA DE CANALES
                if self.target_indices is not None:
                    outputs = outputs[:, :, self.target_indices]
                
                predictions.append(outputs.cpu().numpy())
                
        # Concatenar a arreglo numpy: (Muestras, Horizonte_Prediccion, Caracteristicas)
        final_predictions = np.concatenate(predictions, axis=0)
        
        self.inference_time = time.time() - start_time
        return final_predictions
