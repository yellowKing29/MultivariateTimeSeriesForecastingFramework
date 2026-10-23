"""
Módulo src.visualization.plotters
Generador de visualizaciones de grado científico (estilo IEEE/Springer) 
para la evaluación comparativa de modelos de pronóstico en series de tiempo.
"""

import os
from typing import Dict, List, Union, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def set_scientific_style() -> None:
    """
    Configura los parámetros globales de Matplotlib (rcParams) para cumplir
    con los estándares de publicación académica (IEEE, Springer).
    """
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.figsize': (7.16, 3.5),  # 7.16 pulgadas (ancho doble columna IEEE) x 3.5
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'axes.axisbelow': True,
        'pdf.fonttype': 42,  # Fuentes TrueType para incrustación en PDF
        'ps.fonttype': 42
    })

def plot_forecast_comparisons(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    time_index: Union[List[Any], np.ndarray, pd.DatetimeIndex],
    feature_idx: int,
    feature_name: str,
    save_path: str
) -> None:
    """
    Genera y guarda una gráfica comparativa de alta calidad (Ground Truth vs Modelos).

    Args:
        y_true (np.ndarray): Valores reales desnormalizados. Dim: (Horizonte, Caracteristicas).
        predictions_dict (Dict[str, np.ndarray]): Diccionario de predicciones por modelo.
        time_index (Union[List, np.ndarray, pd.DatetimeIndex]): Eje temporal para el horizonte.
        feature_idx (int): Índice de la columna/característica a graficar.
        feature_name (str): Etiqueta para el eje Y (ej. 'Temperatura (degC)').
        save_path (str): Ruta base donde se guardará la imagen (sin extensión).
    """
    # 1. Configurar el estilo global
    set_scientific_style()
    
    # 2. Inicializar la figura
    fig, ax = plt.subplots()
    
    # Extraer la característica específica del Ground Truth
    y_true_feat = y_true[:, feature_idx]
    
    # 3. Graficar el Ground Truth (Línea base negra, sólida y prominente)
    ax.plot(
        time_index, 
        y_true_feat, 
        color='black', 
        linestyle='-', 
        linewidth=2.0, 
        label='Ground Truth',
        zorder=10  # Asegura que la línea real siempre se dibuje por encima de las predicciones
    )
    
    # 4. Paleta de colores y estilos para los modelos (B/W & Colorblind friendly)
    # Usamos la paleta 'colorblind' de seaborn para alto contraste visual
    colors = sns.color_palette("colorblind", len(predictions_dict))
    
    # Distintos estilos de línea para legibilidad en impresión a escala de grises
    line_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    markers = ['o', 's', '^', 'D', 'v']
    
    # 5. Iterar y graficar cada modelo
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        y_pred_feat = y_pred[:, feature_idx]
        
        # Selección cíclica de estilos para soportar múltiples modelos
        color = colors[idx % len(colors)]
        l_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        # Aplicamos marcadores sutiles (markevery optimiza la densidad visual si hay muchos puntos)
        density = max(1, len(time_index) // 15)  # Mostrar máximo ~15 marcadores a lo largo de la línea
        
        ax.plot(
            time_index, 
            y_pred_feat, 
            color=color, 
            linestyle=l_style, 
            marker=marker,
            markersize=4,
            markevery=density,
            label=model_name.upper(),
            alpha=0.85,
            zorder=5
        )
    
    # 6. Formateo de Ejes y Etiquetas
    ax.set_ylabel(feature_name, fontweight='bold')
    ax.set_xlabel('Tiempo', fontweight='bold', labelpad=10)
    
    # Rotar las fechas en el eje X para evitar solapamiento si son DatetimeIndex
    fig.autofmt_xdate(rotation=45, ha='right')
    
    # 7. Configuración de la Leyenda
    # Se posiciona debajo de la gráfica (fuera del área de dibujo) para no tapar los datos.
    # ncol se ajusta dinámicamente según la cantidad de modelos para mantener un bloque horizontal.
    num_columns = min(3, len(predictions_dict) + 1)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.35), 
        ncol=num_columns, 
        frameon=True, 
        edgecolor='black',
        fancybox=False,
        shadow=False
    )
    
    # Ajuste automático de márgenes para incluir la leyenda externa
    plt.tight_layout()
    
    # 8. Exportación en formatos académicos requeridos
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    #pdf_path = f"{save_path}.pdf"
    png_path = f"{save_path}.png"
    
    #plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    plt.close(fig)
    #print(f"[Visualización] Gráfica guardada en: {pdf_path} y {png_path}")
    print(f"[Visualización] Gráfica guardada en: {png_path}")
    
def plot_forecast_with_zoom(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    time_index: Union[List[Any], np.ndarray, pd.DatetimeIndex],
    feature_idx: int,
    feature_name: str,
    zoom_start_idx: int,
    zoom_end_idx: int,
    save_path: str,
    inset_bounds: Tuple[float, float, float, float] = (0.05, 0.55, 0.35, 0.4)
) -> None:
    """
    Genera y guarda una gráfica comparativa de alta calidad con un recuadro de zoom (Inset Plot).

    Args:
        y_true (np.ndarray): Valores reales desnormalizados. Dim: (Horizonte, Caracteristicas).
        predictions_dict (Dict[str, np.ndarray]): Diccionario de predicciones por modelo.
        time_index (Union[List, np.ndarray, pd.DatetimeIndex]): Eje temporal para el horizonte.
        feature_idx (int): Índice de la columna/característica a graficar.
        feature_name (str): Etiqueta para el eje Y (ej. 'T (degC)').
        zoom_start_idx (int): Índice inicial para la región de zoom.
        zoom_end_idx (int): Índice final para la región de zoom.
        save_path (str): Ruta base donde se guardará la imagen (sin extensión).
        inset_bounds (Tuple[float, float, float, float]): Coordenadas y tamaño del recuadro de zoom 
            en fracciones de los ejes (x0, y0, ancho, alto). Permite mover el recuadro para no tapar datos.
    """
    set_scientific_style()
    fig, ax = plt.subplots()
    
    y_true_feat = y_true[:, feature_idx]
    
    # 1. Eje Principal: Graficar Ground Truth
    ax.plot(
        time_index, y_true_feat, color='black', linestyle='-', 
        linewidth=2.0, label='Ground Truth', zorder=10
    )
    
    colors = sns.color_palette("colorblind", len(predictions_dict))
    line_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    markers = ['o', 's', '^', 'D', 'v']
    density = max(1, len(time_index) // 15)
    
    # 2. Eje Principal: Graficar Modelos
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        y_pred_feat = y_pred[:, feature_idx]
        ax.plot(
            time_index, y_pred_feat, 
            color=colors[idx % len(colors)], 
            linestyle=line_styles[idx % len(line_styles)], 
            marker=markers[idx % len(markers)],
            markersize=4, markevery=density, label=model_name.upper(),
            alpha=0.85, zorder=5
        )
        
    ax.set_ylabel(feature_name, fontweight='bold')
    ax.set_xlabel('Tiempo', fontweight='bold', labelpad=10)
    fig.autofmt_xdate(rotation=45, ha='right')

    # 3. Construcción del Recuadro de Zoom (Inset Plot)
    axins = ax.inset_axes(inset_bounds)
    
    # Re-graficar Ground Truth en el inset
    axins.plot(time_index, y_true_feat, color='black', linestyle='-', linewidth=2.0, zorder=10)
    
    # Re-graficar Modelos en el inset (sin marcadores para evitar saturación en el zoom)
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        y_pred_feat = y_pred[:, feature_idx]
        axins.plot(
            time_index, y_pred_feat, 
            color=colors[idx % len(colors)], 
            linestyle=line_styles[idx % len(line_styles)], 
            alpha=0.85, zorder=5
        )
        
    # Extraer límites temporales y de valor para el recuadro
    x_zoom_start = time_index[zoom_start_idx]
    x_zoom_end = time_index[zoom_end_idx]
    
    # Calcular el rango Y dinámico basado solo en los datos reales dentro de la ventana de zoom
    y_zoom_slice = y_true_feat[zoom_start_idx:zoom_end_idx]
    y_margin = (np.max(y_zoom_slice) - np.min(y_zoom_slice)) * 0.15
    y_zoom_start = np.min(y_zoom_slice) - y_margin
    y_zoom_end = np.max(y_zoom_slice) + y_margin

    axins.set_xlim(x_zoom_start, x_zoom_end)
    axins.set_ylim(y_zoom_start, y_zoom_end)
    
    # Ocultar etiquetas (ticks) del recuadro para mantener limpieza visual
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axins.tick_params(bottom=False, left=False)
    
    # Dibujar líneas de conexión entre el recuadro y la gráfica principal
    # loc1 y loc2 definen qué esquinas conectar (1: superior derecha, 2: superior izquierda, etc.)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", linestyle='--', zorder=4)

    # 4. Configuración de Leyenda y Guardado
    num_columns = min(3, len(predictions_dict) + 1)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.35), 
        ncol=num_columns, frameon=True, edgecolor='black',
        fancybox=False, shadow=False
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[Visualización] Gráfica con zoom guardada en: {save_path}.pdf/.png")
