#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador Avanzado de Experimentos ABC
Welded Beam Design Problem

Este script genera visualizaciones completas de los experimentos ABC:
1. Selección de experimento procesado
2. Estadísticas descriptivas comparativas
3. Análisis de distribuciones
4. Mapas de calor de interacciones
5. Análisis de convergencia
6. CSV completo con estadísticas de todas las 3^5 combinaciones
7. Gráficos de rendimiento y factibilidad

CARACTERÍSTICAS:
- Estilo visual consistente con el análisis estadístico
- Paleta de colores armonizada
- CSV detallado con estadísticas descriptivas de las 243 combinaciones
- Resumen ejecutivo con recomendaciones
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from scipy import stats
from scipy.stats import pearsonr
import itertools

# Importar módulos del proyecto
try:
    from Hive import Hive, Constraints, Utilities
except ImportError:
    print("Error: No se pueden importar los módulos ABC. Asegúrate de que estén en la ruta correcta.")
    exit()

# Configuración de visualización (consistente con análisis estadístico)
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Paleta de colores consistente con el análisis estadístico
COLORES_PRINCIPALES = {
    'azul_claro': '#87CEEB',
    'azul_medio': '#4682B4', 
    'verde': '#32CD32',
    'rojo': '#DC143C',
    'naranja': '#FF8C00',
    'morado': '#9370DB',
    'gris': '#696969'
}

class VisualizadorExperimentos:
    """Clase para generar visualizaciones avanzadas de experimentos ABC."""
    
    def __init__(self, archivo_master_table):
        """
        Inicializa el visualizador cargando los datos.
        
        Args:
            archivo_master_table (str): Ruta al archivo master_table.csv
        """
        self.datos = pd.read_csv(archivo_master_table)
        self.parametros = ['numb_bees', 'max_itrs', 'p_f', 'limit', 'modification_rate']
        self.variable_respuesta = 'fitness_final'
        
        # Separar datos factibles y no factibles
        self.datos_factibles = self.datos[self.datos['factible'] == True].copy()
        self.datos_no_factibles = self.datos[self.datos['factible'] == False].copy()
        
        # Crear carpeta para guardar visualizaciones
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.carpeta_visualizaciones = f"visualizaciones/visualizaciones_{timestamp}"
        os.makedirs(self.carpeta_visualizaciones, exist_ok=True)
        
        print(f"📊 Datos cargados:")
        print(f"   • Total de experimentos: {len(self.datos)}")
        print(f"   • Experimentos factibles: {len(self.datos_factibles)} ({len(self.datos_factibles)/len(self.datos)*100:.1f}%)")
        print(f"   • Experimentos no factibles: {len(self.datos_no_factibles)} ({len(self.datos_no_factibles)/len(self.datos)*100:.1f}%)")
        print(f"📁 Visualizaciones se guardarán en: {self.carpeta_visualizaciones}")
        
        # Calcular estadísticas básicas
        self._calcular_estadisticas_basicas()
    
    def _calcular_estadisticas_basicas(self):
        """Calcula estadísticas básicas de los datos."""
        if len(self.datos_factibles) > 0:
            self.mejor_fitness = self.datos_factibles[self.variable_respuesta].min()
            self.mejor_configuracion = self.datos_factibles.loc[
                self.datos_factibles[self.variable_respuesta].idxmin()
            ]
            self.fitness_promedio = self.datos_factibles[self.variable_respuesta].mean()
            self.fitness_std = self.datos_factibles[self.variable_respuesta].std()
            print(f"✅ Mejor fitness encontrado: {self.mejor_fitness:.6f}")
        else:
            print("⚠️ No hay soluciones factibles en los datos.")
            self.mejor_fitness = None
            self.mejor_configuracion = None
    
    def generar_todas_visualizaciones(self):
        """Genera todas las visualizaciones disponibles."""
        print("\n🎨 Generando visualizaciones completas...")
        print("="*60)
        
        # 1. Estadísticas descriptivas
        self.grafico_estadisticas_descriptivas()
        
        # 2. Distribuciones por parámetro
        self.grafico_distribuciones_parametros()
        
        # 3. Mapas de calor de interacciones
        self.grafico_mapas_calor_interacciones()
        
        # 4. Análisis de violín (distribuciones detalladas)
        self.grafico_violin_distribuciones()
        
        # 5. Análisis de factibilidad
        self.grafico_analisis_factibilidad()
        
        # 6. Rendimiento vs tiempo de ejecución
        self.grafico_rendimiento_tiempo()
        
        # 7. Análisis de convergencia por configuración
        self.grafico_analisis_convergencia()
        
        # 8. Scatter plots multidimensionales
        self.grafico_scatter_multidimensional()
        
        # 9. Análisis de rankings
        self.grafico_rankings_parametros()
        
        # 10. Generar CSV con estadísticas de todas las combinaciones
        self.generar_csv_estadisticas_combinaciones()
        
        print(f"\n✅ Todas las visualizaciones completadas!")
        print(f"📁 Archivos guardados en: {self.carpeta_visualizaciones}")
    
    def grafico_estadisticas_descriptivas(self):
        """Genera gráfico de estadísticas descriptivas por parámetro."""
        print("📊 Generando estadísticas descriptivas...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, param in enumerate(self.parametros):
            # Calcular estadísticas por nivel del parámetro
            stats_param = self.datos_factibles.groupby(param)[self.variable_respuesta].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).reset_index()
            
            # Gráfico de barras para media con barras de error (colores consistentes)
            x_pos = range(len(stats_param))
            bars = axes[i].bar(x_pos, stats_param['mean'], 
                              yerr=stats_param['std'], 
                              capsize=5, alpha=0.8,
                              color=COLORES_PRINCIPALES['azul_medio'])
            
            # Añadir puntos para min y max
            axes[i].scatter(x_pos, stats_param['min'], color=COLORES_PRINCIPALES['rojo'], 
                           marker='v', s=60, label='Mínimo', zorder=5)
            axes[i].scatter(x_pos, stats_param['max'], color=COLORES_PRINCIPALES['verde'], 
                           marker='^', s=60, label='Máximo', zorder=5)
            axes[i].scatter(x_pos, stats_param['median'], color=COLORES_PRINCIPALES['naranja'], 
                           marker='s', s=60, label='Mediana', zorder=5)
            
            axes[i].set_title(f'Estadísticas: {param}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(f'Niveles de {param}', fontsize=12)
            axes[i].set_ylabel('Fitness', fontsize=12)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(stats_param[param], rotation=45)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # Añadir número de muestras
            for j, count in enumerate(stats_param['count']):
                axes[i].text(j, stats_param['mean'].iloc[j] + stats_param['std'].iloc[j], 
                           f'n={count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Ocultar el último subplot si no se usa
        if len(self.parametros) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/01_estadisticas_descriptivas.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_distribuciones_parametros(self):
        """Genera gráficos de distribución para cada parámetro."""
        print("📈 Generando distribuciones por parámetros...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # Colores para diferentes niveles
        colors_palette = [COLORES_PRINCIPALES['azul_medio'], COLORES_PRINCIPALES['verde'], 
                         COLORES_PRINCIPALES['naranja'], COLORES_PRINCIPALES['morado']]
        
        for i, param in enumerate(self.parametros):
            # Histograma con kde para cada nivel del parámetro
            niveles = sorted(self.datos_factibles[param].unique())
            
            for j, nivel in enumerate(niveles):
                datos_nivel = self.datos_factibles[self.datos_factibles[param] == nivel][self.variable_respuesta]
                color = colors_palette[j % len(colors_palette)]
                
                axes[i].hist(datos_nivel, alpha=0.6, label=f'{param}={nivel}', bins=20, 
                            density=True, color=color, edgecolor='black', linewidth=0.5)
                
                # Añadir KDE
                if len(datos_nivel) > 1:
                    kde_x = np.linspace(datos_nivel.min(), datos_nivel.max(), 100)
                    kde = stats.gaussian_kde(datos_nivel)
                    axes[i].plot(kde_x, kde(kde_x), linewidth=2.5, color=color)
            
            axes[i].set_title(f'Distribución de Fitness por {param}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Fitness', fontsize=12)
            axes[i].set_ylabel('Densidad', fontsize=12)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        if len(self.parametros) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/02_distribuciones_parametros.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_mapas_calor_interacciones(self):
        """Genera mapas de calor para analizar interacciones entre parámetros."""
        print("🔥 Generando mapas de calor de interacciones...")
        
        # Seleccionar pares de parámetros más importantes
        pares_parametros = [
            ('numb_bees', 'max_itrs'),
            ('limit', 'modification_rate'),
            ('p_f', 'numb_bees'),
            ('max_itrs', 'limit')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (param1, param2) in enumerate(pares_parametros):
            # Crear tabla pivot para el mapa de calor
            pivot_table = self.datos_factibles.pivot_table(
                values=self.variable_respuesta,
                index=param1,
                columns=param2,
                aggfunc='mean'
            )
            
            # Crear mapa de calor
            sns.heatmap(pivot_table, 
                       annot=True, 
                       fmt='.4f',
                       cmap='viridis_r',  # Colores invertidos (verde=mejor)
                       ax=axes[i],
                       cbar_kws={'label': 'Fitness Promedio'},
                       linewidths=0.5)
            
            axes[i].set_title(f'Interacción: {param1} × {param2}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(param2, fontsize=12)
            axes[i].set_ylabel(param1, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/03_mapas_calor_interacciones.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_violin_distribuciones(self):
        """Genera gráficos de violín para mostrar distribuciones detalladas."""
        print("🎻 Generando gráficos de violín...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, param in enumerate(self.parametros):
            sns.violinplot(data=self.datos_factibles, 
                          x=param, 
                          y=self.variable_respuesta, 
                          ax=axes[i],
                          inner='box',
                          palette=[COLORES_PRINCIPALES['azul_claro']] * len(self.datos_factibles[param].unique()))
            
            axes[i].set_title(f'Distribución Detallada: {param}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(param, fontsize=12)
            axes[i].set_ylabel('Fitness', fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # Añadir puntos para mejores valores (estrella roja)
            mejores = self.datos_factibles.nsmallest(10, self.variable_respuesta)
            mejores_param = mejores[mejores[param].isin(self.datos_factibles[param].unique())]
            
            for _, row in mejores_param.iterrows():
                axes[i].scatter(row[param], row[self.variable_respuesta], 
                              color=COLORES_PRINCIPALES['rojo'], s=80, alpha=0.9, 
                              marker='*', edgecolors='black', linewidth=1, zorder=5)
        
        if len(self.parametros) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/04_violin_distribuciones.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_analisis_factibilidad(self):
        """Analiza la tasa de factibilidad por parámetro."""
        print("✅ Generando análisis de factibilidad...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, param in enumerate(self.parametros):
            # Calcular tasa de factibilidad por nivel
            factibilidad = self.datos.groupby(param).agg({
                'factible': ['sum', 'count']
            }).reset_index()
            
            factibilidad.columns = [param, 'factibles', 'total']
            factibilidad['tasa_factibilidad'] = factibilidad['factibles'] / factibilidad['total'] * 100
            
            # Gráfico de barras para tasa de factibilidad
            bars = axes[i].bar(range(len(factibilidad)), 
                              factibilidad['tasa_factibilidad'],
                              alpha=0.8,
                              color=[COLORES_PRINCIPALES['verde'] if x >= 80 else 
                                    COLORES_PRINCIPALES['naranja'] if x >= 50 else 
                                    COLORES_PRINCIPALES['rojo'] for x in factibilidad['tasa_factibilidad']],
                              edgecolor='black', linewidth=0.5)
            
            axes[i].set_title(f'Tasa de Factibilidad: {param}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(f'Niveles de {param}', fontsize=12)
            axes[i].set_ylabel('Tasa de Factibilidad (%)', fontsize=12)
            axes[i].set_xticks(range(len(factibilidad)))
            axes[i].set_xticklabels(factibilidad[param], rotation=45)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 100)
            
            # Añadir etiquetas con valores
            for j, (bar, valor) in enumerate(zip(bars, factibilidad['tasa_factibilidad'])):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{valor:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        if len(self.parametros) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/05_analisis_factibilidad.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_rendimiento_tiempo(self):
        """Analiza la relación entre rendimiento y tiempo de ejecución."""
        print("⏱️ Generando análisis de rendimiento vs tiempo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scatter plot Fitness vs Tiempo
        scatter = axes[0,0].scatter(self.datos_factibles['tiempo_ejecucion'], 
                                   self.datos_factibles[self.variable_respuesta],
                                   alpha=0.6, c=self.datos_factibles['numb_bees'], 
                                   cmap='viridis', s=40, edgecolors='black', linewidth=0.5)
        axes[0,0].set_xlabel('Tiempo de Ejecución (s)', fontsize=12)
        axes[0,0].set_ylabel('Fitness', fontsize=12)
        axes[0,0].set_title('Fitness vs Tiempo de Ejecución', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Añadir colorbar
        cbar = plt.colorbar(scatter, ax=axes[0,0])
        cbar.set_label('Número de Abejas', fontsize=11)
        
        # 2. Boxplot tiempo por número de abejas
        bp1 = axes[0,1].boxplot([self.datos_factibles[self.datos_factibles['numb_bees'] == val]['tiempo_ejecucion'] 
                                for val in sorted(self.datos_factibles['numb_bees'].unique())],
                               labels=sorted(self.datos_factibles['numb_bees'].unique()),
                               patch_artist=True)
        
        # Colorear cajas
        colors = [COLORES_PRINCIPALES['azul_claro'], COLORES_PRINCIPALES['verde'], COLORES_PRINCIPALES['naranja']]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0,1].set_title('Tiempo de Ejecución por Número de Abejas', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Número de Abejas', fontsize=12)
        axes[0,1].set_ylabel('Tiempo de Ejecución (s)', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Boxplot tiempo por iteraciones
        bp2 = axes[1,0].boxplot([self.datos_factibles[self.datos_factibles['max_itrs'] == val]['tiempo_ejecucion'] 
                                for val in sorted(self.datos_factibles['max_itrs'].unique())],
                               labels=sorted(self.datos_factibles['max_itrs'].unique()),
                               patch_artist=True)
        
        # Colorear cajas
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1,0].set_title('Tiempo de Ejecución por Iteraciones Máximas', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Iteraciones Máximas', fontsize=12)
        axes[1,0].set_ylabel('Tiempo de Ejecución (s)', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Eficiencia (1/fitness) vs tiempo
        eficiencia = 1 / self.datos_factibles[self.variable_respuesta]
        scatter2 = axes[1,1].scatter(self.datos_factibles['tiempo_ejecucion'], eficiencia,
                                    alpha=0.6, c=self.datos_factibles['max_itrs'], 
                                    cmap='plasma', s=40, edgecolors='black', linewidth=0.5)
        axes[1,1].set_xlabel('Tiempo de Ejecución (s)', fontsize=12)
        axes[1,1].set_ylabel('Eficiencia (1/Fitness)', fontsize=12)
        axes[1,1].set_title('Eficiencia vs Tiempo de Ejecución', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # Añadir colorbar
        cbar2 = plt.colorbar(scatter2, ax=axes[1,1])
        cbar2.set_label('Iteraciones Máximas', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/06_rendimiento_tiempo.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_analisis_convergencia(self):
        """Analiza los valores de convergencia por configuración."""
        print("📉 Generando análisis de convergencia...")
        
        # Filtrar datos con convergencia válida
        datos_convergencia = self.datos_factibles[
            self.datos_factibles['convergencia_final'].notna()
        ].copy()
        
        if len(datos_convergencia) == 0:
            print("⚠️ No hay datos de convergencia disponibles.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Convergencia vs Fitness final
        axes[0,0].scatter(datos_convergencia['convergencia_final'], 
                         datos_convergencia[self.variable_respuesta],
                         alpha=0.6, s=40, color=COLORES_PRINCIPALES['azul_medio'],
                         edgecolors='black', linewidth=0.5)
        axes[0,0].plot([datos_convergencia['convergencia_final'].min(), 
                       datos_convergencia['convergencia_final'].max()],
                      [datos_convergencia['convergencia_final'].min(), 
                       datos_convergencia['convergencia_final'].max()], 
                      color=COLORES_PRINCIPALES['rojo'], linestyle='--', alpha=0.8, linewidth=2)
        axes[0,0].set_xlabel('Convergencia Final', fontsize=12)
        axes[0,0].set_ylabel('Fitness Final', fontsize=12)
        axes[0,0].set_title('Convergencia vs Fitness Final', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Distribución de convergencia por número de abejas
        bp1 = axes[0,1].boxplot([datos_convergencia[datos_convergencia['numb_bees'] == val]['convergencia_final'] 
                                for val in sorted(datos_convergencia['numb_bees'].unique())],
                               labels=sorted(datos_convergencia['numb_bees'].unique()),
                               patch_artist=True)
        
        colors = [COLORES_PRINCIPALES['azul_claro'], COLORES_PRINCIPALES['verde'], COLORES_PRINCIPALES['naranja']]
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0,1].set_title('Convergencia por Número de Abejas', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Número de Abejas', fontsize=12)
        axes[0,1].set_ylabel('Convergencia Final', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Distribución de convergencia por iteraciones
        bp2 = axes[1,0].boxplot([datos_convergencia[datos_convergencia['max_itrs'] == val]['convergencia_final'] 
                                for val in sorted(datos_convergencia['max_itrs'].unique())],
                               labels=sorted(datos_convergencia['max_itrs'].unique()),
                               patch_artist=True)
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1,0].set_title('Convergencia por Iteraciones Máximas', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Iteraciones Máximas', fontsize=12)
        axes[1,0].set_ylabel('Convergencia Final', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Mapa de calor convergencia vs parámetros
        pivot_conv = datos_convergencia.pivot_table(
            values='convergencia_final',
            index='numb_bees',
            columns='max_itrs',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_conv, annot=True, fmt='.4f', cmap='viridis_r', ax=axes[1,1],
                   cbar_kws={'label': 'Convergencia Final'}, linewidths=0.5)
        axes[1,1].set_title('Convergencia: Abejas × Iteraciones', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Iteraciones Máximas', fontsize=12)
        axes[1,1].set_ylabel('Número de Abejas', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/07_analisis_convergencia.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_scatter_multidimensional(self):
        """Genera scatter plots multidimensionales."""
        print("🎯 Generando scatter plots multidimensionales...")
        
        # Seleccionar las mejores configuraciones para análisis
        top_10_percent = int(len(self.datos_factibles) * 0.1)
        mejores_datos = self.datos_factibles.nsmallest(top_10_percent, self.variable_respuesta)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 3D-like plot: numb_bees vs max_itrs con color por fitness
        scatter1 = axes[0,0].scatter(mejores_datos['numb_bees'], 
                                    mejores_datos['max_itrs'],
                                    c=mejores_datos[self.variable_respuesta],
                                    s=mejores_datos['limit']*3,  # Tamaño por limit
                                    alpha=0.7, cmap='viridis_r',
                                    edgecolors='black', linewidth=0.5)
        axes[0,0].set_xlabel('Número de Abejas', fontsize=12)
        axes[0,0].set_ylabel('Iteraciones Máximas', fontsize=12)
        axes[0,0].set_title('Top 10%: Abejas vs Iteraciones\n(Color=Fitness, Tamaño=Limit)', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0,0], label='Fitness')
        
        # 2. p_f vs modification_rate
        scatter2 = axes[0,1].scatter(mejores_datos['p_f'], 
                                    mejores_datos['modification_rate'],
                                    c=mejores_datos[self.variable_respuesta],
                                    s=mejores_datos['numb_bees']/3,
                                    alpha=0.7, cmap='plasma_r',
                                    edgecolors='black', linewidth=0.5)
        axes[0,1].set_xlabel('Probabilidad p_f', fontsize=12)
        axes[0,1].set_ylabel('Modification Rate', fontsize=12)
        axes[0,1].set_title('Top 10%: p_f vs Modification Rate\n(Color=Fitness, Tamaño=Abejas)', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0,1], label='Fitness')
        
        # 3. Variables de diseño: x1 vs x3
        scatter3 = axes[1,0].scatter(mejores_datos['x1'], 
                                    mejores_datos['x3'],
                                    c=mejores_datos[self.variable_respuesta],
                                    s=60, alpha=0.7, cmap='viridis_r',
                                    edgecolors='black', linewidth=0.5)
        axes[1,0].set_xlabel('x1 (altura soldadura)', fontsize=12)
        axes[1,0].set_ylabel('x3 (altura barra)', fontsize=12)
        axes[1,0].set_title('Variables de Diseño: x1 vs x3', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[1,0], label='Fitness')
        
        # 4. Variables de diseño: x2 vs x4
        scatter4 = axes[1,1].scatter(mejores_datos['x2'], 
                                    mejores_datos['x4'],
                                    c=mejores_datos[self.variable_respuesta],
                                    s=60, alpha=0.7, cmap='viridis_r',
                                    edgecolors='black', linewidth=0.5)
        axes[1,1].set_xlabel('x2 (longitud soldadura)', fontsize=12)
        axes[1,1].set_ylabel('x4 (espesor barra)', fontsize=12)
        axes[1,1].set_title('Variables de Diseño: x2 vs x4', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=axes[1,1], label='Fitness')
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/08_scatter_multidimensional.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def grafico_rankings_parametros(self):
        """Genera análisis de rankings de parámetros."""
        print("🏆 Generando rankings de parámetros...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Ranking de niveles por parámetro (promedio)
        rankings_data = []
        for param in self.parametros:
            param_stats = self.datos_factibles.groupby(param)[self.variable_respuesta].agg([
                'mean', 'count'
            ]).reset_index()
            param_stats['parametro'] = param
            param_stats['nivel'] = param_stats[param].astype(str)
            rankings_data.append(param_stats)
        
        rankings_df = pd.concat(rankings_data, ignore_index=True)
        
        # Crear gráfico de rankings
        best_levels = rankings_df.loc[rankings_df.groupby('parametro')['mean'].idxmin()]
        
        bars1 = axes[0,0].barh(range(len(best_levels)), best_levels['mean'], 
                              color=[COLORES_PRINCIPALES['verde'], COLORES_PRINCIPALES['azul_medio'], 
                                    COLORES_PRINCIPALES['naranja'], COLORES_PRINCIPALES['morado'],
                                    COLORES_PRINCIPALES['azul_claro']], alpha=0.8)
        axes[0,0].set_yticks(range(len(best_levels)))
        axes[0,0].set_yticklabels([f"{row['parametro']}\n({row['nivel']})" 
                                  for _, row in best_levels.iterrows()], fontsize=11)
        axes[0,0].set_xlabel('Mejor Fitness Promedio', fontsize=12)
        axes[0,0].set_title('Mejores Niveles por Parámetro', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar in bars1:
            width = bar.get_width()
            axes[0,0].text(width, bar.get_y() + bar.get_height()/2.,
                          f'{width:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # 2. Variabilidad (std) por parámetro
        variabilidad = []
        for param in self.parametros:
            std_param = self.datos_factibles.groupby(param)[self.variable_respuesta].std().mean()
            variabilidad.append({'parametro': param, 'std_promedio': std_param})
        
        var_df = pd.DataFrame(variabilidad).sort_values('std_promedio')
        
        bars2 = axes[0,1].bar(range(len(var_df)), var_df['std_promedio'],
                             color=COLORES_PRINCIPALES['azul_medio'], alpha=0.8,
                             edgecolor='black', linewidth=0.5)
        axes[0,1].set_xticks(range(len(var_df)))
        axes[0,1].set_xticklabels(var_df['parametro'], rotation=45, fontsize=11)
        axes[0,1].set_ylabel('Desviación Estándar Promedio', fontsize=12)
        axes[0,1].set_title('Variabilidad por Parámetro', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Correlaciones entre parámetros y fitness
        correlaciones = []
        for param in self.parametros:
            corr, p_val = pearsonr(self.datos_factibles[param], 
                                  self.datos_factibles[self.variable_respuesta])
            correlaciones.append({'parametro': param, 'correlacion': abs(corr), 'p_valor': p_val})
        
        corr_df = pd.DataFrame(correlaciones).sort_values('correlacion', ascending=False)
        
        colors = [COLORES_PRINCIPALES['rojo'] if p < 0.05 else COLORES_PRINCIPALES['gris'] for p in corr_df['p_valor']]
        bars3 = axes[1,0].bar(range(len(corr_df)), corr_df['correlacion'], color=colors, alpha=0.8,
                             edgecolor='black', linewidth=0.5)
        axes[1,0].set_xticks(range(len(corr_df)))
        axes[1,0].set_xticklabels(corr_df['parametro'], rotation=45, fontsize=11)
        axes[1,0].set_ylabel('|Correlación| con Fitness', fontsize=12)
        axes[1,0].set_title('Correlaciones con Fitness\n(Rojo=Significativo p<0.05)', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Matriz de correlación entre parámetros
        corr_matrix = self.datos_factibles[self.parametros + [self.variable_respuesta]].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=axes[1,1], linewidths=0.5,
                   cbar_kws={'label': 'Correlación'})
        axes[1,1].set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_visualizaciones}/09_rankings_parametros.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_csv_estadisticas_combinaciones(self):
        """Genera CSV con estadísticas descriptivas de todas las combinaciones de parámetros."""
        print("📊 Generando CSV con estadísticas de todas las combinaciones...")
        
        # Agrupar por combinación de parámetros
        grupo_combinaciones = self.datos.groupby(self.parametros)
        
        estadisticas_completas = []
        
        for nombre_combo, grupo in grupo_combinaciones:
            # Filtrar datos factibles y no factibles
            factibles = grupo[grupo['factible'] == True]
            no_factibles = grupo[grupo['factible'] == False]
            
            # Crear diccionario con la combinación de parámetros
            stats_combo = {
                'id_combinacion': grupo['id_combinacion'].iloc[0] if 'id_combinacion' in grupo.columns else f"combo_{hash(nombre_combo) % 1000}",
                'numb_bees': nombre_combo[0],
                'max_itrs': nombre_combo[1], 
                'p_f': nombre_combo[2],
                'limit': nombre_combo[3],
                'modification_rate': nombre_combo[4],
            }
            
            # Estadísticas generales
            stats_combo.update({
                'total_experimentos': len(grupo),
                'experimentos_factibles': len(factibles),
                'experimentos_no_factibles': len(no_factibles),
                'tasa_factibilidad': (len(factibles) / len(grupo)) * 100,
            })
            
            # Estadísticas de fitness (solo para factibles)
            if len(factibles) > 0:
                fitness_factibles = factibles[self.variable_respuesta]
                stats_combo.update({
                    'fitness_count': len(fitness_factibles),
                    'fitness_mean': fitness_factibles.mean(),
                    'fitness_std': fitness_factibles.std(),
                    'fitness_median': fitness_factibles.median(),
                    'fitness_min': fitness_factibles.min(),
                    'fitness_max': fitness_factibles.max(),
                    'fitness_q25': fitness_factibles.quantile(0.25),
                    'fitness_q75': fitness_factibles.quantile(0.75),
                    'fitness_iqr': fitness_factibles.quantile(0.75) - fitness_factibles.quantile(0.25),
                    'fitness_cv': (fitness_factibles.std() / fitness_factibles.mean()) * 100 if fitness_factibles.mean() != 0 else np.nan,  # Coeficiente de variación
                })
            else:
                # Si no hay factibles, llenar con valores nulos
                stats_combo.update({
                    'fitness_count': 0,
                    'fitness_mean': np.nan,
                    'fitness_std': np.nan,
                    'fitness_median': np.nan,
                    'fitness_min': np.nan,
                    'fitness_max': np.nan,
                    'fitness_q25': np.nan,
                    'fitness_q75': np.nan,
                    'fitness_iqr': np.nan,
                    'fitness_cv': np.nan,
                })
            
            # Estadísticas de tiempo de ejecución
            if 'tiempo_ejecucion' in grupo.columns:
                tiempo_total = grupo['tiempo_ejecucion']
                stats_combo.update({
                    'tiempo_mean': tiempo_total.mean(),
                    'tiempo_std': tiempo_total.std(),
                    'tiempo_median': tiempo_total.median(),
                    'tiempo_min': tiempo_total.min(),
                    'tiempo_max': tiempo_total.max(),
                })
                
                if len(factibles) > 0:
                    tiempo_factibles = factibles['tiempo_ejecucion']
                    stats_combo.update({
                        'tiempo_factibles_mean': tiempo_factibles.mean(),
                        'tiempo_factibles_std': tiempo_factibles.std(),
                    })
                else:
                    stats_combo.update({
                        'tiempo_factibles_mean': np.nan,
                        'tiempo_factibles_std': np.nan,
                    })
            else:
                stats_combo.update({
                    'tiempo_mean': np.nan,
                    'tiempo_std': np.nan,
                    'tiempo_median': np.nan,
                    'tiempo_min': np.nan,
                    'tiempo_max': np.nan,
                    'tiempo_factibles_mean': np.nan,
                    'tiempo_factibles_std': np.nan,
                })
            
            # Estadísticas de variables de diseño (solo para factibles)
            variables_diseno = ['x1', 'x2', 'x3', 'x4']
            for var in variables_diseno:
                if var in factibles.columns and len(factibles) > 0:
                    var_data = factibles[var]
                    stats_combo.update({
                        f'{var}_mean': var_data.mean(),
                        f'{var}_std': var_data.std(),
                        f'{var}_median': var_data.median(),
                        f'{var}_min': var_data.min(),
                        f'{var}_max': var_data.max(),
                    })
                else:
                    stats_combo.update({
                        f'{var}_mean': np.nan,
                        f'{var}_std': np.nan,
                        f'{var}_median': np.nan,
                        f'{var}_min': np.nan,
                        f'{var}_max': np.nan,
                    })
            
            # Estadísticas de convergencia (si disponible)
            if 'convergencia_final' in grupo.columns:
                conv_factibles = factibles['convergencia_final'].dropna()
                if len(conv_factibles) > 0:
                    stats_combo.update({
                        'convergencia_mean': conv_factibles.mean(),
                        'convergencia_std': conv_factibles.std(),
                        'convergencia_median': conv_factibles.median(),
                        'convergencia_min': conv_factibles.min(),
                        'convergencia_max': conv_factibles.max(),
                    })
                else:
                    stats_combo.update({
                        'convergencia_mean': np.nan,
                        'convergencia_std': np.nan,
                        'convergencia_median': np.nan,
                        'convergencia_min': np.nan,
                        'convergencia_max': np.nan,
                    })
            else:
                stats_combo.update({
                    'convergencia_mean': np.nan,
                    'convergencia_std': np.nan,
                    'convergencia_median': np.nan,
                    'convergencia_min': np.nan,
                    'convergencia_max': np.nan,
                })
            
            # Métricas adicionales
            stats_combo.update({
                'ranking_fitness': 0,  # Se calculará después
                'es_top_10_pct': False,  # Se calculará después
                'es_top_20_pct': False,  # Se calculará después
            })
            
            estadisticas_completas.append(stats_combo)
        
        # Crear DataFrame
        df_stats = pd.DataFrame(estadisticas_completas)
        
        # Calcular rankings (solo para combinaciones con datos factibles)
        df_factibles = df_stats[df_stats['fitness_count'] > 0].copy()
        if len(df_factibles) > 0:
            # Ranking por fitness medio (1 = mejor)
            df_factibles['ranking_fitness'] = df_factibles['fitness_mean'].rank(method='min')
            
            # Marcar top percentiles
            n_total = len(df_factibles)
            top_10_count = max(1, int(n_total * 0.1))
            top_20_count = max(1, int(n_total * 0.2))
            
            df_factibles['es_top_10_pct'] = df_factibles['ranking_fitness'] <= top_10_count
            df_factibles['es_top_20_pct'] = df_factibles['ranking_fitness'] <= top_20_count
            
            # Actualizar el DataFrame principal
            for idx, row in df_factibles.iterrows():
                df_stats.loc[idx, 'ranking_fitness'] = row['ranking_fitness']
                df_stats.loc[idx, 'es_top_10_pct'] = row['es_top_10_pct']
                df_stats.loc[idx, 'es_top_20_pct'] = row['es_top_20_pct']
        
        # Ordenar por ranking de fitness
        df_stats = df_stats.sort_values('ranking_fitness')
        
        # Redondear valores numéricos para mejor legibilidad
        columnas_numericas = df_stats.select_dtypes(include=[np.number]).columns
        df_stats[columnas_numericas] = df_stats[columnas_numericas].round(6)
        
        # Guardar CSV
        archivo_csv = f"{self.carpeta_visualizaciones}/estadisticas_todas_combinaciones.csv"
        df_stats.to_csv(archivo_csv, index=False)
        
        # Crear también un resumen ejecutivo
        resumen_ejecutivo = self._crear_resumen_ejecutivo(df_stats)
        archivo_resumen = f"{self.carpeta_visualizaciones}/resumen_ejecutivo_combinaciones.txt"
        
        with open(archivo_resumen, 'w', encoding='utf-8') as f:
            f.write(resumen_ejecutivo)
        
        # Mostrar información
        print(f"✅ CSV generado exitosamente:")
        print(f"   📁 Archivo principal: {archivo_csv}")
        print(f"   📄 Resumen ejecutivo: {archivo_resumen}")
        print(f"   📊 Total de combinaciones: {len(df_stats)}")
        print(f"   ✅ Combinaciones con datos factibles: {len(df_factibles) if len(df_factibles) > 0 else 0}")
        print(f"   🏆 Mejor combinación: ID {df_stats.iloc[0]['id_combinacion']} (fitness={df_stats.iloc[0]['fitness_mean']:.6f})")
        
        return df_stats
    
    def _crear_resumen_ejecutivo(self, df_stats):
        """Crea un resumen ejecutivo de las estadísticas."""
        resumen = []
        resumen.append("="*80)
        resumen.append("RESUMEN EJECUTIVO - ANÁLISIS DE TODAS LAS COMBINACIONES")
        resumen.append("="*80)
        resumen.append("")
        
        # Estadísticas generales
        total_combinaciones = len(df_stats)
        combinaciones_factibles = len(df_stats[df_stats['fitness_count'] > 0])
        
        resumen.append("📊 ESTADÍSTICAS GENERALES")
        resumen.append("-" * 40)
        resumen.append(f"Total de combinaciones analizadas: {total_combinaciones}")
        resumen.append(f"Combinaciones con soluciones factibles: {combinaciones_factibles}")
        resumen.append(f"Porcentaje de éxito: {(combinaciones_factibles/total_combinaciones)*100:.1f}%")
        resumen.append("")
        
        # Top 10 mejores combinaciones
        if combinaciones_factibles > 0:
            top_10 = df_stats[df_stats['fitness_count'] > 0].head(10)
            
            resumen.append("🏆 TOP 10 MEJORES COMBINACIONES")
            resumen.append("-" * 40)
            resumen.append(f"{'Rank':<5} {'ID':<12} {'Fitness':<12} {'Abejas':<8} {'Iters':<8} {'p_f':<6} {'Limit':<8} {'ModRate':<8} {'Factibles':<10}")
            resumen.append("-" * 80)
            
            for i, (_, row) in enumerate(top_10.iterrows()):
                resumen.append(f"{i+1:<5} {row['id_combinacion']:<12} {row['fitness_mean']:<12.6f} "
                             f"{row['numb_bees']:<8} {row['max_itrs']:<8} {row['p_f']:<6.2f} "
                             f"{row['limit']:<8} {row['modification_rate']:<8.1f} "
                             f"{row['experimentos_factibles']:<10}")
            resumen.append("")
        
        # Análisis por parámetros
        resumen.append("📈 ANÁLISIS POR PARÁMETROS")
        resumen.append("-" * 40)
        
        if combinaciones_factibles > 0:
            factibles_df = df_stats[df_stats['fitness_count'] > 0]
            
            for param in ['numb_bees', 'max_itrs', 'p_f', 'limit', 'modification_rate']:
                param_stats = factibles_df.groupby(param)['fitness_mean'].agg(['mean', 'count']).reset_index()
                mejor_nivel = param_stats.loc[param_stats['mean'].idxmin()]
                
                resumen.append(f"{param}:")
                resumen.append(f"  • Mejor nivel: {mejor_nivel[param]} (fitness promedio: {mejor_nivel['mean']:.6f})")
                resumen.append(f"  • Configuraciones con este nivel: {mejor_nivel['count']}")
        
        resumen.append("")
        
        # Estadísticas de factibilidad
        resumen.append("✅ ANÁLISIS DE FACTIBILIDAD")
        resumen.append("-" * 40)
        factibilidad_promedio = df_stats['tasa_factibilidad'].mean()
        factibilidad_std = df_stats['tasa_factibilidad'].std()
        
        resumen.append(f"Tasa de factibilidad promedio: {factibilidad_promedio:.1f}% ± {factibilidad_std:.1f}%")
        resumen.append(f"Mejor tasa de factibilidad: {df_stats['tasa_factibilidad'].max():.1f}%")
        resumen.append(f"Peor tasa de factibilidad: {df_stats['tasa_factibilidad'].min():.1f}%")
        resumen.append("")
        
        # Recomendaciones
        if combinaciones_factibles > 0:
            mejor_combinacion = df_stats.iloc[0]
            resumen.append("🎯 RECOMENDACIONES")
            resumen.append("-" * 40)
            resumen.append("Configuración recomendada (mejor fitness promedio):")
            resumen.append(f"  • Número de abejas: {mejor_combinacion['numb_bees']}")
            resumen.append(f"  • Iteraciones máximas: {mejor_combinacion['max_itrs']}")
            resumen.append(f"  • Probabilidad p_f: {mejor_combinacion['p_f']}")
            resumen.append(f"  • Límite: {mejor_combinacion['limit']}")
            resumen.append(f"  • Modification rate: {mejor_combinacion['modification_rate']}")
            resumen.append(f"  • Fitness esperado: {mejor_combinacion['fitness_mean']:.6f} ± {mejor_combinacion['fitness_std']:.6f}")
            resumen.append(f"  • Tasa de factibilidad: {mejor_combinacion['tasa_factibilidad']:.1f}%")
        
        resumen.append("")
        resumen.append("="*80)
        resumen.append(f"Archivo generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        resumen.append("="*80)
        
        return "\n".join(resumen)


def buscar_experimentos_disponibles():
    """Busca y lista experimentos disponibles."""
    experimentos = []
    
    if not os.path.exists('resultados'):
        return experimentos
    
    for item in os.listdir('resultados'):
        ruta_carpeta = os.path.join('resultados', item)
        if os.path.isdir(ruta_carpeta) and item.startswith('experimento_'):
            master_file = os.path.join(ruta_carpeta, 'master_table.csv')
            if os.path.exists(master_file):
                # Obtener información básica del experimento
                try:
                    df = pd.read_csv(master_file)
                    info = {
                        'carpeta': item,
                        'archivo': master_file,
                        'total_experimentos': len(df),
                        'experimentos_factibles': df['factible'].sum() if 'factible' in df.columns else 0,
                        'mejor_fitness': df[df['factible']]['fitness_final'].min() if 'factible' in df.columns and df['factible'].any() else 'N/A'
                    }
                    experimentos.append(info)
                except Exception as e:
                    experimentos.append({
                        'carpeta': item,
                        'archivo': master_file,
                        'total_experimentos': 'Error',
                        'experimentos_factibles': 'Error',
                        'mejor_fitness': 'Error'
                    })
    
    return sorted(experimentos, key=lambda x: x['carpeta'], reverse=True)


def main():
    """Función principal para ejecutar el visualizador."""
    print("🎨 VISUALIZADOR AVANZADO DE EXPERIMENTOS ABC")
    print("="*60)
    print("Este script genera visualizaciones completas de tus experimentos")
    print("incluyendo estadísticas, distribuciones, interacciones y convergencia.")
    print("GENERA CSV COMPLETO con estadísticas de todas las 3^5 combinaciones")
    
    # Buscar experimentos disponibles
    experimentos = buscar_experimentos_disponibles()
    
    if not experimentos:
        print("\n❌ No se encontraron experimentos disponibles.")
        print("Ejecuta primero 'experimento_completo.py' para generar datos.")
        return
    
    print(f"\n📁 Experimentos encontrados: {len(experimentos)}")
    print("-" * 80)
    print(f"{'#':<3} {'Experimento':<25} {'Total':<8} {'Factibles':<10} {'Mejor Fitness':<15}")
    print("-" * 80)
    
    for i, exp in enumerate(experimentos):
        porcentaje = f"({exp['experimentos_factibles']/exp['total_experimentos']*100:.1f}%)" if isinstance(exp['total_experimentos'], int) and exp['total_experimentos'] > 0 else ""
        mejor_fitness_str = f"{exp['mejor_fitness']:.6f}" if isinstance(exp['mejor_fitness'], float) else str(exp['mejor_fitness'])
        
        print(f"{i+1:<3} {exp['carpeta']:<25} {exp['total_experimentos']:<8} {exp['experimentos_factibles']:<5}{porcentaje:<5} {mejor_fitness_str:<15}")
    
    print("-" * 80)
    
    # Seleccionar experimento
    while True:
        try:
            if len(experimentos) == 1:
                seleccion = 0
                print(f"\nSeleccionando automáticamente: {experimentos[0]['carpeta']}")
                break
            else:
                seleccion = int(input(f"\nSelecciona un experimento (1-{len(experimentos)}): ")) - 1
                if 0 <= seleccion < len(experimentos):
                    break
                else:
                    print(f"❌ Selección inválida. Debe ser entre 1 y {len(experimentos)}")
        except ValueError:
            print("❌ Entrada inválida. Ingresa un número.")
    
    experimento_seleccionado = experimentos[seleccion]
    archivo_master = experimento_seleccionado['archivo']
    
    print(f"\n✅ Experimento seleccionado: {experimento_seleccionado['carpeta']}")
    print(f"📊 Cargando datos desde: {archivo_master}")
    
    try:
        # Crear visualizador y generar todas las visualizaciones
        visualizador = VisualizadorExperimentos(archivo_master)
        visualizador.generar_todas_visualizaciones()
        
        print(f"\n🎉 ¡Proceso completado exitosamente!")
        print(f"📁 Todas las visualizaciones están disponibles en:")
        print(f"   {visualizador.carpeta_visualizaciones}")
        
        # Mostrar resumen de archivos generados
        archivos_generados = [
            "01_estadisticas_descriptivas.png",
            "02_distribuciones_parametros.png", 
            "03_mapas_calor_interacciones.png",
            "04_violin_distribuciones.png",
            "05_analisis_factibilidad.png",
            "06_rendimiento_tiempo.png",
            "07_analisis_convergencia.png",
            "08_scatter_multidimensional.png",
            "09_rankings_parametros.png",
            "estadisticas_todas_combinaciones.csv",
            "resumen_ejecutivo_combinaciones.txt"
        ]
        
        print(f"\n📋 Archivos generados:")
        for archivo in archivos_generados:
            print(f"   • {archivo}")
        
    except Exception as e:
        print(f"\n❌ Error durante la generación de visualizaciones: {e}")
        print("Verifica que el archivo master_table.csv tenga el formato correcto.")


if __name__ == "__main__":
    main()