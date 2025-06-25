#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Estadístico Completo - Parámetros del Algoritmo ABC
Welded Beam Design Problem

Este script realiza el análisis estadístico completo siguiendo la metodología:
1. Análisis exploratorio y visualización
2. Verificación de supuestos ANOVA
3. ANOVA factorial o alternativas no paramétricas  
4. Pruebas post-hoc
5. Análisis de interacciones
6. Conclusiones y recomendaciones
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
import warnings
import os
from datetime import datetime
import numpy as np

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class AnalizadorEstadistico:
    """Clase para realizar análisis estadístico completo de los experimentos ABC."""
    
    def __init__(self, archivo_master_table):
        """
        Inicializa el analizador cargando los datos.
        
        Args:
            archivo_master_table (str): Ruta al archivo master_table.csv
        """
        self.datos = pd.read_csv(archivo_master_table)
        self.parametros = ['numb_bees', 'max_itrs', 'p_f', 'limit', 'modification_rate']
        self.variable_respuesta = 'fitness_final'
        
        # Filtrar solo soluciones factibles para el análisis principal
        self.datos_factibles = self.datos[self.datos['factible'] == True].copy()
        
        # Crear carpeta para guardar análisis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.carpeta_analisis = f"analisis/analisis_{timestamp}"
        os.makedirs(self.carpeta_analisis, exist_ok=True)
        
        print(f"Datos cargados: {len(self.datos)} ejecuciones totales")
        print(f"Soluciones factibles: {len(self.datos_factibles)} ejecuciones")
        print(f"Resultados del análisis se guardarán en: {self.carpeta_analisis}")
    
    def analisis_exploratorio(self):
        """Paso 2: Análisis exploratorio y visualización de datos."""
        print("\n" + "="*80)
        print("PASO 2: ANÁLISIS EXPLORATORIO Y VISUALIZACIÓN")
        print("="*80)
        
        # Estadísticas descriptivas por combinación
        print("Generando estadísticas descriptivas...")
        
        estadisticas = self.datos_factibles.groupby(self.parametros)[self.variable_respuesta].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(6)
        
        # Guardar estadísticas
        estadisticas.to_csv(f"{self.carpeta_analisis}/estadisticas_descriptivas.csv")
        
        # Mostrar mejores combinaciones
        print("\nTOP 10 MEJORES COMBINACIONES (por media):")
        print("-" * 60)
        mejores = estadisticas.sort_values('mean').head(10)
        print(mejores)
        
        # Estadísticas por parámetro individual
        print("\nEstadísticas por parámetro individual:")
        stats_parametros = {}
        
        for param in self.parametros:
            stats_param = self.datos_factibles.groupby(param)[self.variable_respuesta].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(6)
            stats_parametros[param] = stats_param
            print(f"\n{param.upper()}:")
            print(stats_param)
        
        # Crear visualizaciones
        self._crear_boxplots()
        self._crear_histogramas()
        
        return estadisticas, stats_parametros
    
    def _crear_boxplots(self):
        """Crea diagramas de caja para cada parámetro."""
        print("Generando diagramas de caja...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, param in enumerate(self.parametros):
            sns.boxplot(data=self.datos_factibles, x=param, y=self.variable_respuesta, ax=axes[i])
            axes[i].set_title(f'Distribución de Fitness por {param}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(param, fontsize=10)
            axes[i].set_ylabel('Fitness', fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        # Ocultar el último subplot si no se usa
        if len(self.parametros) < 6:
            axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_analisis}/boxplots_parametros.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _crear_histogramas(self):
        """Crea histogramas de la variable respuesta."""
        print("Generando histogramas...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histograma general
        ax1.hist(self.datos_factibles[self.variable_respuesta], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribución del Fitness (Soluciones Factibles)', fontweight='bold')
        ax1.set_xlabel('Fitness')
        ax1.set_ylabel('Frecuencia')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot para normalidad
        stats.probplot(self.datos_factibles[self.variable_respuesta], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Verificación de Normalidad)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_analisis}/distribucion_fitness.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def verificar_supuestos_anova(self):
        """Paso 3: Verificación de supuestos del ANOVA."""
        print("\n" + "="*80)
        print("PASO 3: VERIFICACIÓN DE SUPUESTOS DEL ANOVA")
        print("="*80)
        
        try:
            # Crear modelo ANOVA factorial completo
            formula = f"{self.variable_respuesta} ~ " + " + ".join(self.parametros)
            from statsmodels.formula.api import ols
            modelo = ols(formula, data=self.datos_factibles).fit()
            residuos = modelo.resid
            
        except Exception as e:
            print(f"Error creando modelo ANOVA: {e}")
            print("Continuando con análisis no paramétrico...")
            return False, None
        
        # 1. Supuesto de normalidad de residuos
        print("1. VERIFICANDO NORMALIDAD DE RESIDUOS")
        print("-" * 40)
        
        try:
            from scipy.stats import shapiro, kstest
            
            # Prueba de Shapiro-Wilk (para muestras pequeñas)
            if len(residuos) <= 5000:  
                stat_shapiro, p_shapiro = shapiro(residuos)
                print(f"Prueba de Shapiro-Wilk:")
                print(f"  Estadístico: {stat_shapiro:.6f}")
                print(f"  p-valor: {p_shapiro:.6f}")
                print(f"  Normalidad: {'SÍ' if p_shapiro > 0.05 else 'NO'} (α = 0.05)")
            else:
                print("Muestra muy grande para Shapiro-Wilk, usando Kolmogorov-Smirnov...")
                stat_ks, p_ks = kstest(residuos, 'norm')
                print(f"Prueba de Kolmogorov-Smirnov:")
                print(f"  Estadístico: {stat_ks:.6f}")
                print(f"  p-valor: {p_ks:.6f}")
                print(f"  Normalidad: {'SÍ' if p_ks > 0.05 else 'NO'} (α = 0.05)")
                p_shapiro = p_ks  # Para consistencia
                
        except Exception as e:
            print(f"Error en prueba de normalidad: {e}")
            p_shapiro = 0.01  # Asumir que no es normal
        
        # 2. Supuesto de homogeneidad de varianzas
        print("\n2. VERIFICANDO HOMOGENEIDAD DE VARIANZAS")
        print("-" * 40)
        
        try:
            from scipy.stats import levene
            
            # Agrupar datos por combinación de parámetros
            grupos = []
            for name, group in self.datos_factibles.groupby(self.parametros):
                if len(group) > 1:  # Necesitamos al menos 2 observaciones por grupo
                    grupos.append(group[self.variable_respuesta].values)
            
            if len(grupos) > 1:
                stat_levene, p_levene = levene(*grupos)
                print(f"Prueba de Levene:")
                print(f"  Estadístico: {stat_levene:.6f}")
                print(f"  p-valor: {p_levene:.6f}")
                print(f"  Homogeneidad: {'SÍ' if p_levene > 0.05 else 'NO'} (α = 0.05)")
            else:
                print("No hay suficientes grupos para la prueba de Levene")
                p_levene = 0.01  # Asumir que no se cumple
                
        except Exception as e:
            print(f"Error en prueba de homogeneidad: {e}")
            p_levene = 0.01  # Asumir que no se cumple
        
        # Crear gráficos de diagnóstico
        try:
            self._crear_graficos_diagnostico(modelo, residuos)
        except Exception as e:
            print(f"Error creando gráficos de diagnóstico: {e}")
        
        # Decisión sobre qué análisis usar
        supuestos_ok = (p_shapiro > 0.05) and (p_levene > 0.05)
        
        print(f"\n{'='*60}")
        print("RESUMEN DE SUPUESTOS:")
        print(f"{'='*60}")
        print(f"Normalidad de residuos: {'✓' if p_shapiro > 0.05 else '✗'}")
        print(f"Homogeneidad de varianzas: {'✓' if p_levene > 0.05 else '✗'}")
        print(f"\nRECOMENDACIÓN: {'ANOVA Paramétrico' if supuestos_ok else 'Análisis No Paramétrico'}")
        print(f"{'='*60}")
        
        return supuestos_ok, modelo
    
    def _crear_graficos_diagnostico(self, modelo, residuos):
        """Crea gráficos de diagnóstico para el modelo."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuos vs valores ajustados
        axes[0,0].scatter(modelo.fittedvalues, residuos, alpha=0.6)
        axes[0,0].axhline(y=0, color='red', linestyle='--')
        axes[0,0].set_xlabel('Valores Ajustados')
        axes[0,0].set_ylabel('Residuos')
        axes[0,0].set_title('Residuos vs Valores Ajustados')
        axes[0,0].grid(True, alpha=0.3)
        
        # Q-Q plot de residuos
        stats.probplot(residuos, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot de Residuos')
        axes[0,1].grid(True, alpha=0.3)
        
        # Histograma de residuos
        axes[1,0].hist(residuos, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1,0].set_xlabel('Residuos')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_title('Distribución de Residuos')
        axes[1,0].grid(True, alpha=0.3)
        
        # Residuos vs orden
        axes[1,1].plot(residuos, 'o', alpha=0.6)
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].set_xlabel('Orden de Observación')
        axes[1,1].set_ylabel('Residuos')
        axes[1,1].set_title('Residuos vs Orden')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_analisis}/diagnosticos_modelo.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def anova_factorial(self, modelo):
        """Paso 4A: Análisis ANOVA factorial (si se cumplen supuestos)."""
        print("\n" + "="*80)
        print("PASO 4A: ANÁLISIS ANOVA FACTORIAL")
        print("="*80)
        
        # ANOVA tabla
        anova_tabla = sm.stats.anova_lm(modelo, typ=2)
        print("TABLA ANOVA:")
        print("-" * 60)
        print(anova_tabla)
        
        # Guardar tabla ANOVA
        anova_tabla.to_csv(f"{self.carpeta_analisis}/tabla_anova.csv")
        
        # Identificar efectos significativos
        efectos_significativos = anova_tabla[anova_tabla['PR(>F)'] < 0.05].index.tolist()
        
        print(f"\nEFECTOS SIGNIFICATIVOS (α = 0.05):")
        print("-" * 40)
        for efecto in efectos_significativos:
            p_valor = anova_tabla.loc[efecto, 'PR(>F)']
            print(f"  {efecto}: p = {p_valor:.6f}")
        
        if not efectos_significativos:
            print("  Ningún efecto es estadísticamente significativo.")
        
        # Pruebas post-hoc para efectos principales significativos
        self._pruebas_posthoc_parametricas(efectos_significativos)
        
        return anova_tabla, efectos_significativos
    
    def analisis_no_parametrico(self):
        """Paso 4B: Análisis no paramétrico (si no se cumplen supuestos)."""
        print("\n" + "="*80)
        print("PASO 4B: ANÁLISIS NO PARAMÉTRICO")
        print("="*80)
        
        resultados_kruskal = {}
        efectos_significativos = []
        
        print("PRUEBAS DE KRUSKAL-WALLIS POR PARÁMETRO:")
        print("-" * 50)
        
        for param in self.parametros:
            try:
                # Agrupar datos por niveles del parámetro
                grupos = []
                niveles = []
                
                for nivel, group in self.datos_factibles.groupby(param):
                    if len(group) > 0:  # Verificar que el grupo no esté vacío
                        grupos.append(group[self.variable_respuesta].values)
                        niveles.append(nivel)
                
                # Verificar que tenemos al menos 2 grupos con datos
                if len(grupos) < 2:
                    print(f"{param}:")
                    print(f"  Error: Solo hay {len(grupos)} grupo(s) con datos. Se necesitan al menos 2.")
                    print()
                    continue
                
                # Verificar que los grupos tienen suficientes observaciones
                tamaños_grupos = [len(g) for g in grupos]
                if any(t < 1 for t in tamaños_grupos):
                    print(f"{param}:")
                    print(f"  Error: Algunos grupos están vacíos. Tamaños: {tamaños_grupos}")
                    print()
                    continue
                
                # Prueba de Kruskal-Wallis
                from scipy.stats import kruskal
                stat, p_valor = kruskal(*grupos)
                resultados_kruskal[param] = {'estadistico': stat, 'p_valor': p_valor}
                
                significativo = p_valor < 0.05
                if significativo:
                    efectos_significativos.append(param)
                
                print(f"{param}:")
                print(f"  Niveles analizados: {niveles}")
                print(f"  Tamaños de grupos: {tamaños_grupos}")
                print(f"  Estadístico H: {stat:.6f}")
                print(f"  p-valor: {p_valor:.6f}")
                print(f"  Significativo: {'SÍ' if significativo else 'NO'} (α = 0.05)")
                print()
                
            except Exception as e:
                print(f"{param}:")
                print(f"  Error en análisis: {e}")
                print(f"  Tipo de error: {type(e).__name__}")
                print()
        
        # Guardar resultados
        if resultados_kruskal:
            import pandas as pd
            df_kruskal = pd.DataFrame(resultados_kruskal).T
            df_kruskal.to_csv(f"{self.carpeta_analisis}/resultados_kruskal_wallis.csv")
        
        # Pruebas post-hoc para efectos significativos
        if efectos_significativos:
            self._pruebas_posthoc_no_parametricas(efectos_significativos)
        else:
            print("No hay efectos significativos para pruebas post-hoc.")
        
        return resultados_kruskal, efectos_significativos
    
    def _pruebas_posthoc_parametricas(self, efectos_significativos):
        """Realiza pruebas post-hoc de Tukey para efectos significativos."""
        print("\nPRUEBAS POST-HOC (TUKEY HSD):")
        print("-" * 40)
        
        for param in efectos_significativos:
            if param in self.parametros:  # Solo para efectos principales
                print(f"\n{param.upper()}:")
                
                # Prueba de Tukey
                tukey = pairwise_tukeyhsd(
                    endog=self.datos_factibles[self.variable_respuesta],
                    groups=self.datos_factibles[param],
                    alpha=0.05
                )
                
                print(tukey)
                
                # Guardar resultados
                with open(f"{self.carpeta_analisis}/tukey_{param}.txt", 'w') as f:
                    f.write(str(tukey))
    
    def _pruebas_posthoc_no_parametricas(self, efectos_significativos):
        """Realiza pruebas post-hoc de Dunn para efectos significativos."""
        print("\nPRUEBAS POST-HOC (DUNN):")
        print("-" * 30)
        
        for param in efectos_significativos:
            print(f"\n{param.upper()}:")
            
            # Preparar datos para la prueba de Dunn
            grupos_data = []
            grupos_labels = []
            
            for nivel, group in self.datos_factibles.groupby(param):
                # CORRECCIÓN COMPLETA: Manejo seguro de diferentes tipos de datos
                try:
                    # Convertir nivel a valor escalar si es necesario
                    if hasattr(nivel, 'iloc'):  # Es pandas Series
                        nivel_valor = nivel.iloc[0]
                    elif hasattr(nivel, '__len__') and not isinstance(nivel, str):  # Es array/lista
                        nivel_valor = nivel[0] if len(nivel) > 0 else nivel
                    else:  # Es scalar
                        nivel_valor = nivel
                    
                    # Convertir a tipo Python estándar
                    if isinstance(nivel_valor, (int, float, np.integer, np.floating)):
                        # Verificar si es entero de forma segura
                        try:
                            float_val = float(nivel_valor)
                            if float_val.is_integer():
                                nivel_str = str(int(float_val))
                            else:
                                nivel_str = str(float_val)
                        except (ValueError, OverflowError):
                            nivel_str = str(nivel_valor)
                    else:
                        nivel_str = str(nivel_valor)
                    
                    # Verificar que el grupo no esté vacío
                    if len(group) == 0:
                        print(f"Advertencia: Grupo vacío para {param}={nivel_str}")
                        continue
                    
                    # Agregar datos del grupo
                    valores_grupo = group[self.variable_respuesta].values
                    if len(valores_grupo) > 0:  # Solo agregar si hay datos
                        grupos_data.extend(valores_grupo)
                        grupos_labels.extend([nivel_str] * len(valores_grupo))
                    
                except Exception as e:
                    print(f"Error procesando nivel {nivel} para {param}: {e}")
                    continue
            
            # Verificar que tenemos datos suficientes
            if len(grupos_data) == 0:
                print(f"Error: No hay datos para {param}")
                continue
            
            grupos_unicos = list(set(grupos_labels))
            if len(grupos_unicos) < 2:
                print(f"Advertencia: Solo hay {len(grupos_unicos)} grupo(s) para {param}. Se necesitan al menos 2 grupos.")
                continue
            
            # Verificar tamaños de grupos
            try:
                import pandas as pd
                grupos_info = pd.Series(grupos_labels).value_counts()
                print(f"Tamaños de grupos: {dict(grupos_info)}")
                
                # Verificar que todos los grupos tienen al menos 1 observación
                if (grupos_info < 1).any():
                    print(f"Advertencia: Algunos grupos tienen 0 observaciones")
                    continue
                    
            except Exception as e:
                print(f"Error calculando tamaños de grupos: {e}")
                continue
            
            # Prueba de Dunn
            try:
                from scikit_posthocs import posthoc_dunn
                
                # Convertir a arrays de numpy para consistencia
                grupos_data_array = np.array(grupos_data, dtype=float)
                grupos_labels_array = np.array(grupos_labels, dtype=str)
                
                # Verificar datos antes de la prueba
                if len(grupos_data_array) != len(grupos_labels_array):
                    print(f"Error: Inconsistencia en longitudes de datos ({len(grupos_data_array)}) y etiquetas ({len(grupos_labels_array)})")
                    continue
                
                # Verificar que no hay NaN o infinitos
                if np.any(np.isnan(grupos_data_array)) or np.any(np.isinf(grupos_data_array)):
                    print(f"Advertencia: Se encontraron valores NaN o infinitos en los datos")
                    # Filtrar valores problemáticos
                    mask = np.isfinite(grupos_data_array)
                    grupos_data_array = grupos_data_array[mask]
                    grupos_labels_array = grupos_labels_array[mask]
                
                # 1) Construir un DataFrame de dos columnas: valor y grupo
                df = pd.DataFrame({
                    self.variable_respuesta: grupos_data_array,
                    'grupo': grupos_labels_array
                })

                print(f"Ejecutando prueba de Dunn con {len(df)} observaciones en {len(grupos_unicos)} grupos...")

                # 2) Llamar a Dunn con val_col y group_col explícitos
                dunn_results = posthoc_dunn(
                    df,
                    val_col=self.variable_respuesta,
                    group_col='grupo',
                    p_adjust='bonferroni'
                )

                print("Resultados de la prueba de Dunn:")
                print(dunn_results)

                # 3) Guardar CSV y procesar comparaciones significativas como ya lo hacías
                dunn_results.to_csv(f"{self.carpeta_analisis}/dunn_{param}.csv")
                
                # Mostrar comparaciones significativas
                print(f"\nComparaciones significativas (p < 0.05) para {param}:")
                significativas = []
                
                try:
                    for i in dunn_results.index:
                        for j in dunn_results.columns:
                            if i != j:
                                p_val = dunn_results.loc[i, j]
                                if not np.isnan(p_val) and p_val < 0.05:
                                    significativas.append(f"{i} vs {j}: p = {p_val:.6f}")
                    
                    if significativas:
                        for comp in significativas:
                            print(f"  • {comp}")
                    else:
                        print("  • No se encontraron diferencias significativas entre grupos")
                        
                except Exception as e:
                    print(f"Error analizando significancia: {e}")
                    
            except Exception as e:
                print(f"Error en prueba de Dunn para {param}: {e}")
                print(f"Tipo de error: {type(e).__name__}")
                
                # Información de diagnóstico detallada
                print(f"Información de diagnóstico:")
                print(f"  • Número total de observaciones: {len(grupos_data) if grupos_data else 0}")
                print(f"  • Número de grupos únicos: {len(grupos_unicos)}")
                print(f"  • Grupos únicos: {grupos_unicos}")
                
                if grupos_data:
                    print(f"  • Tipo de datos (muestra): {type(grupos_data[0])}")
                    print(f"  • Rango de valores: [{min(grupos_data):.6f}, {max(grupos_data):.6f}]")
                
                if grupos_labels:
                    print(f"  • Tipo de etiquetas (muestra): {type(grupos_labels[0])}")
                    print(f"  • Etiquetas únicas: {set(grupos_labels)}")
    
    def analisis_interacciones(self):
        """Paso 5: Análisis visual de interacciones."""
        print("\n" + "="*80)
        print("PASO 5: ANÁLISIS DE INTERACCIONES")
        print("="*80)
        
        print("Generando gráficos de interacciones...")
        
        # Seleccionar pares de parámetros más relevantes
        pares_importantes = [
            ('numb_bees', 'max_itrs'),
            ('limit', 'modification_rate'),
            ('p_f', 'numb_bees'),
            ('max_itrs', 'limit')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (param1, param2) in enumerate(pares_importantes):
            # Calcular medias por combinación
            medias = self.datos_factibles.groupby([param1, param2])[self.variable_respuesta].mean().reset_index()
            
            # Crear gráfico de interacción
            for nivel2 in medias[param2].unique():
                datos_nivel = medias[medias[param2] == nivel2]
                axes[i].plot(datos_nivel[param1], datos_nivel[self.variable_respuesta], 
                           marker='o', label=f'{param2}={nivel2}')
            
            axes[i].set_xlabel(param1)
            axes[i].set_ylabel('Fitness Promedio')
            axes[i].set_title(f'Interacción: {param1} × {param2}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.carpeta_analisis}/graficos_interacciones.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generar_conclusiones(self, efectos_significativos):
        """Paso 6: Generar conclusiones y recomendaciones."""
        print("\n" + "="*80)
        print("PASO 6: CONCLUSIONES Y RECOMENDACIONES")
        print("="*80)
        
        # Encontrar la mejor configuración
        mejor_combinacion = self.datos_factibles.loc[
            self.datos_factibles[self.variable_respuesta].idxmin()
        ]
        
        # Estadísticas de las mejores configuraciones
        top_10_combinaciones = self.datos_factibles.groupby(self.parametros)[self.variable_respuesta].mean().nsmallest(10)
        
        conclusiones = []
        conclusiones.append("ANÁLISIS ESTADÍSTICO COMPLETO - ALGORITMO ABC")
        conclusiones.append("="*60)
        conclusiones.append("")
        
        # Parámetros más influyentes
        conclusiones.append("1. PARÁMETROS MÁS INFLUYENTES:")
        conclusiones.append("-" * 35)
        if efectos_significativos:
            for param in efectos_significativos:
                conclusiones.append(f"   • {param}")
        else:
            conclusiones.append("   • Ningún parámetro mostró efectos significativos")
        conclusiones.append("")
        
        # Mejor configuración encontrada
        conclusiones.append("2. MEJOR CONFIGURACIÓN ENCONTRADA:")
        conclusiones.append("-" * 37)
        conclusiones.append(f"   • Fitness: {mejor_combinacion[self.variable_respuesta]:.6f}")
        for param in self.parametros:
            conclusiones.append(f"   • {param}: {mejor_combinacion[param]}")
        conclusiones.append("")
        
        # Variables de diseño de la mejor solución
        conclusiones.append("3. VARIABLES DE DISEÑO (MEJOR SOLUCIÓN):")
        conclusiones.append("-" * 42)
        conclusiones.append(f"   • x1 (altura soldadura): {mejor_combinacion['x1']:.4f}")
        conclusiones.append(f"   • x2 (longitud soldadura): {mejor_combinacion['x2']:.4f}")
        conclusiones.append(f"   • x3 (altura barra): {mejor_combinacion['x3']:.4f}")
        conclusiones.append(f"   • x4 (espesor barra): {mejor_combinacion['x4']:.4f}")
        conclusiones.append("")
        
        # Recomendaciones finales
        conclusiones.append("4. RECOMENDACIONES FINALES:")
        conclusiones.append("-" * 28)
        
        # Analizar tendencias en los mejores resultados
        mejores_datos = self.datos_factibles.nsmallest(int(len(self.datos_factibles) * 0.1), self.variable_respuesta)
        
        for param in self.parametros:
            moda_mejores = mejores_datos[param].mode()[0] if not mejores_datos[param].mode().empty else "N/A"
            conclusiones.append(f"   • {param}: {moda_mejores} (valor más frecuente en top 10%)")
        
        conclusiones.append("")
        conclusiones.append("5. RESUMEN ESTADÍSTICO:")
        conclusiones.append("-" * 23)
        conclusiones.append(f"   • Total ejecuciones: {len(self.datos)}")
        conclusiones.append(f"   • Soluciones factibles: {len(self.datos_factibles)} ({len(self.datos_factibles)/len(self.datos)*100:.1f}%)")
        conclusiones.append(f"   • Mejor fitness global: {self.datos_factibles[self.variable_respuesta].min():.6f}")
        conclusiones.append(f"   • Fitness promedio: {self.datos_factibles[self.variable_respuesta].mean():.6f}")
        conclusiones.append("")
        
        # Imprimir y guardar conclusiones
        texto_conclusiones = "\n".join(conclusiones)
        print(texto_conclusiones)
        
        with open(f"{self.carpeta_analisis}/conclusiones_finales.txt", 'w', encoding='utf-8') as f:
            f.write(texto_conclusiones)
        
        return mejor_combinacion, top_10_combinaciones
    
    def ejecutar_analisis_completo(self):
        """Ejecuta el análisis estadístico completo."""
        print("INICIANDO ANÁLISIS ESTADÍSTICO COMPLETO")
        print("="*80)
        
        # Paso 2: Análisis exploratorio
        stats_generales, stats_parametros = self.analisis_exploratorio()
        
        # Paso 3: Verificación de supuestos
        supuestos_ok, modelo = self.verificar_supuestos_anova()
        
        # Paso 4: Análisis estadístico (paramétrico o no paramétrico)
        if supuestos_ok:
            anova_tabla, efectos_significativos = self.anova_factorial(modelo)
            self.analisis_interacciones()  # Solo si ANOVA es válido
        else:
            resultados_kruskal, efectos_significativos = self.analisis_no_parametrico()
            self.analisis_interacciones()  # Análisis visual de interacciones
        
        # Paso 6: Conclusiones y recomendaciones
        mejor_config, top_configs = self.generar_conclusiones(efectos_significativos)
        
        print(f"\nAnálisis completo guardado en: {self.carpeta_analisis}")
        
        return {
            'estadisticas': stats_generales,
            'efectos_significativos': efectos_significativos,
            'mejor_configuracion': mejor_config,
            'carpeta_resultados': self.carpeta_analisis
        }

def main():
    """Función principal para ejecutar el análisis."""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python analisis_estadistico.py <ruta_al_master_table.csv>")
        print("Ejemplo: python analisis_estadistico.py resultados/experimento_20240625_1530/master_table.csv")
        return
    
    archivo_master = sys.argv[1]
    
    if not os.path.exists(archivo_master):
        print(f"Error: No se encuentra el archivo {archivo_master}")
        return
    
    # Crear analizador y ejecutar análisis completo
    analizador = AnalizadorEstadistico(archivo_master)
    resultados = analizador.ejecutar_analisis_completo()
    
    print("\n" + "="*80)
    print("ANÁLISIS ESTADÍSTICO COMPLETADO EXITOSAMENTE")
    print("="*80)

if __name__ == "__main__":
    main()
