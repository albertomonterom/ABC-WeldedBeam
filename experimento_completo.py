#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento Completo - Análisis de Parámetros del Algoritmo ABC
Welded Beam Design Problem

Este script ejecuta todas las combinaciones de parámetros (3^5 = 243)
con 30 repeticiones cada una usando diferentes semillas.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count
import itertools
import sys
from tqdm import tqdm
import time

# Importar módulos del proyecto
from Hive import Constraints
from Hive import Hive

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================

# Parámetros del problema
LOWER = [0.1, 0.1, 0.1, 0.1]
UPPER = [2.0, 10.0, 10.0, 2.0]

def objective(x):
    """Función objetivo del Welded Beam Design Problem"""
    x1, x2, x3, x4 = x
    return (1.10471 * (x1 ** 2) * x2) + (0.04811 * x3 * x4 * (14.0 + x2))

# Definición de parámetros y sus niveles
PARAMETROS = {
    'numb_bees': [50, 150, 450],           # A: Tamaño de población
    'max_itrs': [100, 500, 1000],          # B: Número de iteraciones  
    'p_f': [0.25, 0.45, 0.75],            # C: Probabilidad jerarquización estocástica
    'limit': [10, 50, 150],               # D: Límite de abandono
    'modification_rate': [0.1, 0.5, 0.9]  # E: Factor de modificación
}

# Semillas: Los primeros 30 números primos
SEMILLAS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 
           61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

def configurar_procesamiento():
    """Permite al usuario configurar el número de procesos a usar."""
    cores_disponibles = cpu_count()
    print(f"\n{'='*60}")
    print("CONFIGURACIÓN DE PROCESAMIENTO")
    print(f"{'='*60}")
    print(f"Cores/procesadores disponibles: {cores_disponibles}")
    print(f"Recomendaciones:")
    print(f"  • Uso ligero: {max(1, cores_disponibles // 4)} cores (75% disponible para otras tareas)")
    print(f"  • Uso moderado: {max(1, cores_disponibles // 2)} cores (50% disponible)")
    print(f"  • Uso intensivo: {max(1, cores_disponibles - 1)} cores (máximo rendimiento)")
    print(f"  • Uso máximo: {cores_disponibles} cores (100% del sistema)")
    
    while True:
        try:
            opcion = input(f"\nSelecciona número de cores a usar (1-{cores_disponibles}) o 'auto' para automático: ").strip().lower()
            
            if opcion == 'auto':
                # Usar configuración automática inteligente
                if cores_disponibles >= 8:
                    num_procesos = cores_disponibles - 2  # Dejar 2 cores libres
                elif cores_disponibles >= 4:
                    num_procesos = cores_disponibles - 1  # Dejar 1 core libre
                else:
                    num_procesos = cores_disponibles  # Usar todos si hay pocos
                
                print(f"Configuración automática: {num_procesos} cores")
                return num_procesos
            
            num_procesos = int(opcion)
            if 1 <= num_procesos <= cores_disponibles:
                # Estimar tiempo aproximado
                tiempo_estimado = estimar_tiempo_ejecucion(num_procesos)
                print(f"\nConfiguración seleccionada: {num_procesos} cores")
                print(f"Tiempo estimado: {tiempo_estimado}")
                
                confirmar = input("¿Continuar con esta configuración? (s/N): ").strip().lower()
                if confirmar == 's':
                    return num_procesos
                else:
                    continue
            else:
                print(f"Número inválido. Debe estar entre 1 y {cores_disponibles}")
                
        except ValueError:
            print("Entrada inválida. Ingresa un número o 'auto'")

def estimar_tiempo_ejecucion(num_procesos, total_ejecuciones=None):
    """Estima el tiempo de ejecución basado en el número de procesos."""
    if total_ejecuciones is None:
        total_ejecuciones = len(generar_todas_combinaciones()) * len(SEMILLAS)
    
    # Tiempo promedio por ejecución (segundos) - estimación conservadora
    tiempo_por_ejecucion = 1.5  # 1.5 segundos promedio por experimento
    
    # Tiempo total considerando paralelización
    tiempo_total_segundos = (total_ejecuciones * tiempo_por_ejecucion) / num_procesos
    
    # Convertir a formato legible
    horas = int(tiempo_total_segundos // 3600)
    minutos = int((tiempo_total_segundos % 3600) // 60)
    
    if horas > 0:
        return f"~{horas}h {minutos}m"
    else:
        return f"~{minutos}m"

def generar_todas_combinaciones():
    """Genera todas las combinaciones posibles de parámetros."""
    nombres_params = list(PARAMETROS.keys())
    valores_params = list(PARAMETROS.values())
    
    combinaciones = []
    for combo in itertools.product(*valores_params):
        combinacion = dict(zip(nombres_params, combo))
        combinaciones.append(combinacion)
    
    return combinaciones

def ejecutar_experimento_individual(args):
    """
    Ejecuta una sola instancia del algoritmo ABC con parámetros específicos.
    
    Args:
        args: tupla (combinacion_params, semilla, id_combinacion)
    
    Returns:
        dict: Resultados del experimento
    """
    combinacion, semilla, id_combo = args
    
    try:
        # Crear modelo ABC con los parámetros especificados
        modelo = Hive.BeeHive(
            lower=LOWER,
            upper=UPPER,
            fun=objective,
            funcon=Constraints.evaluate_constraints,
            seed=semilla,
            verbose=False,  # Desactivar verbose para ejecución en lote
            **combinacion  # Desempaqueta todos los parámetros de la combinación
        )
        
        # Medir tiempo de ejecución
        inicio = datetime.now()
        cost = modelo.run()
        fin = datetime.now()
        tiempo_ejecucion = (fin - inicio).total_seconds()
        
        # Evaluar la mejor solución encontrada
        if modelo.solution:
            constraint_result = Constraints.evaluate_constraints(modelo.solution)
            feasible = constraint_result['feasible']
            violacion_total = constraint_result['violation']
        else:
            feasible = False
            violacion_total = float('inf')
        
        # Recopilar resultados
        resultado = {
            'id_combinacion': id_combo,
            'semilla': semilla,
            'numb_bees': combinacion['numb_bees'],
            'max_itrs': combinacion['max_itrs'],
            'p_f': combinacion['p_f'],
            'limit': combinacion['limit'],
            'modification_rate': combinacion['modification_rate'],
            'fitness_final': modelo.best,
            'factible': feasible,
            'violacion_total': violacion_total,
            'tiempo_ejecucion': tiempo_ejecucion,
            'x1': modelo.solution[0] if modelo.solution else None,
            'x2': modelo.solution[1] if modelo.solution else None,
            'x3': modelo.solution[2] if modelo.solution else None,
            'x4': modelo.solution[3] if modelo.solution else None,
            'convergencia_final': cost["best"][-1] if cost["best"] else None
        }
        
        return resultado
        
    except Exception as e:
        # Retornar resultado con error
        return {
            'id_combinacion': id_combo,
            'semilla': semilla,
            'numb_bees': combinacion['numb_bees'],
            'max_itrs': combinacion['max_itrs'],
            'p_f': combinacion['p_f'],
            'limit': combinacion['limit'],
            'modification_rate': combinacion['modification_rate'],
            'fitness_final': float('inf'),
            'factible': False,
            'violacion_total': float('inf'),
            'tiempo_ejecucion': 0,
            'x1': None, 'x2': None, 'x3': None, 'x4': None,
            'convergencia_final': None,
            'error': str(e)
        }

def ejecutar_experimento_completo(num_procesos=None):
    """Ejecuta el experimento completo con todas las combinaciones."""
    
    # Crear timestamp para carpeta única
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    carpeta_experimento = f"resultados/experimento_{timestamp}"
    
    # Crear estructura de carpetas
    os.makedirs(carpeta_experimento, exist_ok=True)
    
    print("="*80)
    print("INICIANDO EXPERIMENTO COMPLETO - ANÁLISIS DE PARÁMETROS ABC")
    print("="*80)
    print(f"Carpeta de resultados: {carpeta_experimento}")
    
    # Generar todas las combinaciones
    combinaciones = generar_todas_combinaciones()
    total_combinaciones = len(combinaciones)
    total_ejecuciones = total_combinaciones * len(SEMILLAS)
    
    print(f"Total de combinaciones de parámetros: {total_combinaciones}")
    print(f"Repeticiones por combinación: {len(SEMILLAS)}")
    print(f"Total de ejecuciones: {total_ejecuciones}")
    
    # Configurar procesamiento si no se especifica
    if num_procesos is None:
        num_procesos = configurar_procesamiento()
    
    print(f"Procesadores a utilizar: {num_procesos}/{cpu_count()}")
    tiempo_estimado = estimar_tiempo_ejecucion(num_procesos, total_ejecuciones)
    print(f"Tiempo estimado: {tiempo_estimado}")
    print("="*80)
    
    # Guardar configuración del experimento
    config_df = pd.DataFrame([
        {'parametro': k, 'niveles': str(v)} for k, v in PARAMETROS.items()
    ])
    config_info = pd.DataFrame([
        {'configuracion': 'num_procesos', 'valor': num_procesos},
        {'configuracion': 'total_ejecuciones', 'valor': total_ejecuciones},
        {'configuracion': 'tiempo_estimado', 'valor': tiempo_estimado},
        {'configuracion': 'timestamp', 'valor': timestamp}
    ])
    
    config_df.to_csv(f"{carpeta_experimento}/configuracion_experimento.csv", index=False)
    config_info.to_csv(f"{carpeta_experimento}/info_ejecucion.csv", index=False)
    
    # Preparar argumentos para procesamiento paralelo
    argumentos = []
    for i, combinacion in enumerate(combinaciones):
        for semilla in SEMILLAS:
            argumentos.append((combinacion, semilla, i))
    
    # Ejecutar con barra de progreso
    print("Iniciando ejecución paralela...")
    inicio_total = time.time()
    
    resultados_completos = []
    
    # Usar Pool con barra de progreso
    with Pool(processes=num_procesos) as pool:
        # Crear barra de progreso
        with tqdm(total=total_ejecuciones, 
                 desc="Procesando experimentos", 
                 unit="exp",
                 ncols=100,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            
            # Procesar en lotes para actualizar progreso
            tamaño_lote = max(1, num_procesos * 2)  # Lotes adaptativos
            
            for i in range(0, len(argumentos), tamaño_lote):
                lote_args = argumentos[i:i + tamaño_lote]
                
                # Procesar lote
                resultados_lote = pool.map(ejecutar_experimento_individual, lote_args)
                resultados_completos.extend(resultados_lote)
                
                # Actualizar barra de progreso
                pbar.update(len(lote_args))
                
                # Calcular estadísticas de progreso
                tiempo_transcurrido = time.time() - inicio_total
                experimentos_completados = len(resultados_completos)
                
                if experimentos_completados > 0:
                    tiempo_por_exp = tiempo_transcurrido / experimentos_completados
                    tiempo_restante = tiempo_por_exp * (total_ejecuciones - experimentos_completados)
                    
                    # Contar soluciones factibles hasta ahora
                    factibles = sum(1 for r in resultados_completos if r.get('factible', False))
                    porcentaje_factibles = (factibles / experimentos_completados) * 100
                    
                    # Actualizar información de la barra
                    pbar.set_postfix({
                        'Factibles': f'{porcentaje_factibles:.1f}%',
                        'ETA': f'{tiempo_restante/60:.1f}m'
                    })
                
                # Guardar progreso intermedio cada 1000 experimentos
                if len(resultados_completos) % 1000 == 0:
                    df_intermedio = pd.DataFrame(resultados_completos)
                    df_intermedio.to_csv(f"{carpeta_experimento}/resultados_intermedio.csv", index=False)
    
    # Crear DataFrame final
    df_resultados = pd.DataFrame(resultados_completos)
    
    # Guardar resultados completos
    archivo_master = f"{carpeta_experimento}/master_table.csv"
    df_resultados.to_csv(archivo_master, index=False)
    
    tiempo_total = time.time() - inicio_total
    
    print("\n" + "="*80)
    print("EXPERIMENTO COMPLETADO")
    print("="*80)
    print(f"Resultados guardados en: {archivo_master}")
    print(f"Total de filas en master_table: {len(df_resultados)}")
    print(f"Tiempo total de ejecución: {tiempo_total/3600:.2f} horas ({tiempo_total/60:.1f} minutos)")
    
    # Estadísticas rápidas
    factibles = df_resultados['factible'].sum()
    mejor_fitness = df_resultados[df_resultados['factible']]['fitness_final'].min() if factibles > 0 else "N/A"
    
    print(f"Soluciones factibles encontradas: {factibles}/{total_ejecuciones} ({factibles/total_ejecuciones*100:.1f}%)")
    print(f"Mejor fitness encontrado: {mejor_fitness}")
    print(f"Eficiencia de procesamiento: {total_ejecuciones/tiempo_total:.1f} experimentos/segundo")
    print("="*80)
    
    return df_resultados, carpeta_experimento

def ejecutar_prueba_rapida(num_procesos=None):
    """Ejecuta una prueba rápida con parámetros reducidos."""
    
    # Parámetros reducidos para prueba rápida
    global PARAMETROS, SEMILLAS
    PARAMETROS_ORIGINAL = PARAMETROS.copy()
    SEMILLAS_ORIGINAL = SEMILLAS.copy()
    
    # Modificar temporalmente para prueba rápida
    PARAMETROS = {
        'numb_bees': [50, 150],           # Solo 2 niveles
        'max_itrs': [100, 500],           # Solo 2 niveles
        'p_f': [0.25, 0.45],             # Solo 2 niveles
        'limit': [10, 50],               # Solo 2 niveles
        'modification_rate': [0.1, 0.5]  # Solo 2 niveles
    }
    SEMILLAS = [2, 3, 5, 7, 11]  # Solo 5 semillas
    
    try:
        print("="*80)
        print("EJECUTANDO PRUEBA RÁPIDA")
        print("="*80)
        print(f"Configuración reducida: 2^5 = 32 combinaciones × 5 repeticiones = 160 experimentos")
        
        if num_procesos is None:
            num_procesos = configurar_procesamiento()
        
        resultados, carpeta = ejecutar_experimento_completo(num_procesos)
        
        print("\n✅ PRUEBA RÁPIDA COMPLETADA EXITOSAMENTE")
        print(f"Resultados guardados en: {carpeta}")
        
        return resultados, carpeta
        
    finally:
        # Restaurar parámetros originales
        PARAMETROS = PARAMETROS_ORIGINAL
        SEMILLAS = SEMILLAS_ORIGINAL

if __name__ == "__main__":
    # Verificar si es una prueba rápida
    if len(sys.argv) > 1 and sys.argv[1] == "--prueba-rapida":
        resultados, carpeta = ejecutar_prueba_rapida()
    else:
        # Ejecutar experimento completo
        resultados, carpeta = ejecutar_experimento_completo()
    
    # Mostrar resumen por combinación
    print("\nResumen por combinación (Top 10 mejores):")
    print("-" * 50)
    
    resumen = resultados.groupby('id_combinacion').agg({
        'fitness_final': ['mean', 'std', 'min'],
        'factible': 'sum',
        'tiempo_ejecucion': 'mean'
    }).round(4)
    
    resumen.columns = ['fitness_mean', 'fitness_std', 'fitness_min', 'factibles', 'tiempo_mean']
    resumen = resumen.sort_values('fitness_mean')
    
    print(resumen.head(10))
    
    print(f"\nTodos los resultados guardados en: {carpeta}")
