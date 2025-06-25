#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba rapida de experimentos ABC
"""
import sys
sys.path.append('.')
from experimento_completo import *

# Modificar parametros para prueba rapida
PARAMETROS_PRUEBA = {
    'numb_bees': [50, 150],           # Solo 2 niveles en lugar de 3
    'max_itrs': [100, 500],           # Solo 2 niveles en lugar de 3
    'p_f': [0.25, 0.45],             # Solo 2 niveles en lugar de 3
    'limit': [10, 50],               # Solo 2 niveles en lugar de 3
    'modification_rate': [0.1, 0.5]  # Solo 2 niveles en lugar de 3
}

# Usar solo las primeras 5 semillas
SEMILLAS_PRUEBA = [2, 3, 5, 7, 11]

def generar_combinaciones_prueba():
    import itertools
    nombres_params = list(PARAMETROS_PRUEBA.keys())
    valores_params = list(PARAMETROS_PRUEBA.values())
    
    combinaciones = []
    for combo in itertools.product(*valores_params):
        combinacion = dict(zip(nombres_params, combo))
        combinaciones.append(combinacion)
    
    return combinaciones

if __name__ == "__main__":
    # Usar parametros de prueba
    global PARAMETROS, SEMILLAS
    PARAMETROS = PARAMETROS_PRUEBA
    SEMILLAS = SEMILLAS_PRUEBA
    
    # Ejecutar experimento
    print("Iniciando prueba rapida...")
    print(f"Total combinaciones: {2**5} = 32")
    print(f"Total ejecuciones: 32 * 5 = 160")
    
    resultados, carpeta = ejecutar_experimento_completo()
    print(f"Prueba rapida completada exitosamente!")
    print(f"Resultados guardados en: {carpeta}")
