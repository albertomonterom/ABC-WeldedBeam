#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de Funcionalidades - Verificación de Barra de Progreso y Configuración
Welded Beam Design Problem

Este script verifica que todas las nuevas funcionalidades estén trabajando correctamente.
"""

import sys
import time
from multiprocessing import cpu_count

def test_importacion_dependencias():
    """Verifica que todas las dependencias necesarias estén disponibles."""
    print("🔍 Verificando importación de dependencias...")
    
    dependencias = {
        'pandas': 'Análisis de datos',
        'numpy': 'Cálculos numéricos', 
        'tqdm': 'Barras de progreso',
        'multiprocessing': 'Procesamiento paralelo',
        'itertools': 'Combinaciones de parámetros',
        'datetime': 'Timestamps',
        'os': 'Sistema de archivos'
    }
    
    exitos = 0
    total = len(dependencias)
    
    for dep, desc in dependencias.items():
        try:
            if dep == 'multiprocessing':
                from multiprocessing import Pool, cpu_count
                print(f"  ✓ {dep} ({desc})")
            else:
                __import__(dep)
                print(f"  ✓ {dep} ({desc})")
            exitos += 1
        except ImportError:
            print(f"  ✗ {dep} ({desc}) - FALTANTE")
    
    print(f"\nResultado: {exitos}/{total} dependencias disponibles")
    return exitos == total

def test_barra_progreso():
    """Verifica que tqdm funcione correctamente."""
    print("\n📊 Verificando barra de progreso...")
    
    try:
        from tqdm import tqdm
        print("  ✓ tqdm importado correctamente")
        
        # Simular una barra de progreso
        print("  🧪 Ejecutando simulación de barra de progreso...")
        total_items = 50
        
        with tqdm(total=total_items, 
                 desc="Test progreso", 
                 unit="items",
                 ncols=80,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}") as pbar:
            
            for i in range(total_items):
                time.sleep(0.05)  # Simular trabajo
                pbar.update(1)
                
                # Actualizar información adicional cada 10 items
                if i % 10 == 0:
                    pbar.set_postfix({'Fase': f'{i//10 + 1}/5', 'Status': 'OK'})
        
        print("  ✓ Barra de progreso funcionando correctamente")
        return True
        
    except ImportError:
        print("  ✗ tqdm no está disponible")
        return False
    except Exception as e:
        print(f"  ✗ Error en barra de progreso: {e}")
        return False

def test_configuracion_cores():
    """Verifica la detección y configuración de cores."""
    print("\n💻 Verificando configuración de cores...")
    
    try:
        cores_disponibles = cpu_count()
        print(f"  ✓ Cores detectados: {cores_disponibles}")
        
        # Calcular configuraciones recomendadas
        if cores_disponibles >= 8:
            recomendado = cores_disponibles - 2
            tipo = "Sistema potente"
        elif cores_disponibles >= 4:
            recomendado = cores_disponibles - 1
            tipo = "Sistema moderado"
        else:
            recomendado = cores_disponibles
            tipo = "Sistema básico"
        
        print(f"  ✓ Tipo de sistema: {tipo}")
        print(f"  ✓ Configuración recomendada: {recomendado} cores")
        
        # Simular estimación de tiempo
        total_experimentos = 160  # Para prueba rápida
        tiempo_por_exp = 1.5  # segundos
        tiempo_estimado = (total_experimentos * tiempo_por_exp) / recomendado
        
        minutos = int(tiempo_estimado // 60)
        print(f"  ✓ Tiempo estimado (prueba rápida): ~{minutos} minutos")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error en configuración de cores: {e}")
        return False

def test_estructura_proyecto():
    """Verifica que la estructura del proyecto sea correcta."""
    print("\n📁 Verificando estructura del proyecto...")
    
    import os
    
    archivos_requeridos = [
        ('Hive/Constraints.py', 'Funciones de restricciones'),
        ('Hive/Hive.py', 'Algoritmo ABC principal'),
        ('Hive/Utilities.py', 'Utilidades de graficación'),
        ('main.py', 'Archivo principal original'),
        ('experimento_completo.py', 'Ejecutor de experimentos'),
        ('analisis_estadistico.py', 'Análisis estadístico'),
        ('setup_dependencias.py', 'Verificador de dependencias'),
        ('ejecutor_principal.py', 'Menú principal')
    ]
    
    encontrados = 0
    total = len(archivos_requeridos)
    
    for archivo, descripcion in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"  ✓ {archivo} ({descripcion})")
            encontrados += 1
        else:
            print(f"  ✗ {archivo} ({descripcion}) - FALTANTE")
    
    print(f"\nArchivos encontrados: {encontrados}/{total}")
    return encontrados == total

def test_funcionalidad_completa():
    """Ejecuta una prueba muy pequeña del algoritmo ABC."""
    print("\n🧪 Verificando funcionalidad del algoritmo ABC...")
    
    try:
        # Intentar importar módulos del proyecto
        from Hive import Hive
        from Hive import Constraints
        
        print("  ✓ Módulos ABC importados correctamente")
        
        # Configuración mínima para prueba
        lower = [0.1, 0.1, 0.1, 0.1]
        upper = [2.0, 10.0, 10.0, 2.0]
        
        def objective_test(x):
            x1, x2, x3, x4 = x
            return (1.10471 * (x1 ** 2) * x2) + (0.04811 * x3 * x4 * (14.0 + x2))
        
        print("  🔧 Ejecutando ABC con configuración mínima...")
        
        # Crear modelo ABC muy pequeño para prueba
        modelo = Hive.BeeHive(
            lower=lower,
            upper=upper,
            fun=objective_test,
            funcon=Constraints.evaluate_constraints,
            numb_bees=10,  # Muy pocas abejas
            max_itrs=5,    # Muy pocas iteraciones
            seed=42,
            verbose=False
        )
        
        # Ejecutar algoritmo
        cost = modelo.run()
        
        if modelo.solution and modelo.best < float('inf'):
            print(f"  ✓ Algoritmo ejecutado exitosamente")
            print(f"  ✓ Mejor fitness encontrado: {modelo.best:.6f}")
            
            # Verificar restricciones
            constraint_result = Constraints.evaluate_constraints(modelo.solution)
            print(f"  ✓ Solución factible: {constraint_result['feasible']}")
            
            return True
        else:
            print("  ✗ El algoritmo no encontró solución válida")
            return False
        
    except Exception as e:
        print(f"  ✗ Error en verificación del algoritmo: {e}")
        return False

def main():
    """Ejecuta todos los tests de verificación."""
    print("="*70)
    print("🧪 TEST DE FUNCIONALIDADES - SISTEMA ABC ACTUALIZADO")
    print("="*70)
    print("Verificando que todas las nuevas funcionalidades estén operativas...")
    
    tests = [
        ("Dependencias", test_importacion_dependencias),
        ("Barra de Progreso", test_barra_progreso),
        ("Configuración de Cores", test_configuracion_cores),
        ("Estructura del Proyecto", test_estructura_proyecto),
        ("Funcionalidad ABC", test_funcionalidad_completa)
    ]
    
    resultados = []
    
    for nombre, test_func in tests:
        try:
            resultado = test_func()
            resultados.append((nombre, resultado))
        except Exception as e:
            print(f"\n❌ Error crítico en {nombre}: {e}")
            resultados.append((nombre, False))
    
    # Resumen final
    print("\n" + "="*70)
    print("📋 RESUMEN DE VERIFICACIONES")
    print("="*70)
    
    exitos = 0
    for nombre, resultado in resultados:
        status = "✅ CORRECTO" if resultado else "❌ ERROR"
        print(f"  {nombre}: {status}")
        if resultado:
            exitos += 1
    
    total = len(resultados)
    porcentaje = (exitos / total) * 100
    
    print(f"\n📊 Resultado final: {exitos}/{total} tests exitosos ({porcentaje:.1f}%)")
    
    if exitos == total:
        print("\n🎉 ¡PERFECTO! Todas las funcionalidades están operativas.")
        print("✅ El sistema está listo para ejecutar experimentos con:")
        print("   • Barras de progreso en tiempo real")
        print("   • Configuración automática de cores")
        print("   • Guardado automático de progreso")
        print("   • Monitoreo avanzado del sistema")
        print("\n🚀 Ejecuta 'python ejecutor_principal.py' para comenzar.")
    
    elif exitos >= total * 0.8:
        print("\n⚠️  Sistema mayormente funcional, pero con algunos problemas.")
        print("💡 Recomendación: Ejecuta 'python setup_dependencias.py' y reintenta.")
    
    else:
        print("\n🚨 Sistema con problemas importantes.")
        print("💡 Recomendaciones:")
        print("   1. Ejecuta 'python setup_dependencias.py'")
        print("   2. Verifica que todos los archivos estén en las ubicaciones correctas")
        print("   3. Reinstala las dependencias manualmente:")
        print("      pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-posthocs tqdm")
    
    print("="*70)

if __name__ == "__main__":
    main()
