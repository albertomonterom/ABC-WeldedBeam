#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejecutor Principal - Análisis Completo de Parámetros ABC (VERSIÓN CORREGIDA)
Welded Beam Design Problem

CAMBIOS:
- Función ejecutar_comando_interactivo() para ver salida en tiempo real
- Función ejecutar_comando() para verificaciones silenciosas
- Mejor manejo de la salida de experimentos con barras de progreso
"""

import os
import sys
import subprocess
from multiprocessing import cpu_count

def ejecutar_comando_interactivo(comando, descripcion):
    """
    Ejecuta un comando mostrando la salida en tiempo real.
    Ideal para experimentos con barras de progreso.
    """
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {descripcion}")
    print(f"{'='*60}")
    
    try:
        if isinstance(comando, list):
            # Ejecutar sin capturar salida para ver progreso en tiempo real
            resultado = subprocess.run(comando, check=True)
        else:
            resultado = subprocess.run(comando, shell=True, check=True)
        
        print(f"\n✅ {descripcion} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en {descripcion}:")
        print(f"Código de error: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {descripcion} interrumpido por el usuario")
        return False

def ejecutar_comando(comando, descripcion):
    """
    Ejecuta un comando capturando la salida (para verificaciones silenciosas).
    """
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {descripcion}")
    print(f"{'='*60}")
    
    try:
        if isinstance(comando, list):
            resultado = subprocess.run(comando, check=True, capture_output=True, text=True)
        else:
            resultado = subprocess.run(comando, shell=True, check=True, capture_output=True, text=True)
        
        if resultado.stdout:
            print(resultado.stdout)
        print(f"✅ {descripcion} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {descripcion}:")
        print(f"Código de error: {e.returncode}")
        if e.stdout:
            print(f"Salida: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def verificar_estructura_proyecto():
    """Verifica que la estructura del proyecto sea correcta."""
    archivos_requeridos = [
        'Hive/Constraints.py',
        'Hive/Hive.py', 
        'Hive/Utilities.py',
        'main.py',
        'experimento_completo.py',
        'analisis_estadistico.py',
        'setup_dependencias.py'
    ]
    
    print("Verificando estructura del proyecto...")
    faltantes = []
    
    for archivo in archivos_requeridos:
        if os.path.exists(archivo):
            print(f"  ✅ {archivo}")
        else:
            print(f"  ❌ {archivo} (faltante)")
            faltantes.append(archivo)
    
    if faltantes:
        print(f"\n❌ Faltan {len(faltantes)} archivos requeridos.")
        print("Asegúrate de que todos los archivos estén en sus ubicaciones correctas.")
        return False
    
    print("\n✅ Estructura del proyecto verificada correctamente")
    return True

def mostrar_menu():
    """Muestra el menú de opciones."""
    cores_disponibles = cpu_count()
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO DE PARÁMETROS - ALGORITMO ABC")
    print("Welded Beam Design Problem")
    print("="*80)
    print(f"💻 Sistema detectado: {cores_disponibles} cores/procesadores disponibles")
    print("\nOpciones disponibles:")
    print("  1. Verificar e instalar dependencias")
    print("  2. Ejecutar experimentos completos (243 combinaciones × 30 repeticiones)")
    print("  3. Ejecutar análisis estadístico (requiere datos previos)")
    print("  4. Proceso completo (1 + 2 + 3)")
    print("  5. Prueba rápida (32 combinaciones × 5 repeticiones)")
    print("  6. Configuración de rendimiento")
    print("  7. Estado del sistema")
    print("  8. Salir")
    print("="*80)

def mostrar_configuracion_rendimiento():
    """Muestra opciones de configuración de rendimiento."""
    cores_disponibles = cpu_count()
    
    print(f"\n{'='*70}")
    print("CONFIGURACIÓN DE RENDIMIENTO")
    print(f"{'='*70}")
    print(f"💻 Cores/procesadores disponibles: {cores_disponibles}")
    print(f"\n📊 Opciones de configuración:")
    print(f"  • Uso LIGERO:     {max(1, cores_disponibles // 4)} cores (75% del sistema libre)")
    print(f"  • Uso MODERADO:   {max(1, cores_disponibles // 2)} cores (50% del sistema libre)")
    print(f"  • Uso INTENSIVO:  {max(1, cores_disponibles - 1)} cores (mínimo sistema libre)")
    print(f"  • Uso MÁXIMO:     {cores_disponibles} cores (100% del sistema)")
    
    print(f"\n⏱️  Estimaciones de tiempo (experimento completo - 7,290 ejecuciones):")
    print(f"  • {max(1, cores_disponibles // 4)} cores: ~6-8 horas")
    print(f"  • {max(1, cores_disponibles // 2)} cores: ~3-4 horas")
    print(f"  • {max(1, cores_disponibles - 1)} cores: ~1.5-2 horas")
    print(f"  • {cores_disponibles} cores: ~1-1.5 horas")
    
    print(f"\n💡 Recomendaciones:")
    if cores_disponibles >= 8:
        print(f"  ✅ Tu sistema tiene suficientes cores. Recomendado: {cores_disponibles - 2} cores")
        print(f"    (Deja 2 cores libres para el sistema operativo)")
    elif cores_disponibles >= 4:
        print(f"  ✅ Tu sistema es adecuado. Recomendado: {cores_disponibles - 1} cores")
        print(f"    (Deja 1 core libre para el sistema operativo)")
    else:
        print(f"  ⚠️  Tu sistema tiene pocos cores. Considera usar todos ({cores_disponibles})")
        print(f"    (Los experimentos pueden tomar más tiempo)")
    
    print(f"\n🔧 Configuración se realizará antes de cada experimento")
    print(f"{'='*70}")

def mostrar_estado_sistema():
    """Muestra el estado actual del sistema y resultados previos."""
    print(f"\n{'='*70}")
    print("ESTADO DEL SISTEMA")
    print(f"{'='*70}")
    
    cores_disponibles = cpu_count()
    print(f"💻 Cores/procesadores: {cores_disponibles}")
    
    # Verificar dependencias críticas
    print(f"\n📦 Dependencias críticas:")
    dependencias_criticas = ['pandas', 'numpy', 'tqdm', 'multiprocessing']
    
    for dep in dependencias_criticas:
        try:
            if dep == 'multiprocessing':
                import multiprocessing
                print(f"  ✅ {dep}")
            else:
                __import__(dep)
                print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} (faltante)")
    
    # Verificar experimentos previos
    print(f"\n📊 Experimentos previos:")
    if os.path.exists('resultados'):
        experimentos = []
        for item in os.listdir('resultados'):
            ruta_carpeta = os.path.join('resultados', item)
            if os.path.isdir(ruta_carpeta) and item.startswith('experimento_'):
                master_file = os.path.join(ruta_carpeta, 'master_table.csv')
                if os.path.exists(master_file):
                    # Obtener información del experimento
                    try:
                        import pandas as pd
                        df = pd.read_csv(master_file)
                        factibles = df['factible'].sum() if 'factible' in df.columns else 0
                        total = len(df)
                        experimentos.append({
                            'carpeta': item,
                            'total': total,
                            'factibles': factibles,
                            'porcentaje': (factibles/total)*100 if total > 0 else 0
                        })
                    except:
                        experimentos.append({
                            'carpeta': item,
                            'total': 'Error',
                            'factibles': 'Error',
                            'porcentaje': 0
                        })
        
        if experimentos:
            experimentos.sort(key=lambda x: x['carpeta'], reverse=True)
            for exp in experimentos[:5]:  # Mostrar últimos 5
                print(f"  📁 {exp['carpeta']}")
                print(f"     └─ {exp['total']} ejecuciones, {exp['factibles']} factibles ({exp['porcentaje']:.1f}%)")
        else:
            print("  📭 No se encontraron experimentos completados")
    else:
        print("  📭 Carpeta 'resultados' no existe")
    
    # Verificar análisis previos
    print(f"\n📈 Análisis estadísticos previos:")
    if os.path.exists('analisis'):
        analisis = [item for item in os.listdir('analisis') 
                   if os.path.isdir(os.path.join('analisis', item)) and item.startswith('analisis_')]
        if analisis:
            analisis.sort(reverse=True)
            for anal in analisis[:3]:  # Mostrar últimos 3
                print(f"  📈 {anal}")
        else:
            print("  📭 No se encontraron análisis completados")
    else:
        print("  📭 Carpeta 'analisis' no existe")
    
    # Espacio en disco
    print(f"\n💾 Espacio en disco:")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        gb = 1024**3
        print(f"  📊 Total: {total/gb:.1f} GB")
        print(f"  📊 Usado: {used/gb:.1f} GB")
        print(f"  📊 Libre: {free/gb:.1f} GB")
        
        if free/gb < 1:
            print(f"  ⚠️  Espacio bajo (se requieren ~500MB para experimentos)")
        else:
            print(f"  ✅ Espacio suficiente")
    except:
        print("  ❓ No se pudo verificar espacio en disco")
    
    print(f"{'='*70}")

def ejecutar_prueba_rapida():
    """Ejecuta una prueba rápida con configuración de cores."""
    print("💡 Ejecutando prueba rápida...")
    print("Esta prueba usa 32 combinaciones × 5 repeticiones = 160 experimentos")
    
    confirmar = input("¿Continuar? (s/N): ").strip().lower()
    
    if confirmar == 's':
        print("\n🚀 Iniciando prueba rápida con salida en tiempo real...")
        print("📊 Verás la barra de progreso y configuración de cores a continuación:")
        print("-" * 60)
        
        # USAR ejecutar_comando_interactivo en lugar de ejecutar_comando
        exito = ejecutar_comando_interactivo([sys.executable, 'experimento_completo.py', '--prueba-rapida'], 
                                           "Prueba rápida de experimentos")
        if exito:
            print("\n✅ Prueba rápida completada exitosamente.")
            print("El sistema está funcionando correctamente.")
            return True
        else:
            print("\n❌ Error en la prueba rápida.")
            print("Verifica que todos los archivos estén correctamente instalados.")
            return False
    else:
        print("Operación cancelada.")
        return False

def main():
    """Función principal del ejecutor."""
    print("🚀 Bienvenido al Sistema de Análisis de Parámetros ABC")
    print("    Con barras de progreso y configuración de rendimiento")
    
    # Verificar estructura del proyecto
    if not verificar_estructura_proyecto():
        print("\n❌ Estructura del proyecto incorrecta. Abortando.")
        return
    
    while True:
        mostrar_menu()
        
        try:
            opcion = input("\nSelecciona una opción (1-8): ").strip()
            
            if opcion == '1':
                # Verificar dependencias (usar comando silencioso)
                exito = ejecutar_comando([sys.executable, 'setup_dependencias.py'], 
                                       "Verificación de dependencias")
                if not exito:
                    print("❌ Instala las dependencias manualmente antes de continuar:")
                    print("pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-posthocs tqdm")
            
            elif opcion == '2':
                # Ejecutar experimentos completos (usar comando interactivo)
                print("\n⚠️  EXPERIMENTOS COMPLETOS")
                print("Se ejecutarán 243 × 30 = 7,290 experimentos individuales.")
                print("Este proceso incluye:")
                print("  • Configuración automática de cores/procesadores")
                print("  • Barra de progreso en tiempo real")
                print("  • Estimación de tiempo restante")
                print("  • Guardado automático de progreso")
                print("  • Estadísticas de rendimiento")
                
                confirmar = input("\n¿Continuar? (s/N): ").strip().lower()
                
                if confirmar == 's':
                    print("\n🚀 Iniciando experimentos completos...")
                    print("📊 Verás la configuración de cores y barra de progreso:")
                    print("-" * 60)
                    
                    exito = ejecutar_comando_interactivo([sys.executable, 'experimento_completo.py'], 
                                                       "Experimentos completos")
                    if exito:
                        print("\n✅ Experimentos completados exitosamente!")
                        print("Busca la carpeta 'resultados/' para los archivos generados.")
                else:
                    print("Operación cancelada.")
            
            elif opcion == '3':
                # Ejecutar análisis estadístico (usar comando silencioso)
                print("🔍 Buscando archivos master_table.csv...")
                
                # Buscar automáticamente el archivo más reciente
                carpetas_resultados = []
                if os.path.exists('resultados'):
                    for item in os.listdir('resultados'):
                        ruta_carpeta = os.path.join('resultados', item)
                        if os.path.isdir(ruta_carpeta) and item.startswith('experimento_'):
                            master_file = os.path.join(ruta_carpeta, 'master_table.csv')
                            if os.path.exists(master_file):
                                carpetas_resultados.append((item, master_file))
                
                if carpetas_resultados:
                    carpetas_resultados.sort(reverse=True)  # Más reciente primero
                    print(f"📁 Encontrados {len(carpetas_resultados)} experimentos:")
                    
                    for i, (carpeta, archivo) in enumerate(carpetas_resultados[:5]):  # Mostrar máximo 5
                        print(f"  {i+1}. {carpeta}")
                    
                    if len(carpetas_resultados) == 1:
                        seleccion = 0
                        print(f"Seleccionando automáticamente: {carpetas_resultados[0][0]}")
                    else:
                        try:
                            seleccion = int(input("Selecciona experimento (número): ")) - 1
                            if seleccion < 0 or seleccion >= len(carpetas_resultados):
                                print("Selección inválida.")
                                continue
                        except ValueError:
                            print("Entrada inválida.")
                            continue
                    
                    archivo_master = carpetas_resultados[seleccion][1]
                    exito = ejecutar_comando([sys.executable, 'analisis_estadistico.py', archivo_master], 
                                           "Análisis estadístico")
                    
                    if exito:
                        print("\n📊 Análisis estadístico completado exitosamente!")
                        print("Busca la carpeta 'analisis/' para los resultados.")
                else:
                    print("📭 No se encontraron archivos master_table.csv.")
                    print("Ejecuta primero los experimentos (opción 2).")
            
            elif opcion == '4':
                # Proceso completo (usar comandos apropiados)
                print("\n🚀 PROCESO COMPLETO")
                print("Esto ejecutará todo el análisis desde cero:")
                print("  1. ✅ Verificación de dependencias")
                print("  2. ⚙️  Configuración de procesamiento")
                print("  3. 🧪 Experimentos completos (3-6 horas)")
                print("  4. 📊 Análisis estadístico (2-5 minutos)")
                print("  5. 📄 Generación de conclusiones")
                
                confirmar = input("\n¿Continuar con el proceso completo? (s/N): ").strip().lower()
                
                if confirmar == 's':
                    # Paso 1: Dependencias (silencioso)
                    print("\n🔧 Paso 1/4: Verificando dependencias...")
                    if not ejecutar_comando([sys.executable, 'setup_dependencias.py'], 
                                          "Verificación de dependencias"):
                        print("❌ Error en dependencias. Abortando.")
                        continue
                    
                    # Paso 2: Experimentos (interactivo)
                    print("\n🧪 Paso 2/4: Ejecutando experimentos completos...")
                    print("📊 Verás la configuración de cores y barra de progreso:")
                    print("-" * 60)
                    
                    if not ejecutar_comando_interactivo([sys.executable, 'experimento_completo.py'], 
                                                      "Experimentos completos"):
                        print("❌ Error en experimentos. Abortando.")
                        continue
                    
                    # Paso 3: Análisis (silencioso)
                    print("\n📊 Paso 3/4: Ejecutando análisis estadístico...")
                    
                    # Buscar el master_table.csv más reciente
                    carpetas_resultados = []
                    if os.path.exists('resultados'):
                        for item in os.listdir('resultados'):
                            ruta_carpeta = os.path.join('resultados', item)
                            if os.path.isdir(ruta_carpeta) and item.startswith('experimento_'):
                                master_file = os.path.join(ruta_carpeta, 'master_table.csv')
                                if os.path.exists(master_file):
                                    carpetas_resultados.append((item, master_file))
                    
                    if carpetas_resultados:
                        carpetas_resultados.sort(reverse=True)  # Más reciente primero
                        archivo_master = carpetas_resultados[0][1]
                        
                        if ejecutar_comando([sys.executable, 'analisis_estadistico.py', archivo_master], 
                                          "Análisis estadístico"):
                            print("\n" + "="*80)
                            print("🎉 ¡PROCESO COMPLETO FINALIZADO EXITOSAMENTE!")
                            print("="*80)
                            print("📁 Revisa las carpetas 'resultados/' y 'analisis/' para los resultados.")
                            print("📄 Lee 'conclusiones_finales.txt' para el resumen ejecutivo.")
                            print("📊 Examina los gráficos PNG para análisis visual.")
                            print("="*80)
                    else:
                        print("❌ Error: No se encontró master_table.csv después de los experimentos.")
                else:
                    print("Operación cancelada.")
            
            elif opcion == '5':
                # Prueba rápida (usar función corregida)
                print("\n🧪 PRUEBA RÁPIDA")
                print("Esta prueba verifica que todo funcione correctamente:")
                print("  • 32 combinaciones de parámetros")
                print("  • 5 repeticiones por combinación")
                print("  • Total: 160 experimentos (~5-15 minutos)")
                print("  • Incluye configuración de cores y barra de progreso")
                
                ejecutar_prueba_rapida()
            
            elif opcion == '6':
                # Configuración de rendimiento
                mostrar_configuracion_rendimiento()
            
            elif opcion == '7':
                # Estado del sistema
                mostrar_estado_sistema()
            
            elif opcion == '8':
                print("\n👋 ¡Hasta luego!")
                print("Gracias por usar el Sistema de Análisis de Parámetros ABC")
                break
            
            else:
                print("❌ Opción inválida. Selecciona un número del 1 al 8.")
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Proceso interrumpido por el usuario.")
            print("Presiona Ctrl+C de nuevo para salir completamente, o continúa con el menú.")
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("Si el problema persiste, verifica la estructura de archivos.")

if __name__ == "__main__":
    main()