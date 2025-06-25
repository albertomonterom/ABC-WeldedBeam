#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup de Dependencias para Análisis Estadístico ABC
Verifica e instala las librerías necesarias para el análisis completo.
"""

import subprocess
import sys
import importlib
import os

# Lista de dependencias requeridas
DEPENDENCIAS = [
    'pandas',
    'numpy', 
    'matplotlib',
    'seaborn',
    'scipy',
    'statsmodels',
    'scikit-posthocs',
    'tqdm'  # Añadido para barras de progreso
]

def configurar_encoding():
    """Configura el encoding para compatibilidad con Windows."""
    if os.name == 'nt':  # Windows
        try:
            # Intentar configurar UTF-8 en Windows
            import locale
            import codecs
            
            # Configurar encoding de salida
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
                return True
        except:
            pass
    return False

def get_symbols():
    """Retorna símbolos compatibles según el sistema."""
    # Intentar usar Unicode primero
    try:
        test_str = "✓ ✗"
        # Probar si se puede codificar
        test_str.encode(sys.stdout.encoding or 'cp1252')
        return {"ok": "✓", "error": "✗", "bullet": "•"}
    except (UnicodeEncodeError, AttributeError):
        # Usar ASCII si hay problemas
        return {"ok": "[OK]", "error": "[X]", "bullet": "*"}

def verificar_dependencia(nombre_paquete):
    """Verifica si un paquete está instalado."""
    try:
        importlib.import_module(nombre_paquete.replace('-', '_'))
        return True
    except ImportError:
        return False

def instalar_paquete(nombre_paquete):
    """Instala un paquete usando pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", nombre_paquete], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Función principal para setup de dependencias."""
    # Configurar encoding si es posible
    encoding_ok = configurar_encoding()
    
    # Obtener símbolos compatibles
    symbols = get_symbols()
    
    print("="*60)
    print("VERIFICACION E INSTALACION DE DEPENDENCIAS")
    print("="*60)
    
    if not encoding_ok and os.name == 'nt':
        print("NOTA: Usando simbolos ASCII para compatibilidad con Windows")
        print("-" * 60)
    
    faltantes = []
    
    # Verificar dependencias
    print("Verificando dependencias existentes...")
    for dep in DEPENDENCIAS:
        if verificar_dependencia(dep):
            print(f"  {symbols['ok']} {dep}")
        else:
            print(f"  {symbols['error']} {dep} (faltante)")
            faltantes.append(dep)
    
    # Instalar faltantes
    if faltantes:
        print(f"\nInstalando {len(faltantes)} dependencias faltantes...")
        for dep in faltantes:
            print(f"  Instalando {dep}...")
            if instalar_paquete(dep):
                print(f"    {symbols['ok']} {dep} instalado correctamente")
            else:
                print(f"    {symbols['error']} Error instalando {dep}")
    else:
        print(f"\n{symbols['ok']} Todas las dependencias estan instaladas!")
    
    print("\n" + "="*60)
    print("VERIFICACION FINAL")
    print("="*60)
    
    # Verificación final
    todas_ok = True
    for dep in DEPENDENCIAS:
        if verificar_dependencia(dep):
            print(f"  {symbols['ok']} {dep}")
        else:
            print(f"  {symbols['error']} {dep} (problema)")
            todas_ok = False
    
    if todas_ok:
        print(f"\n{symbols['ok']} Sistema listo para ejecutar los experimentos!")
        print("\nProximos pasos:")
        print(f"  {symbols['bullet']} Ejecuta: python ejecutor_principal.py")
        print(f"  {symbols['bullet']} Selecciona opcion 5 para prueba rapida")
        print(f"  {symbols['bullet']} O opcion 2 para experimento completo")
    else:
        print(f"\n{symbols['error']} Algunos paquetes fallan. Instalacion manual:")
        print("pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-posthocs tqdm")
        print("\nO ejecuta este script nuevamente.")

if __name__ == "__main__":
    main()