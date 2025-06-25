from Hive import Hive
from Hive import Constraints
from Hive import Utilities

# Límites del vector solución
lower = [0.1, 0.1, 0.1, 0.1]
upper = [2.0, 10.0, 10.0, 2.0]

def objective(x):
    x1, x2, x3, x4 = x
    return (1.10471 * (x1 ** 2) * x2) + (0.04811 * x3 * x4 * (14.0 + x2))

def reproducir_mejor_resultado():
    """
    Reproduce exactamente el mejor resultado del experimento.
    
    Datos del mejor resultado:
    - id_combinacion: 72
    - semilla: 43
    - fitness_final: 1.801761544556665
    - Variables: x1=0.1918, x2=3.9510, x3=8.8788, x4=0.2140
    """
    
    print("="*70)
    print("REPRODUCIENDO EL MEJOR RESULTADO DEL EXPERIMENTO")
    print("="*70)
    print("Configuración exacta del experimento id_combinacion=72, semilla=43")
    print("Fitness objetivo: 1.801761544556665")
    print("="*70)
    
    # Configuración exacta del mejor resultado
    model = Hive.BeeHive(
        lower=lower,
        upper=upper,
        fun=objective,
        funcon=Constraints.evaluate_constraints,
        numb_bees=50,              # Parámetros exactos del mejor resultado
        max_itrs=1000,
        p_f=0.75,
        limit=10,
        modification_rate=0.1,
        seed=43,                   # ← ¡CLAVE! Semilla exacta
        verbose=True
    )
    
    print(f"Semilla configurada: {model.seed}")
    print("Ejecutando algoritmo...")
    
    # Ejecutar el algoritmo
    cost = model.run()
    
    # Resultados
    print("\n" + "="*70)
    print("RESULTADOS DE LA REPRODUCCIÓN")
    print("="*70)
    print(f"Fitness obtenido:    {model.best:.15f}")
    print(f"Fitness esperado:    1.801761544556665")
    print(f"Diferencia:          {abs(model.best - 1.801761544556665):.2e}")
    
    if model.solution:
        x1, x2, x3, x4 = model.solution
        print(f"\nVariables obtenidas:")
        print(f"  x1: {x1:.15f}")
        print(f"  x2: {x2:.15f}")
        print(f"  x3: {x3:.15f}")
        print(f"  x4: {x4:.15f}")
        
        print(f"\nVariables esperadas:")
        print(f"  x1: 0.191847521546638")
        print(f"  x2: 3.951021247960733")
        print(f"  x3: 8.878841203666800")
        print(f"  x4: 0.214021961821953")
        
        # Verificar factibilidad
        constraint_result = Constraints.evaluate_constraints(model.solution)
        print(f"\nFactibilidad: {constraint_result['feasible']}")
        print(f"Violación total: {constraint_result['violation']:.10f}")
        
        # Verificar si la reproducción fue exitosa
        tolerancia = 1e-10
        if abs(model.best - 1.801761544556665) < tolerancia:
            print("\n✅ ¡REPRODUCCIÓN EXITOSA!")
            print("Se obtuvo exactamente el mismo resultado.")
        else:
            print("\n⚠️  REPRODUCCIÓN PARCIAL")
            print("El resultado es similar pero no idéntico.")
            print("Esto puede deberse a diferencias en:")
            print("  • Precisión numérica del sistema")
            print("  • Versiones de librerías")
            print("  • Optimizaciones del compilador")
    
    else:
        print("\n❌ ERROR: No se encontró solución")
    
    print("="*70)
    
    # Mostrar gráfica de convergencia
    Utilities.ConvergencePlot(cost)
    
    return model.best, model.solution

def run():
    # Parámetros de configuración del algoritmo ABC
    # ==============================================
    
    # PARÁMETRO D: limit - Límite de abandono de fuente
    # Control del balance exploración/explotación
    # - Valores pequeños (10-20): Mayor exploración, más diversidad
    # - Valores grandes (100+): Mayor explotación, convergencia más lenta
    limit = 50  # Número máximo de intentos sin mejora antes de abandonar una fuente
    
    # PARÁMETRO E: modification_rate - Factor de modificación de la solución
    # Control de la intensidad de mutación
    # - Valores pequeños (0.1-0.5): Pasos pequeños, búsqueda local fina
    # - Valores grandes (1.0-2.0): Pasos grandes, mayor diversificación
    modification_rate = 1.0  # Factor que controla el tamaño del paso en la mutación
    
    # Inicializar el modelo con la función unificada y los nuevos parámetros
    model = Hive.BeeHive(
        lower     = lower,  # Límite inferior del vector de búsqueda
        upper     = upper,  # Límite superior del vector de búsqueda
        fun       = objective,  # Función objetivo a minimizar
        numb_bees = 30,     # Número total de abejas
        max_itrs  = 100,    # Número máximo de iteraciones
        funcon    = Constraints.evaluate_constraints,  # Función unificada de restricciones
        p_f       = 0.45,   # Probabilidad de jerarquización estocástica
        limit     = limit,  # NUEVO: Límite de abandono de fuente
        modification_rate = modification_rate,  # NUEVO: Factor de modificación
        verbose   = True    # Si es True, imprime información de cada iteración
    )
    
    # Imprimir configuración de parámetros
    print("="*60)
    print("CONFIGURACIÓN DEL ALGORITMO ABC")
    print("="*60)
    print(f"Número de abejas: {model.size}")
    print(f"Iteraciones máximas: {model.max_itrs}")
    print(f"Límite de abandono (limit): {model.limit}")
    print(f"Factor de modificación (modification_rate): {model.modification_rate}")
    print(f"Probabilidad p_f: {model.p_f}")
    print(f"Semilla aleatoria: {model.seed}")
    print("="*60)
    
    # Correr el algoritmo
    cost = model.run()

    # Gráfica de convergencia
    Utilities.ConvergencePlot(cost)

    # Imprimir nuestra mejor solución
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"Fitness Value ABC (menos es mejor): {model.best:.6f}")
    print(f"Mejor solución (x) encontrada: {model.solution}")
    
    # Verificar que la mejor solución cumple las restricciones
    if model.solution:
        constraint_result = Constraints.evaluate_constraints(model.solution)
        
        print(f"\n¿La mejor solución es factible? {constraint_result['feasible']}")
        print(f"Violación total de restricciones: {constraint_result['violation']:.6f}")
        
        # Mostrar valores de las variables
        x1, x2, x3, x4 = model.solution
        print(f"\nVariables de diseño:")
        print(f"  x1 (h - altura de soldadura): {x1:.4f}")
        print(f"  x2 (l - longitud de soldadura): {x2:.4f}")
        print(f"  x3 (t - altura de la barra): {x3:.4f}")
        print(f"  x4 (b - espesor de la barra): {x4:.4f}")
        
        # Mostrar restricciones individuales para debugging
        if not constraint_result['feasible']:
            print(f"\nViolaciones individuales:")
            constraint_names = [
                "g1: Esfuerzo cortante",
                "g2: Esfuerzo de flexión", 
                "g3: Restricción geométrica",
                "g4: Restricción de costo",
                "g5: Límite mínimo x1",
                "g6: Deflexión máxima",
                "g7: Carga de pandeo"
            ]
            for i, (name, g_val) in enumerate(zip(constraint_names, constraint_result['constraints'])):
                if g_val > 0:
                    print(f"  {name}: {g_val:.6f} (VIOLADA)")
                else:
                    print(f"  {name}: {g_val:.6f} (OK)")
    else:
        print("No se encontró ninguna solución válida.")
    
    print("="*60)

def test_parameter_combinations():
    """
    Función opcional para probar diferentes combinaciones de parámetros
    y observar su efecto en el rendimiento del algoritmo.
    """
    print("\n" + "="*80)
    print("PRUEBA DE DIFERENTES COMBINACIONES DE PARÁMETROS")
    print("="*80)
    
    # Combinaciones de parámetros a probar
    parameter_combinations = [
        {"limit": 30, "modification_rate": 0.5, "description": "Exploración alta, pasos pequeños"},
        {"limit": 50, "modification_rate": 1.0, "description": "Balance estándar"},
        {"limit": 100, "modification_rate": 1.5, "description": "Explotación alta, pasos grandes"},
    ]
    
    results = []
    
    for i, params in enumerate(parameter_combinations):
        print(f"\nPrueba {i+1}: {params['description']}")
        print(f"  limit = {params['limit']}, modification_rate = {params['modification_rate']}")
        
        model = Hive.BeeHive(
            lower=lower,
            upper=upper,
            fun=objective,
            numb_bees=20,
            max_itrs=50,
            funcon=Constraints.evaluate_constraints,
            limit=params['limit'],
            modification_rate=params['modification_rate'],
            verbose=False
        )
        
        cost = model.run()
        
        results.append({
            'params': params,
            'best_value': model.best,
            'solution': model.solution,
            'feasible': model.solution and Constraints.evaluate_constraints(model.solution)['feasible']
        })
        
        print(f"  Mejor valor: {model.best:.6f}")
        print(f"  Factible: {results[-1]['feasible']}")
    
    # Mostrar resumen de resultados
    print(f"\n{'='*80}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*80}")
    for i, result in enumerate(results):
        print(f"Prueba {i+1}: {result['best_value']:.6f} (Factible: {result['feasible']})")
    
    return results

if __name__ == "__main__":
    # Ejecutar optimización principal
    # run()

    # Reproducir mejor
    reproducir_mejor_resultado()
    
    # Opcional: Probar diferentes combinaciones de parámetros
    # Descomenta la siguiente línea para ejecutar las pruebas
    #test_parameter_combinations()