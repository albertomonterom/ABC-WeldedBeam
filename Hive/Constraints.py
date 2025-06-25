import numpy as np

def evaluate_constraints(x):
    """
    Evalúa todas las restricciones del problema Welded Beam Design.
    
    Parámetros:
    ----------
    x : list
        Vector de diseño [x1, x2, x3, x4] = [h, l, t, b]
    
    Retorna:
    -------
    dict
        Diccionario con:
        - 'feasible': bool, True si cumple todas las restricciones
        - 'violation': float, suma total de violaciones (0 si es factible)
        - 'constraints': list, valores de cada restricción g_i
    """
    x1, x2, x3, x4 = x  # h, l, t, b

    # Parámetros constantes
    P = 6000       # carga
    L = 14         # longitud
    E = 30e6       # módulo de elasticidad
    G = 12e6       # módulo de corte
    tau_max = 13600
    sigma_max = 30000
    delta_max = 0.25

    # Cálculos intermedios
    M = P * (L + x2 / 2)
    R = np.sqrt((x2**2) / 4 + ((x1 + x3)**2) / 4)
    J = 2 * (np.sqrt(2) * x1 * x2 * ((x2**2) / 12 + ((x1 + x3)**2) / 4))
    
    tau_p = P / (np.sqrt(2) * x1 * x2)
    tau_pp = M * R / J
    # CORRECCIÓN: Fórmula correcta del esfuerzo cortante
    tau = np.sqrt(tau_p**2 + 2 * tau_p * tau_pp * x2 / (2 * R) + tau_pp**2)
    
    sigma = 6 * P * L / (x4 * x3**2)
    delta = 4 * P * L**3 / (E * x4 * x3**3)

    # Carga crítica de pandeo P_c (fórmula exacta)
    term1 = (x3**2) * (x4**6) / 36
    term2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
    Pc = (4.013 * E * np.sqrt(term1) / (L**2)) * (1 - term2)

    # Restricciones (cada una debe ser <= 0)
    g = [
        tau - tau_max,                                          # g1: esfuerzo cortante
        sigma - sigma_max,                                      # g2: esfuerzo de flexión
        x1 - x4,                                               # g3: restricción geométrica
        0.10471 * x1**2 + 0.04811 * x3 * x4 * (14 + x2) - 5.0, # g4: restricción de costo
        0.125 - x1,                                            # g5: límite mínimo x1
        delta - delta_max,                                      # g6: deflexión máxima
        P - Pc,                                                # g7: carga de pandeo
    ]

    # Calcular violaciones (solo valores positivos cuentan como violación)
    violations = [max(0, gi) for gi in g]
    total_violation = sum(violations)
    
    # Es factible si todas las restricciones se cumplen (g_i <= 0)
    feasible = all(gi <= 0 for gi in g)

    return {
        'feasible': feasible,
        'violation': total_violation,
        'constraints': g
    }

# Funciones de compatibilidad (para no romper código existente)
def constraints(x):
    """Función de compatibilidad: retorna solo si es factible."""
    return evaluate_constraints(x)['feasible']

def constraint_violation(x):
    """Función de compatibilidad: retorna solo la violación total."""
    return evaluate_constraints(x)['violation']