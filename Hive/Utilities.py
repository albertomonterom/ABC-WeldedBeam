# ---- MODULE DOCSTRING

__doc__ = """

Descripción:
-----------

Una serie de funciones utilitarias (funciones para graficar).

"""

# ---- IMPORTACIÓN DE MÓDULOS

try:
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
except:
    raise ImportError("Install 'matplotlib' to plot convergence results.")

# ---- GRÁFICA DE CONVERGENCIA

def ConvergencePlot(cost, save_path=None, show_stats=True, titulo_personalizado=None):
    """
    Monitorea la convergencia del algoritmo con visualización moderna y estadísticas.

    Parámetros:
    ----------
        :param dict cost: diccionario con los valores medios y mínimos de evaluación
                          por iteración, devueltos por el optimizador.
        :param str save_path: ruta opcional para guardar el gráfico (ej: "convergencia.png")
        :param bool show_stats: si mostrar estadísticas adicionales
        :param str titulo_personalizado: título personalizado para el gráfico
    """
    
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Limpiar y procesar datos
    def limpiar_datos(valores):
        """Limpia valores None e infinitos, interpolando cuando es necesario."""
        datos_limpios = []
        
        for i, val in enumerate(valores):
            if val is None or val == sys.float_info.max or np.isnan(val) or np.isinf(val):
                # Buscar el valor válido más cercano hacia atrás
                valor_previo = None
                for j in range(i-1, -1, -1):
                    if (valores[j] is not None and 
                        valores[j] != sys.float_info.max and 
                        not np.isnan(valores[j]) and 
                        not np.isinf(valores[j])):
                        valor_previo = valores[j]
                        break
                
                # Si no hay valor previo, buscar hacia adelante
                if valor_previo is None:
                    for j in range(i+1, len(valores)):
                        if (valores[j] is not None and 
                            valores[j] != sys.float_info.max and 
                            not np.isnan(valores[j]) and 
                            not np.isinf(valores[j])):
                            valor_previo = valores[j]
                            break
                
                datos_limpios.append(valor_previo if valor_previo is not None else 0)
            else:
                datos_limpios.append(val)
        
        return datos_limpios
    
    # Procesar datos
    best_values = limpiar_datos(cost["best"])
    mean_values = limpiar_datos(cost.get("mean", []))
    
    # Verificar que tenemos datos válidos
    if not best_values:
        print("⚠️ No hay datos válidos para graficar")
        return
    
    # Calcular estadísticas
    fitness_inicial = best_values[0] if best_values else 0
    fitness_final = best_values[-1] if best_values else 0
    mejora_absoluta = fitness_inicial - fitness_final
    mejora_porcentual = (mejora_absoluta / fitness_inicial * 100) if fitness_inicial != 0 else 0
    
    # Crear figura con diseño moderno
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    # Colores modernos
    color_best = '#2E86AB'      # Azul principal
    color_mean = '#F24236'      # Rojo para promedio
    color_inicio = '#A23B72'    # Morado para inicio
    color_final = '#F18F01'     # Naranja para final
    
    # Graficar líneas principales
    iteraciones = range(len(best_values))
    
    # Línea del mejor valor
    ax.plot(iteraciones, best_values, color=color_best, linewidth=3, 
            label='Mejor fitness', alpha=0.9, zorder=3)
    
    # Línea del valor promedio (si existe)
    if mean_values and len(mean_values) == len(best_values):
        ax.plot(iteraciones, mean_values, color=color_mean, linewidth=2.5, 
                linestyle='--', alpha=0.7, label='Fitness promedio', zorder=2)
    
    # Área de relleno para mostrar la mejora
    if len(best_values) > 1:
        ax.fill_between(iteraciones, best_values, alpha=0.2, color=color_best, zorder=1)
    
    # Puntos destacados para inicio y final
    ax.scatter([0], [fitness_inicial], color=color_inicio, s=120, zorder=5, 
               edgecolors='white', linewidth=2, label=f'Inicial: {fitness_inicial:.4f}')
    
    if len(best_values) > 1:
        ax.scatter([len(best_values)-1], [fitness_final], color=color_final, s=120, zorder=5, 
                   edgecolors='white', linewidth=2, label=f'Final: {fitness_final:.4f}')
    
    # Configurar ejes y etiquetas
    ax.set_xlabel('Iteración', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=14, fontweight='bold')
    
    # Título dinámico
    if titulo_personalizado:
        titulo = titulo_personalizado
    else:
        titulo = f'Convergencia del Algoritmo ABC\nMejora: {mejora_porcentual:.1f}% ({mejora_absoluta:.4f})'
    
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)
    
    # Configurar leyenda
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, 
              shadow=True, framealpha=0.9)
    
    # Configurar grilla
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Mejorar límites de ejes
    ax.set_xlim(0, max(1, len(best_values)-1))
    
    # Ajustar límites Y para mejor visualización
    if len(best_values) > 1:
        y_min, y_max = min(best_values), max(best_values)
        y_range = y_max - y_min
        if y_range > 0:
            margin = y_range * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
    
    # Añadir estadísticas adicionales si se solicita
    if show_stats and len(best_values) > 1:
        # Calcular estadísticas adicionales
        convergencia_final = best_values[-10:] if len(best_values) >= 10 else best_values
        estabilidad = np.std(convergencia_final) / np.mean(convergencia_final) * 100 if np.mean(convergencia_final) != 0 else 0
        
        # Encontrar iteración donde se alcanzó el 90% de la mejora
        if mejora_absoluta > 0:
            objetivo_90 = fitness_inicial - (mejora_absoluta * 0.9)
            iter_90 = next((i for i, val in enumerate(best_values) if val <= objetivo_90), len(best_values))
        else:
            iter_90 = 0
        
        # Crear texto de estadísticas
        stats_text = (f'Estadísticas:\n'
                      f'• Iteraciones: {len(best_values)}\n'
                      f'• 90% mejora en: {iter_90} iter\n'
                      f'• Estabilidad final: {estabilidad:.2f}%')
        
        # Añadir caja de estadísticas
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightblue', alpha=0.8), zorder=6)
    
    # Añadir anotaciones para puntos clave si hay suficiente espacio
    if len(best_values) > 10:
        # Anotar punto inicial
        ax.annotate(f'Inicio\n{fitness_inicial:.4f}', 
                    xy=(0, fitness_inicial), 
                    xytext=(len(best_values)*0.15, fitness_inicial + (fitness_inicial-fitness_final)*0.1),
                    arrowprops=dict(arrowstyle='->', color=color_inicio, lw=2),
                    fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Anotar punto final
        if len(best_values) > 1:
            ax.annotate(f'Final\n{fitness_final:.4f}', 
                        xy=(len(best_values)-1, fitness_final), 
                        xytext=(len(best_values)*0.85, fitness_final + (fitness_inicial-fitness_final)*0.1),
                        arrowprops=dict(arrowstyle='->', color=color_final, lw=2),
                        fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Configurar diseño
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"📊 Gráfico guardado en: {save_path}")
        except Exception as e:
            print(f"⚠️ Error al guardar gráfico: {e}")
    
    # Mostrar gráfico
    plt.show()
    
    # Imprimir resumen en consola
    print(f"\n📈 Resumen de Convergencia:")
    print(f"   • Fitness inicial: {fitness_inicial:.6f}")
    print(f"   • Fitness final: {fitness_final:.6f}")
    print(f"   • Mejora absoluta: {mejora_absoluta:.6f}")
    print(f"   • Mejora porcentual: {mejora_porcentual:.2f}%")
    print(f"   • Iteraciones totales: {len(best_values)}")
    
    return fig, ax


# Función de compatibilidad con la versión original
def ConvergencePlotOriginal(cost):
    """Versión original simplificada para compatibilidad."""
    return ConvergencePlot(cost, show_stats=False)

# ---- FIN