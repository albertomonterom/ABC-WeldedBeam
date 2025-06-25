# Análisis Estadístico Completo - Algoritmo ABC
## Welded Beam Design Problem

Este proyecto implementa un análisis estadístico completo para optimizar los parámetros del algoritmo ABC (Artificial Bee Colony) aplicado al problema Welded Beam Design.

## 📁 Estructura del Proyecto

```
proyecto/
├── Hive/
│   ├── Constraints.py      # Funciones de evaluación de restricciones
│   ├── Hive.py            # Implementación del algoritmo ABC
│   └── Utilities.py       # Funciones de graficación
├── main.py                # Archivo principal original
├── experimento_completo.py # Ejecutor de experimentos (243 combinaciones × 30 repeticiones)
├── analisis_estadistico.py # Análisis estadístico completo (ANOVA, post-hoc, etc.)
├── setup_dependencias.py  # Verificador e instalador de dependencias
├── ejecutor_principal.py  # Script principal integrado
└── README.md              # Este archivo
```

## 🚀 Instalación y Configuración

### 1. Instalar los Scripts

Coloca todos los archivos generados en tu estructura de proyecto:

- `experimento_completo.py` → raíz del proyecto
- `analisis_estadistico.py` → raíz del proyecto  
- `setup_dependencias.py` → raíz del proyecto
- `ejecutor_principal.py` → raíz del proyecto

### 2. Verificar Dependencias

```bash
python setup_dependencias.py
```

### 3. Instalación Manual de Dependencias (si es necesario)

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-posthocs
```

## 🧪 Metodología del Experimento

### Parámetros Analizados (5 factores × 3 niveles = 3⁵ = 243 combinaciones)

| Parámetro | Nivel 1 (Bajo) | Nivel 2 (Medio) | Nivel 3 (Alto) | Descripción |
|-----------|----------------|------------------|----------------|-------------|
| **numb_bees** | 50 | 150 | 450 | Número de abejas (tamaño población) |
| **max_itrs** | 100 | 500 | 1000 | Número máximo de iteraciones |
| **p_f** | 0.25 | 0.45 | 0.75 | Probabilidad jerarquización estocástica |
| **limit** | 10 | 50 | 150 | Límite de abandono de fuente |
| **modification_rate** | 0.1 | 0.5 | 0.9 | Factor de modificación de solución |

### Diseño Experimental

- **Total de combinaciones:** 243
- **Repeticiones por combinación:** 30 (usando semillas fijas)
- **Total de ejecuciones:** 7,290
- **Semillas utilizadas:** Primeros 30 números primos [2, 3, 5, 7, 11, ..., 113]

## 🔧 Uso del Sistema

### Opción 1: Ejecutor Principal (Recomendado)

```bash
python ejecutor_principal.py
```

El ejecutor principal te guiará a través de un menú interactivo con las siguientes opciones:

1. **Verificar dependencias** - Instala librerías necesarias
2. **Ejecutar experimentos completos** - 7,290 ejecuciones (3-6 horas)
3. **Ejecutar análisis estadístico** - Analiza resultados previos
4. **Proceso completo** - Ejecuta todo desde cero
5. **Prueba rápida** - Verificación con 160 ejecuciones (5 minutos)
6. **Salir**

### Opción 2: Ejecución Manual Paso a Paso

#### Paso 1: Experimentos Completos
```bash
python experimento_completo.py
```

#### Paso 2: Análisis Estadístico
```bash
python analisis_estadistico.py resultados/experimento_YYYYMMDD_HHMM/master_table.csv
```

### Opción 3: Prueba Rápida

Para verificar que todo funciona correctamente:
```bash
# Usar opción 5 del ejecutor principal
python ejecutor_principal.py
```

## 📊 Análisis Estadístico Implementado

### 1. Análisis Exploratorio
- Estadísticas descriptivas por combinación
- Diagramas de caja (boxplots) por parámetro
- Histogramas y distribuciones

### 2. Verificación de Supuestos ANOVA
- **Normalidad de residuos:** Prueba de Shapiro-Wilk
- **Homogeneidad de varianzas:** Prueba de Levene
- Gráficos de diagnóstico

### 3. Análisis Inferencial

#### Si se cumplen supuestos (Análisis Paramétrico):
- **ANOVA Factorial Completo** con efectos principales e interacciones
- **Pruebas Post-hoc de Tukey** para comparaciones múltiples
- **Análisis de interacciones** con gráficos

#### Si NO se cumplen supuestos (Análisis No Paramétrico):
- **Pruebas de Kruskal-Wallis** por parámetro
- **Pruebas Post-hoc de Dunn** con corrección Bonferroni
- **Análisis visual de interacciones**

### 4. Resultados y Conclusiones
- Identificación de parámetros más influyentes
- Mejor configuración encontrada
- Recomendaciones finales
- Variables de diseño óptimas

## 📁 Estructura de Resultados

### Después de Ejecutar Experimentos:
```
resultados/
└── experimento_YYYYMMDD_HHMM/
    ├── master_table.csv              # Datos completos (7,290 filas)
    ├── configuracion_experimento.csv # Parámetros usados
    └── resultados_intermedio.csv     # Backup de progreso
```

### Después del Análisis Estadístico:
```
analisis/
└── analisis_YYYYMMDD_HHMM/
    ├── estadisticas_descriptivas.csv # Estadísticas por combinación
    ├── tabla_anova.csv               # Resultados ANOVA (si aplica)
    ├── resultados_kruskal_wallis.csv # Resultados no paramétricos (si aplica)
    ├── tukey_[parametro].txt         # Pruebas post-hoc Tukey
    ├── dunn_[parametro].csv          # Pruebas post-hoc Dunn
    ├── conclusiones_finales.txt      # Resumen y recomendaciones
    ├── boxplots_parametros.png       # Diagramas de caja
    ├── distribucion_fitness.png      # Histogramas y Q-Q plots
    ├── diagnosticos_modelo.png       # Gráficos de diagnóstico ANOVA
    └── graficos_interacciones.png    # Análisis de interacciones
```

## ⏱️ Tiempos de Ejecución Estimados

| Proceso | Tiempo Aproximado | Descripción |
|---------|-------------------|-------------|
| **Prueba rápida** | 5-10 minutos | 32 combinaciones × 5 repeticiones = 160 ejecuciones |
| **Experimentos completos** | 3-6 horas | 243 combinaciones × 30 repeticiones = 7,290 ejecuciones |
| **Análisis estadístico** | 2-5 minutos | Procesamiento de resultados y generación de gráficos |

*Los tiempos varían según el hardware utilizado.*

## 🔍 Interpretación de Resultados

### Archivo `conclusiones_finales.txt`

Este archivo contiene:

1. **Parámetros más influyentes** - Factores con efectos significativos
2. **Mejor configuración encontrada** - Combinación óptima de parámetros
3. **Variables de diseño** - Valores óptimos para el problema Welded Beam
4. **Recomendaciones finales** - Configuración sugerida para uso práctico
5. **Resumen estadístico** - Métricas generales del experimento

### Interpretación de p-valores

- **p < 0.05:** Efecto estadísticamente significativo
- **p ≥ 0.05:** No hay evidencia de efecto significativo

### Gráficos de Interacción

- **Líneas paralelas:** No hay interacción
- **Líneas que se cruzan:** Interacción presente
- **Líneas con pendientes diferentes:** Posible interacción

## 🚨 Solución de Problemas

### Error: "Module not found"
```bash
python setup_dependencias.py
```

### Error: "No space left on device"
- Los experimentos generan ~100MB de datos
- Asegúrate de tener al menos 500MB libres

### Error: "Process killed" 
- Reduce el número de procesos paralelos en `experimento_completo.py`
- Modifica la línea: `num_procesos = min(cpu_count(), 4)`

### Experimentos muy lentos
- Usa la **prueba rápida** para verificar funcionamiento
- Considera ejecutar en horarios de menor carga del sistema

## 📞 Soporte

Para problemas específicos:

1. Verifica que todos los archivos estén en las ubicaciones correctas
2. Ejecuta `python setup_dependencias.py` para verificar librerías
3. Usa la **prueba rápida** para verificar funcionamiento básico
4. Revisa los logs de error en la consola

## 🎯 Próximos Pasos

Después de completar el análisis:

1. **Revisa `conclusiones_finales.txt`** para los resultados principales
2. **Examina los gráficos** para entender las tendencias
3. **Implementa la configuración recomendada** en tu algoritmo ABC
4. **Documenta los hallazgos** para futuras referencias

¡Buena suerte con tu análisis estadístico! 🚀
