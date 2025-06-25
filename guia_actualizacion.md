# 🚀 Guía de Actualización - Sistema ABC con Barras de Progreso

## 📋 Nuevas Funcionalidades Añadidas

### ✨ Características Principales

1. **🎯 Barra de Progreso en Tiempo Real**
   - Muestra progreso visual durante experimentos
   - Estimación de tiempo restante (ETA)
   - Porcentaje de soluciones factibles en tiempo real
   - Velocidad de procesamiento (experimentos/segundo)

2. **⚙️ Configuración Avanzada de Procesamiento**
   - Selección manual del número de cores a usar
   - Configuración automática inteligente
   - Estimaciones de tiempo basadas en hardware
   - Recomendaciones de rendimiento por sistema

3. **📊 Monitoreo de Sistema Mejorado**
   - Estado de dependencias en tiempo real
   - Información de experimentos previos
   - Espacio en disco disponible
   - Estadísticas de rendimiento

4. **💾 Guardado Automático de Progreso**
   - Respaldo cada 1000 experimentos
   - Recuperación automática en caso de interrupción
   - Información detallada de configuración

## 🔧 Instalación de Actualizaciones

### Paso 1: Reemplazar Archivos

Reemplaza estos archivos en tu proyecto con las versiones actualizadas:

```
📁 proyecto/
├── setup_dependencias.py      ← REEMPLAZAR (nueva dependencia: tqdm)
├── experimento_completo.py    ← REEMPLAZAR (barra progreso + config cores)
├── ejecutor_principal.py      ← REEMPLAZAR (menú mejorado + opciones)
└── GUIA_ACTUALIZACION.md      ← NUEVO ARCHIVO
```

### Paso 2: Instalar Nueva Dependencia

```bash
# Opción A: Automático
python setup_dependencias.py

# Opción B: Manual
pip install tqdm
```

### Paso 3: Verificar Instalación

```bash
python ejecutor_principal.py
# Selecciona opción 7 "Estado del sistema"
```

## 🎮 Nuevas Opciones del Menú

### Menú Principal Actualizado

```
ANÁLISIS COMPLETO DE PARÁMETROS - ALGORITMO ABC
================================================================
💻 Sistema detectado: 8 cores/procesadores disponibles

Opciones disponibles:
  1. Verificar e instalar dependencias
  2. Ejecutar experimentos completos (243 combinaciones × 30 repeticiones)
  3. Ejecutar análisis estadístico (requiere datos previos)
  4. Proceso completo (1 + 2 + 3)
  5. Prueba rápida (32 combinaciones × 5 repeticiones)
  6. Configuración de rendimiento          ← NUEVO
  7. Estado del sistema                    ← NUEVO
  8. Salir
```

### 🆕 Opción 6: Configuración de Rendimiento

Muestra información detallada sobre:
- Cores disponibles en tu sistema
- Recomendaciones de uso (ligero/moderado/intensivo/máximo)
- Estimaciones de tiempo por configuración
- Consejos específicos para tu hardware

### 🆕 Opción 7: Estado del Sistema

Proporciona un resumen completo:
- Hardware disponible (cores/procesadores)
- Estado de dependencias críticas
- Experimentos previos realizados
- Análisis estadísticos completados
- Espacio en disco disponible

## ⚡ Configuración de Cores Durante Experimentos

### Proceso de Configuración Automática

Cuando ejecutes experimentos (opciones 2, 4, o 5), verás:

```
============================================================
CONFIGURACIÓN DE PROCESAMIENTO
============================================================
Cores/procesadores disponibles: 8
Recomendaciones:
  • Uso ligero: 2 cores (75% disponible para otras tareas)
  • Uso moderado: 4 cores (50% disponible)
  • Uso intensivo: 7 cores (máximo rendimiento)
  • Uso máximo: 8 cores (100% del sistema)

Selecciona número de cores a usar (1-8) o 'auto' para automático: 
```

### Opciones de Configuración

- **Número específico (1-8)**: Control manual exacto
- **'auto'**: Configuración automática inteligente
  - ≥8 cores: usa todos menos 2
  - 4-7 cores: usa todos menos 1  
  - <4 cores: usa todos disponibles

### Estimaciones de Tiempo

El sistema calcula automáticamente:
- Tiempo estimado basado en cores seleccionados
- Confirmación antes de proceder
- ETA actualizado durante ejecución

## 📊 Nueva Barra de Progreso

### Información Mostrada

```
Procesando experimentos: 45%|████████████▌             | 3285/7290 [1:23:45<1:45:32, 25.3exp/s] Factibles: 78.2%, ETA: 105.5m
```

**Elementos de la barra:**
- **45%**: Porcentaje completado
- **████████████▌**: Barra visual
- **3285/7290**: Experimentos completados/total
- **[1:23:45<1:45:32]**: Tiempo transcurrido < tiempo restante
- **25.3exp/s**: Velocidad de procesamiento
- **Factibles: 78.2%**: Porcentaje de soluciones válidas
- **ETA: 105.5m**: Tiempo estimado restante

### Guardado Automático

- **Progreso intermedio**: Cada 1000 experimentos
- **Configuración**: Al inicio de cada ejecución
- **Recuperación**: Automática si se interrumpe

## 🧪 Prueba Rápida Mejorada

### Nueva Funcionalidad

```bash
# Opción A: Desde el menú principal
python ejecutor_principal.py
# Selecciona opción 5

# Opción B: Línea de comandos directa
python experimento_completo.py --prueba-rapida
```

### Configuración de Prueba Rápida

- **Combinaciones**: 2^5 = 32 (en lugar de 3^5 = 243)
- **Repeticiones**: 5 (en lugar de 30)
- **Total**: 160 experimentos (en lugar de 7,290)
- **Tiempo**: 5-15 minutos (en lugar de 3-6 horas)

## 🚨 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'tqdm'"

```bash
pip install tqdm
# o
python setup_dependencias.py
```

### Barra de Progreso No Aparece

1. Verifica que `tqdm` esté instalado
2. Ejecuta desde terminal/cmd (no desde IDE)
3. Actualiza tqdm: `pip install --upgrade tqdm`

### Experimentos Muy Lentos

1. Usa **opción 6** para ver recomendaciones de rendimiento
2. Reduce número de cores si hay otros procesos ejecutándose
3. Verifica espacio en disco con **opción 7**

### Interrupción de Experimentos

- Los experimentos se guardan cada 1000 ejecuciones
- Busca archivo `resultados_intermedio.csv` para recuperar progreso
- Puedes reanudar desde el progreso guardado

## 📈 Mejoras de Rendimiento

### Optimizaciones Implementadas

1. **Procesamiento Adaptativo**
   - Lotes dinámicos basados en número de cores
   - Balanceamiento automático de carga

2. **Memoria Optimizada**
   - Procesamiento por lotes para evitar sobrecarga
   - Liberación automática de memoria

3. **Monitoreo Inteligente**
   - Actualización eficiente de progreso
   - Cálculos optimizados de estadísticas

### Recomendaciones de Uso

**Para sistemas con 8+ cores:**
```
Configuración recomendada: 6-7 cores
Tiempo estimado: 1-2 horas (experimento completo)
```

**Para sistemas con 4-7 cores:**
```
Configuración recomendada: cores-1
Tiempo estimado: 2-4 horas (experimento completo)
```

**Para sistemas con <4 cores:**
```
Configuración recomendada: todos los cores
Tiempo estimado: 4-8 horas (experimento completo)
```

## 🔍 Archivos de Configuración Generados

### Nuevos Archivos en `resultados/experimento_YYYYMMDD_HHMM/`

```
📁 resultados/experimento_20241220_1430/
├── master_table.csv              # Datos principales
├── configuracion_experimento.csv # Parámetros usados
├── info_ejecucion.csv           # ← NUEVO: Info de rendimiento
└── resultados_intermedio.csv    # ← NUEVO: Backup de progreso
```

### Contenido de `info_ejecucion.csv`

```csv
configuracion,valor
num_procesos,6
total_ejecuciones,7290
tiempo_estimado,~2h 15m
timestamp,20241220_1430
```

## 🎯 Próximos Pasos Recomendados

1. **Ejecuta una prueba rápida** (opción 5) para verificar funcionamiento
2. **Revisa la configuración de rendimiento** (opción 6) para tu sistema
3. **Ejecuta el experimento completo** cuando tengas tiempo suficiente
4. **Monitorea el progreso** usando la nueva barra de progreso

## 📞 Soporte

Si encuentras problemas:

1. **Verifica estado del sistema**: opción 7
2. **Ejecuta prueba rápida**: opción 5  
3. **Reinstala dependencias**: opción 1
4. **Revisa logs de error** en la consola

¡Las nuevas funcionalidades están listas para usar! 🚀