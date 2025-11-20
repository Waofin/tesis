# PACO Performance Benchmark

Este sistema implementa un análisis de rendimiento riguroso del algoritmo PACO comparando diferentes estrategias de paralelización, siguiendo la metodología científica del **Exoplanet Imaging Data Challenge Phase 1**.

## Objetivos

1. **Comparar rendimiento** entre versiones serial, multiprocessing y loky/joblib
2. **Analizar speedup** según la ley de Amdahl
3. **Mantener rigurosidad científica** siguiendo estándares del EIDC
4. **Generar reportes** con visualizaciones y métricas

## Referencias

- [EIDC Phase 1 Submission Guidelines](https://exoplanet-imaging-challenge.github.io/submission1/)
- [EIDC Phase 1 Datasets](https://exoplanet-imaging-challenge.github.io/datasets1/)
- [EIDC Starting Kit](https://exoplanet-imaging-challenge.github.io/startingkit/)
- Flasseur et al. 2018: PACO algorithm

## Estructura

```
PACO-master/
├── paco_performance_benchmark.py  # Script principal de benchmarking
├── PACO_Performance_Analysis.ipynb  # Notebook interactivo
└── PERFORMANCE_BENCHMARK_README.md  # Este archivo
```

## Requisitos

```bash
pip install numpy matplotlib pandas joblib
```

Opcional (para profiling avanzado):
```bash
pip install line_profiler
```

## Uso

### Opción 1: Script Python

Ejecutar directamente el script:

```bash
cd PACO-master
python paco_performance_benchmark.py
```

El script:
1. Genera datos de prueba sintéticos
2. Ejecuta benchmarks con múltiples trials
3. Analiza resultados según ley de Amdahl
4. Genera visualizaciones
5. Guarda resultados en JSON y CSV

### Opción 2: Notebook Jupyter

Abrir `PACO_Performance_Analysis.ipynb` y ejecutar las celdas interactivamente.

### Opción 3: Uso como módulo

```python
from paco_performance_benchmark import (
    generate_test_data,
    run_benchmark,
    analyze_amdahl,
    plot_benchmark_results,
    save_results
)

# Configurar benchmark
config = {
    'nFrames': 50,
    'image_size': (100, 100),
    'angles_range': 60,
    'patch_size': 49,
    'psf_rad': 4,
    'sigma': 2.0,
    'n_trials': 5,
    'cpu_counts': [1, 2, 4, 8],
    'output_dir': 'output/performance_analysis/'
}

# Generar datos
test_data = generate_test_data(
    config['nFrames'],
    config['image_size'],
    config['angles_range'],
    seed=42
)

# Ejecutar benchmark
results = run_benchmark(config, test_data, n_trials=5)

# Analizar
analysis = analyze_amdahl(results)

# Visualizar
plot_benchmark_results(results, analysis)

# Guardar
save_results(results, analysis, config, config['output_dir'])
```

## Configuración

Los parámetros del benchmark se pueden ajustar en el diccionario `BENCHMARK_CONFIG`:

- `nFrames`: Número de frames en la secuencia ADI (típico: 50-100)
- `image_size`: Tamaño de cada imagen (height, width)
- `angles_range`: Rango total de rotación en grados
- `patch_size`: Tamaño del patch circular (recomendado: 49 para radio 4)
- `psf_rad`: Radio del PSF en píxeles
- `sigma`: Parámetro sigma para modelo gaussiano del PSF
- `n_trials`: Número de ejecuciones para estadísticas robustas
- `cpu_counts`: Lista de números de CPUs a probar
- `output_dir`: Directorio para guardar resultados

## Métricas Generadas

### 1. Tiempos de Ejecución
- Tiempo promedio y desviación estándar para cada configuración
- Comparación entre métodos (serial, multiprocessing, loky)

### 2. Speedup
- Speedup real vs teórico
- Comparación con speedup perfecto (lineal)
- Predicción según ley de Amdahl

### 3. Eficiencia
- Eficiencia = Speedup / Número de CPUs
- Indica qué tan bien se aprovechan los recursos

### 4. Fracción Paralelizable (p)
- Estimación de la fracción del código que es paralelizable
- Usado para predicciones teóricas según Amdahl

## Ley de Amdahl

El speedup teórico según la ley de Amdahl es:

```
Speedup = 1 / ((1 - p) + p / n)
```

Donde:
- `p`: Fracción paralelizable del código (0-1)
- `n`: Número de procesadores

El script estima `p` a partir de los resultados experimentales y compara con el speedup teórico.

## Salidas

El benchmark genera:

1. **JSON**: Resultados completos con todos los datos (`benchmark_results_TIMESTAMP.json`)
2. **CSV**: Tabla resumen con métricas principales (`benchmark_summary_TIMESTAMP.csv`)
3. **PNG**: Gráficos de visualización (`benchmark_plots.png`)

Todos los archivos se guardan en el directorio especificado en `output_dir`.

## Interpretación de Resultados

### Speedup > 1
- Indica mejora con paralelización
- Valores cercanos al número de CPUs indican buena escalabilidad

### Eficiencia < 1
- Indica overhead de paralelización
- Valores > 0.7 generalmente se consideran buenos

### Fracción Paralelizable (p)
- `p` cercano a 1: Código altamente paralelizable
- `p` cercano a 0: Código con mucho overhead serial
- `p` intermedio: Mejoras posibles pero limitadas por parte serial

## Notas Importantes

1. **Reproducibilidad**: Usar `seed` fijo en `generate_test_data()` para resultados reproducibles
2. **Ambiente Controlado**: Las variables de entorno `MKL_NUM_THREADS`, etc. se configuran a 1 para evitar interferencias
3. **Múltiples Trials**: Se recomienda al menos 5 trials para estadísticas robustas
4. **Tiempo de Ejecución**: El benchmark completo puede tomar varios minutos dependiendo de la configuración

## Troubleshooting

### Error: "loky no disponible"
```bash
pip install joblib
```

### Error: "Cannot import paco"
Asegurarse de estar en el directorio correcto y que el módulo `paco` esté en el PYTHONPATH.

### Resultados inconsistentes
- Aumentar `n_trials` para más robustez estadística
- Verificar que no haya otros procesos usando CPU
- Asegurar que las variables de entorno de threading estén configuradas

## Próximos Pasos

1. Probar con datasets reales del EIDC Phase 1
2. Comparar con otros algoritmos del challenge
3. Optimizar según resultados del profiling
4. Integrar métricas de detección (SNR maps) para validación científica

## Contacto

Para preguntas o mejoras, referirse al repositorio principal del proyecto.

