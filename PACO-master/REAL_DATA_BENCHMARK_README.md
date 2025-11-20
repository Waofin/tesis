# PACO Benchmark con Datos Reales del EIDC Phase 1

Este módulo adapta el benchmark de rendimiento de PACO para trabajar con los datos reales del **Exoplanet Imaging Data Challenge Phase 1**.

## Referencias

- [EIDC Phase 1 Datasets](https://exoplanet-imaging-challenge.github.io/datasets1/)
- [EIDC Phase 1 Submission](https://exoplanet-imaging-challenge.github.io/submission1/)
- [EIDC Starting Kit](https://exoplanet-imaging-challenge.github.io/startingkit/)

## Estructura de Datos

Los datasets del challenge están organizados según la convención:

```
subchallenge1/
├── sphere_irdis_cube_1.fits      # Cubo de imágenes (3D: n_frames, height, width)
├── sphere_irdis_pa_1.fits        # Ángulos de paralaje (1D: n_frames)
├── sphere_irdis_psf_1.fits       # PSF template (2D: height, width)
├── sphere_irdis_pxscale_1.fits   # Pixel scale (float)
├── nirc2_cube_1.fits
├── nirc2_pa_1.fits
├── nirc2_psf_1.fits
├── nirc2_pxscale_1.fits
└── ... (más datasets)
```

### Instrumentos Disponibles

- **VLT/SPHERE-IRDIS**: 3 datasets (sphere_irdis_*)
- **Keck/NIRC2**: 3 datasets (nirc2_*)
- **LBT/LMIRCam**: 3 datasets (lmircam_*)

## Uso

### Opción 1: Notebook Interactivo (Recomendado)

Abre y ejecuta `PACO_Real_Data_Benchmark.ipynb`:

```bash
jupyter notebook PACO_Real_Data_Benchmark.ipynb
```

El notebook incluye:
- Carga y exploración de datasets
- Configuración del benchmark
- Ejecución del benchmark
- Análisis de resultados
- Visualización y guardado

### Opción 2: Script Python

```python
from pathlib import Path
from paco_real_data_benchmark import benchmark_real_dataset
from paco_performance_benchmark import BENCHMARK_CONFIG

# Configuración
data_dir = Path("../subchallenge1")
config = BENCHMARK_CONFIG.copy()
config['cpu_counts'] = [1, 2, 4, 8]
n_trials = 3

# Ejecutar benchmark en un dataset
results = benchmark_real_dataset(
    data_dir=data_dir,
    instrument='sphere_irdis',
    dataset_id=1,
    config=config,
    crop_size=None,  # None = tamaño completo, (100, 100) = recortar
    n_trials=n_trials
)
```

### Opción 3: Benchmark en Todos los Datasets

```python
from paco_real_data_benchmark import benchmark_all_datasets

all_results = benchmark_all_datasets(
    data_dir=Path("../subchallenge1"),
    config=config,
    instruments=['sphere_irdis', 'nirc2'],  # Especificar instrumentos
    dataset_ids=[1, 2, 3],  # Especificar IDs
    crop_size=None,
    n_trials=3
)
```

## Funciones Principales

### `load_challenge_dataset(data_dir, instrument, dataset_id)`

Carga un dataset completo desde archivos FITS.

**Parámetros:**
- `data_dir`: Path al directorio con los archivos FITS
- `instrument`: Nombre del instrumento ('sphere_irdis', 'nirc2', 'lmircam')
- `dataset_id`: ID del dataset (1, 2, 3, etc.)

**Retorna:**
- Diccionario con: `cube`, `angles`, `psf`, `pxscale`, `instrument`, `dataset_id`

### `prepare_dataset_for_paco(dataset, crop_size=None)`

Prepara un dataset para ser usado con PACO, determinando automáticamente parámetros como `psf_rad` y `patch_size`.

**Parámetros:**
- `dataset`: Dataset cargado con `load_challenge_dataset`
- `crop_size`: Opcional, tupla (height, width) para recortar imágenes grandes

**Retorna:**
- Diccionario con datos formateados para PACO

### `benchmark_real_dataset(data_dir, instrument, dataset_id, config, crop_size=None, n_trials=3)`

Ejecuta el benchmark completo en un dataset específico.

**Retorna:**
- Diccionario con resultados del benchmark (mismo formato que `paco_performance_benchmark.run_benchmark`)

## Configuración

### Parámetros del Benchmark

```python
config = {
    'cpu_counts': [1, 2, 4, 8],  # CPUs a probar
    'psf_rad': None,              # Se calcula automáticamente desde el PSF
    'patch_size': None,           # Se calcula automáticamente
    'image_size': None,           # Se obtiene del dataset
    'sigma': 2.0,                 # Parámetro del modelo gaussiano
    'n_frames': None             # Se obtiene del dataset
}
```

### Recorte de Imágenes

Para datasets grandes (ej: LMIRCam con 160x160 píxeles), puedes recortar las imágenes para acelerar el benchmark:

```python
crop_size = (100, 100)  # Recortar a 100x100 píxeles centrado
```

Esto es útil para:
- Pruebas rápidas
- Sistemas con memoria limitada
- Análisis de escalabilidad sin procesar imágenes completas

## Análisis de Resultados

Los resultados se pueden analizar usando las mismas funciones del benchmark original:

```python
from paco_performance_benchmark import analyze_amdahl, plot_benchmark_results, save_results

# Análisis
analysis = analyze_amdahl(results)

# Visualización
plot_benchmark_results(results, analysis, output_dir)

# Guardar
json_path, csv_path = save_results(results, analysis, output_dir)
```

## Notas Importantes

1. **Tiempo de Ejecución**: Los datasets reales pueden ser grandes (ej: LMIRCam tiene ~700MB por dataset). El benchmark puede tomar varias horas dependiendo del número de CPUs y trials.

2. **Memoria**: Asegúrate de tener suficiente RAM. Para datasets grandes, considera usar `crop_size`.

3. **Formato de Datos**: Los archivos FITS deben seguir la convención de nombres del challenge. Si tus archivos tienen nombres diferentes, ajusta la función `load_challenge_dataset`.

4. **Normalización del PSF**: El PSF se normaliza automáticamente (suma = 1.0) para compatibilidad con PACO.

5. **Compatibilidad**: Este módulo requiere:
   - `astropy` para cargar archivos FITS
   - `paco_performance_benchmark` (el benchmark original)
   - Todas las dependencias de PACO

## Ejemplo Completo

```python
from pathlib import Path
from paco_real_data_benchmark import (
    load_challenge_dataset,
    prepare_dataset_for_paco,
    benchmark_real_dataset
)
from paco_performance_benchmark import (
    BENCHMARK_CONFIG,
    analyze_amdahl,
    plot_benchmark_results,
    save_results
)

# 1. Cargar dataset
data_dir = Path("../subchallenge1")
dataset = load_challenge_dataset(data_dir, 'sphere_irdis', 1)

# 2. Preparar para PACO
paco_data = prepare_dataset_for_paco(dataset, crop_size=None)

# 3. Configurar benchmark
config = BENCHMARK_CONFIG.copy()
config['cpu_counts'] = [1, 2, 4, 8]

# 4. Ejecutar benchmark
results = benchmark_real_dataset(
    data_dir, 'sphere_irdis', 1, config, n_trials=3
)

# 5. Analizar
analysis = analyze_amdahl(results)

# 6. Visualizar y guardar
output_dir = Path("output/real_data_benchmark")
plot_benchmark_results(results, analysis, output_dir)
save_results(results, analysis, output_dir)
```

## Troubleshooting

### Error: "Archivos faltantes"
- Verifica que los archivos FITS estén en el directorio correcto
- Verifica que los nombres de archivos sigan la convención: `instrument_type_id.fits`

### Error: "Cube tiene forma inesperada"
- Algunos datasets pueden tener formatos ligeramente diferentes
- Revisa el contenido del archivo FITS con `astropy.io.fits`

### Error: "Número de ángulos != número de frames"
- El dataset puede tener inconsistencias
- El script ajusta automáticamente truncando al mínimo común

### Memoria insuficiente
- Usa `crop_size` para recortar imágenes grandes
- Reduce el número de CPUs o trials
- Procesa un dataset a la vez

