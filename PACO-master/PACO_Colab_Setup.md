# Configuración de PACO Benchmark en Google Colab

Esta guía explica cómo ejecutar el benchmark de PACO con datos reales en Google Colab.

## Ventajas de Google Colab

- ✅ **Gratis** con GPU/TPU disponible
- ✅ **Almacenamiento temporal** de ~80 GB
- ✅ **Sin límites de RAM** (hasta cierto punto)
- ✅ **Acceso a Google Drive** para datos persistentes
- ✅ **No ocupa espacio en tu disco local**

## Opción 1: Subir datos a Google Drive

### Paso 1: Subir archivos FITS a Google Drive

1. Crea una carpeta en Google Drive: `PACO_Challenge_Data`
2. Sube los archivos `.fits` del challenge:
   - `sphere_irdis_cube_1.fits`
   - `sphere_irdis_pa_1.fits`
   - `sphere_irdis_psf_1.fits`
   - `sphere_irdis_pxscale_1.fits`
   - (y otros datasets que necesites)

### Paso 2: Notebook de Colab

```python
# Instalar dependencias
!pip install astropy numpy scipy matplotlib joblib

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configurar rutas
import sys
from pathlib import Path

# Ruta a tus datos en Drive
data_dir = Path("/content/drive/MyDrive/PACO_Challenge_Data")

# Clonar o subir el código de PACO
!git clone https://github.com/tu-repo/PACO.git /content/PACO
# O subir el código manualmente a Drive

# Agregar al path
sys.path.insert(0, '/content/PACO')

# Importar funciones
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

# Configurar benchmark
config = BENCHMARK_CONFIG.copy()
config['cpu_counts'] = [4, 8]  # Colab tiene múltiples CPUs
n_trials = 3
crop_size = (100, 100)  # Reducir tamaño para evitar problemas de memoria

# Ejecutar benchmark
results = benchmark_real_dataset(
    data_dir=data_dir,
    instrument='sphere_irdis',
    dataset_id=1,
    config=config,
    crop_size=crop_size,
    n_trials=n_trials
)

# Guardar resultados en Drive
output_dir = Path("/content/drive/MyDrive/PACO_Results")
output_dir.mkdir(exist_ok=True)

if results:
    analysis = analyze_amdahl(results)
    json_path, csv_path = save_results(results, analysis, config, output_dir)
    plot_benchmark_results(results, analysis, save_path=str(output_dir / "benchmark_plots.png"))
```

## Opción 2: Usar datos directamente desde URL

Si los datos están disponibles públicamente:

```python
import urllib.request
from pathlib import Path

# Crear directorio para datos
data_dir = Path("/content/challenge_data")
data_dir.mkdir(exist_ok=True)

# Descargar archivos (ejemplo)
urls = {
    'cube': 'https://ejemplo.com/sphere_irdis_cube_1.fits',
    'pa': 'https://ejemplo.com/sphere_irdis_pa_1.fits',
    'psf': 'https://ejemplo.com/sphere_irdis_psf_1.fits',
    'pxscale': 'https://ejemplo.com/sphere_irdis_pxscale_1.fits',
}

for name, url in urls.items():
    filepath = data_dir / f"sphere_irdis_{name}_1.fits"
    print(f"Descargando {name}...")
    urllib.request.urlretrieve(url, filepath)
    print(f"  OK: {filepath}")
```

## Opción 3: Usar almacenamiento temporal de Colab

Colab tiene ~80 GB de almacenamiento temporal. Puedes:

1. Subir archivos directamente a Colab
2. Usar el almacenamiento temporal (se pierde al desconectar)

```python
# Subir archivos manualmente
from google.colab import files
uploaded = files.upload()  # Selecciona archivos .fits

# Los archivos se guardan en /content/
data_dir = Path("/content")
```

## Consideraciones importantes

### Memoria en Colab

- **RAM libre**: ~12-15 GB (suficiente para el benchmark)
- **GPU RAM**: Si usas GPU, tiene su propia memoria
- **Recomendación**: Usa `crop_size=(100, 100)` para datasets grandes

### Tiempo de ejecución

- **Sesiones gratuitas**: Se desconectan después de ~90 minutos de inactividad
- **Solución**: Guarda resultados frecuentemente en Drive

### Instalación de PACO

```python
# Opción A: Si PACO está en un repositorio
!git clone https://github.com/tu-repo/PACO.git /content/PACO
sys.path.insert(0, '/content/PACO')

# Opción B: Subir código a Drive y copiar
!cp -r /content/drive/MyDrive/PACO /content/PACO
sys.path.insert(0, '/content/PACO')

# Opción C: Instalar como paquete (si está en PyPI)
!pip install paco-astronomy
```

## Script completo para Colab

Crea un notebook con este contenido:

```python
# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================
!pip install -q astropy numpy scipy matplotlib joblib pandas

from google.colab import drive
drive.mount('/content/drive')

import sys
from pathlib import Path
import numpy as np

# Configurar rutas
PACO_DIR = Path("/content/drive/MyDrive/PACO")  # O donde tengas el código
DATA_DIR = Path("/content/drive/MyDrive/PACO_Challenge_Data")
OUTPUT_DIR = Path("/content/drive/MyDrive/PACO_Results")

sys.path.insert(0, str(PACO_DIR))

# ============================================================================
# IMPORTAR MÓDULOS
# ============================================================================
from paco_real_data_benchmark import (
    load_challenge_dataset,
    prepare_dataset_for_paco,
    benchmark_real_dataset
)
from paco_performance_benchmark import (
    BENCHMARK_CONFIG,
    analyze_amdahl,
    plot_benchmark_results,
    save_results,
    analyze_scalability
)

# ============================================================================
# CONFIGURACIÓN DEL BENCHMARK
# ============================================================================
config = BENCHMARK_CONFIG.copy()
config['cpu_counts'] = [4, 8]  # Colab tiene múltiples CPUs
n_trials = 3
crop_size = (100, 100)  # Reducir para evitar problemas de memoria

print("Configuración:")
print(f"  CPU counts: {config['cpu_counts']}")
print(f"  N trials: {n_trials}")
print(f"  Crop size: {crop_size}")

# ============================================================================
# EJECUTAR BENCHMARK
# ============================================================================
results = benchmark_real_dataset(
    data_dir=DATA_DIR,
    instrument='sphere_irdis',
    dataset_id=1,
    config=config,
    crop_size=crop_size,
    n_trials=n_trials
)

# ============================================================================
# ANALIZAR Y GUARDAR RESULTADOS
# ============================================================================
if results:
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Análisis
    analysis = analyze_amdahl(results)
    
    # Guardar
    json_path, csv_path = save_results(results, analysis, config, OUTPUT_DIR)
    
    # Gráficos
    plot_path = OUTPUT_DIR / "benchmark_plots.png"
    plot_benchmark_results(results, analysis, save_path=str(plot_path))
    
    print(f"\n✅ Resultados guardados en: {OUTPUT_DIR}")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Gráficos: {plot_path}")
```

## Ventajas vs Desventajas

### ✅ Ventajas de Colab
- No ocupa espacio en tu disco
- Acceso a GPU/TPU (si lo necesitas)
- Fácil de compartir
- Resultados guardados en Drive

### ⚠️ Desventajas
- Requiere conexión a internet
- Sesiones pueden desconectarse
- Límite de tiempo en sesiones gratuitas
- Primera ejecución más lenta (instalación)

## Recomendación

**Usa Google Colab si:**
- Tienes buena conexión a internet
- No necesitas ejecutar benchmarks muy largos
- Quieres liberar espacio en tu disco local

**Mantén ejecución local si:**
- Necesitas ejecuciones muy largas
- Tienes suficiente espacio en disco
- Prefieres no depender de internet

## Siguiente paso

¿Quieres que cree un notebook de Colab listo para usar? Puedo generarlo con toda la configuración necesaria.

