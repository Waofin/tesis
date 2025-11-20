# Análisis y Benchmark de PACO (VIP-master)

Este repositorio contiene scripts para analizar, corregir errores y hacer benchmarking completo de la implementación de PACO en VIP-master.

## 📋 Contenido

1. **`paco_analysis_benchmark.py`** - Análisis de errores y benchmark completo
2. **`paco_profiling_detailed.py`** - Profiling detallado línea por línea
3. **`load_datasets_github.py`** - Cargador de datasets desde GitHub
4. **`fix_paco_sample_covariance.py`** - Script para corregir el error en `sample_covariance`

## 🔍 Errores Detectados

### Error Crítico: `sample_covariance` (Línea ~1304)

**Problema:**
```python
# IMPLEMENTACIÓN ACTUAL (INCORRECTA):
S = (1.0/T)*np.sum([np.cov(np.stack((p, m)), rowvar=False, bias=False) for p in r], axis=0)
```

**Análisis:**
- `np.cov(np.stack((p, m)))` calcula la covarianza **ENTRE** `p` y `m`
- Esto es **matemáticamente incorrecto**
- Debería calcular la covarianza de `(p-m)` consigo mismo

**Implementación Correcta:**
```python
# VERSIÓN CORREGIDA (VECTORIZADA):
r_centered = r - m[np.newaxis, :]  # Broadcasting
S = (1.0 / T) * np.dot(r_centered.T, r_centered)
```

**Impacto:**
- Resultados numéricos incorrectos
- Matrices de covarianza mal calculadas
- SNR y flux estimates incorrectos
- Speedup adicional: ~4.9x (por vectorización)

### Problema de Memoria: `compute_statistics` (Línea ~821)

**Problema:**
- Pre-asigna memoria para **TODOS** los píxeles de la imagen
- Memoria necesaria: `(height, width, patch_area_pixels, patch_area_pixels) * 8 bytes`
- Ejemplo: Imagen 321x321, patch_area_pixels=99481 → **7.25 PiB** (!!)

**Solución:**
- Solo pre-asignar memoria para los píxeles especificados en `phi0s`
- Usar procesamiento por lotes (batching)
- Implementar procesamiento lazy/on-demand

## 🚀 Uso

### 1. Análisis de Errores y Benchmark

```bash
python paco_analysis_benchmark.py
```

Este script:
- Detecta errores en la implementación
- Ejecuta benchmarks con datasets sintéticos
- Genera reporte completo en `paco_benchmark_report.txt`

### 2. Profiling Detallado

```bash
python paco_profiling_detailed.py
```

Este script:
- Hace profiling línea por línea de funciones críticas
- Analiza tiempos de cada paso del algoritmo
- Genera reporte en `paco_profiling_report.txt`

### 3. Cargar Datasets desde GitHub

```python
from load_datasets_github import GitHubDatasetLoader

loader = GitHubDatasetLoader()

config = {
    'name': 'Mi Dataset',
    'cube_url': 'https://github.com/user/repo/blob/main/data/cube.fits',
    'pa_url': 'https://github.com/user/repo/blob/main/data/angles.fits',
    'psf_url': 'https://github.com/user/repo/blob/main/data/psf.fits',
    'pixscale': 0.027
}

cube, pa, psf, pixscale = loader.load_dataset(config)
```

### 4. Corregir Error en `sample_covariance`

```bash
python fix_paco_sample_covariance.py
```

Este script:
- Crea un backup del archivo original
- Genera versión corregida
- Muestra comparación entre implementaciones

**Para aplicar la corrección:**
1. Revisa el archivo generado: `VIP-master/VIP-master/src/vip_hci/invprob/paco.py.fixed`
2. Si está correcto, reemplaza el original:
   ```bash
   cp VIP-master/VIP-master/src/vip_hci/invprob/paco.py.fixed VIP-master/VIP-master/src/vip_hci/invprob/paco.py
   ```

## 📊 Resultados Esperados

### Speedup con Corrección

- **sample_covariance**: ~4.9x más rápido (vectorización)
- **Inversión de matrices**: ~1.6x más rápido (scipy.linalg.inv con regularización)
- **Total estimado**: ~6.5x speedup (CPU optimizado)

### Tiempos Típicos (CPU)

- **sample_covariance** (100 iteraciones):
  - Incorrecta: ~X ms
  - Correcta: ~X/4.9 ms

- **compute_statistics_at_pixel** (50 iteraciones):
  - Total: ~Y ms
  - Descomposición:
    - Mean: ~Z% del tiempo
    - Sample covariance: ~W% del tiempo
    - Inverse: ~V% del tiempo

## 📝 Reportes Generados

1. **`paco_benchmark_report.txt`** - Reporte completo de análisis y benchmarks
2. **`paco_profiling_report.txt`** - Reporte detallado de profiling

## 🔧 Dependencias

```bash
pip install numpy scipy astropy matplotlib
pip install line_profiler memory_profiler  # Opcional, para profiling avanzado
```

## ⚠️ Advertencias

1. **Backup automático**: El script de corrección crea un backup automático antes de modificar archivos
2. **Verificación**: Siempre revisa los archivos generados antes de aplicar correcciones
3. **Testing**: Prueba la implementación corregida con tus datos antes de usarla en producción

## 📚 Referencias

- [FLA18] Flasseur et al. 2018 - "Exoplanet detection in angular differential imaging by statistical learning of the nonstationary patch covariances. The PACO algorithm"
- VIP Documentation: https://vip.readthedocs.io/

## 👤 Autor

César Cerda - Universidad del Bío-Bío

## 📄 Licencia

Este código es para análisis y corrección de errores. Consulta la licencia de VIP para el código base.

