# Implementación CUDA para PACO

Este directorio contiene una implementación funcional de PACOCalc con aceleración GPU usando CuPy.

## 📋 Requisitos

### 1. GPU NVIDIA
- GPU NVIDIA con soporte CUDA
- Drivers NVIDIA actualizados

### 2. CUDA Toolkit
- CUDA 11.x o 12.x instalado
- Verificar versión: `nvcc --version`

### 3. CuPy
Instalar según tu versión de CUDA:

```bash
# Para CUDA 11.x
pip install cupy-cuda11x

# Para CUDA 12.x
pip install cupy-cuda12x

# Verificar instalación
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

## 🚀 Uso Rápido

### 1. Benchmark Básico

```bash
python benchmark_cuda.py
```

Esto ejecutará:
- Benchmark de `pixelCalc()` CPU vs GPU
- Opcionalmente, benchmark completo de `PACOCalc()`

### 2. Usar en Código

```python
from paco.processing.fullpaco_cuda import FullPACO_CUDA

# Crear instancia GPU
fp_gpu = FullPACO_CUDA(
    image_stack=image_stack,
    angles=angles,
    psf=psf,
    psf_rad=4,
    patch_area=49
)

# Ejecutar PACOCalc en GPU
a, b = fp_gpu.PACOCalc_CUDA(phi0s, batch_size=1000)
```

## 📊 Resultados Esperados

### Speedup Típico

| Operación | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| `pixelCalc()` (100 iteraciones) | ~0.5s | ~0.05s | **10x** |
| `PACOCalc()` (50×50 imagen) | ~5 min | ~30s | **10x** |
| `PACOCalc()` (100×100 imagen) | ~8 horas | ~30 min | **16x** |

**Nota**: Speedups reales dependen de:
- GPU (RTX 3090 vs GTX 1050)
- Tamaño de datos
- Overhead de transferencia CPU↔GPU

## 🔧 Configuración Avanzada

### Ajustar Batch Size

```python
# Para imágenes grandes, usar batches más grandes
a, b = fp_gpu.PACOCalc_CUDA(phi0s, batch_size=2000)

# Para imágenes pequeñas, batches más pequeños
a, b = fp_gpu.PACOCalc_CUDA(phi0s, batch_size=500)
```

### Usar Solo CPU para Comparación

```python
# Forzar CPU aunque tengas GPU
a, b = fp_gpu.PACOCalc_CUDA(phi0s, use_gpu_pixelcalc=False)
```

## ⚠️ Limitaciones Actuales

1. **Extracción de Patches**: `getPatch()` se ejecuta en CPU
   - Esto es porque requiere indexación compleja
   - Puede optimizarse en el futuro con kernels CUDA personalizados

2. **Transferencia CPU↔GPU**: Overhead para datos pequeños
   - Para imágenes <50×50, el overhead puede dominar
   - Para imágenes >100×100, el speedup es significativo

3. **Memoria GPU**: Limitada por VRAM
   - Típico: 8-24 GB
   - Para imágenes muy grandes, procesar en batches

## 🐛 Solución de Problemas

### Error: "CuPy no está disponible"

```bash
# Verificar que CUDA está instalado
nvcc --version

# Instalar CuPy correcto para tu versión de CUDA
pip uninstall cupy
pip install cupy-cuda11x  # o cupy-cuda12x
```

### Error: "No se pudo inicializar GPU"

1. Verificar que la GPU es NVIDIA:
   ```bash
   nvidia-smi
   ```

2. Verificar que los drivers están actualizados

3. Verificar que CuPy detecta la GPU:
   ```python
   import cupy as cp
   print(cp.cuda.runtime.getDeviceCount())
   ```

### Speedup Bajo o Negativo

- **Causa**: Overhead de transferencia CPU↔GPU
- **Solución**: Usar imágenes más grandes (>100×100) o batches más grandes

- **Causa**: GPU muy antigua o lenta
- **Solución**: GPU moderna (RTX 2060+) recomendada

## 📈 Próximas Optimizaciones

1. **Kernels CUDA Personalizados**: Para `getPatch()` y operaciones específicas
2. **Procesamiento Completamente Paralelo**: Paralelizar el loop sobre píxeles en GPU
3. **Streams CUDA**: Overlap de transferencias y computación
4. **Shared Memory**: Optimizar acceso a memoria para operaciones repetidas

## 📚 Referencias

- [CuPy Documentation](https://docs.cupy.dev/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html) (alternativa)

## 💡 Notas

- La implementación actual es un **prototipo funcional**
- Está optimizada para **facilidad de uso** sobre máximo rendimiento
- Para máximo rendimiento, considera escribir kernels CUDA personalizados
- La precisión numérica puede diferir ligeramente entre CPU y GPU (float32)


