# Análisis: Paralelización de PACOCalc con GPU CUDA

## 📊 Resumen Ejecutivo

**Dificultad**: ⭐⭐⭐ (Media-Alta)
- **Fácil**: Usar CuPy (drop-in replacement de NumPy)
- **Medio**: Optimizar con Numba CUDA
- **Difícil**: Escribir kernels CUDA desde cero

**Speedup Esperado**: 10-50x (dependiendo del tamaño de datos y GPU)

---

## 🔍 Análisis de Operaciones

### Operaciones Principales en PACOCalc

#### 1. **pixelCalc()** - Cuello de Botella Principal (93-98% del tiempo)

```python
def pixelCalc(patch):
    # patch: (T, P) donde T=frames, P=píxeles en patch (típico 49)
    m = np.mean(patch, axis=0)                    # ✅ Fácil en GPU
    S = sampleCovariance(patch, m, T)            # ⚠️ Operaciones matriciales
    rho = shrinkageFactor(S, T)                  # ⚠️ Trazas y productos
    F = diagSampleCovariance(S)                   # ✅ Fácil
    C = covariance(rho, S, F)                     # ✅ Fácil
    Cinv = np.linalg.inv(C)                       # ⚠️ O(P³) - Costoso pero paralelizable
    return m, Cinv
```

**Operaciones por llamada**:
- `np.mean()`: O(T×P) - **Muy fácil en GPU**
- `sampleCovariance()`: O(T×P²) - **Paralelizable en GPU**
- `shrinkageFactor()`: Trazas y productos matriciales - **Paralelizable**
- `np.linalg.inv()`: O(P³) donde P=49 - **GPU excelente para esto**

#### 2. **al() y bl()** - Operaciones Finales

```python
def al(self, hfl, Cfl_inv):
    # hfl: lista de (P,) arrays
    # Cfl_inv: lista de (P, P) matrices
    a = np.sum([np.dot(hfl[i], np.dot(Cfl_inv[i], hfl[i]).T) 
                for i in range(len(hfl))])
    return a

def bl(self, hfl, Cfl_inv, r_fl, m_fl):
    b = np.sum([np.dot(np.dot(Cfl_inv[i], hfl[i]).T, (r_fl[i][i]-m_fl[i]))
                for i in range(len(hfl))])
    return b
```

**Operaciones**: Productos matriciales (np.dot) - **Excelente para GPU**

#### 3. **Loop Principal** - Perfectamente Paralelizable

```python
for i, p0 in enumerate(phi0s):  # 10,000 píxeles independientes
    angles_px = getRotatedPixels(x, y, p0, angles)
    for l, ang in enumerate(angles_px):  # 252 frames
        patch[l] = getPatch(ang, pwidth, mask)
        m[l], Cinv[l] = pixelCalc(patch[l])  # ← BOTTLENECK
        h[l] = psf[mask]
    a[i] = al(h, Cinv)
    b[i] = bl(h, Cinv, patch, m)
```

**Paralelización**: Cada iteración de `i` es **completamente independiente** → ideal para GPU

---

## 🎯 Estrategias de Implementación

### Opción 1: CuPy (Más Fácil) ⭐⭐⭐

**Dificultad**: Baja
**Tiempo estimado**: 2-4 horas
**Speedup esperado**: 10-30x

**Ventajas**:
- Drop-in replacement de NumPy
- Misma API que NumPy
- Cambios mínimos en código

**Implementación**:
```python
import cupy as cp

def pixelCalc_CUDA(patch):
    """Versión GPU usando CuPy"""
    # Transferir a GPU
    patch_gpu = cp.asarray(patch)
    
    # Operaciones en GPU (misma sintaxis que NumPy)
    m = cp.mean(patch_gpu, axis=0)
    S = sampleCovariance_CUDA(patch_gpu, m, T)
    rho = shrinkageFactor_CUDA(S, T)
    F = diagSampleCovariance_CUDA(S)
    C = covariance_CUDA(rho, S, F)
    Cinv = cp.linalg.inv(C)  # ← GPU excelente para esto
    
    # Transferir de vuelta
    return cp.asnumpy(m), cp.asnumpy(Cinv)

def PACOCalc_CUDA(self, phi0s):
    """Versión GPU del loop principal"""
    # Transferir datos grandes a GPU una vez
    im_stack_gpu = cp.asarray(self.m_im_stack)
    psf_gpu = cp.asarray(self.m_psf)
    
    # Paralelizar loop sobre píxeles
    # Usar CUDA kernels o procesar en batches
    ...
```

**Desafíos**:
- Transferencia CPU↔GPU puede ser costosa
- Matrices pequeñas (49×49) pueden no aprovechar bien la GPU
- Necesita gestión de memoria GPU

---

### Opción 2: Numba CUDA (Medio) ⭐⭐⭐⭐

**Dificultad**: Media
**Tiempo estimado**: 1-2 días
**Speedup esperado**: 20-50x

**Ventajas**:
- Optimización automática
- Puede compilar kernels CUDA desde Python
- Mejor control que CuPy

**Implementación**:
```python
from numba import cuda
import numba

@cuda.jit
def pixelCalc_kernel(patch, m_out, Cinv_out):
    """Kernel CUDA para pixelCalc"""
    idx = cuda.grid(1)
    if idx < patch.shape[0]:
        # Cálculos en GPU
        ...

@cuda.jit
def PACOCalc_kernel(phi0s, im_stack, angles, ...):
    """Kernel principal paralelizado"""
    idx = cuda.grid(1)
    if idx < len(phi0s):
        # Procesar píxel idx en paralelo
        ...
```

**Desafíos**:
- Requiere conocimiento de CUDA
- Debugging más difícil
- Optimización manual necesaria

---

### Opción 3: CUDA Puro (Difícil) ⭐⭐⭐⭐⭐

**Dificultad**: Alta
**Tiempo estimado**: 1-2 semanas
**Speedup esperado**: 30-100x (con optimización)

**Ventajas**:
- Control total
- Máxima optimización posible

**Desventajas**:
- Requiere C/CUDA
- Desarrollo largo
- Mantenimiento complejo

---

## ⚠️ Desafíos y Consideraciones

### 1. **Tamaño de Matrices Pequeñas**

**Problema**: Matrices de 49×49 pueden ser pequeñas para GPU
- GPU es eficiente con matrices grandes (>100×100)
- Overhead de transferencia puede dominar

**Solución**: Procesar múltiples píxeles en paralelo (batches)

### 2. **Transferencia de Datos**

**Problema**: CPU↔GPU transfer es costosa
- Para 100×100 imagen: ~40 MB de datos
- Transferencia inicial: ~10-50ms
- Puede ser significativo si se hace muchas veces

**Solución**: 
- Transferir datos grandes una vez
- Mantener datos en GPU durante todo el procesamiento
- Usar streams CUDA para overlap

### 3. **Memoria GPU**

**Problema**: GPU tiene memoria limitada (típico 8-24 GB)
- Para 100×100×252: ~10 MB (fácil)
- Para 160×160×252: ~26 MB (fácil)
- Pero con múltiples batches puede crecer

**Solución**: Procesar en batches si es necesario

### 4. **Operaciones Específicas**

**Problema**: Algunas operaciones pueden no tener equivalente directo en GPU
- `getRotatedPixels()`: Operaciones de coordenadas
- `getPatch()`: Indexación compleja

**Solución**: Implementar kernels personalizados o mantener en CPU

---

## 📈 Estimación de Speedup

### Escenario: Imagen 100×100, 252 frames, patch_size=49

| Método | Tiempo Actual | Tiempo GPU | Speedup |
|--------|---------------|------------|---------|
| **Serial CPU** | ~8 horas | - | 1x |
| **Loky 8 CPUs** | ~10 minutos | - | ~48x |
| **CuPy (GPU)** | - | ~5-15 min | 30-100x |
| **Numba CUDA** | - | ~2-8 min | 60-240x |
| **CUDA Optimizado** | - | ~1-3 min | 160-480x |

**Nota**: Speedups reales dependen de:
- GPU (RTX 3090 vs GTX 1050)
- Tamaño de datos
- Overhead de transferencia
- Optimización del código

---

## 🛠️ Recomendación de Implementación

### Fase 1: Prototipo con CuPy (2-4 horas)

1. **Reemplazar NumPy por CuPy en funciones críticas**:
   ```python
   # En pixelCalc
   import cupy as cp
   patch_gpu = cp.asarray(patch)
   m = cp.mean(patch_gpu, axis=0)
   Cinv = cp.linalg.inv(C)
   ```

2. **Paralelizar loop principal**:
   - Procesar píxeles en batches
   - Usar `cupy.RawKernel` o procesar en paralelo

3. **Medir speedup real**

### Fase 2: Optimización con Numba (1-2 días)

Si CuPy da buen speedup pero necesita más optimización:
- Escribir kernels Numba CUDA personalizados
- Optimizar transferencias de memoria
- Usar shared memory cuando sea posible

### Fase 3: CUDA Puro (solo si necesario)

Solo si se necesita máximo rendimiento y hay tiempo disponible.

---

## 💡 Alternativa: JAX (Más Moderno)

**JAX** es otra opción interesante:
- Compilación JIT automática
- Puede usar GPU/TPU automáticamente
- API similar a NumPy
- Gradientes automáticos (útil para optimización)

```python
import jax.numpy as jnp
from jax import jit, vmap

@jit
def pixelCalc_JAX(patch):
    m = jnp.mean(patch, axis=0)
    Cinv = jnp.linalg.inv(C)
    return m, Cinv

# Paralelizar automáticamente
PACOCalc_vectorized = vmap(process_pixel)
```

---

## ✅ Conclusión

**Dificultad General**: ⭐⭐⭐ (Media)

**Recomendación**:
1. **Empezar con CuPy** (más fácil, cambios mínimos)
2. **Medir speedup real** con tus datos
3. **Optimizar si es necesario** con Numba CUDA

**Tiempo estimado para implementación básica**: 2-4 horas
**Speedup esperado**: 10-30x (dependiendo de GPU y datos)

**¿Vale la pena?**
- ✅ **SÍ** si tienes GPU NVIDIA moderna (RTX 2060+)
- ✅ **SÍ** si procesas muchas imágenes
- ⚠️ **Quizás** si solo procesas ocasionalmente
- ❌ **NO** si no tienes GPU o es muy antigua


