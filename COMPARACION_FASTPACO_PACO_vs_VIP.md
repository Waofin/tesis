# Comparación: FastPACO - PACO-master vs VIP-master

## Resumen Ejecutivo

Este documento compara las implementaciones de **FastPACO** (Algoritmo 2 del paper PACO) en dos proyectos diferentes:
- **PACO-master**: Implementación independiente con múltiples optimizaciones (CPU, CUDA, PyTorch)
- **VIP-master**: Implementación integrada en el paquete VIP (Very high contrast Imaging Package)

---

## 1. Arquitectura y Estructura de Clases

### PACO-master

**Jerarquía de clases:**
```
PACO (clase base)
  └── FastPACO (CPU básico)
      ├── FastPACO_CUDA (GPU con CuPy)
      │   ├── FastPACO_CUDA_Optimized (altamente optimizado)
      │   └── FastPACO_CUDA_PreExtract (pre-extracción de patches)
      └── FastPACO_PyTorch (GPU con PyTorch)
```

**Características:**
- Múltiples implementaciones especializadas
- Optimizaciones progresivas (CPU → CUDA básico → CUDA optimizado)
- Separación clara entre algoritmos base y optimizaciones

### VIP-master

**Jerarquía de clases:**
```
PACO (clase base abstracta)
  ├── FastPACO (CPU con paralelización opcional)
  └── FullPACO (Algoritmo 1 completo)
```

**Características:**
- Implementación más unificada
- Integrada en el ecosistema VIP
- Soporte para subpixel PSF astrometry
- Funciones de utilidad integradas (normalización PSF, detección, etc.)

---

## 2. Implementación del Algoritmo FastPACO

### 2.1 Método Principal: `PACOCalc`

#### PACO-master (`fastpaco.py`)

```python
def PACOCalc(self, phi0s, cpu=1):
    # 1. Precomputar estadísticas (Cinv, m, h)
    if cpu == 1:
        Cinv, m, h = self.computeStatistics(phi0s)
    else:
        Cinv, m, h = self.computeStatisticsParallel(phi0s, cpu=cpu)
    
    # 2. Loop sobre píxeles
    for i, p0 in enumerate(phi0s):
        # Calcular ángulos rotados
        angles_px = getRotatedPixels(x, y, p0, self.m_angles)
        
        # Extraer patches y estadísticas
        for l, ang in enumerate(angles_px):
            patch[l] = self.getPatch(ang, self.m_pwidth, mask)
            Cinlst.append(Cinv[int(ang[0])][int(ang[1])])
            mlst.append(m[int(ang[0])][int(ang[1])])
            hlst.append(h[int(ang[0])][int(ang[1])])
        
        # Calcular a y b
        a[i] = self.al(hlst, Cinlst)
        b[i] = self.bl(hlst, Cinlst, patch, mlst)
    
    return a, b
```

**Características:**
- Loop secuencial sobre píxeles
- Precomputación de estadísticas para todos los píxeles
- Sin subpixel PSF astrometry (PSF fija en el centro del patch)

#### VIP-master (`paco.py`)

```python
def PACOCalc(self, phi0s, use_subpixel_psf_astrometry=True, cpu=1):
    # 1. Precomputar estadísticas
    if cpu == 1:
        Cinv, m, patches = self.compute_statistics(phi0s)
    else:
        Cinv, m, patches = self.compute_statistics_parallel(phi0s, cpu=cpu)
    
    # 2. Normalizar PSF
    normalised_psf = normalize_psf(self.psf, ...)
    
    # 3. Loop sobre píxeles
    for i, p0 in enumerate(phi0s):
        angles_px = get_rotated_pixel_coords(x, y, p0, self.angles)
        
        for l, ang in enumerate(angles_px):
            # OPCIÓN: Subpixel PSF astrometry
            if use_subpixel_psf_astrometry:
                offax = frame_shift(normalised_psf,
                                    ang[1]-int(ang[1]),  # Subpixel shift
                                    ang[0]-int(ang[0]),
                                    ...)[psf_mask]
            else:
                offax = normalised_psf[psf_mask]
            
            hlst.append(offax)
            patch.append(patches[int(ang[0]), int(ang[1]), l])
        
        a[i] = self.al(hlst, Cinlst)
        b[i] = self.bl(hlst, Cinlst, patch, mlst)
    
    return a, b
```

**Características:**
- **Subpixel PSF astrometry**: Opción para desplazar PSF según posición subpixel
- Normalización automática de PSF
- Almacenamiento de patches completos en memoria

### 2.2 Diferencias Clave

| Aspecto | PACO-master | VIP-master |
|---------|-------------|------------|
| **Subpixel PSF** | ❌ No soportado | ✅ Opcional (`use_subpixel_psf_astrometry`) |
| **Normalización PSF** | Manual | Automática con `normalize_psf()` |
| **Almacenamiento patches** | Solo extrae cuando se necesita | Pre-almacena todos los patches |
| **Rotación de coordenadas** | `getRotatedPixels()` | `get_rotated_pixel_coords()` |
| **Convención coordenadas** | (y, x) | (x, y) con conversión |

---

## 3. Precomputación de Estadísticas

### 3.1 PACO-master: `computeStatistics`

```python
def computeStatistics(self, phi0s):
    # Crear arrays de salida
    h = np.zeros((H, W, psf_area))
    m = np.zeros((H, W, psf_area))
    Cinv = np.zeros((H, W, psf_area, psf_area))
    
    # Loop serial sobre píxeles
    for p0 in phi0s:
        apatch = self.getPatch(p0, self.m_pwidth, mask)
        m[p0[0]][p0[1]], Cinv[p0[0]][p0[1]] = pixelCalc(apatch)
        h[p0[0]][p0[1]] = self.m_psf[mask]
    
    return Cinv, m, h
```

**Versión paralela:**
- Usa `multiprocessing.Pool`
- Procesa patches en paralelo
- Nota: "currently seems slower than computing in serial"

### 3.2 VIP-master: `compute_statistics`

```python
def compute_statistics(self, phi0s):
    # Crear arrays de salida (incluye patches)
    patch = np.zeros((W, H, T, psf_area))
    m = np.zeros((H, W, psf_area))
    Cinv = np.zeros((H, W, psf_area, psf_area))
    
    # Loop serial
    for p0 in phi0s:
        apatch = self.get_patch(p0)
        m[p0[1]][p0[0]], Cinv[p0[1]][p0[0]] = compute_statistics_at_pixel(apatch)
        patch[p0[1]][p0[0]] = apatch  # Almacena patch completo
    
    return Cinv, m, patch
```

**Versión paralela:**
- Usa `pool_map` de VIP (wrapper de multiprocessing)
- Procesa patches y estadísticas en paralelo
- Mejor integrado con el sistema VIP

### 3.3 Comparación de Cálculo de Estadísticas

| Función | PACO-master | VIP-master |
|---------|-------------|------------|
| **Media** | `np.mean(patch, axis=0)` | `np.mean(patch, axis=0)` |
| **Covarianza** | `pixelCalc()` → `sampleCovariance()` | `compute_statistics_at_pixel()` → `sample_covariance()` |
| **Shrinkage** | Implementado | Implementado (mismo algoritmo Ledoit-Wolf) |
| **Inversión matriz** | `np.linalg.inv()` | `np.linalg.inv()` |

**Nota:** Ambos usan el mismo algoritmo de shrinkage de Ledoit-Wolf para regularizar la matriz de covarianza.

---

## 4. Optimizaciones GPU/CUDA

### 4.1 PACO-master: Versiones CUDA

#### FastPACO_CUDA (básico)
- **Optimización**: Solo `computeStatistics` en GPU
- **Tecnología**: CuPy
- **Estrategia**: Procesa batches de patches en GPU
- **Speedup esperado**: 50-200x vs CPU

#### FastPACO_CUDA_Optimized
- **Optimización**: `PACOCalc` completo en GPU
- **Características**:
  - Vectorización masiva con `einsum`
  - Procesamiento en batches grandes (10,000 píxeles)
  - Múltiples streams CUDA para procesamiento asíncrono
  - Eliminación de llamadas redundantes a `getPatch`
- **Métodos optimizados**:
  - `al_CUDA()`: Vectorizado para múltiples píxeles
  - `bl_CUDA()`: Vectorizado para múltiples píxeles
  - `PACOCalc_CUDA_Optimized()`: Todo el pipeline en GPU

#### FastPACO_CUDA_PreExtract
- **Optimización**: Pre-extrae todos los patches antes de procesar
- **Ventaja**: Reduce overhead de extracción durante el cálculo

### 4.2 VIP-master: GPU

**Estado actual**: ❌ **No tiene implementación GPU/CUDA**

VIP-master solo tiene:
- Versión CPU serial
- Versión CPU paralela (multiprocessing)

---

## 5. Funciones Matemáticas: `al()` y `bl()`

### 5.1 Implementación

#### PACO-master

```python
def al(self, hlst, Cinlst):
    # hlst: lista de PSFs
    # Cinlst: lista de matrices de covarianza inversa
    a = np.sum([np.dot(hfl[i], np.dot(Cinv[i], hfl[i]).T)
                for i in range(len(hlst))], axis=0)
    return a

def bl(self, hlst, Cinlst, patch, mlst):
    # Calcula b_l para cada frame
    b = np.sum([np.dot(np.dot(Cinv[i], hfl[i]).T, (patch[i]-mlst[i]))
                for i in range(len(hlst))], axis=0)
    return b
```

#### VIP-master

```python
def al(self, hfl, Cfl_inv, method=""):
    if method == "einsum":
        d1 = np.einsum('ijk,gj', Cfl_inv, hfl)
        return np.einsum('ml,ml', hfl, np.diagonal(d1).T)
    
    a = np.sum(np.array([np.dot(hfl[i], np.dot(Cfl_inv[i], hfl[i]).T)
                         for i in range(len(hfl))]), axis=0)
    return a

def bl(self, hfl, Cfl_inv, r_fl, m_fl, method=""):
    if method == "einsum":
        d1 = np.einsum('ijk,gj', Cfl_inv, r_fl-m_fl)
        return np.einsum('ml,ml', hfl, np.diagonal(d1).T)
    
    b = np.sum(np.array([np.dot(np.dot(Cfl_inv[i], hfl[i]).T, (r_fl[i]-m_fl[i]))
                         for i in range(len(hfl))]), axis=0)
    return b
```

**Diferencias:**
- VIP tiene opción `einsum` (aunque comentan que puede ser más lento)
- Misma lógica matemática en ambos casos

---

## 6. Utilidades y Funciones Auxiliares

### 6.1 Extracción de Patches

#### PACO-master: `getPatch()`

```python
def getPatch(self, px, width, mask=None):
    k = int(width/2)
    # Extrae región rectangular y aplica máscara circular
    patch = np.array([self.m_im_stack[i][y-k:y+k2, x-k:x+k2][mask]
                      for i in range(len(self.m_im_stack))])
    return patch
```

#### VIP-master: `get_patch()`

```python
def get_patch(self, px, width=None, mask=None):
    # Similar pero con manejo de bordes más robusto
    if px fuera de límites:
        return np.ones((T, psf_area)) * np.nan
    # Usa broadcasting para aplicar máscara
    patch = self.cube[np.broadcast_to(mask, self.cube.shape)]
    return patch.reshape(T, psf_area)
```

### 6.2 Rotación de Coordenadas

#### PACO-master: `getRotatedPixels()`
- Implementación propia
- Convierte a coordenadas polares, rota, convierte de vuelta

#### VIP-master: `get_rotated_pixel_coords()`
- Usa funciones VIP: `cart_to_pol()` y `pol_to_cart()`
- Soporte para convención astronómica opcional

---

## 7. Ventajas y Desventajas

### PACO-master

**Ventajas:**
- ✅ **Múltiples optimizaciones GPU**: CUDA básico, optimizado, pre-extract
- ✅ **Alto rendimiento**: Versiones CUDA optimizadas con speedups de 50-200x
- ✅ **Flexibilidad**: Múltiples backends (CPU, CUDA, PyTorch)
- ✅ **Código modular**: Fácil de extender y optimizar
- ✅ **Procesamiento en batches**: Optimizado para datasets grandes

**Desventajas:**
- ❌ **Sin subpixel PSF astrometry**: Menor precisión en algunos casos
- ❌ **Dependencias adicionales**: Requiere CuPy/PyTorch para GPU
- ❌ **Menos integrado**: No tiene funciones de detección/post-procesamiento
- ❌ **Documentación limitada**: Menos ejemplos y tutoriales

### VIP-master

**Ventajas:**
- ✅ **Subpixel PSF astrometry**: Mayor precisión en detección
- ✅ **Integración completa**: Parte del ecosistema VIP
- ✅ **Funciones adicionales**: Detección, estimación de flujo, métricas
- ✅ **Bien documentado**: Integrado en documentación VIP
- ✅ **Normalización automática**: PSF normalizada automáticamente
- ✅ **Estabilidad**: Código más maduro y probado

**Desventajas:**
- ❌ **Sin GPU**: Solo CPU (serial o paralelo)
- ❌ **Menor rendimiento**: No tiene optimizaciones GPU
- ❌ **Menos flexible**: Menos opciones de optimización
- ❌ **Mayor uso de memoria**: Almacena todos los patches

---

## 8. Casos de Uso Recomendados

### Usar PACO-master cuando:
- Necesitas **máximo rendimiento** con datasets grandes
- Tienes acceso a **GPU CUDA**
- Quieres **procesar muchos frames** rápidamente
- No necesitas subpixel PSF astrometry
- Quieres **flexibilidad** para optimizar

### Usar VIP-master cuando:
- Necesitas **máxima precisión** (subpixel PSF)
- Ya usas el **ecosistema VIP** para otras tareas
- Necesitas **funciones integradas** (detección, métricas)
- Tienes datasets **pequeños a medianos**
- Prefieres **código estable y probado**

---

## 9. Rendimiento Esperado

### PACO-master (CPU)
- **Serial**: ~1x (baseline)
- **Paralelo (8 cores)**: ~3-5x
- **CUDA básico**: ~50-100x
- **CUDA optimizado**: ~100-200x

### VIP-master
- **Serial**: ~1x (baseline)
- **Paralelo (8 cores)**: ~4-6x
- **GPU**: ❌ No disponible

**Nota**: Los speedups dependen del tamaño del dataset, número de frames, y hardware.

---

## 10. Código de Ejemplo

### PACO-master (FastPACO_CUDA_Optimized)

```python
from paco.processing.fastpaco_cuda_optimized import FastPACO_CUDA_Optimized

paco = FastPACO_CUDA_Optimized(
    image_stack=cube,
    angles=angles,
    psf=psf,
    psf_rad=4
)

a, b = paco.PACOCalc(phi0s, cpu=1)  # Usa GPU automáticamente
snr = b / np.sqrt(a)
```

### VIP-master

```python
from vip_hci.invprob.paco import FastPACO

paco = FastPACO(
    cube=cube,
    angles=angles,
    psf=psf,
    fwhm=4.0,
    pixscale=1.0,
    verbose=True
)

snr, flux = paco.run(
    cpu=8,
    use_subpixel_psf_astrometry=True
)
```

---

## 11. Conclusiones

1. **PACO-master** es superior para **rendimiento** gracias a sus optimizaciones GPU
2. **VIP-master** es superior para **precisión** gracias a subpixel PSF astrometry
3. **PACO-master** es más **flexible** y extensible
4. **VIP-master** está mejor **integrado** en un ecosistema completo
5. La elección depende de tus **prioridades**: velocidad vs precisión

**Recomendación híbrida**: 
- Usar **PACO-master CUDA** para exploración rápida y datasets grandes
- Usar **VIP-master** para análisis final y detección precisa

---

## Referencias

- **Paper PACO**: Flasseur et al. 2018, A&A, 618, A138
- **PACO-master**: Implementación independiente con optimizaciones
- **VIP-master**: Paquete VIP (Very high contrast Imaging Package)

---

*Documento generado: 2024*
*Última actualización: Comparación entre PACO-master y VIP-master*

