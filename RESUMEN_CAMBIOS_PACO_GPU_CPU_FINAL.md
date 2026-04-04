# Resumen Final de Cambios - PACO CPU vs GPU

Fecha: 2026-03-17

## Objetivo del trabajo

Construir una comparacion CPU vs GPU de PACO en condiciones 1:1 (mismos hiperparametros, mismos datos, misma logica), corregir inconsistencias cientificas entre backends, y optimizar rendimiento GPU manteniendo consistencia.

---

## 1) Notebooks creados/ajustados para benchmark reproducible

Se crearon y/o ajustaron notebooks para trabajar desde Google Drive (sin dependencia de clonacion desde GitHub en runtime):

- `PACO_VIP_CPU_drive.ipynb`
- `PACO_VIP_GPU_drive.ipynb`
- `PACO_VIP_COMPARE_drive.ipynb`

Tambien se adaptaron versiones de resultados:

- `PACO_VIP_CPU_results.ipynb`
- `PACO_VIP_GPU_results.ipynb`
- `PACO_VIP_COMPARE_results.ipynb` (y corrida en `PACO_VIP_COMPARE_results (1).ipynb`)

### Mejoras clave en notebooks

- Setup robusto para Drive (evita fallo por `drive.mount` cuando ya esta montado).
- Auto-deteccion de rutas candidatas para:
  - `VIP-master/VIP-master`
  - `subchallenge1`
- Bloqueo de hiperparametros via `LOCKED_HPARAMS`.
- Fingerprint (`HP_FINGERPRINT`) para validar comparacion 1:1.
- Export de artefactos (CSV, figuras, perfilado, metadata) a Drive.
- Notebook comparador final con:
  - validacion de fingerprint CPU/GPU
  - tabla consolidada
  - speedup global
  - chequeo de robustez SNR
  - zip final para descarga local opcional

---

## 2) Causa raiz de inconsistencia CPU vs GPU

Archivo analizado:

- `VIP-master/VIP-master/src/vip_hci/invprob/paco.py`

### Hallazgo

La implementacion CPU de covarianza muestral no estaba alineada con la formula usada en el backend GPU (y con la formulacion esperada de PACO para la matriz de covarianza centrada).

### Correccion aplicada

En `sample_covariance(...)` se unifico la formula a:

- `S = (R - m)^T (R - m) / T`

y en `compute_statistics_at_pixel(...)`:

- conversion a `float64` en CPU para estabilidad numerica
- fallback a `np.linalg.pinv` si `np.linalg.inv` falla por singularidad

### Resultado

Se recupero consistencia cientifica entre CPU y GPU (SNR alineados en comparacion 1:1).

---

## 3) Optimizaciones de rendimiento que SI quedaron aplicadas

Todas en:

- `VIP-master/VIP-master/src/vip_hci/invprob/paco.py`

### 3.1 `get_patch(...)` optimizado

- Se elimino la construccion repetida de mascara global completa por pixel.
- Se usa ventana local + mascara circular local cacheada.
- Se agrego cache interno (`_patch_mask_cache`) y reinicio de cache al reescalar.

Impacto: menor overhead CPU/memoria en extraccion de patches.

### 3.2 `al(...)` y `bl(...)` vectorizados

- Se agrego ruta vectorizada con `np.einsum` para evitar loops Python internos por frame.
- Se mantiene fallback compatible.

Impacto: menor tiempo en operaciones matriciales por pixel.

### 3.3 `FastPACO.PACOCalc(...)` con path batched GPU (benchmark mode)

- Para `_GPU_BACKEND_ENABLED=True` y `use_subpixel_psf_astrometry=False`:
  - gather en bloques
  - computo de `a` y `b` con `cp.einsum`
- Se mantiene fallback completo para modo subpixel (`True`) y/o CPU.

Impacto: paralelizacion efectiva de parte importante de PACOCalc en GPU.

### 3.4 Autotuning de batch size GPU

- Nueva funcion helper: `_resolve_gpu_batch_size(...)`
- Prioridad:
  1. override por variable de entorno `PACO_GPU_BATCH_SIZE`
  2. estimacion por memoria libre GPU (`cupy.cuda.runtime.memGetInfo`)
  3. fallback default

Aplicado en:

- batching de `compute_statistics` GPU
- batching de `PACOCalc` GPU

---

## 4) Cambios experimentales que se probaron y luego se revirtieron

Se probaron dos cambios agresivos y luego se deshicieron por solicitud:

1. Vectorizacion global alternativa de trayectorias rotadas (`nx_all/ny_all`) reemplazando `get_rotated_pixel_coords` por pixel.
2. Reuso de buffers GPU prealocados en `PACOCalc` + transferencia de `diff = patch - m` en vez de `patch` y `m` por separado.

Estado final: ambos cambios **revertidos**. El resto de optimizaciones listadas arriba se mantiene.

---

## 5) Estado final reportado

Comparacion CPU/GPU validada con consistencia cientifica:

- Datasets robustos SNR: `2/2`
- Speedup global observado: alrededor de `2.37x` (valor reportado en la ultima corrida estable)

---

## 6) Archivos principales modificados

- `VIP-master/VIP-master/src/vip_hci/invprob/paco.py`
- `PACO_VIP_CPU_drive.ipynb`
- `PACO_VIP_GPU_drive.ipynb`
- `PACO_VIP_COMPARE_drive.ipynb`
- `PACO_VIP_CPU_results.ipynb`
- `PACO_VIP_GPU_results.ipynb`

---

## 7) Nota operativa para futuras corridas

Para forzar un batch fijo en GPU (si se desea comparar contra autotuning):

```python
import os
os.environ["PACO_GPU_BATCH_SIZE"] = "128"
```

y luego ejecutar el notebook GPU desde cero.

