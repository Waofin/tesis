# Informe de Cambios: Optimizacion PACO (CPU/GPU) y Validacion de Robustez

## Objetivo

Acelerar la ejecucion de PACO en Colab (GPU T4) manteniendo la formulacion cientifica y verificando que los resultados de deteccion sigan siendo robustos.

Este informe documenta:

- Que hacia el flujo original.
- Que se cambio exactamente (codigo y notebooks).
- Por que se paralelizo de esa forma.
- Como validar que los resultados siguen siendo cientificamente consistentes.


## Estado original (antes de optimizar)

El flujo base en `tesisfinal (4).ipynb` usaba `vip_hci.invprob.paco.FastPACO` con backend GPU opcional (`enable_paco_gpu_backend(True)`), pero:

- El benchmark corria con una carga pequena (aprox. 100 pixeles por dataset).
- El tiempo medido no estaba completamente estabilizado para GPU (sin warmup/sincronizacion explicita).
- La ruta GPU de covarianzas trabajaba de forma mayormente por pixel (alto overhead de transferencias CPU<->GPU por llamada).

En el codigo original de `VIP-master/VIP-master/src/vip_hci/invprob/paco.py`, `compute_statistics()` hacia un loop pixel a pixel y llamaba `compute_statistics_at_pixel(...)` por cada parche.


## Cambios implementados

## 1) Cambios en el algoritmo PACO (VIP)

Archivo modificado:

- `VIP-master/VIP-master/src/vip_hci/invprob/paco.py`

### 1.1 Paralelizacion en lote de estadisticas sobre GPU

Se agrego una nueva funcion:

- `_compute_statistics_batch_gpu(patches)`

Esta funcion recibe un batch de parches con forma `(N, T, P)` y calcula en GPU, para todos los pixeles del batch:

- media `m`,
- covarianza de muestra `S`,
- factor de shrinkage `rho`,
- covarianza regularizada `C`,
- inversa `Cinv`.

Esto reemplaza el patron anterior de calcular cada pixel por separado con ida/vuelta CPU<->GPU, reduciendo overhead de lanzamientos y transferencias.

### 1.2 Integracion del batching en `compute_statistics`

En `compute_statistics(...)` se agrego una rama cuando el backend GPU esta habilitado:

- Recolecta parches validos y sus coordenadas.
- Los procesa en lotes (`batch_size=128`) con `_compute_statistics_batch_gpu`.
- Escribe los resultados `m` y `Cinv` en las posiciones correspondientes.

Si GPU no esta habilitada, se mantiene el flujo original serial (compatibilidad).

### 1.3 Conservacion de la formulacion cientifica

No se cambio la ecuacion base del metodo PACO; se mantuvo la misma estructura estadistica:

- covarianza de muestra,
- shrinkage,
- inversion de covarianza.

El cambio es de implementacion computacional (batching/vectorizacion), no de criterio de deteccion.


## 2) Cambios en notebook principal de benchmark

Archivo modificado:

- `tesisfinal (4).ipynb`

### 2.1 Benchmark mas representativo (carga real)

Se agrego una funcion para construir `phi0s` en anillo:

- `build_phi0_annulus(...)`

Motivacion:

- Evitar medir solo una grilla minima cercana al centro.
- Aumentar cobertura espacial y carga realista para evaluar GPU T4.

### 2.2 Parametrizacion explicita del benchmark

Se incorporaron parametros en la celda principal:

- `BENCHMARK_CPU = 1`
- `BENCHMARK_PIXELS = 2500`
- `BENCHMARK_INNER_RADIUS = 8`
- `BENCHMARK_OUTER_RADIUS = 55`

`benchmark_paco_simple(...)` ahora acepta:

- `cpu`,
- `n_pixels_target`,
- `inner_radius`,
- `outer_radius`,
- `warmup_gpu`,
- `use_subpixel_psf_astrometry`.

### 2.3 Medicion de tiempos correcta para GPU

Se agrego:

- warmup de GPU,
- sincronizacion explicita antes y despues de `PACOCalc` usando `cp.cuda.Stream.null.synchronize`.

Esto evita subestimar/sobreestimar tiempo por asincronia de CUDA.

### 2.4 Profiling alineado al nuevo muestreo

`profile_paco_detailed(...)` se adapto para usar un patron anular representativo (con menor numero de pixeles para que el profiling sea practicable).


## 3) Notebook de validacion A/B de robustez

Archivo nuevo:

- `mini_robustez_paco_colab.ipynb`

Este notebook permite verificar robustez con el mismo dataset y exactamente los mismos `phi0s`, ejecutando:

- CPU,
- GPU_1,
- GPU_2 (repeticion).

Mide:

- diferencias absolutas en `a`, `b`, `snr`,
- correlacion de `snr`,
- solapamiento Top-K candidatos,
- conteos sobre umbrales (`SNR > 3, 5, 7`).

Incluye checks `OK/REVISAR` para evaluación rapida.


## Justificacion tecnica de la paralelizacion

La parte dominante en PACO para FastPACO es el calculo de estadisticas locales (medias/covarianzas/inversiones) sobre muchos parches.

El enfoque original por pixel en GPU tiene dos costos grandes:

- muchas llamadas pequenas a kernels,
- muchas conversiones CPU<->GPU.

El enfoque por lotes reduce ambos costos y mejora ocupacion de GPU:

- mas trabajo por kernel,
- menos transferencias por unidad de trabajo,
- mejor throughput total.


## Validaciones realizadas y criterio de robustez

## Validaciones ya observadas en resultados

- `RESULTADOS1.ipynb` muestra mejora importante de speedup GPU/CPU respecto al estado previo.
- La corrida produce validaciones de SNR y detecciones coherentes con mayor cobertura de pixeles.

Importante: si se comparan notebooks con distinta cantidad de pixeles evaluados (ej. 100 vs 2500), los conteos absolutos de deteccion no son 1:1.

## Validacion recomendada (A/B estricta)

Para corroborar robustez sin ambiguedad:

1. Mismo dataset.
2. Mismo recorte.
3. Mismos `phi0s`.
4. Mismos parametros PACO (`use_subpixel_psf_astrometry`, FWHM, etc.).
5. Comparar CPU vs GPU con las metricas del mini notebook.

Criterios sugeridos:

- `snr_corr > 0.99` (CPU vs GPU),
- `topk_overlap >= 0.8`,
- diferencias medias bajas en `a`, `b`, `snr`,
- repetibilidad alta GPU_1 vs GPU_2 (`snr_corr > 0.999`, `topk_overlap >= 0.95`).


## Lista concreta de archivos afectados

- Modificado: `VIP-master/VIP-master/src/vip_hci/invprob/paco.py`
- Modificado: `tesisfinal (4).ipynb`
- Nuevo: `mini_robustez_paco_colab.ipynb`
- Nuevo: `INFORME_CAMBIOS_PACO_OPTIMIZACION.md` (este documento)


## Riesgos conocidos y notas

- En la implementacion actual de FastPACO, el camino `cpu > 1` para ciertos subconjuntos de pixeles puede presentar problemas de reshape; por eso el benchmark se dejo en `cpu=1` para estabilidad.
- Las comparaciones de calidad cientifica deben hacerse siempre con mismas condiciones de muestreo para evitar interpretaciones incorrectas.


## Resumen ejecutivo

Se aplico una optimizacion real del nucleo de PACO (estadisticas por lotes en GPU) y una mejora metodologica del benchmark (carga representativa + timing correcto en CUDA).  
El resultado esperado es mantener la logica cientifica de PACO y obtener mejor eficiencia computacional en T4.  
La robustez se valida con comparacion A/B controlada (CPU/GPU) en el mini notebook de validacion.
