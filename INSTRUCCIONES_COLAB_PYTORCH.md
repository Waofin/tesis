# Instrucciones para Ejecutar PACO con PyTorch GPU en Google Colab

## Configuración Inicial

### 1. Configurar GPU en Colab

1. Ve a **Runtime > Change runtime type**
2. En **Hardware accelerator**, selecciona:
   - **GPU T4** (gratis, recomendado para empezar)
   - **GPU A100** (más rápida, requiere Colab Pro)
   - **GPU L4** (buena opción intermedia)
3. Activa **High RAM** si vas a procesar imágenes grandes
4. Haz clic en **Save**

### 2. Subir el Código a Colab

Tienes dos opciones:

#### Opción A: Desde GitHub (Recomendado)
```python
# Ejecutar en la primera celda del notebook
!git clone https://github.com/tu-usuario/PACO-master.git
# o sube el repositorio completo
```

#### Opción B: Subir Archivos Manualmente
1. Clic en el ícono de carpeta (Files) en el panel izquierdo
2. Sube la carpeta `PACO-master` completa
3. Asegúrate de que la estructura sea: `/content/PACO-master/`

### 3. Instalar Dependencias

```python
# Ejecutar en una celda del notebook
!pip install torch scipy joblib matplotlib numpy astropy
```

## Uso del Notebook

### Estructura del Notebook

El notebook `PACO_Comparison_Optimized_vs_VIP.ipynb` incluye:

1. **Detección automática de hardware**: Detecta GPU, CPU, PyTorch, etc.
2. **Carga de datos**: Sintéticos o reales
3. **Ejecución CPU optimizada**: FullPACO con optimizaciones
4. **Ejecución GPU PyTorch**: FastPACO_PyTorch con aceleración GPU
5. **Comparación con VIP**: Si está disponible
6. **Visualización y validación**: Comparación de resultados

### Ejecutar Todas las Celdas

1. Abre el notebook en Colab
2. Ejecuta todas las celdas: **Runtime > Run all**
3. O ejecuta celda por celda con `Shift + Enter`

## Resultados Esperados

### Con GPU T4 (Gratis)
- **Speedup GPU vs CPU secuencial**: 50-100×
- **Speedup GPU vs CPU paralelo (8 cores)**: 5-10×
- **Tiempo típico** (900 píxeles, 20 frames): 2-5 segundos

### Con GPU A100 (Colab Pro)
- **Speedup GPU vs CPU secuencial**: 100-200×
- **Speedup GPU vs CPU paralelo (8 cores)**: 10-20×
- **Tiempo típico** (900 píxeles, 20 frames): 1-2 segundos

### Con CPU solamente
- **Speedup CPU paralelo vs secuencial**: 6-8×
- **Tiempo típico** (900 píxeles, 20 frames): 30-60 segundos

## Solución de Problemas

### Error: "PyTorch no disponible"
```python
!pip install torch
```

### Error: "GPU no disponible"
- Verifica que seleccionaste GPU en Runtime settings
- Reinicia el runtime: **Runtime > Restart runtime**
- Verifica con:
```python
import torch
print(torch.cuda.is_available())
```

### Error: "Module not found: paco"
- Verifica que `PACO-master` está en el path correcto
- Ajusta el path en la celda de importación:
```python
paco_master_path = Path('/content/PACO-master')  # Para Colab
```

### Error: "Out of memory"
- Reduce el `batch_size` en `computeStatistics_PyTorch`
- Procesa menos píxeles a la vez
- Usa `batch_mode=False` en lugar de `batch_mode=True`

## Optimizaciones Disponibles

### FastPACO_PyTorch

**Modo Normal** (`batch_mode=False`):
- Procesa píxeles en batches pequeños
- Menor uso de memoria
- Bueno para GPUs con poca memoria (T4)

**Modo Batch** (`batch_mode=True`):
- Procesa múltiples píxeles simultáneamente
- Mayor uso de memoria
- Más rápido en GPUs potentes (A100)
- Requiere más memoria GPU

### Ajustar Batch Size

Si tienes problemas de memoria, ajusta el `batch_size`:

```python
# En la celda de ejecución GPU
Cinv, m, h = paco_gpu.computeStatistics_PyTorch(phi0s, batch_size=500)  # Reducir si hay OOM
```

## Comparación de Métodos

| Método | Dispositivo | Speedup | Uso de Memoria |
|--------|-------------|---------|----------------|
| FullPACO Original | CPU (1 core) | 1× (baseline) | Bajo |
| FullPACO Optimizado | CPU (8 cores) | ~15× | Medio |
| FastPACO PyTorch | GPU T4 | ~50-100× | Alto |
| FastPACO PyTorch | GPU A100 | ~100-200× | Alto |
| FastPACO PyTorch Batch | GPU A100 | ~150-250× | Muy Alto |

## Notas Importantes

1. **Primera ejecución**: La primera vez puede ser más lenta (compilación JIT de PyTorch)
2. **Warm-up**: Ejecuta una vez pequeña antes de medir tiempos
3. **Precisión**: Los resultados GPU vs CPU deben ser idénticos (diferencia < 0.01%)
4. **Colab Pro**: Para acceso consistente a GPU A100, considera Colab Pro

## Contacto

Para problemas o preguntas, contacta a:
- César Cerda - Universidad del Bío-Bío

