# Instrucciones para Instalar CuPy con Python 3.12

## ✅ Tu Sistema

- **GPU**: NVIDIA GeForce RTX 2060 (6 GB VRAM) ✅ Excelente para PACO
- **CUDA**: 13.0 (según driver)
- **Python disponible**: 3.12 ✅ (compatible con CuPy)

## 🚀 Pasos Rápidos

### Opción 1: Script Automático (Recomendado)

```powershell
# Ejecutar el script que creé
.\setup_cuda_env.bat
```

### Opción 2: Manual

```powershell
# 1. Crear entorno virtual con Python 3.12
py -3.12 -m venv venv_paco_cuda

# 2. Activar entorno
venv_paco_cuda\Scripts\activate

# 3. Verificar versión (debe ser 3.12.x)
python --version

# 4. Actualizar pip
python -m pip install --upgrade pip

# 5. Instalar dependencias básicas
pip install numpy scipy matplotlib astropy

# 6. Instalar CuPy
# NOTA: CuPy soporta hasta CUDA 12.x, pero CUDA 13.0 es compatible hacia atrás
# Usa cupy-cuda12x que debería funcionar con CUDA 13.0
pip install cupy-cuda12x

# 7. Verificar instalación
python -c "import cupy as cp; print('GPUs:', cp.cuda.runtime.getDeviceCount())"
```

## ⚠️ Nota sobre CUDA 13.0

Tu sistema muestra CUDA 13.0, pero CuPy oficialmente soporta hasta CUDA 12.x. Sin embargo:
- **CUDA 13.0 es compatible hacia atrás** con CUDA 12.x
- **Usa `cupy-cuda12x`** - debería funcionar
- Si no funciona, prueba `cupy-cuda11x` como alternativa

## 🔍 Verificar Instalación

Después de instalar, ejecuta:

```powershell
python test_cuda_simple.py
```

Debería mostrar:
```
✅ CuPy importado correctamente
✅ GPUs detectadas: 1
✅ GPU 0: NVIDIA GeForce RTX 2060
```

## 📝 Uso del Entorno

Cada vez que quieras usar CUDA:

```powershell
# Activar entorno
venv_paco_cuda\Scripts\activate

# Ejecutar código
python benchmark_cuda.py
```

## 🐛 Si Hay Problemas

### Error: "No module named 'cupy'"

- Verifica que estás en el entorno correcto: `python --version` debe ser 3.12.x
- Reinstala: `pip install cupy-cuda12x`

### Error: "CUDA runtime not found"

- Verifica que CUDA está instalado: `nvcc --version`
- Si no está, instala CUDA Toolkit desde NVIDIA

### Error: "Out of memory"

- Tu RTX 2060 tiene 6 GB - suficiente para imágenes hasta ~150×150
- Para imágenes más grandes, reduce `batch_size` en `PACOCalc_CUDA()`

## 🎯 Próximos Pasos

Una vez instalado:

1. **Probar implementación básica:**
   ```powershell
   python test_cuda_simple.py
   ```

2. **Ejecutar benchmark:**
   ```powershell
   python benchmark_cuda.py
   ```

3. **Usar en tu código:**
   ```python
   from paco.processing.fullpaco_cuda import FullPACO_CUDA
   # ... tu código ...
   ```

## 💡 Alternativa: Continuar con CPU

Si prefieres no instalar CuPy ahora, puedes continuar usando:
- **FullPACO con Loky** (paralelización CPU) - ya funciona bien
- Speedup: ~8-12x con 8 CPUs
- No requiere GPU ni CuPy

La implementación CUDA es opcional y da speedups adicionales si tienes GPU.


