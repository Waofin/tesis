# Solución: Error nvrtc64_120_0.dll

## 🔍 Problema

CuPy necesita la DLL `nvrtc64_120_0.dll` que es parte del **CUDA Toolkit**, no solo del driver.

**Error:**
```
RuntimeError: CuPy failed to load nvrtc64_120_0.dll
```

## ✅ Solución

### Opción 1: Instalar CUDA Toolkit (Recomendado)

1. **Descargar CUDA Toolkit 12.x:**
   - Ir a: https://developer.nvidia.com/cuda-downloads
   - Seleccionar Windows > x86_64 > 10/11 > exe (local)
   - Descargar CUDA Toolkit 12.6 o 12.7 (compatible con CUDA 13.0 driver)

2. **Instalar:**
   - Ejecutar el instalador
   - Seleccionar "Custom installation"
   - Asegurarse de que "NVRTC" esté marcado
   - Instalar en ubicación por defecto (normalmente `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`)

3. **Configurar variables de entorno:**
   ```powershell
   # Agregar al PATH del sistema:
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp
   
   # Crear variable CUDA_PATH:
   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
   ```

4. **Reiniciar terminal** y probar de nuevo

### Opción 2: Usar CuPy sin NVRTC (Limitado)

Si no puedes instalar CUDA Toolkit, puedes usar CuPy en modo "precompiled kernels only", pero tendrás funcionalidad limitada.

```python
import os
os.environ['CUPY_ACCELERATORS'] = 'cub'  # Usar solo kernels precompilados
```

**Limitación:** Algunas operaciones avanzadas pueden no funcionar.

### Opción 3: Usar CPU (Temporal)

Mientras instalas CUDA Toolkit, puedes usar la versión CPU:

```python
# En fullpaco_cuda.py, cambiar use_gpu_pixelcalc=False
fp_cuda.PACOCalc_CUDA(phi0s, use_gpu_pixelcalc=False)
```

## 🔧 Verificación

Después de instalar CUDA Toolkit:

```powershell
# Verificar nvcc
nvcc --version

# Verificar DLL
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\nvrtc64_120_0.dll"
```

## 📝 Nota

- **CUDA Driver** (ya instalado): Permite ejecutar programas CUDA
- **CUDA Toolkit** (falta): Proporciona compiladores y librerías como NVRTC

Para CuPy necesitas **ambos**.

