# Guía de Instalación: CUDA Toolkit 12.6

## 🎯 Objetivo

Instalar CUDA Toolkit 12.6 para habilitar NVRTC que requiere CuPy.

## 📥 Paso 1: Descargar CUDA Toolkit

1. **Abre la página de descarga** (ya debería estar abierta en tu navegador):
   ```
   https://developer.nvidia.com/cuda-12-6-0-download-archive
   ```

2. **Selecciona las opciones:**
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **10** o **11** (según tu Windows)
   - Installer Type: **exe (local)**

3. **Click en "Download"**
   - Tamaño aproximado: ~3 GB
   - Tiempo estimado: 10-30 minutos (depende de tu conexión)

## 🔧 Paso 2: Instalar CUDA Toolkit

1. **Ejecuta el instalador descargado:**
   - Archivo: `cuda_12.6.0_560.26.03_windows.exe`
   - Ubicación: `C:\Users\cesar\Downloads\`

2. **Durante la instalación:**
   - Selecciona **"Custom (Advanced)"** en lugar de "Express"
   - Asegúrate de que **"NVRTC"** esté marcado
   - También marca:
     - ✅ CUDA Runtime
     - ✅ CUDA Development Tools
     - ✅ CUDA Samples (opcional)
   - Desmarca componentes que no necesites para ahorrar espacio

3. **Ubicación de instalación:**
   - Por defecto: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
   - Puedes dejarlo así o cambiar si prefieres

4. **Completa la instalación** (puede tomar 10-20 minutos)

## ⚙️ Paso 3: Configurar Variables de Entorno

Después de instalar, ejecuta este script para configurar automáticamente:

```powershell
cd e:\TESIS\PACO-master
.\configurar_cuda_env.ps1
```

O manualmente:

1. **Abrir Variables de Entorno:**
   - Presiona `Win + R`
   - Escribe: `sysdm.cpl`
   - Ve a "Opciones avanzadas" > "Variables de entorno"

2. **Agregar al PATH (Usuario):**
   - Selecciona "Path" en "Variables de usuario"
   - Click "Editar" > "Nuevo"
   - Agregar: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
   - Agregar: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\libnvvp`
   - Click "Aceptar"

3. **Crear variable CUDA_PATH:**
   - Click "Nueva" en "Variables de usuario"
   - Nombre: `CUDA_PATH`
   - Valor: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
   - Click "Aceptar"

4. **Aceptar todos los diálogos**

## ✅ Paso 4: Verificar Instalación

**Cierra y reabre tu terminal PowerShell**, luego ejecuta:

```powershell
# Verificar nvcc
nvcc --version

# Debe mostrar algo como:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on ...
# Cuda compilation tools, release 12.6, V12.6.xxx
```

```powershell
# Verificar NVRTC
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvrtc64_120_0.dll"
# Debe retornar: True
```

## 🚀 Paso 5: Probar CuPy

```powershell
cd e:\TESIS\PACO-master
venv_paco_cuda\Scripts\activate
python test_cuda_simple.py
```

Debería funcionar sin errores de NVRTC.

## 🔍 Solución de Problemas

### Error: "nvcc no se reconoce"
- **Causa**: PATH no configurado o terminal no reiniciada
- **Solución**: Reinicia la terminal y verifica PATH

### Error: "nvrtc64_120_0.dll no encontrado"
- **Causa**: NVRTC no instalado o en ubicación incorrecta
- **Solución**: Reinstala CUDA Toolkit asegurándote de marcar "NVRTC"

### Error: "CUDA version mismatch"
- **Causa**: Driver y Toolkit no coinciden
- **Solución**: Tu driver es CUDA 13.0, Toolkit 12.6 es compatible (backward compatible)

## 📝 Notas

- **Tamaño total**: ~3-4 GB (instalador + instalación)
- **Tiempo total**: 20-40 minutos (descarga + instalación)
- **Requisitos**: Administrador (puede pedir permisos)

## 🎉 Después de Instalar

Una vez instalado y configurado, podrás:
- ✅ Usar CuPy sin errores de NVRTC
- ✅ Ejecutar `test_cuda_simple.py` exitosamente
- ✅ Ejecutar `benchmark_cuda.py` para comparar CPU vs GPU
- ✅ Usar `FullPACO_CUDA` con aceleración GPU completa

