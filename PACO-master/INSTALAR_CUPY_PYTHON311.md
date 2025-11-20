# Guía: Instalar CuPy con Python 3.11/3.12

## 🔴 Problema Actual

- **Python actual**: 3.13.9
- **Problema**: CuPy no es compatible con Python 3.13
- **Solución**: Usar Python 3.11 o 3.12

## ✅ Solución Paso a Paso

### Opción A: Crear Nuevo Entorno Virtual con Python 3.11/3.12

#### 1. Verificar si tienes Python 3.11 o 3.12 instalado

```powershell
# Verificar versiones disponibles
py -0  # Lista todas las versiones de Python instaladas
```

Si no tienes Python 3.11 o 3.12:

#### 2. Instalar Python 3.11 o 3.12

- Descargar desde: https://www.python.org/downloads/
- O usar pyenv-win: `pyenv install 3.11.9`

#### 3. Crear nuevo entorno virtual

```powershell
# Con Python 3.11
py -3.11 -m venv venv_paco_cuda

# O con Python 3.12
py -3.12 -m venv venv_paco_cuda
```

#### 4. Activar el nuevo entorno

```powershell
venv_paco_cuda\Scripts\activate
```

#### 5. Verificar versión

```powershell
python --version
# Debe mostrar: Python 3.11.x o 3.12.x
```

#### 6. Instalar dependencias

```powershell
# Instalar dependencias básicas
pip install numpy scipy matplotlib astropy

# Instalar CuPy (según tu versión de CUDA)
pip install cupy-cuda11x  # Para CUDA 11.x
# O
pip install cupy-cuda12x  # Para CUDA 12.x
```

#### 7. Verificar instalación

```powershell
python -c "import cupy as cp; print('CuPy OK:', cp.cuda.runtime.getDeviceCount(), 'GPUs')"
```

### Opción B: Usar Conda (Más Fácil)

Conda maneja mejor las dependencias y tiene wheels precompilados:

```powershell
# Instalar Miniconda si no lo tienes
# https://docs.conda.io/en/latest/miniconda.html

# Crear entorno
conda create -n paco_cuda python=3.11
conda activate paco_cuda

# Instalar CuPy
conda install -c conda-forge cupy
```

### Opción C: Continuar con CPU (Sin GPU)

Si no puedes instalar CuPy, puedes continuar usando la paralelización CPU con Loky que ya funciona bien:

```powershell
# Tu entorno actual (Python 3.13) funciona perfectamente para CPU
# Solo usa FullPACO con Loky en lugar de FullPACO_CUDA
```

## 🔍 Verificar Versión de CUDA

Antes de instalar CuPy, verifica tu versión de CUDA:

```powershell
nvidia-smi
```

Busca la línea "CUDA Version" para saber qué versión de CuPy instalar.

## 📝 Nota Importante

- **Python 3.13**: Muy nuevo, muchas librerías aún no compatibles
- **Python 3.11/3.12**: Versiones estables, mejor compatibilidad
- **Python 3.10 o anterior**: También funcionan pero menos optimizados

## 🚀 Después de Instalar

Una vez que tengas CuPy instalado:

```powershell
# Probar la implementación
python test_cuda_simple.py

# Ejecutar benchmark
python benchmark_cuda.py
```


