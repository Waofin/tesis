@echo off
REM Script para crear entorno virtual con Python 3.12 e instalar CuPy

echo ============================================================
echo CREANDO ENTORNO VIRTUAL PARA CUDA (Python 3.12)
echo ============================================================
echo.

REM Crear entorno virtual con Python 3.12 de Anaconda
echo [1/5] Creando entorno virtual con Python 3.12 (Anaconda)...
C:\Users\cesar\anaconda3\python.exe -m venv venv_paco_cuda
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual
    echo Verificando si Python 3.12 de Anaconda existe...
    C:\Users\cesar\anaconda3\python.exe --version
    pause
    exit /b 1
)

REM Activar entorno
echo [2/5] Activando entorno...
call venv_paco_cuda\Scripts\activate.bat

REM Actualizar pip
echo [3/5] Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias básicas
echo [4/5] Instalando dependencias básicas...
pip install numpy scipy matplotlib astropy

REM Instalar CuPy (el usuario debe elegir según su CUDA)
echo [5/5] Instalando CuPy...
echo.
echo IMPORTANTE: Elige según tu versión de CUDA:
echo   - Si tienes CUDA 11.x: pip install cupy-cuda11x
echo   - Si tienes CUDA 12.x: pip install cupy-cuda12x
echo.
echo Para verificar tu versión de CUDA, ejecuta: nvidia-smi
echo.
echo.
echo NOTA: Tu sistema tiene CUDA 13.0, pero CuPy soporta hasta CUDA 12.x
echo       CUDA 13.0 es compatible hacia atrás, así que usaremos cupy-cuda12x
echo.
echo Instalando cupy-cuda12x...
pip install cupy-cuda12x

if errorlevel 1 (
    echo.
    echo ERROR: No se pudo instalar cupy-cuda12x
    echo Intentando con cupy-cuda11x como alternativa...
    pip install cupy-cuda11x
)

echo.
echo ============================================================
echo INSTALACIÓN COMPLETADA
echo ============================================================
echo.
echo Para usar este entorno en el futuro:
echo   1. Activar: venv_paco_cuda\Scripts\activate
echo   2. Verificar: python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
echo.
pause

