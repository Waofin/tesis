@echo off
REM Script para configurar entorno CUDA con Python 3.12 de Anaconda

echo ============================================================
echo CONFIGURANDO ENTORNO CUDA CON PYTHON 3.12 (ANACONDA)
echo ============================================================
echo.

cd /d "%~dp0"

REM Verificar si el entorno existe
if not exist "venv_paco_cuda" (
    echo Creando entorno virtual con Python 3.12 de Anaconda...
    C:\Users\cesar\anaconda3\python.exe -m venv venv_paco_cuda
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual
        pause
        exit /b 1
    )
    echo [OK] Entorno creado
) else (
    echo [OK] Entorno ya existe
)

echo.
echo Activando entorno...
call venv_paco_cuda\Scripts\activate.bat

echo.
echo Verificando versión de Python...
python --version

echo.
echo ============================================================
echo INSTALANDO CUPY
echo ============================================================
echo.

REM Detectar versión de CUDA
echo Detectando versión de CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ADVERTENCIA: nvidia-smi no encontrado. Instalando cupy-cuda12x por defecto...
    set CUPY_PACKAGE=cupy-cuda12x
) else (
    REM Intentar detectar versión de CUDA desde nvidia-smi
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2^>nul') do set DRIVER_VERSION=%%i
    
    REM Por defecto usar CUDA 12.x (compatible con CUDA 13.0)
    set CUPY_PACKAGE=cupy-cuda12x
    echo Usando: %CUPY_PACKAGE%
)

echo.
echo Instalando %CUPY_PACKAGE%...
pip install %CUPY_PACKAGE%

if errorlevel 1 (
    echo.
    echo ERROR: Fallo la instalación de CuPy
    echo.
    echo Intentando con cupy-cuda11x como alternativa...
    pip install cupy-cuda11x
    if errorlevel 1 (
        echo ERROR: No se pudo instalar CuPy
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo VERIFICANDO INSTALACIÓN
echo ============================================================
echo.

python -c "import cupy as cp; print('CuPy version:', cp.__version__); print('CUDA version:', cp.cuda.runtime.runtimeGetVersion()); print('GPU disponible:', cp.cuda.is_available())"

if errorlevel 1 (
    echo ERROR: CuPy no se importó correctamente
    pause
    exit /b 1
)

echo.
echo ============================================================
echo INSTALACIÓN COMPLETADA
echo ============================================================
echo.
echo Para usar este entorno en el futuro:
echo   1. cd e:\TESIS\PACO-master
echo   2. venv_paco_cuda\Scripts\activate
echo   3. python --version (debe ser 3.12.x)
echo.
pause

