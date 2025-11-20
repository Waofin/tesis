# Script para configurar variables de entorno de CUDA después de la instalación

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CONFIGURANDO VARIABLES DE ENTORNO DE CUDA" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Buscar instalación de CUDA 12.x
$cudaPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
)

$cuda12Path = $null
foreach ($basePath in $cudaPaths) {
    if (Test-Path $basePath) {
        $versions = Get-ChildItem $basePath -Directory -ErrorAction SilentlyContinue | 
                    Where-Object { $_.Name -match "v12\." } |
                    Sort-Object Name -Descending |
                    Select-Object -First 1
        if ($versions) {
            $cuda12Path = $versions.FullName
            break
        }
    }
}

if (-not $cuda12Path) {
    Write-Host "[ERROR] No se encontro CUDA Toolkit 12.x instalado" -ForegroundColor Red
    Write-Host ""
    Write-Host "Por favor:" -ForegroundColor Yellow
    Write-Host "  1. Instala CUDA Toolkit 12.6 primero" -ForegroundColor White
    Write-Host "  2. Verifica que este en:" -ForegroundColor White
    Write-Host "     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "[OK] CUDA encontrado en: $cuda12Path" -ForegroundColor Green

$binPath = Join-Path $cuda12Path "bin"
$libPath = Join-Path $cuda12Path "libnvvp"

# Verificar que existan
if (-not (Test-Path $binPath)) {
    Write-Host "[ERROR] Directorio bin no encontrado: $binPath" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Directorio bin: $binPath" -ForegroundColor Green

# Verificar NVRTC
$nvrtcPath = Join-Path $binPath "nvrtc64_120_0.dll"
if (Test-Path $nvrtcPath) {
    Write-Host "[OK] NVRTC encontrado: $nvrtcPath" -ForegroundColor Green
} else {
    Write-Host "[WARNING] NVRTC no encontrado en ubicacion esperada" -ForegroundColor Yellow
    Write-Host "  Verifica que NVRTC este instalado" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Configurando variables de entorno..." -ForegroundColor Yellow

# Obtener PATH actual del usuario
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Verificar si ya está en PATH
if ($userPath -like "*$binPath*") {
    Write-Host "[OK] PATH ya contiene CUDA bin" -ForegroundColor Green
} else {
    Write-Host "Agregando al PATH del usuario..." -ForegroundColor White
    $newPath = if ($userPath) { "$userPath;$binPath;$libPath" } else { "$binPath;$libPath" }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "[OK] PATH actualizado" -ForegroundColor Green
}

# Configurar CUDA_PATH
[Environment]::SetEnvironmentVariable("CUDA_PATH", $cuda12Path, "User")
Write-Host "[OK] CUDA_PATH configurado: $cuda12Path" -ForegroundColor Green

# También configurar para esta sesión
$env:Path = "$env:Path;$binPath;$libPath"
$env:CUDA_PATH = $cuda12Path

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CONFIGURACION COMPLETADA" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANTE:" -ForegroundColor Yellow
Write-Host "  1. Cierra y reabre esta terminal para que los cambios surtan efecto" -ForegroundColor White
Write-Host "  2. Verifica con: nvcc --version" -ForegroundColor White
Write-Host "  3. Prueba CuPy: python test_cuda_simple.py" -ForegroundColor White
Write-Host ""

# Verificar nvcc en esta sesión
Write-Host "Verificando nvcc en esta sesion..." -ForegroundColor Yellow
try {
    $nvccVersion = & nvcc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] nvcc funciona:" -ForegroundColor Green
        Write-Host $nvccVersion -ForegroundColor White
    } else {
        Write-Host "[WARNING] nvcc no encontrado en esta sesion" -ForegroundColor Yellow
        Write-Host "  Reinicia la terminal para que funcione" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] nvcc no encontrado en esta sesion" -ForegroundColor Yellow
    Write-Host "  Reinicia la terminal para que funcione" -ForegroundColor Yellow
}

Write-Host ""

