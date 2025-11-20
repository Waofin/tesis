# Script para descargar e instalar CUDA Toolkit 12.6

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "INSTALACION DE CUDA TOOLKIT 12.6" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar si ya está instalado
Write-Host "1. Verificando instalaciones existentes..." -ForegroundColor Yellow
$cudaPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
)

$installed = $false
foreach ($basePath in $cudaPaths) {
    if (Test-Path $basePath) {
        $versions = Get-ChildItem $basePath -Directory -ErrorAction SilentlyContinue
        if ($versions) {
            Write-Host "   [INFO] Encontradas versiones de CUDA:" -ForegroundColor White
            foreach ($ver in $versions) {
                Write-Host "      - $($ver.Name)" -ForegroundColor White
                if ($ver.Name -match "v12\.") {
                    $installed = $true
                    Write-Host "   [OK] CUDA Toolkit 12.x ya esta instalado!" -ForegroundColor Green
                    Write-Host "      Ruta: $($ver.FullName)" -ForegroundColor White
                    
                    # Verificar NVRTC
                    $nvrtcPath = Join-Path $ver.FullName "bin\nvrtc64_120_0.dll"
                    if (Test-Path $nvrtcPath) {
                        Write-Host "   [OK] NVRTC encontrado: $nvrtcPath" -ForegroundColor Green
                    } else {
                        Write-Host "   [WARNING] NVRTC no encontrado en ubicacion esperada" -ForegroundColor Yellow
                    }
                }
            }
        }
    }
}

if ($installed) {
    Write-Host ""
    Write-Host "¿Deseas continuar con la instalación de CUDA 12.6 de todas formas? (y/N): " -NoNewline -ForegroundColor Yellow
    $respuesta = Read-Host
    if ($respuesta -ne "y" -and $respuesta -ne "Y") {
        Write-Host "Instalacion cancelada." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "2. Informacion de descarga..." -ForegroundColor Yellow
Write-Host "   URL: https://developer.nvidia.com/cuda-12-6-0-download-archive" -ForegroundColor White
Write-Host "   Version: CUDA Toolkit 12.6" -ForegroundColor White
Write-Host "   Tamaño aproximado: ~3 GB" -ForegroundColor White
Write-Host ""

# Crear directorio de descargas
$downloadDir = "$env:USERPROFILE\Downloads"
if (-not (Test-Path $downloadDir)) {
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
}

Write-Host "3. Opciones de instalacion:" -ForegroundColor Yellow
Write-Host "   A) Descargar instalador automaticamente (requiere wget o Invoke-WebRequest)" -ForegroundColor White
Write-Host "   B) Abrir pagina de descarga en navegador" -ForegroundColor White
Write-Host "   C) Usar instalador local si ya lo descargaste" -ForegroundColor White
Write-Host ""
Write-Host "Selecciona opcion (A/B/C): " -NoNewline -ForegroundColor Yellow
$opcion = Read-Host

switch ($opcion.ToUpper()) {
    "A" {
        Write-Host ""
        Write-Host "Descargando CUDA Toolkit 12.6..." -ForegroundColor Yellow
        
        # URL de descarga directa (Windows 10/11, x86_64, exe local)
        $url = "https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.26.03_windows.exe"
        $outputFile = Join-Path $downloadDir "cuda_12.6.0_windows.exe"
        
        try {
            Write-Host "   Descargando desde: $url" -ForegroundColor White
            Write-Host "   Guardando en: $outputFile" -ForegroundColor White
            Write-Host "   (Esto puede tomar varios minutos...)" -ForegroundColor Yellow
            
            # Usar Invoke-WebRequest con barra de progreso
            $ProgressPreference = 'Continue'
            Invoke-WebRequest -Uri $url -OutFile $outputFile -UseBasicParsing
            
            Write-Host "   [OK] Descarga completada!" -ForegroundColor Green
            Write-Host ""
            Write-Host "4. Ejecutando instalador..." -ForegroundColor Yellow
            Write-Host "   Archivo: $outputFile" -ForegroundColor White
            
            Start-Process -FilePath $outputFile -Wait
            
            Write-Host "   [OK] Instalacion completada!" -ForegroundColor Green
        } catch {
            Write-Host "   [ERROR] Error en descarga: $_" -ForegroundColor Red
            Write-Host "   Por favor descarga manualmente desde:" -ForegroundColor Yellow
            Write-Host "   https://developer.nvidia.com/cuda-12-6-0-download-archive" -ForegroundColor White
            exit 1
        }
    }
    "B" {
        Write-Host ""
        Write-Host "Abriendo pagina de descarga..." -ForegroundColor Yellow
        Start-Process "https://developer.nvidia.com/cuda-12-6-0-download-archive"
        Write-Host ""
        Write-Host "Instrucciones:" -ForegroundColor Yellow
        Write-Host "   1. Selecciona: Windows > x86_64 > 10/11 > exe (local)" -ForegroundColor White
        Write-Host "   2. Descarga el archivo (aprox. 3 GB)" -ForegroundColor White
        Write-Host "   3. Ejecuta el instalador" -ForegroundColor White
        Write-Host "   4. Asegurate de marcar 'NVRTC' en la instalacion custom" -ForegroundColor White
        Write-Host "   5. Ejecuta este script de nuevo y selecciona opcion C" -ForegroundColor White
        exit 0
    }
    "C" {
        Write-Host ""
        Write-Host "Buscando instalador local..." -ForegroundColor Yellow
        
        # Buscar en Downloads
        $installer = Get-ChildItem $downloadDir -Filter "*cuda*12.6*.exe" -ErrorAction SilentlyContinue | 
                     Sort-Object LastWriteTime -Descending | 
                     Select-Object -First 1
        
        if (-not $installer) {
            # Buscar en directorio actual
            $installer = Get-ChildItem "." -Filter "*cuda*12.6*.exe" -ErrorAction SilentlyContinue | 
                         Sort-Object LastWriteTime -Descending | 
                         Select-Object -First 1
        }
        
        if ($installer) {
            Write-Host "   [OK] Instalador encontrado: $($installer.FullName)" -ForegroundColor Green
            Write-Host ""
            Write-Host "Ejecutando instalador..." -ForegroundColor Yellow
            Start-Process -FilePath $installer.FullName -Wait
            Write-Host "   [OK] Instalacion completada!" -ForegroundColor Green
        } else {
            Write-Host "   [ERROR] No se encontro instalador de CUDA 12.6" -ForegroundColor Red
            Write-Host "   Buscando en: $downloadDir" -ForegroundColor White
            Write-Host "   Buscando en: $(Get-Location)" -ForegroundColor White
            Write-Host ""
            Write-Host "   Por favor:" -ForegroundColor Yellow
            Write-Host "   1. Descarga CUDA Toolkit 12.6 desde:" -ForegroundColor White
            Write-Host "      https://developer.nvidia.com/cuda-12-6-0-download-archive" -ForegroundColor White
            Write-Host "   2. Guardalo en Downloads o en este directorio" -ForegroundColor White
            Write-Host "   3. Ejecuta este script de nuevo y selecciona opcion C" -ForegroundColor White
            exit 1
        }
    }
    default {
        Write-Host "   [ERROR] Opcion invalida" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "5. Configurando variables de entorno..." -ForegroundColor Yellow

# Buscar instalación de CUDA 12.x
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

if ($cuda12Path) {
    Write-Host "   [OK] CUDA encontrado en: $cuda12Path" -ForegroundColor Green
    
    $binPath = Join-Path $cuda12Path "bin"
    $libPath = Join-Path $cuda12Path "libnvvp"
    
    # Verificar PATH actual
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
    
    if ($currentPath -notlike "*$binPath*") {
        Write-Host "   Agregando al PATH del usuario..." -ForegroundColor White
        [Environment]::SetEnvironmentVariable("Path", "$currentPath;$binPath;$libPath", "User")
        Write-Host "   [OK] PATH actualizado" -ForegroundColor Green
    } else {
        Write-Host "   [OK] PATH ya contiene CUDA" -ForegroundColor Green
    }
    
    # Configurar CUDA_PATH
    [Environment]::SetEnvironmentVariable("CUDA_PATH", $cuda12Path, "User")
    Write-Host "   [OK] CUDA_PATH configurado: $cuda12Path" -ForegroundColor Green
    
    # Verificar NVRTC
    $nvrtcPath = Join-Path $binPath "nvrtc64_120_0.dll"
    if (Test-Path $nvrtcPath) {
        Write-Host "   [OK] NVRTC encontrado: $nvrtcPath" -ForegroundColor Green
    } else {
        Write-Host "   [WARNING] NVRTC no encontrado. Verifica la instalacion." -ForegroundColor Yellow
    }
    
} else {
    Write-Host "   [WARNING] No se encontro CUDA 12.x instalado" -ForegroundColor Yellow
    Write-Host "   Configura manualmente las variables de entorno:" -ForegroundColor White
    Write-Host "   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" -ForegroundColor White
    Write-Host "   PATH += C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin" -ForegroundColor White
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "INSTALACION COMPLETADA" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "IMPORTANTE:" -ForegroundColor Yellow
Write-Host "   1. Cierra y reabre esta terminal para que los cambios surtan efecto" -ForegroundColor White
Write-Host "   2. Verifica con: nvcc --version" -ForegroundColor White
Write-Host "   3. Prueba CuPy de nuevo: python test_cuda_simple.py" -ForegroundColor White
Write-Host ""

