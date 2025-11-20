# Script PowerShell para cambiar Python default a 3.12

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CAMBIAR PYTHON DEFAULT A 3.12" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar versión actual
Write-Host "1. Verificando versión actual..." -ForegroundColor Yellow
$currentVersion = python --version 2>&1
Write-Host "   Python actual: $currentVersion" -ForegroundColor White

# Ver todas las versiones
Write-Host "`n2. Versiones de Python instaladas:" -ForegroundColor Yellow
py --list

# Verificar si Python 3.12 está disponible
Write-Host "`n3. Verificando Python 3.12..." -ForegroundColor Yellow
$py312 = py -3.12 --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Python 3.12 está disponible: $py312" -ForegroundColor Green
} else {
    Write-Host "   ❌ Python 3.12 NO está disponible" -ForegroundColor Red
    Write-Host "   Instala Python 3.12 desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Crear/editar py.ini
Write-Host "`n4. Configurando py.ini..." -ForegroundColor Yellow
$pyIniPath = "$env:USERPROFILE\py.ini"

if (Test-Path $pyIniPath) {
    Write-Host "   py.ini ya existe, editando..." -ForegroundColor White
    $content = Get-Content $pyIniPath -Raw
    
    if ($content -match "\[defaults\]") {
        # Ya tiene sección [defaults], actualizar
        $content = $content -replace "python=\d+\.\d+", "python=3.12"
        if ($content -notmatch "python=3.12") {
            $content = $content + "`npython=3.12"
        }
    } else {
        # Agregar sección [defaults]
        $content = $content + "`n[defaults]`npython=3.12"
    }
    
    Set-Content -Path $pyIniPath -Value $content
} else {
    Write-Host "   Creando nuevo py.ini..." -ForegroundColor White
    $content = "[defaults]`npython=3.12"
    Set-Content -Path $pyIniPath -Value $content
}

Write-Host "   ✅ py.ini configurado en: $pyIniPath" -ForegroundColor Green

# Verificar cambio
Write-Host "`n5. Verificando cambio..." -ForegroundColor Yellow
Start-Sleep -Seconds 1
$newVersion = py --version 2>&1
Write-Host "   Versión con 'py': $newVersion" -ForegroundColor White

if ($newVersion -match "3\.12") {
    Write-Host "   ✅ Python 3.12 es ahora el default" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ El cambio puede requerir reiniciar la terminal" -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "CONFIGURACIÓN COMPLETADA" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Yellow
Write-Host "  1. Cierra y reabre esta terminal" -ForegroundColor White
Write-Host "  2. Verifica: py --version (debe mostrar 3.12)" -ForegroundColor White
Write-Host '  3. Crea entorno: py -m venv venv_paco_cuda' -ForegroundColor White
Write-Host ""

