# ✅ Solución Rápida: Cambiar Python Default a 3.12

## ✅ Ya está configurado

He creado el archivo `py.ini` en tu directorio de usuario que cambia el default a Python 3.12.

## 🔄 Para aplicar los cambios

**Cierra y reabre tu terminal PowerShell**, luego verifica:

```powershell
# Debe mostrar Python 3.12.x
py --version

# Debe mostrar Python 3.12.x
python --version
```

## 🚀 Crear Entorno Virtual con Python 3.12

Una vez que reabras la terminal:

```powershell
# Crear entorno (ahora usará 3.12 por defecto)
py -m venv venv_paco_cuda

# O explícitamente
py -3.12 -m venv venv_paco_cuda

# Activar
venv_paco_cuda\Scripts\activate

# Verificar versión
python --version  # Debe ser 3.12.x

# Instalar CuPy
pip install cupy-cuda12x
```

## 📝 Archivo py.ini

El archivo está en: `C:\Users\cesar\py.ini`

Contenido:
```ini
[defaults]
python=3.12
```

Si necesitas editarlo manualmente:
```powershell
notepad $env:USERPROFILE\py.ini
```

## ⚠️ Si No Funciona

Si después de reabrir la terminal sigue usando Python 3.13:

1. **Verifica que py.ini existe:**
   ```powershell
   Get-Content $env:USERPROFILE\py.ini
   ```

2. **Fuerza el uso de Python 3.12:**
   ```powershell
   # Siempre usa -3.12 explícitamente
   py -3.12 -m venv venv_paco_cuda
   ```

3. **O desinstala Python 3.13** (solo si no lo necesitas):
   - Settings > Apps > Buscar "Python 3.13" > Uninstall

## 💡 Recomendación

**NO desinstales Python 3.13** a menos que estés seguro. Es mejor:
- Mantener ambas versiones
- Usar `py.ini` para cambiar el default
- Usar `py -3.12` cuando necesites 3.12 explícitamente
- Usar `py -3.13` si alguna vez necesitas 3.13


