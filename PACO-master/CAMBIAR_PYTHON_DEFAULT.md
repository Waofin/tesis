# Cambiar Python por Defecto de 3.13 a 3.12

## 🔍 Verificar Configuración Actual

```powershell
# Ver qué versión se usa por defecto
python --version

# Ver todas las versiones instaladas
py --list

# Ver qué versión usa 'py' sin especificar
py --version
```

## ✅ Solución: Cambiar Versión por Defecto

### Opción 1: Usar py.ini (Recomendado - No requiere desinstalar)

Crea/edita el archivo `py.ini` en tu directorio de usuario:

```powershell
# Crear archivo py.ini
notepad $env:USERPROFILE\py.ini
```

Agrega este contenido:

```ini
[defaults]
python=3.12
```

Ahora `py` usará Python 3.12 por defecto.

### Opción 2: Cambiar Variable de Entorno PATH

1. **Abrir Variables de Entorno:**
   - Presiona `Win + R`
   - Escribe: `sysdm.cpl`
   - Ve a la pestaña "Opciones avanzadas"
   - Click en "Variables de entorno"

2. **Editar PATH:**
   - Busca la variable `Path` en "Variables del sistema"
   - Edita y mueve la ruta de Python 3.12 **ANTES** de la de Python 3.13
   - Ejemplo:
     ```
     C:\Python312\Scripts
     C:\Python312\
     C:\Python313\Scripts  <- Mover después
     C:\Python313\
     ```

3. **Reiniciar terminal** para que los cambios surtan efecto

### Opción 3: Desinstalar Python 3.13 (Si no lo necesitas)

#### Método A: Desde Settings de Windows

1. Abre **Settings** (Configuración)
2. Ve a **Apps** > **Installed apps** (Aplicaciones instaladas)
3. Busca "Python 3.13"
4. Click en los tres puntos (...) > **Uninstall**

#### Método B: Desde Panel de Control

1. Abre **Panel de Control**
2. Ve a **Programs** > **Uninstall a program**
3. Busca "Python 3.13.x"
4. Click derecho > **Uninstall**

#### Método C: Desde PowerShell (Administrador)

```powershell
# Listar paquetes de Python instalados
Get-Package | Where-Object {$_.Name -like "*Python*3.13*"}

# Desinstalar (reemplaza con el nombre exacto)
Uninstall-Package -Name "Python.3.13" -Force
```

## 🎯 Verificar Cambio

Después de hacer los cambios:

```powershell
# Debe mostrar Python 3.12.x
python --version

# Debe mostrar Python 3.12.x
py --version
```

## 💡 Recomendación

**NO desinstales Python 3.13** a menos que estés seguro de que no lo necesitas.

**Mejor opción**: Usa `py.ini` para cambiar el default a 3.12, pero mantén ambas versiones instaladas. Así puedes:
- Usar `py` → Python 3.12 (default)
- Usar `py -3.13` → Python 3.13 (si lo necesitas)
- Usar `py -3.12` → Python 3.12 (explícito)

## 🚀 Para el Entorno Virtual de CuPy

Una vez que cambies el default a 3.12:

```powershell
# Crear entorno con Python 3.12 (ahora es el default)
py -m venv venv_paco_cuda

# O explícitamente
py -3.12 -m venv venv_paco_cuda

# Activar
venv_paco_cuda\Scripts\activate

# Verificar
python --version  # Debe ser 3.12.x
```


