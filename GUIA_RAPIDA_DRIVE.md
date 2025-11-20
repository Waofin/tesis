# Guía Rápida: Usar PACO desde Google Drive en Colab

## Pasos Rápidos

### 1. Preparar Archivos en Drive
- Sube la carpeta completa `PACO-master` a tu Google Drive
- Puedes ponerla en cualquier ubicación, por ejemplo:
  - `/MyDrive/PACO-master` (directamente en MyDrive)
  - `/MyDrive/Tesis/PACO-master` (en una carpeta Tesis)
  - `/MyDrive/Colab Notebooks/PACO-master` (en Colab Notebooks)

### 2. Abrir Notebook en Colab
- Sube `PACO_Comparison_Optimized_vs_VIP.ipynb` a Colab
- O crea un nuevo notebook y copia las celdas

### 3. Configurar GPU
- **Runtime > Change runtime type**
- **Hardware accelerator: GPU** (T4 o A100)
- **High RAM: ON** (recomendado)
- **Save**

### 4. Ejecutar Celdas de Configuración

Ejecuta en orden las siguientes celdas:

#### Celda 0.1: Montar Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Se abrirá una ventana para autorizar acceso
- Copia el código de autorización cuando se solicite

#### Celda 0.2: Configurar Path
```python
# Ajusta esta ruta según donde esté PACO-master en tu Drive
DRIVE_PACO_PATH = '/content/drive/MyDrive/PACO-master'
```
- Si está en otra ubicación, cambia la ruta
- Ejemplo: `/content/drive/MyDrive/Tesis/PACO-master`

#### Celda 0.3: Copiar a /content
- Esta celda copia los archivos desde Drive a `/content`
- Es más rápido trabajar desde `/content` que directamente desde Drive
- Puede tardar 1-2 minutos dependiendo del tamaño

#### Celda 0.4: Instalar Dependencias
- Verifica e instala PyTorch, scipy, joblib, etc.
- Solo instala lo que falta

### 5. Continuar con el Notebook
- Las siguientes celdas detectarán automáticamente los archivos
- El notebook buscará en:
  1. `/content/PACO-master` (si se copió desde Drive)
  2. `/content/drive/MyDrive/PACO-master` (directo desde Drive)
  3. Otros paths comunes

## Solución de Problemas

### "No se encontró PACO-master"
**Solución:**
1. Verifica que la carpeta existe en Drive
2. Lista el contenido de Drive:
   ```python
   !ls /content/drive/MyDrive/
   ```
3. Ajusta `DRIVE_PACO_PATH` con la ruta correcta

### "Permission denied"
**Solución:**
- Vuelve a ejecutar `drive.mount('/content/drive')`
- Asegúrate de autorizar el acceso correctamente

### "Out of memory" al copiar
**Solución:**
- Trabaja directamente desde Drive (comenta la celda 0.3)
- O comprime PACO-master antes de subirlo y descomprímelo en Colab

### Datos muy grandes
**Solución:**
- Si PACO-master es muy grande, considera:
  - Subir solo las carpetas necesarias (`paco/`, `testData/`)
  - Usar datos sintéticos en lugar de datos reales grandes

## Estructura Recomendada en Drive

```
MyDrive/
├── PACO-master/          # Código principal
│   ├── paco/
│   ├── testData/
│   └── ...
├── Datos/                # Datos de prueba (opcional)
│   ├── naco_betapic_cube.fits
│   ├── naco_betapic_pa.fits
│   └── naco_betapic_psf.fits
└── Notebooks/            # Tus notebooks de Colab
    └── PACO_Comparison_Optimized_vs_VIP.ipynb
```

## Tips

1. **Primera vez**: La primera ejecución puede tardar más (instalación, copia de archivos)
2. **Re-ejecutar**: Si reinicias el runtime, solo necesitas volver a montar Drive
3. **Persistencia**: Los archivos en `/content` se pierden al reiniciar, pero Drive permanece
4. **Velocidad**: Trabajar desde `/content` es más rápido que desde Drive directamente

## Comandos Útiles

```python
# Ver qué hay en Drive
!ls /content/drive/MyDrive/

# Ver tamaño de PACO-master
!du -sh /content/drive/MyDrive/PACO-master

# Verificar estructura
!ls /content/PACO-master/paco/processing/

# Limpiar espacio si es necesario
!rm -rf /content/PACO-master  # Solo si ya no lo necesitas
```


