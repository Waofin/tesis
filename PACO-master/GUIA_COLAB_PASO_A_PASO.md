# Guía Paso a Paso: Ejecutar PACO Benchmark en Google Colab

## Paso 1: Preparar Datos en Google Drive

### 1.1 Subir archivos FITS a Google Drive

1. Ve a [Google Drive](https://drive.google.com)
2. Crea una carpeta llamada: `PACO_Challenge_Data`
3. Sube los archivos `.fits` del challenge a esa carpeta:
   - `sphere_irdis_cube_1.fits`
   - `sphere_irdis_pa_1.fits`
   - `sphere_irdis_psf_1.fits`
   - `sphere_irdis_pxscale_1.fits`
   - (y otros datasets que tengas)

### 1.2 Subir código PACO a Google Drive

**Opción A: Subir carpeta completa**
1. Comprime la carpeta `PACO-master` en un archivo ZIP
2. Súbelo a Google Drive
3. Descomprímelo en Drive (click derecho > "Extraer todo")

**Opción B: Usar desde tu repositorio (si tienes Git)**
- Solo necesitas subir los archivos `.fits`, el código se clonará desde GitHub

## Paso 2: Crear Notebook en Colab

1. Ve a [Google Colab](https://colab.research.google.com)
2. Click en "Nuevo notebook"
3. Cambia el nombre a: `PACO_Real_Data_Benchmark_Colab`

## Paso 3: Copiar el Código

Copia el contenido del archivo `PACO_Colab_Notebook_Completo.ipynb` que creé para ti, o sigue las celdas de abajo.

## Paso 4: Ajustar Rutas

En la celda de configuración, ajusta estas rutas según tu estructura en Drive:

```python
PACO_DIR = Path("/content/drive/MyDrive/PACO-master")  # Ruta al código
DATA_DIR = Path("/content/drive/MyDrive/PACO_Challenge_Data")  # Ruta a datos
OUTPUT_DIR = Path("/content/drive/MyDrive/PACO_Results")  # Donde guardar resultados
```

## Paso 5: Ejecutar

1. Ejecuta las celdas en orden (Shift+Enter)
2. La primera vez te pedirá autorización para acceder a Google Drive
3. El benchmark puede tardar varios minutos

## Ventajas de Colab

✅ No ocupa espacio en tu disco local
✅ Más RAM disponible (~12-15 GB)
✅ Resultados guardados automáticamente en Drive
✅ Puedes compartir el notebook fácilmente

## Solución de Problemas

### Error: "No se encuentra PACO_DIR"
- Verifica que la carpeta `PACO-master` esté en Drive
- Ajusta la ruta en la celda de configuración

### Error: "No se encuentran archivos .fits"
- Verifica que los archivos estén en `PACO_Challenge_Data`
- Revisa que los nombres sean exactos: `sphere_irdis_cube_1.fits`

### Error de memoria
- Reduce `crop_size` a `(80, 80)` o `(60, 60)`
- Reduce `n_trials` a 2

