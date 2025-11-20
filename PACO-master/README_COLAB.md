# Cómo usar PACO Benchmark en Google Colab

## Resumen

Sí, puedes usar Google Colab para ejecutar el benchmark sin ocupar espacio en tu disco local. Los datos se almacenan en Google Drive.

## Pasos rápidos

### 1. Preparar datos en Google Drive

1. Ve a [Google Drive](https://drive.google.com)
2. Crea una carpeta llamada `PACO_Challenge_Data`
3. Sube los archivos `.fits` del challenge a esa carpeta:
   - `sphere_irdis_cube_1.fits`
   - `sphere_irdis_pa_1.fits`
   - `sphere_irdis_psf_1.fits`
   - `sphere_irdis_pxscale_1.fits`

### 2. Subir código PACO a Google Drive

1. Comprime la carpeta `PACO-master` en un archivo ZIP
2. Súbelo a Google Drive
3. Descomprímelo en Drive (o déjalo como ZIP y descomprímelo en Colab)

### 3. Crear notebook en Colab

1. Ve a [Google Colab](https://colab.research.google.com)
2. Crea un nuevo notebook
3. Copia el código del archivo `PACO_Colab_Notebook.ipynb` (que creé para ti)
4. Ajusta las rutas según tu estructura en Drive

## Ventajas

✅ **No ocupa espacio local**: Todo en la nube
✅ **Más RAM disponible**: Colab tiene ~12-15 GB RAM
✅ **GPU disponible**: Si necesitas aceleración
✅ **Resultados persistentes**: Se guardan en Drive

## Desventajas

⚠️ **Requiere internet**: Necesitas conexión estable
⚠️ **Sesiones limitadas**: Se desconectan después de ~90 min inactividad
⚠️ **Primera ejecución lenta**: Instalación de dependencias

## Archivos creados

He creado estos archivos para ti:

1. **`PACO_Colab_Setup.md`** - Guía detallada completa
2. **`PACO_Colab_Notebook.ipynb`** - Notebook listo para usar (necesita ser creado con edit_notebook)

## Alternativa: Usar directamente desde Drive

Si prefieres, puedes:
1. Subir el notebook `PACO_Real_Data_Benchmark.ipynb` a Colab
2. Modificar las rutas para apuntar a Drive
3. Ejecutar directamente

¿Quieres que cree el notebook de Colab completo ahora?

