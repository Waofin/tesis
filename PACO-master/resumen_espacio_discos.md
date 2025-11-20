# Resumen de Espacio en Discos

## Estado Actual

| Disco | Total | Usado | Libre | % Usado |
|-------|-------|-------|-------|---------|
| **C:** | 236.47 GB | 232.99 GB | 3.48 GB | 98.5% |
| **D:** | 931.50 GB | 928.10 GB | 3.40 GB | 99.6% |
| **E:** | 464.60 GB | 450.42 GB | 14.18 GB | 96.9% |

## Problema Principal

**El disco D está casi lleno (99.6%)** pero NO por archivos de PACO/benchmarks.

### Archivos grandes en D:\

Los archivos más grandes encontrados son:
- **Videos de juegos** (Valorant, Overwatch): ~200+ GB
- **Steam Library**: Juegos instalados
- **Videos de captura**: Outplayed recordings

**Estos NO son archivos generados por los benchmarks de PACO.**

## Archivos de PACO

Los archivos relacionados con PACO están en:
- **E:\TESIS\PACO-master\output\**: ~1.41 MB
- **E:\TESIS\** (raíz): Archivos JSON/PNG de benchmarks anteriores (~95 MB)

## Recomendaciones

### Para liberar espacio en D:\

1. **Videos de juegos antiguos**: 
   - `D:\Valorant\` - Videos de partidas
   - `D:\BACKUP\videos\` - Backups de videos
   - `D:\valo\Outplayed\` - Grabaciones antiguas
   
   **Puedes eliminar videos antiguos que no necesites**

2. **Juegos de Steam no usados**:
   - `D:\SteamLibrary\` - Juegos instalados
   - Desinstala juegos que no uses

### Para liberar espacio en C:\

1. **Archivos temporales de Windows**: Ya limpiados parcialmente
2. **Caché de Python**: Ya limpiado (__pycache__)
3. **Logs de Windows**: Algunos logs grandes encontrados

### Para liberar espacio en E:\

1. **Archivos de benchmarks antiguos**: Ya identificados
2. **Entornos virtuales duplicados**: Ya limpiados (2.59 GB liberados)

## Conclusión

**Los benchmarks de PACO NO están llenando el disco D.**

El disco D está lleno principalmente por:
- Videos de juegos (~200+ GB)
- Juegos de Steam
- Archivos personales

**Solución recomendada:**
1. Mover videos de juegos a un disco externo o eliminarlos
2. Desinstalar juegos no usados de Steam
3. Usar Google Colab para ejecutar benchmarks (no ocupa espacio local)

