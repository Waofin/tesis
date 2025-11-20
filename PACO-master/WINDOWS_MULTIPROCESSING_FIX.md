# Fix para Multiprocessing en Windows

## Problema

En Windows, cuando se ejecuta código que usa `multiprocessing` desde un notebook o script interactivo, aparece el error:

```
An attempt has been made to start a new process before the current process has finished its bootstrapping phase.
```

## Solución

Se han aplicado dos correcciones:

1. **En `fastpaco.py`**: Cambiado de `Pool().map()` + `close()` + `join()` a usar contexto manager `with Pool():`
2. **En scripts principales**: Agregado `multiprocessing.freeze_support()` antes de importar módulos que usan multiprocessing

## Nota Importante

Para ejecutar benchmarks desde un notebook en Windows, es mejor usar el script `run_extended_benchmark.py` directamente desde la línea de comandos en lugar de ejecutarlo desde el notebook, ya que Windows tiene limitaciones con multiprocessing en entornos interactivos.

## Alternativa: Usar solo Loky

Si los problemas persisten, se recomienda usar solo la versión con Loky, que maneja mejor los procesos en Windows:

```python
# En la configuración, solo usar loky
config['skip_multiprocessing'] = True  # Si se implementa esta opción
```

