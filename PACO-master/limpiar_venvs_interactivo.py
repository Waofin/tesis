#!/usr/bin/env python3
"""
Script interactivo para limpiar entornos virtuales no usados
"""

from limpiar_entornos_virtuales import find_venvs, print_venv_summary, check_venv_usage, clean_venvs
from pathlib import Path

if __name__ == "__main__":
    base_dir = Path("E:/TESIS")
    
    print("=" * 80)
    print("LIMPIEZA DE ENTORNOS VIRTUALES")
    print("=" * 80)
    
    # Buscar entornos virtuales
    venvs = find_venvs(base_dir)
    venvs_sorted = print_venv_summary(venvs)
    
    if not venvs_sorted:
        print("\n[OK] No hay entornos virtuales para limpiar")
        exit(0)
    
    # Identificar candidatos para eliminar
    candidates = []
    for venv in venvs_sorted:
        in_use = check_venv_usage(venv['path'])
        if not in_use and venv['size_gb'] > 0.1:
            candidates.append(venv)
    
    if not candidates:
        print("\n[OK] No hay entornos virtuales no usados para eliminar")
        exit(0)
    
    # Mostrar candidatos
    print("\n" + "=" * 80)
    print("CANDIDATOS PARA ELIMINAR")
    print("=" * 80)
    total_to_free = 0
    for i, venv in enumerate(candidates, 1):
        print(f"\n[{i}] {venv['name']}")
        print(f"    Ruta: {venv['full_path']}")
        print(f"    Tamaño: {venv['size_gb']:.2f} GB")
        total_to_free += venv['size_gb']
    
    print("\n" + "=" * 80)
    print(f"ESPACIO A LIBERAR: {total_to_free:.2f} GB")
    print("=" * 80)
    
    # Preguntar confirmación
    print("\n¿Deseas eliminar estos entornos virtuales?")
    print("  [S] Sí, eliminar")
    print("  [N] No, cancelar")
    
    # Para ejecución automática, descomenta la siguiente línea:
    # response = 'S'
    # Para modo interactivo, usa:
    response = input("\nTu respuesta (S/N): ").strip().upper()
    
    if response == 'S':
        print("\n[ELIMINANDO] Eliminando entornos virtuales...")
        print("-" * 80)
        
        deleted_count = 0
        deleted_size = 0
        
        import shutil
        for venv in candidates:
            try:
                shutil.rmtree(venv['path'])
                print(f"  [OK] Eliminado: {venv['name']} ({venv['size_gb']:.2f} GB)")
                deleted_count += 1
                deleted_size += venv['size_gb']
            except Exception as e:
                print(f"  [ERROR] No se pudo eliminar {venv['name']}: {e}")
        
        print("\n" + "=" * 80)
        print(f"ELIMINADOS: {deleted_count} entornos virtuales, {deleted_size:.2f} GB")
        print("=" * 80)
    else:
        print("\n[CANCELADO] No se eliminaron entornos virtuales")

