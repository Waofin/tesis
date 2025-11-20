#!/usr/bin/env python3
"""
Script para limpiar automáticamente entornos virtuales no usados
"""

from limpiar_entornos_virtuales import find_venvs, print_venv_summary, check_venv_usage
from pathlib import Path
import shutil

if __name__ == "__main__":
    base_dir = Path("E:/TESIS")
    
    print("=" * 80)
    print("LIMPIEZA AUTOMATICA DE ENTORNOS VIRTUALES")
    print("=" * 80)
    
    # Buscar entornos virtuales
    venvs = find_venvs(base_dir)
    venvs_sorted = print_venv_summary(venvs)
    
    if not venvs_sorted:
        print("\n[OK] No hay entornos virtuales para limpiar")
        exit(0)
    
    # Identificar candidatos para eliminar (no usados y > 0.1 GB)
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
    print("ELIMINANDO ENTORNOS VIRTUALES NO USADOS")
    print("=" * 80)
    total_to_free = 0
    for venv in candidates:
        print(f"\n  - {venv['name']}: {venv['size_gb']:.2f} GB")
        total_to_free += venv['size_gb']
    
    print(f"\nEspacio total a liberar: {total_to_free:.2f} GB")
    print("\n[ELIMINANDO]...")
    print("-" * 80)
    
    # Eliminar
    deleted_count = 0
    deleted_size = 0
    
    for venv in candidates:
        try:
            shutil.rmtree(venv['path'])
            print(f"  [OK] Eliminado: {venv['name']} ({venv['size_gb']:.2f} GB)")
            deleted_count += 1
            deleted_size += venv['size_gb']
        except Exception as e:
            print(f"  [ERROR] No se pudo eliminar {venv['name']}: {e}")
    
    print("\n" + "=" * 80)
    print(f"COMPLETADO: {deleted_count} entornos virtuales eliminados, {deleted_size:.2f} GB liberados")
    print("=" * 80)

