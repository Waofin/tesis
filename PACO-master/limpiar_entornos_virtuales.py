#!/usr/bin/env python3
"""
Script para identificar y limpiar entornos virtuales duplicados
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def get_size_gb(path):
    """Obtiene el tamaño de un directorio en GB"""
    if not path.exists():
        return 0
    
    total = 0
    try:
        for item in path.rglob('*'):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except (PermissionError, OSError):
                    pass
    except (PermissionError, OSError):
        pass
    
    return total / (1024 ** 3)

def find_venvs(base_dir):
    """Encuentra todos los entornos virtuales"""
    venv_patterns = [
        'venv*',
        '.venv*',
        'env*',
        '.env*',
        '*venv*',
        '*_env*',
    ]
    
    venvs = []
    
    # Buscar en el directorio base
    for pattern in venv_patterns:
        for venv_path in base_dir.glob(pattern):
            if venv_path.is_dir():
                # Verificar que sea un entorno virtual (tiene Scripts o bin)
                if (venv_path / 'Scripts').exists() or (venv_path / 'bin').exists() or (venv_path / 'Lib').exists():
                    size_gb = get_size_gb(venv_path)
                    venvs.append({
                        'path': venv_path,
                        'name': venv_path.name,
                        'size_gb': size_gb,
                        'full_path': str(venv_path)
                    })
    
    # Buscar en subdirectorios también
    for root, dirs, files in os.walk(base_dir):
        # Saltar algunos directorios para acelerar
        dirs[:] = [d for d in dirs if not d.startswith('.') and 'venv' not in d.lower() or d in ['venv', '.venv']]
        
        for dir_name in dirs:
            if any(pattern.replace('*', '') in dir_name.lower() for pattern in venv_patterns):
                venv_path = Path(root) / dir_name
                if (venv_path / 'Scripts').exists() or (venv_path / 'bin').exists() or (venv_path / 'Lib').exists():
                    size_gb = get_size_gb(venv_path)
                    venvs.append({
                        'path': venv_path,
                        'name': venv_path.name,
                        'size_gb': size_gb,
                        'full_path': str(venv_path)
                    })
    
    # Eliminar duplicados
    seen = set()
    unique_venvs = []
    for venv in venvs:
        if venv['full_path'] not in seen:
            seen.add(venv['full_path'])
            unique_venvs.append(venv)
    
    return unique_venvs

def check_venv_usage(venv_path):
    """Verifica si un entorno virtual está en uso"""
    # Verificar si hay un archivo de activación reciente
    activation_files = [
        venv_path / 'pyvenv.cfg',
        venv_path / 'Scripts' / 'activate.bat',
        venv_path / 'bin' / 'activate',
    ]
    
    for af in activation_files:
        if af.exists():
            try:
                # Verificar fecha de modificación
                mtime = af.stat().st_mtime
                days_since_mod = (datetime.now().timestamp() - mtime) / (24 * 3600)
                return days_since_mod < 30  # Usado en los últimos 30 días
            except:
                pass
    
    return False

def print_venv_summary(venvs):
    """Imprime un resumen de los entornos virtuales encontrados"""
    print("=" * 80)
    print("ENTORNOS VIRTUALES ENCONTRADOS")
    print("=" * 80)
    
    if not venvs:
        print("\n[OK] No se encontraron entornos virtuales")
        return
    
    # Ordenar por tamaño
    venvs_sorted = sorted(venvs, key=lambda x: x['size_gb'], reverse=True)
    
    total_size = 0
    for i, venv in enumerate(venvs_sorted, 1):
        print(f"\n[{i}] {venv['name']}")
        print(f"    Ruta: {venv['full_path']}")
        print(f"    Tamaño: {venv['size_gb']:.2f} GB")
        
        # Verificar uso
        in_use = check_venv_usage(venv['path'])
        status = "[EN USO]" if in_use else "[NO USADO RECIENTEMENTE]"
        print(f"    Estado: {status}")
        
        total_size += venv['size_gb']
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {len(venvs)} entornos virtuales, {total_size:.2f} GB")
    print("=" * 80)
    
    return venvs_sorted

def clean_venvs(venvs, dry_run=True, min_size_gb=0.1):
    """Limpia entornos virtuales no usados"""
    if dry_run:
        print("\n[ADVERTENCIA] MODO DRY-RUN: No se eliminarán entornos virtuales")
        print("   Ejecuta con dry_run=False para eliminar realmente")
        return
    
    print("\n[ELIMINANDO] Eliminando entornos virtuales no usados...")
    print("-" * 80)
    
    deleted_count = 0
    deleted_size = 0
    
    for venv in venvs:
        # Solo eliminar si no está en uso y es grande
        if venv['size_gb'] >= min_size_gb:
            in_use = check_venv_usage(venv['path'])
            
            if not in_use:
                try:
                    shutil.rmtree(venv['path'])
                    print(f"  [OK] Eliminado: {venv['name']} ({venv['size_gb']:.2f} GB)")
                    deleted_count += 1
                    deleted_size += venv['size_gb']
                except Exception as e:
                    print(f"  [ERROR] No se pudo eliminar {venv['name']}: {e}")
            else:
                print(f"  [SKIP] {venv['name']} está en uso, no se eliminará")
    
    print("\n" + "=" * 80)
    print(f"ELIMINADOS: {deleted_count} entornos virtuales, {deleted_size:.2f} GB")
    print("=" * 80)

if __name__ == "__main__":
    base_dir = Path("E:/TESIS")
    
    print("Buscando entornos virtuales...")
    venvs = find_venvs(base_dir)
    
    venvs_sorted = print_venv_summary(venvs)
    
    if venvs_sorted:
        print("\n¿Deseas eliminar entornos virtuales no usados?")
        print("  - Ejecuta: clean_venvs(venvs_sorted, dry_run=False)")
        print("  - O modifica el script para limpiar automáticamente")
        
        # Mostrar recomendaciones
        print("\n" + "=" * 80)
        print("RECOMENDACIONES")
        print("=" * 80)
        
        large_unused = [v for v in venvs_sorted if v['size_gb'] > 0.5 and not check_venv_usage(v['path'])]
        if large_unused:
            print("\nEntornos virtuales grandes no usados (candidatos para eliminar):")
            for venv in large_unused:
                print(f"  - {venv['name']}: {venv['size_gb']:.2f} GB")
        
        # Descomentar para limpiar automáticamente (¡CUIDADO!)
        # clean_venvs(venvs_sorted, dry_run=False, min_size_gb=0.5)

