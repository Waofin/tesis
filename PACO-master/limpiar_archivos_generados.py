#!/usr/bin/env python3
"""
Script para identificar y limpiar archivos generados por los benchmarks
"""

import os
from pathlib import Path
from datetime import datetime

def get_size_mb(path):
    """Obtiene el tamaño de un archivo o directorio en MB"""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except (PermissionError, OSError):
            pass
        return total / (1024 * 1024)
    return 0

def find_generated_files(base_dir):
    """Encuentra todos los archivos generados por benchmarks"""
    
    # Directorios de salida conocidos
    output_dirs = [
        base_dir / "PACO-master" / "output",
        base_dir / "figures",
        base_dir / "__pycache__",
    ]
    
    # Patrones de archivos generados
    file_patterns = {
        'JSON results': ['*.json'],
        'CSV summaries': ['*.csv'],
        'PNG plots': ['*.png'],
        'FITS data': ['*.fits'],
    }
    
    results = {
        'directories': [],
        'files_by_type': {k: [] for k in file_patterns.keys()},
        'other_files': []
    }
    
    # Buscar directorios de salida
    for output_dir in output_dirs:
        if output_dir.exists():
            size_mb = get_size_mb(output_dir)
            results['directories'].append({
                'path': str(output_dir),
                'size_mb': size_mb
            })
    
    # Buscar archivos por patrón en el directorio raíz
    for file_type, patterns in file_patterns.items():
        for pattern in patterns:
            for file_path in base_dir.glob(pattern):
                if file_path.is_file() and 'venv' not in str(file_path):
                    size_mb = get_size_mb(file_path)
                    results['files_by_type'][file_type].append({
                        'path': str(file_path),
                        'size_mb': size_mb
                    })
    
    # Buscar otros archivos grandes generados
    for file_path in base_dir.glob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in ['.log', '.txt', '.zip'] and 'venv' not in str(file_path):
                size_mb = get_size_mb(file_path)
                if size_mb > 1:  # Solo archivos > 1 MB
                    results['other_files'].append({
                        'path': str(file_path),
                        'size_mb': size_mb
                    })
    
    return results

def print_summary(results):
    """Imprime un resumen de los archivos encontrados"""
    print("=" * 80)
    print("ARCHIVOS GENERADOS POR BENCHMARKS")
    print("=" * 80)
    
    total_size = 0
    
    # Directorios
    print("\n[DIRECTORIOS] DIRECTORIOS DE SALIDA:")
    print("-" * 80)
    for dir_info in results['directories']:
        size = dir_info['size_mb']
        total_size += size
        print(f"  {dir_info['path']}")
        print(f"    Tamaño: {size:.2f} MB")
    
    # Archivos por tipo
    for file_type, files in results['files_by_type'].items():
        if files:
            type_total = sum(f['size_mb'] for f in files)
            total_size += type_total
            print(f"\n[ARCHIVOS] {file_type.upper()} ({len(files)} archivos):")
            print("-" * 80)
            # Mostrar los 10 más grandes
            sorted_files = sorted(files, key=lambda x: x['size_mb'], reverse=True)[:10]
            for file_info in sorted_files:
                print(f"  {file_info['path']}")
                print(f"    Tamaño: {file_info['size_mb']:.2f} MB")
            if len(files) > 10:
                print(f"  ... y {len(files) - 10} archivos más")
    
    # Otros archivos
    if results['other_files']:
        other_total = sum(f['size_mb'] for f in results['other_files'])
        total_size += other_total
        print(f"\n[ARCHIVOS] OTROS ARCHIVOS GRANDES ({len(results['other_files'])} archivos):")
        print("-" * 80)
        sorted_other = sorted(results['other_files'], key=lambda x: x['size_mb'], reverse=True)[:10]
        for file_info in sorted_other:
            print(f"  {file_info['path']}")
            print(f"    Tamaño: {file_info['size_mb']:.2f} MB")
    
    print("\n" + "=" * 80)
    print(f"TAMAÑO TOTAL: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    print("=" * 80)
    
    return total_size

def clean_files(results, dry_run=True):
    """Limpia los archivos encontrados"""
    if dry_run:
        print("\n[ADVERTENCIA] MODO DRY-RUN: No se eliminarán archivos")
        print("   Ejecuta con dry_run=False para eliminar realmente")
        return
    
    print("\n[ELIMINANDO] ELIMINANDO ARCHIVOS...")
    print("-" * 80)
    
    deleted_count = 0
    deleted_size = 0
    
    # Eliminar directorios
    for dir_info in results['directories']:
        dir_path = Path(dir_info['path'])
        if dir_path.exists():
            try:
                import shutil
                size = dir_info['size_mb']
                shutil.rmtree(dir_path)
                print(f"  [OK] Eliminado: {dir_info['path']} ({size:.2f} MB)")
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                print(f"  [ERROR] Error eliminando {dir_info['path']}: {e}")
    
    # Eliminar archivos
    for file_type, files in results['files_by_type'].items():
        for file_info in files:
            file_path = Path(file_info['path'])
            if file_path.exists() and file_path.is_file():
                try:
                    size = file_info['size_mb']
                    file_path.unlink()
                    print(f"  [OK] Eliminado: {file_info['path']} ({size:.2f} MB)")
                    deleted_count += 1
                    deleted_size += size
                except Exception as e:
                    print(f"  [ERROR] Error eliminando {file_info['path']}: {e}")
    
    print("\n" + "=" * 80)
    print(f"ELIMINADOS: {deleted_count} elementos, {deleted_size:.2f} MB ({deleted_size/1024:.2f} GB)")
    print("=" * 80)

if __name__ == "__main__":
    base_dir = Path("E:/TESIS")
    
    print("Buscando archivos generados por benchmarks...")
    results = find_generated_files(base_dir)
    
    total_size = print_summary(results)
    
    # Preguntar si quiere limpiar
    if total_size > 0:
        print("\n¿Deseas eliminar estos archivos?")
        print("  - Ejecuta: clean_files(results, dry_run=False)")
        print("  - O modifica el script para limpiar automáticamente")
        
        # Descomentar para limpiar automáticamente (¡CUIDADO!)
        # clean_files(results, dry_run=False)
    else:
        print("\n[OK] No se encontraron archivos generados para limpiar")

