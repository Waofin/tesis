#!/usr/bin/env python3
"""
Script para buscar archivos grandes que puedan estar ocupando espacio en disco
"""

import os
from pathlib import Path
import tempfile

def get_size_gb(path):
    """Obtiene el tamaño en GB"""
    if path.is_file():
        return path.stat().st_size / (1024 ** 3)
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except (PermissionError, OSError):
            pass
        return total / (1024 ** 3)
    return 0

def find_large_files_and_dirs(base_dir, min_size_gb=0.1):
    """Encuentra archivos y directorios grandes"""
    
    print("=" * 80)
    print("BUSCANDO ARCHIVOS Y DIRECTORIOS GRANDES")
    print("=" * 80)
    print(f"Buscando en: {base_dir}")
    print(f"Tamaño mínimo: {min_size_gb} GB\n")
    
    large_files = []
    large_dirs = {}
    
    # Buscar archivos grandes
    print("Buscando archivos grandes...")
    for root, dirs, files in os.walk(base_dir):
        # Saltar directorios de venv y otros que no nos interesan
        dirs[:] = [d for d in dirs if 'venv' not in d and '__pycache__' not in d]
        
        for file in files:
            file_path = Path(root) / file
            try:
                if file_path.is_file():
                    size_gb = get_size_gb(file_path)
                    if size_gb >= min_size_gb:
                        large_files.append({
                            'path': str(file_path),
                            'size_gb': size_gb
                        })
            except (PermissionError, OSError):
                continue
    
    # Buscar directorios grandes
    print("Buscando directorios grandes...")
    for root, dirs, files in os.walk(base_dir):
        # Saltar algunos directorios
        if 'venv' in root or '__pycache__' in root:
            continue
            
        dir_path = Path(root)
        try:
            total_size = 0
            for item in dir_path.rglob('*'):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except:
                        pass
            
            size_gb = total_size / (1024 ** 3)
            if size_gb >= min_size_gb:
                large_dirs[str(dir_path)] = size_gb
        except:
            pass
    
    return large_files, large_dirs

def check_temp_dirs():
    """Verifica directorios temporales"""
    print("\nVerificando directorios temporales...")
    
    temp_dirs = [
        Path(tempfile.gettempdir()),
        Path(os.environ.get('TMP', '')),
        Path(os.environ.get('TEMP', '')),
        Path('C:/Windows/Temp'),
    ]
    
    temp_files = []
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            print(f"  Revisando: {temp_dir}")
            try:
                for item in temp_dir.rglob('*'):
                    if item.is_file():
                        try:
                            size_gb = item.stat().st_size / (1024 ** 3)
                            if size_gb > 0.01:  # > 10 MB
                                temp_files.append({
                                    'path': str(item),
                                    'size_gb': size_gb
                                })
                        except:
                            pass
            except:
                pass
    
    return temp_files

if __name__ == "__main__":
    base_dir = Path("E:/TESIS")
    
    # Buscar archivos y directorios grandes
    large_files, large_dirs = find_large_files_and_dirs(base_dir, min_size_gb=0.1)
    
    # Ordenar por tamaño
    large_files.sort(key=lambda x: x['size_gb'], reverse=True)
    large_dirs_sorted = sorted(large_dirs.items(), key=lambda x: x[1], reverse=True)
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print("ARCHIVOS GRANDES (> 0.1 GB)")
    print("=" * 80)
    total_files_size = 0
    for file_info in large_files[:30]:  # Top 30
        print(f"\n{file_info['path']}")
        print(f"  Tamaño: {file_info['size_gb']:.2f} GB")
        total_files_size += file_info['size_gb']
    
    if len(large_files) > 30:
        print(f"\n... y {len(large_files) - 30} archivos más")
    
    print("\n" + "=" * 80)
    print("DIRECTORIOS GRANDES (> 0.1 GB)")
    print("=" * 80)
    total_dirs_size = 0
    for dir_path, size_gb in large_dirs_sorted[:20]:  # Top 20
        print(f"\n{dir_path}")
        print(f"  Tamaño: {size_gb:.2f} GB")
        total_dirs_size += size_gb
    
    if len(large_dirs_sorted) > 20:
        print(f"\n... y {len(large_dirs_sorted) - 20} directorios más")
    
    # Verificar directorios temporales
    temp_files = check_temp_dirs()
    
    if temp_files:
        print("\n" + "=" * 80)
        print("ARCHIVOS TEMPORALES GRANDES")
        print("=" * 80)
        temp_files.sort(key=lambda x: x['size_gb'], reverse=True)
        total_temp_size = 0
        for file_info in temp_files[:20]:
            print(f"\n{file_info['path']}")
            print(f"  Tamaño: {file_info['size_gb']:.2f} GB")
            total_temp_size += file_info['size_gb']
        
        if len(temp_files) > 20:
            print(f"\n... y {len(temp_files) - 20} archivos más")
    
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"Archivos grandes: {len(large_files)} archivos, {total_files_size:.2f} GB")
    print(f"Directorios grandes: {len(large_dirs_sorted)} directorios, {total_dirs_size:.2f} GB")
    if temp_files:
        print(f"Archivos temporales: {len(temp_files)} archivos, {total_temp_size:.2f} GB")
    print(f"TOTAL: {total_files_size + total_dirs_size + (total_temp_size if temp_files else 0):.2f} GB")
    print("=" * 80)

