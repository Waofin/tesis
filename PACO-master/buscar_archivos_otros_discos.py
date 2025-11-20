#!/usr/bin/env python3
"""
Script para buscar archivos generados por benchmarks en otros discos
"""

import os
from pathlib import Path
import shutil

def get_size_gb(path):
    """Obtiene el tamaño de un archivo o directorio en GB"""
    if path.is_file():
        try:
            return path.stat().st_size / (1024 ** 3)
        except:
            return 0
    elif path.is_dir():
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except:
                        pass
        except:
            pass
        return total / (1024 ** 3)
    return 0

def find_files_in_drive(drive_letter, patterns, min_size_gb=0.1):
    """Busca archivos en un disco específico"""
    drive_path = Path(f"{drive_letter}:\\")
    
    if not drive_path.exists():
        return []
    
    print(f"\n{'='*80}")
    print(f"BUSCANDO EN DISCO {drive_letter}:\\")
    print(f"{'='*80}")
    
    found_items = []
    
    # Buscar directorios comunes
    common_dirs = [
        drive_path / "Temp",
        drive_path / "tmp",
        drive_path / "AppData",
        drive_path / "Users",
        drive_path / "ProgramData",
        drive_path / "Windows" / "Temp",
    ]
    
    for dir_path in common_dirs:
        if dir_path.exists():
            try:
                size_gb = get_size_gb(dir_path)
                if size_gb >= min_size_gb:
                    found_items.append({
                        'path': str(dir_path),
                        'type': 'directory',
                        'size_gb': size_gb
                    })
            except:
                pass
    
    # Buscar archivos por patrón
    file_patterns = {
        'JSON results': ['*.json'],
        'CSV summaries': ['*.csv'],
        'PNG plots': ['*.png'],
        'FITS data': ['*.fits'],
        'Temporary files': ['*.dat', '*temp*', '*memmap*', '*.log'],
    }
    
    for file_type, patterns_list in file_patterns.items():
        for pattern in patterns_list:
            try:
                for file_path in drive_path.rglob(pattern):
                    if file_path.is_file():
                        try:
                            size_gb = get_size_gb(file_path)
                            if size_gb >= min_size_gb:
                                found_items.append({
                                    'path': str(file_path),
                                    'type': 'file',
                                    'size_gb': size_gb,
                                    'file_type': file_type
                                })
                        except:
                            pass
            except:
                pass
    
    # Buscar entornos virtuales
    venv_patterns = ['venv*', '.venv*', 'env*', '*venv*']
    for pattern in venv_patterns:
        try:
            for venv_path in drive_path.rglob(pattern):
                if venv_path.is_dir():
                    if (venv_path / 'Scripts').exists() or (venv_path / 'bin').exists() or (venv_path / 'Lib').exists():
                        size_gb = get_size_gb(venv_path)
                        if size_gb >= min_size_gb:
                            found_items.append({
                                'path': str(venv_path),
                                'type': 'venv',
                                'size_gb': size_gb
                            })
        except:
            pass
    
    return found_items

def check_all_drives():
    """Verifica todos los discos disponibles"""
    import string
    
    drives = []
    for letter in string.ascii_uppercase:
        drive_path = Path(f"{letter}:\\")
        if drive_path.exists():
            try:
                stat = shutil.disk_usage(drive_path)
                total_gb = stat.total / (1024 ** 3)
                used_gb = stat.used / (1024 ** 3)
                free_gb = stat.free / (1024 ** 3)
                drives.append({
                    'letter': letter,
                    'total_gb': total_gb,
                    'used_gb': used_gb,
                    'free_gb': free_gb,
                    'percent_used': (used_gb / total_gb * 100) if total_gb > 0 else 0
                })
            except:
                pass
    
    return drives

if __name__ == "__main__":
    print("=" * 80)
    print("BÚSQUEDA DE ARCHIVOS EN TODOS LOS DISCOS")
    print("=" * 80)
    
    # Verificar espacio en todos los discos
    print("\nESPACIO EN DISCOS:")
    print("-" * 80)
    drives = check_all_drives()
    for drive in drives:
        print(f"\nDisco {drive['letter']}:\\")
        print(f"  Total: {drive['total_gb']:.2f} GB")
        print(f"  Usado: {drive['used_gb']:.2f} GB ({drive['percent_used']:.1f}%)")
        print(f"  Libre: {drive['free_gb']:.2f} GB")
    
    # Buscar archivos en discos no principales (C:, D:, etc.)
    drives_to_check = ['D', 'C']
    
    all_found = {}
    
    for drive_letter in drives_to_check:
        found = find_files_in_drive(drive_letter, [], min_size_gb=0.1)
        if found:
            all_found[drive_letter] = found
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print("ARCHIVOS ENCONTRADOS EN OTROS DISCOS")
    print("=" * 80)
    
    total_size = 0
    
    for drive_letter, items in all_found.items():
        if items:
            print(f"\n[DISCO {drive_letter}:\\]")
            print("-" * 80)
            
            # Agrupar por tipo
            by_type = {}
            for item in items:
                item_type = item.get('file_type', item['type'])
                if item_type not in by_type:
                    by_type[item_type] = []
                by_type[item_type].append(item)
            
            for item_type, type_items in by_type.items():
                type_total = sum(i['size_gb'] for i in type_items)
                total_size += type_total
                print(f"\n  {item_type.upper()} ({len(type_items)} elementos, {type_total:.2f} GB):")
                
                # Mostrar los más grandes
                sorted_items = sorted(type_items, key=lambda x: x['size_gb'], reverse=True)[:10]
                for item in sorted_items:
                    print(f"    {item['path']}")
                    print(f"      Tamaño: {item['size_gb']:.2f} GB")
    
    if total_size > 0:
        print("\n" + "=" * 80)
        print(f"TOTAL ENCONTRADO EN OTROS DISCOS: {total_size:.2f} GB")
        print("=" * 80)
    else:
        print("\n[OK] No se encontraron archivos grandes en otros discos")

