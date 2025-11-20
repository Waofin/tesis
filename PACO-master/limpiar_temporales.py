#!/usr/bin/env python3
"""
Script para limpiar archivos temporales que están ocupando espacio
"""

import os
import shutil
from pathlib import Path
import tempfile

def clean_vscode_temp_files():
    """Limpia archivos temporales de VS Code"""
    temp_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'Temp'
    vscode_dirs = list(temp_dir.glob('vscode-*'))
    
    print(f"Encontrados {len(vscode_dirs)} directorios temporales de VS Code")
    
    total_size = 0
    deleted_count = 0
    
    for vscode_dir in vscode_dirs:
        try:
            # Calcular tamaño
            size = sum(f.stat().st_size for f in vscode_dir.rglob('*') if f.is_file())
            size_gb = size / (1024 ** 3)
            
            # Eliminar
            shutil.rmtree(vscode_dir)
            print(f"  [OK] Eliminado: {vscode_dir.name} ({size_gb:.2f} GB)")
            total_size += size_gb
            deleted_count += 1
        except Exception as e:
            print(f"  [ERROR] No se pudo eliminar {vscode_dir.name}: {e}")
    
    return deleted_count, total_size

def clean_python_temp_files(base_dir):
    """Limpia archivos temporales de Python (memmap, .dat, etc.)"""
    print("\nBuscando archivos temporales de Python...")
    
    temp_patterns = ['*.dat', '*memmap*', 'temp_*', '*_temp.*']
    found_files = []
    
    for pattern in temp_patterns:
        for file_path in base_dir.rglob(pattern):
            if file_path.is_file():
                try:
                    size_gb = file_path.stat().st_size / (1024 ** 3)
                    if size_gb > 0.001:  # > 1 MB
                        found_files.append((file_path, size_gb))
                except:
                    pass
    
    print(f"Encontrados {len(found_files)} archivos temporales")
    
    total_size = 0
    deleted_count = 0
    
    for file_path, size_gb in found_files:
        try:
            file_path.unlink()
            print(f"  [OK] Eliminado: {file_path} ({size_gb:.2f} GB)")
            total_size += size_gb
            deleted_count += 1
        except Exception as e:
            print(f"  [ERROR] No se pudo eliminar {file_path}: {e}")
    
    return deleted_count, total_size

def clean_windows_temp():
    """Limpia archivos temporales de Windows"""
    windows_temp = Path('C:/Windows/Temp')
    
    if not windows_temp.exists():
        return 0, 0
    
    print("\nLimpiando archivos temporales de Windows...")
    
    total_size = 0
    deleted_count = 0
    
    try:
        for item in windows_temp.iterdir():
            try:
                if item.is_file():
                    size_gb = item.stat().st_size / (1024 ** 3)
                    if size_gb > 0.01:  # > 10 MB
                        item.unlink()
                        total_size += size_gb
                        deleted_count += 1
                elif item.is_dir():
                    try:
                        shutil.rmtree(item)
                        deleted_count += 1
                    except:
                        pass
            except (PermissionError, OSError):
                pass
    except Exception as e:
        print(f"  [ERROR] Error limpiando Windows Temp: {e}")
    
    return deleted_count, total_size

if __name__ == "__main__":
    print("=" * 80)
    print("LIMPIEZA DE ARCHIVOS TEMPORALES")
    print("=" * 80)
    
    # Limpiar archivos temporales de VS Code
    print("\n[1] Limpiando archivos temporales de VS Code...")
    vscode_count, vscode_size = clean_vscode_temp_files()
    
    # Limpiar archivos temporales de Python
    base_dir = Path("E:/TESIS")
    print("\n[2] Limpiando archivos temporales de Python...")
    python_count, python_size = clean_python_temp_files(base_dir)
    
    # Limpiar Windows Temp (opcional, puede requerir permisos)
    print("\n[3] Limpiando archivos temporales de Windows...")
    windows_count, windows_size = clean_windows_temp()
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE LIMPIEZA")
    print("=" * 80)
    print(f"VS Code temporales: {vscode_count} elementos, {vscode_size:.2f} GB")
    print(f"Python temporales: {python_count} archivos, {python_size:.2f} GB")
    print(f"Windows temporales: {windows_count} elementos, {windows_size:.2f} GB")
    print(f"TOTAL LIBERADO: {vscode_size + python_size + windows_size:.2f} GB")
    print("=" * 80)

