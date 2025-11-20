#!/usr/bin/env python3
"""
Script para limpiar archivos temporales y grandes en todos los discos
"""

import os
import shutil
from pathlib import Path
import tempfile

def get_size_gb(path):
    """Obtiene el tamaño en GB"""
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

def clean_windows_temp(drive='C'):
    """Limpia archivos temporales de Windows"""
    temp_dirs = [
        Path(f"{drive}:\\Windows\\Temp"),
        Path(f"{drive}:\\Temp"),
        Path(f"{drive}:\\tmp"),
    ]
    
    if drive == 'C':
        user_temp = Path(os.environ.get('LOCALAPPDATA', '')) / 'Temp'
        if user_temp.exists():
            temp_dirs.append(user_temp)
    
    print(f"\n[LIMPIEZA] Archivos temporales en disco {drive}:\\")
    print("-" * 80)
    
    total_freed = 0
    deleted_count = 0
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            try:
                size_before = get_size_gb(temp_dir)
                
                # Eliminar archivos antiguos (> 7 días)
                import time
                current_time = time.time()
                seven_days = 7 * 24 * 3600
                
                for item in temp_dir.iterdir():
                    try:
                        if item.is_file():
                            # Verificar si es antiguo
                            mtime = item.stat().st_mtime
                            if current_time - mtime > seven_days:
                                size = get_size_gb(item)
                                item.unlink()
                                total_freed += size
                                deleted_count += 1
                        elif item.is_dir():
                            # Eliminar directorios vacíos o antiguos
                            try:
                                if not any(item.iterdir()):
                                    shutil.rmtree(item)
                                    deleted_count += 1
                            except:
                                pass
                    except (PermissionError, OSError):
                        pass
                
                size_after = get_size_gb(temp_dir)
                freed = size_before - size_after
                if freed > 0.01:
                    print(f"  [OK] {temp_dir.name}: {freed:.2f} GB liberados")
                    total_freed += freed
            except Exception as e:
                print(f"  [ERROR] {temp_dir}: {e}")
    
    return total_freed, deleted_count

def clean_python_cache(drive='C'):
    """Limpia archivos __pycache__ y .pyc"""
    print(f"\n[LIMPIEZA] Archivos Python cache en disco {drive}:\\")
    print("-" * 80)
    
    total_freed = 0
    deleted_count = 0
    
    drive_path = Path(f"{drive}:\\")
    
    # Buscar __pycache__
    try:
        for pycache in drive_path.rglob('__pycache__'):
            try:
                size = get_size_gb(pycache)
                if size > 0.001:  # > 1 MB
                    shutil.rmtree(pycache)
                    total_freed += size
                    deleted_count += 1
                    print(f"  [OK] Eliminado: {pycache} ({size:.2f} GB)")
            except:
                pass
    except:
        pass
    
    # Buscar .pyc individuales
    try:
        for pyc in drive_path.rglob('*.pyc'):
            try:
                size = get_size_gb(pyc)
                pyc.unlink()
                total_freed += size
                deleted_count += 1
            except:
                pass
    except:
        pass
    
    return total_freed, deleted_count

def clean_browser_cache(drive='C'):
    """Limpia caché de navegadores"""
    if drive != 'C':
        return 0, 0
    
    print(f"\n[LIMPIEZA] Caché de navegadores")
    print("-" * 80)
    
    appdata = Path(os.environ.get('LOCALAPPDATA', ''))
    cache_dirs = [
        appdata / 'Google' / 'Chrome' / 'User Data' / 'Default' / 'Cache',
        appdata / 'Microsoft' / 'Edge' / 'User Data' / 'Default' / 'Cache',
        appdata / 'Mozilla' / 'Firefox' / 'Profiles',
    ]
    
    total_freed = 0
    deleted_count = 0
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                size = get_size_gb(cache_dir)
                if size > 0.1:  # > 100 MB
                    shutil.rmtree(cache_dir)
                    total_freed += size
                    deleted_count += 1
                    print(f"  [OK] Eliminado: {cache_dir.name} ({size:.2f} GB)")
            except Exception as e:
                print(f"  [ERROR] {cache_dir.name}: {e}")
    
    return total_freed, deleted_count

def clean_windows_logs(drive='C'):
    """Limpia logs grandes de Windows"""
    if drive != 'C':
        return 0, 0
    
    print(f"\n[LIMPIEZA] Logs de Windows")
    print("-" * 80)
    
    log_dirs = [
        Path('C:\\Windows\\Logs'),
        Path('C:\\Windows\\Temp'),
    ]
    
    total_freed = 0
    deleted_count = 0
    
    for log_dir in log_dirs:
        if log_dir.exists():
            try:
                for log_file in log_dir.rglob('*.log'):
                    try:
                        size = get_size_gb(log_file)
                        if size > 0.1:  # > 100 MB
                            log_file.unlink()
                            total_freed += size
                            deleted_count += 1
                            print(f"  [OK] Eliminado: {log_file.name} ({size:.2f} GB)")
                    except:
                        pass
            except:
                pass
    
    return total_freed, deleted_count

if __name__ == "__main__":
    print("=" * 80)
    print("LIMPIEZA DE ARCHIVOS TEMPORALES EN TODOS LOS DISCOS")
    print("=" * 80)
    
    drives_to_clean = ['C', 'D']
    
    total_freed_all = 0
    total_deleted_all = 0
    
    for drive in drives_to_clean:
        print(f"\n{'='*80}")
        print(f"PROCESANDO DISCO {drive}:\\")
        print(f"{'='*80}")
        
        # Limpiar temporales
        freed, count = clean_windows_temp(drive)
        total_freed_all += freed
        total_deleted_all += count
        
        # Limpiar Python cache (solo en C normalmente)
        if drive == 'C':
            freed, count = clean_python_cache(drive)
            total_freed_all += freed
            total_deleted_all += count
            
            # Limpiar caché de navegadores
            freed, count = clean_browser_cache(drive)
            total_freed_all += freed
            total_deleted_all += count
            
            # Limpiar logs de Windows
            freed, count = clean_windows_logs(drive)
            total_freed_all += freed
            total_deleted_all += count
    
    print("\n" + "=" * 80)
    print("RESUMEN DE LIMPIEZA")
    print("=" * 80)
    print(f"Archivos eliminados: {total_deleted_all}")
    print(f"Espacio liberado: {total_freed_all:.2f} GB")
    print("=" * 80)
    
    # Mostrar espacio actual
    print("\nESPACIO ACTUAL EN DISCOS:")
    print("-" * 80)
    import shutil
    for drive in ['C', 'D', 'E']:
        try:
            drive_path = Path(f"{drive}:\\")
            if drive_path.exists():
                stat = shutil.disk_usage(drive_path)
                total_gb = stat.total / (1024 ** 3)
                used_gb = stat.used / (1024 ** 3)
                free_gb = stat.free / (1024 ** 3)
                percent = (used_gb / total_gb * 100) if total_gb > 0 else 0
                print(f"\nDisco {drive}:\\")
                print(f"  Total: {total_gb:.2f} GB")
                print(f"  Usado: {used_gb:.2f} GB ({percent:.1f}%)")
                print(f"  Libre: {free_gb:.2f} GB")
        except:
            pass

