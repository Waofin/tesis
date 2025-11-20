"""
Script para corregir el error en sample_covariance de VIP PACO
===============================================================

Este script crea una versión corregida de la función sample_covariance
que puede ser usada como parche o referencia para la corrección.
"""

import numpy as np
from pathlib import Path
import shutil
from datetime import datetime


def create_fixed_sample_covariance():
    """Crea la versión corregida de sample_covariance"""
    
    fixed_code = '''
def sample_covariance(r: np.ndarray, m: np.ndarray,
                      T: int) -> np.ndarray:
    """
    Ŝ: Sample covariance matrix
    
    VERSIÓN CORREGIDA: Implementación vectorizada correcta
    
    La implementación original tenía un error crítico:
    - Original: np.cov(np.stack((p, m))) calcula covarianza ENTRE p y m (INCORRECTO)
    - Corregida: Calcula covarianza de (p-m) consigo mismo (CORRECTO)
    
    Parameters
    ----------
    r : numpy.ndarray
        Observed intensity at position θk and time tl
        Shape: (T, P) where T=number of frames, P=pixels in patch
    m : numpy.ndarray
        Mean of all background patches at position θk
        Shape: (P,)
    T : int
        Number of temporal frames

    Returns
    -------
    S : numpy.ndarray
        Sample covariance matrix
        Shape: (P, P)
    """
    
    # VERSIÓN CORREGIDA: Vectorizada y matemáticamente correcta
    # Centrar los patches restando la media
    r_centered = r - m[np.newaxis, :]  # Broadcasting: (T, P) - (P,) -> (T, P)
    
    # Calcular matriz de covarianza muestral
    # S = (1/T) * sum((p - m)^T * (p - m)) para cada frame p
    # Versión vectorizada: (1/T) * r_centered^T @ r_centered
    S = (1.0 / T) * np.dot(r_centered.T, r_centered)
    
    return S
'''
    
    return fixed_code


def create_patch_file():
    """Crea un archivo de parche que puede ser aplicado"""
    
    paco_file = Path("VIP-master/VIP-master/src/vip_hci/invprob/paco.py")
    
    if not paco_file.exists():
        print(f"✗ No se encontró el archivo: {paco_file}")
        return False
    
    # Leer el archivo original
    with open(paco_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Crear backup
    backup_file = paco_file.with_suffix('.py.backup')
    shutil.copy2(paco_file, backup_file)
    print(f"✓ Backup creado: {backup_file}")
    
    # Buscar la función sample_covariance
    start_line = None
    end_line = None
    
    for i, line in enumerate(lines):
        if 'def sample_covariance' in line:
            start_line = i
            # Buscar el final de la función (siguiente def o línea en blanco después de return)
            for j in range(i + 1, min(i + 30, len(lines))):
                if lines[j].strip().startswith('def ') or (lines[j].strip().startswith('return') and j > i + 5):
                    # Verificar si hay más líneas después del return
                    if 'return' in lines[j]:
                        # Buscar la siguiente línea no vacía o siguiente función
                        for k in range(j + 1, min(j + 5, len(lines))):
                            if lines[k].strip() == '' or lines[k].strip().startswith('def '):
                                end_line = k
                                break
                        if end_line is None:
                            end_line = j + 2
                        break
            break
    
    if start_line is None:
        print("✗ No se encontró la función sample_covariance")
        return False
    
    if end_line is None:
        end_line = start_line + 25  # Fallback
    
    print(f"  Función encontrada en líneas {start_line+1}-{end_line+1}")
    
    # Crear versión corregida
    fixed_code = create_fixed_sample_covariance()
    
    # Reemplazar la función
    new_lines = lines[:start_line]
    new_lines.extend(fixed_code.split('\n'))
    new_lines.append('')  # Línea en blanco
    new_lines.extend(lines[end_line:])
    
    # Guardar archivo corregido
    fixed_file = paco_file.with_suffix('.py.fixed')
    with open(fixed_file, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"✓ Archivo corregido creado: {fixed_file}")
    print(f"\nPara aplicar la corrección:")
    print(f"  1. Revisa el archivo: {fixed_file}")
    print(f"  2. Si está correcto, reemplaza el original:")
    print(f"     cp {fixed_file} {paco_file}")
    print(f"  3. O restaura el backup si algo sale mal:")
    print(f"     cp {backup_file} {paco_file}")
    
    return True


def show_comparison():
    """Muestra comparación entre implementación incorrecta y correcta"""
    
    print("\n" + "="*70)
    print("COMPARACIÓN: Implementación Incorrecta vs Correcta")
    print("="*70)
    
    print("\n1. IMPLEMENTACIÓN ORIGINAL (INCORRECTA):")
    print("-"*70)
    print("""
    S = (1.0/T)*np.sum([np.cov(np.stack((p, m)),
                               rowvar=False, bias=False) for p in r], axis=0)
    
    PROBLEMA:
    - np.cov(np.stack((p, m))) calcula la covarianza ENTRE p y m
    - Esto es matemáticamente incorrecto
    - Debería calcular la covarianza de (p-m) consigo mismo
    """)
    
    print("\n2. IMPLEMENTACIÓN CORREGIDA (VECTORIZADA):")
    print("-"*70)
    print("""
    r_centered = r - m[np.newaxis, :]  # Broadcasting
    S = (1.0 / T) * np.dot(r_centered.T, r_centered)
    
    VENTAJAS:
    - Matemáticamente correcta
    - Vectorizada (más rápida)
    - Más legible
    - Speedup estimado: ~4.9x
    """)
    
    print("\n3. PRUEBA NUMÉRICA:")
    print("-"*70)
    
    # Generar datos de prueba
    T = 10
    P = 5
    r = np.random.randn(T, P)
    m = np.mean(r, axis=0)
    
    # Implementación incorrecta
    try:
        S_wrong = (1.0/T)*np.sum([np.cov(np.stack((p, m)), rowvar=False, bias=False) for p in r], axis=0)
        print(f"   Implementación incorrecta - Shape: {S_wrong.shape}")
        print(f"   Valores ejemplo: {S_wrong[0, :3]}")
    except Exception as e:
        print(f"   Error: {e}")
        S_wrong = None
    
    # Implementación correcta
    r_centered = r - m[np.newaxis, :]
    S_correct = (1.0 / T) * np.dot(r_centered.T, r_centered)
    print(f"\n   Implementación correcta - Shape: {S_correct.shape}")
    print(f"   Valores ejemplo: {S_correct[0, :3]}")
    
    # Verificar propiedades
    print(f"\n   Propiedades de matriz de covarianza:")
    print(f"   - Simétrica: {np.allclose(S_correct, S_correct.T)}")
    print(f"   - Valores propios positivos: {np.all(np.linalg.eigvals(S_correct) > -1e-10)}")
    
    if S_wrong is not None:
        print(f"\n   Comparación:")
        print(f"   - Diferencia máxima: {np.max(np.abs(S_correct - S_wrong)):.6f}")
        if not np.allclose(S_correct, S_wrong, atol=1e-5):
            print(f"   ⚠ Las implementaciones producen resultados diferentes!")


def main():
    """Función principal"""
    print("="*70)
    print("CORRECCIÓN DE ERROR EN sample_covariance")
    print("="*70)
    
    # Mostrar comparación
    show_comparison()
    
    # Crear archivo de parche
    print("\n" + "="*70)
    print("CREANDO ARCHIVO DE PARCHE")
    print("="*70)
    
    success = create_patch_file()
    
    if success:
        print("\n" + "="*70)
        print("CORRECCIÓN PREPARADA")
        print("="*70)
        print("\nRevisa los archivos generados antes de aplicar la corrección.")
    else:
        print("\n" + "="*70)
        print("ERROR AL CREAR PARCHE")
        print("="*70)
        print("\nNo se pudo crear el archivo de parche. Verifica la ruta del archivo.")


if __name__ == "__main__":
    main()

