"""
Profiling de PACOCalc para identificar cuellos de botella.

Analiza qué partes de PACOCalc consumen más tiempo.
"""

import numpy as np
import time
import cProfile
import pstats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from paco.processing.fastpaco import FastPACO
from paco.util.util import getRotatedPixels, createCircularMask


def profile_pacocalc():
    """Perfilar PACOCalc para identificar cuellos de botella."""
    
    # Crear datos de prueba medianos
    image_size = (200, 200)
    n_frames = 100
    psf_size = (25, 25)
    
    print("=" * 70)
    print("PROFILING DE PACOCalc")
    print("=" * 70)
    print(f"\nDatos de prueba:")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    
    # Generar datos sintéticos
    image_stack = np.random.randn(n_frames, image_size[0], image_size[1]).astype(np.float32)
    angles = np.linspace(0, 360, n_frames)
    psf = np.random.randn(psf_size[0], psf_size[1]).astype(np.float32)
    psf = psf / np.sum(psf)  # Normalizar
    
    # Preparar coordenadas (solo algunos píxeles para profiling rápido)
    x, y = np.meshgrid(np.arange(20, image_size[0]-20, 10), 
                       np.arange(20, image_size[1]-20, 10))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"  Píxeles a procesar: {len(phi0s)}")
    
    # Crear instancia FastPACO
    fp = FastPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf,
        psf_rad=4,
        patch_area=49
    )
    
    # Profiling con cProfile
    print("\n" + "-" * 70)
    print("Ejecutando profiling...")
    print("-" * 70)
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.perf_counter()
    a, b = fp.PACOCalc(phi0s, cpu=1)
    time_total = time.perf_counter() - start
    
    profiler.disable()
    
    print(f"\nTiempo total: {time_total:.2f} segundos")
    print(f"Throughput: {len(phi0s)/time_total:.1f} píxeles/segundo")
    
    # Análisis detallado
    print("\n" + "=" * 70)
    print("ANÁLISIS DE CUellOS DE BOTELLA (Top 20 funciones)")
    print("=" * 70)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    # Análisis por tiempo total
    print("\n" + "=" * 70)
    print("ANÁLISIS POR TIEMPO TOTAL (Top 20 funciones)")
    print("=" * 70)
    
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    # Guardar resultados
    output_file = "pacocalc_profile.txt"
    with open(output_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        stats.print_stats()
    
    print(f"\n[OK] Perfil completo guardado en: {output_file}")
    
    # Análisis manual de funciones clave
    print("\n" + "=" * 70)
    print("ANÁLISIS MANUAL DE FUNCIONES CLAVE")
    print("=" * 70)
    
    # Medir tiempo de cada parte manualmente
    print("\nMediendo tiempos de componentes clave...")
    
    # 1. computeStatistics
    start = time.perf_counter()
    Cinv, m, h = fp.computeStatistics(phi0s)
    time_stats = time.perf_counter() - start
    print(f"  computeStatistics: {time_stats:.2f}s ({time_stats/time_total*100:.1f}%)")
    
    # 2. getRotatedPixels (promedio)
    dim = int(image_size[0] / 2)
    x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
    times_rot = []
    for i in range(min(100, len(phi0s))):
        start = time.perf_counter()
        _ = getRotatedPixels(x, y, phi0s[i], angles)
        times_rot.append(time.perf_counter() - start)
    time_rot_avg = np.mean(times_rot) * len(phi0s)
    print(f"  getRotatedPixels (estimado): {time_rot_avg:.2f}s ({time_rot_avg/time_total*100:.1f}%)")
    
    # 3. getPatch (promedio)
    mask = createCircularMask((fp.m_pwidth, fp.m_pwidth), radius=fp.m_psf_rad)
    times_patch = []
    for i in range(min(100, len(phi0s))):
        start = time.perf_counter()
        _ = fp.getPatch(phi0s[i], fp.m_pwidth, mask)
        times_patch.append(time.perf_counter() - start)
    time_patch_avg = np.mean(times_patch) * len(phi0s)
    print(f"  getPatch (estimado): {time_patch_avg:.2f}s ({time_patch_avg/time_total*100:.1f}%)")
    
    # 4. al y bl (promedio)
    # Necesitamos medir dentro del loop de PACOCalc
    print(f"  al/bl: ~{time_total - time_stats - time_rot_avg - time_patch_avg:.2f}s (resto)")


if __name__ == "__main__":
    profile_pacocalc()

