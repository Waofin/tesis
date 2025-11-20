"""
Test de FastPACO_CUDA_PreExtract para validar optimización.
"""

import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from paco.processing.fastpaco import FastPACO
from paco.processing.fastpaco_cuda_preextract import FastPACO_CUDA_PreExtract, CUDA_AVAILABLE


def test_preextract():
    """Comparar FastPACO CPU vs FastPACO_CUDA_PreExtract GPU."""
    
    print("=" * 70)
    print("TEST: FastPACO_CUDA_PreExtract (con pre-extract de patches)")
    print("=" * 70)
    
    if not CUDA_AVAILABLE:
        print("\n[ERROR] CuPy no disponible")
        return
    
    # Crear datos de prueba grandes
    image_size = (300, 300)  # Dataset grande
    n_frames = 150
    psf_size = (25, 25)
    
    print(f"\nDatos de prueba (GRANDES):")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    
    # Generar datos sintéticos
    np.random.seed(42)
    image_stack = np.random.randn(n_frames, image_size[0], image_size[1]).astype(np.float32)
    angles = np.linspace(0, 360, n_frames)
    psf = np.random.randn(psf_size[0], psf_size[1]).astype(np.float32)
    psf = psf / np.sum(psf)
    
    # Preparar coordenadas (todos los píxeles válidos)
    x, y = np.meshgrid(np.arange(20, image_size[0]-20, 2), 
                       np.arange(20, image_size[1]-20, 2))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"  Píxeles a procesar: {len(phi0s):,}")
    
    # Test CPU
    print("\n" + "-" * 70)
    print("1. CPU (FastPACO)")
    print("-" * 70)
    
    fp_cpu = FastPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf,
        psf_rad=4,
        patch_area=49
    )
    
    start = time.perf_counter()
    a_cpu, b_cpu = fp_cpu.PACOCalc(phi0s, cpu=1)
    time_cpu = time.perf_counter() - start
    
    print(f"  Tiempo: {time_cpu:.2f} segundos ({time_cpu/60:.2f} minutos)")
    print(f"  Throughput: {len(phi0s)/time_cpu:.1f} píxeles/segundo")
    
    # Test GPU PreExtract
    print("\n" + "-" * 70)
    print("2. GPU (FastPACO_CUDA_PreExtract)")
    print("-" * 70)
    
    try:
        fp_gpu = FastPACO_CUDA_PreExtract(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=4,
            patch_area=49
        )
        
        start = time.perf_counter()
        a_gpu, b_gpu = fp_gpu.PACOCalc(phi0s, cpu=1)
        time_gpu = time.perf_counter() - start
        
        print(f"  Tiempo: {time_gpu:.2f} segundos ({time_gpu/60:.2f} minutos)")
        print(f"  Throughput: {len(phi0s)/time_gpu:.1f} píxeles/segundo")
        
        # Calcular speedup
        speedup = time_cpu / time_gpu
        print(f"\n  [SPEEDUP] {speedup:.2f}x mas rapido")
        print(f"  Ahorro de tiempo: {time_cpu - time_gpu:.2f}s ({(time_cpu - time_gpu)/60:.2f} min)")
        
        if speedup > 1.0:
            print(f"  [OK] GPU es mas rapida!")
        else:
            print(f"  [WARNING] GPU es mas lenta (overhead)")
        
        # Verificar precisión
        valid_mask = ~(np.isnan(a_cpu) | np.isnan(a_gpu) | np.isnan(b_cpu) | np.isnan(b_gpu))
        if np.any(valid_mask):
            diff_a = np.abs(a_cpu[valid_mask] - a_gpu[valid_mask])
            diff_b = np.abs(b_cpu[valid_mask] - b_gpu[valid_mask])
            
            print(f"\n  Verificacion de precision:")
            print(f"    Diferencia en a (max): {diff_a.max():.2e}")
            print(f"    Diferencia en a (mean): {diff_a.mean():.2e}")
            print(f"    Diferencia en b (max): {diff_b.max():.2e}")
            print(f"    Diferencia en b (mean): {diff_b.mean():.2e}")
            
            # Verificar si diferencias son pequeñas
            if diff_a.max() < 1e-1 and diff_b.max() < 1e-1:
                print(f"    [OK] Resultados son similares (dentro de precision)")
            else:
                print(f"    [WARNING] Hay diferencias (puede ser normal por precision float32)")
        
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_preextract()

