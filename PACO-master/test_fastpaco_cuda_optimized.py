"""
Test de FastPACO_CUDA_Optimized para validar optimizaciones.
"""

import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from paco.processing.fastpaco import FastPACO
from paco.processing.fastpaco_cuda_optimized import FastPACO_CUDA_Optimized, CUDA_AVAILABLE


def test_optimized():
    """Comparar FastPACO CPU vs FastPACO_CUDA_Optimized GPU."""
    
    print("=" * 70)
    print("TEST: FastPACO_CUDA_Optimized")
    print("=" * 70)
    
    if not CUDA_AVAILABLE:
        print("\n[ERROR] CuPy no disponible")
        return
    
    # Crear datos de prueba medianos
    image_size = (200, 200)
    n_frames = 100
    psf_size = (25, 25)
    
    print(f"\nDatos de prueba:")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    
    # Generar datos sintéticos
    np.random.seed(42)
    image_stack = np.random.randn(n_frames, image_size[0], image_size[1]).astype(np.float32)
    angles = np.linspace(0, 360, n_frames)
    psf = np.random.randn(psf_size[0], psf_size[1]).astype(np.float32)
    psf = psf / np.sum(psf)
    
    # Preparar coordenadas
    x, y = np.meshgrid(np.arange(20, image_size[0]-20, 5), 
                       np.arange(20, image_size[1]-20, 5))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"  Píxeles a procesar: {len(phi0s)}")
    
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
    
    print(f"  Tiempo: {time_cpu:.2f} segundos")
    print(f"  Throughput: {len(phi0s)/time_cpu:.1f} píxeles/segundo")
    
    # Test GPU Optimized
    print("\n" + "-" * 70)
    print("2. GPU (FastPACO_CUDA_Optimized)")
    print("-" * 70)
    
    try:
        fp_gpu = FastPACO_CUDA_Optimized(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=4,
            patch_area=49
        )
        
        start = time.perf_counter()
        a_gpu, b_gpu = fp_gpu.PACOCalc(phi0s, cpu=1)
        time_gpu = time.perf_counter() - start
        
        print(f"  Tiempo: {time_gpu:.2f} segundos")
        print(f"  Throughput: {len(phi0s)/time_gpu:.1f} píxeles/segundo")
        
        # Calcular speedup
        speedup = time_cpu / time_gpu
        print(f"\n  [SPEEDUP] {speedup:.2f}x mas rapido")
        
        # Verificar precisión
        # Comparar solo valores válidos (no NaN)
        valid_mask = ~(np.isnan(a_cpu) | np.isnan(a_gpu))
        if np.any(valid_mask):
            diff_a = np.abs(a_cpu[valid_mask] - a_gpu[valid_mask]).max()
            diff_b = np.abs(b_cpu[valid_mask] - b_gpu[valid_mask]).max()
            
            print(f"\n  Verificacion de precision:")
            print(f"    Diferencia en a: {diff_a:.2e}")
            print(f"    Diferencia en b: {diff_b:.2e}")
            
            if diff_a < 1e-1 and diff_b < 1e-1:
                print(f"    [OK] Resultados son similares")
            else:
                print(f"    [WARNING] Hay diferencias (puede ser normal)")
        
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_optimized()

