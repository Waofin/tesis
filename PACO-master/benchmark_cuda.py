"""
Script de benchmark para comparar rendimiento CPU vs GPU en PACOCalc.

Uso:
    python benchmark_cuda.py

Requisitos:
    pip install cupy-cuda11x  # Para CUDA 11.x
    # o
    pip install cupy-cuda12x  # Para CUDA 12.x
"""

import numpy as np
import time
from pathlib import Path
import sys

# Agregar ruta de PACO
sys.path.insert(0, str(Path(__file__).parent))

from paco.processing.fullpaco import FullPACO
from paco.processing.fullpaco_cuda import FullPACO_CUDA, CUDA_AVAILABLE
from paco.util.util import pixelCalc


def benchmark_pixelCalc():
    """Compara pixelCalc CPU vs GPU."""
    print("=" * 70)
    print("BENCHMARK: pixelCalc CPU vs GPU")
    print("=" * 70)
    
    # Crear datos de prueba
    T = 252  # frames
    P = 49   # píxeles en patch
    
    print(f"\nDatos de prueba:")
    print(f"  Frames (T): {T}")
    print(f"  Píxeles en patch (P): {P}")
    print(f"  Shape del patch: ({T}, {P})")
    
    patch = np.random.randn(T, P).astype(np.float32)
    
    # Benchmark CPU
    print("\n" + "-" * 70)
    print("1. CPU (NumPy)")
    print("-" * 70)
    
    n_iterations = 100
    start = time.perf_counter()
    for _ in range(n_iterations):
        m_cpu, Cinv_cpu = pixelCalc(patch)
    time_cpu = time.perf_counter() - start
    
    print(f"  Tiempo total: {time_cpu:.4f} segundos")
    print(f"  Tiempo por iteración: {time_cpu/n_iterations*1000:.2f} ms")
    print(f"  Throughput: {n_iterations/time_cpu:.1f} iteraciones/segundo")
    
    # Benchmark GPU
    if not CUDA_AVAILABLE:
        print("\n" + "-" * 70)
        print("2. GPU (CuPy) - NO DISPONIBLE")
        print("-" * 70)
        print("  Instala CuPy con: pip install cupy-cuda11x")
        return
    
    print("\n" + "-" * 70)
    print("2. GPU (CuPy)")
    print("-" * 70)
    
    # Crear instancia para acceder a métodos GPU
    # Necesitamos datos mínimos para inicializar
    dummy_stack = np.random.randn(10, 50, 50).astype(np.float32)
    dummy_angles = np.linspace(0, 360, 10)
    dummy_psf = np.random.randn(10, 10).astype(np.float32)
    
    try:
        fp_cuda = FullPACO_CUDA(
            image_stack=dummy_stack,
            angles=dummy_angles,
            psf=dummy_psf,
            psf_rad=4,
            patch_area=49
        )
        
        # Calentar GPU (primera ejecución es más lenta)
        _ = fp_cuda.pixelCalc_CUDA(patch)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            m_gpu, Cinv_gpu = fp_cuda.pixelCalc_CUDA(patch)
        time_gpu = time.perf_counter() - start
        
        print(f"  Tiempo total: {time_gpu:.4f} segundos")
        print(f"  Tiempo por iteración: {time_gpu/n_iterations*1000:.2f} ms")
        print(f"  Throughput: {n_iterations/time_gpu:.1f} iteraciones/segundo")
        
        # Calcular speedup
        speedup = time_cpu / time_gpu
        print(f"\n  [SPEEDUP] {speedup:.2f}x mas rapido")
        
        # Verificar precisión
        diff_m = np.abs(m_cpu - m_gpu).max()
        diff_Cinv = np.abs(Cinv_cpu - Cinv_gpu).max()
        
        print(f"\n  Verificación de precisión:")
        print(f"    Diferencia en m: {diff_m:.2e}")
        print(f"    Diferencia en Cinv: {diff_Cinv:.2e}")
        
        if diff_m < 1e-4 and diff_Cinv < 1e-4:
            print(f"    [OK] Resultados son identicos (dentro de precision)")
        else:
            print(f"    [WARNING] Hay diferencias (puede ser normal por precision float32)")
            
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


def benchmark_PACOCalc_small():
    """Compara PACOCalc completo con datos pequeños."""
    print("\n" + "=" * 70)
    print("BENCHMARK: PACOCalc Completo (CPU vs GPU)")
    print("=" * 70)
    
    # Crear datos de prueba pequeños
    image_size = (50, 50)  # Imagen pequeña para prueba rápida
    n_frames = 50
    psf_size = (10, 10)
    
    print(f"\nDatos de prueba:")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    print(f"  Total píxeles: {image_size[0] * image_size[1]}")
    
    # Generar datos sintéticos
    image_stack = np.random.randn(n_frames, image_size[0], image_size[1]).astype(np.float32)
    angles = np.linspace(0, 360, n_frames)
    psf = np.random.randn(psf_size[0], psf_size[1]).astype(np.float32)
    
    # Preparar coordenadas de píxeles
    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"  Píxeles a procesar: {len(phi0s)}")
    
    # Benchmark CPU
    print("\n" + "-" * 70)
    print("1. CPU (FullPACO Serial)")
    print("-" * 70)
    
    fp_cpu = FullPACO(
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
    
    # Benchmark GPU
    if not CUDA_AVAILABLE:
        print("\n" + "-" * 70)
        print("2. GPU (FullPACO_CUDA) - NO DISPONIBLE")
        print("-" * 70)
        print("  Instala CuPy con: pip install cupy-cuda11x")
        return
    
    print("\n" + "-" * 70)
    print("2. GPU (FullPACO_CUDA)")
    print("-" * 70)
    
    try:
        fp_gpu = FullPACO_CUDA(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=4,
            patch_area=49
        )
        
        # Calentar GPU
        _ = fp_gpu.PACOCalc_CUDA(phi0s[:10], batch_size=10, use_gpu_pixelcalc=True)
        
        # Benchmark
        start = time.perf_counter()
        a_gpu, b_gpu = fp_gpu.PACOCalc_CUDA(phi0s, batch_size=500, use_gpu_pixelcalc=True)
        time_gpu = time.perf_counter() - start
        
        print(f"  Tiempo: {time_gpu:.2f} segundos ({time_gpu/60:.2f} minutos)")
        print(f"  Throughput: {len(phi0s)/time_gpu:.1f} píxeles/segundo")
        
        # Calcular speedup
        speedup = time_cpu / time_gpu
        print(f"\n  [SPEEDUP] {speedup:.2f}x mas rapido")
        
        # Verificar precisión
        diff_a = np.abs(a_cpu - a_gpu).max()
        diff_b = np.abs(b_cpu - b_gpu).max()
        
        print(f"\n  Verificacion de precision:")
        print(f"    Diferencia en a: {diff_a:.2e}")
        print(f"    Diferencia en b: {diff_b:.2e}")
        
        if diff_a < 1e-3 and diff_b < 1e-3:
            print(f"    [OK] Resultados son similares (dentro de precision)")
        else:
            print(f"    [WARNING] Hay diferencias (puede ser normal por precision float32)")
            
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Ejecutar todos los benchmarks."""
    print("\n" + "=" * 70)
    print("BENCHMARK CUDA PARA PACO")
    print("=" * 70)
    
    if CUDA_AVAILABLE:
        import cupy as cp
        try:
            device = cp.cuda.Device(0)
            device.use()
            props = cp.cuda.runtime.getDeviceProperties(0)
            print(f"\nGPU Detectada: {props['name'].decode()}")
            print(f"  Memoria: {props['totalGlobalMem'] / 1024**3:.1f} GB")
        except:
            print("\n[WARNING] GPU detectada pero no se pudo obtener informacion")
    else:
        print("\n[WARNING] CuPy no esta disponible")
        print("  Instala con: pip install cupy-cuda11x")
        print("  (o cupy-cuda12x para CUDA 12.x)")
    
    # Benchmark pixelCalc
    benchmark_pixelCalc()
    
    # Benchmark PACOCalc completo (solo si GPU disponible)
    if CUDA_AVAILABLE:
        print("\n")
        respuesta = input("¿Ejecutar benchmark completo de PACOCalc? (puede tomar varios minutos) [y/N]: ")
        if respuesta.lower() == 'y':
            benchmark_PACOCalc_small()
    
    print("\n" + "=" * 70)
    print("Benchmark completado")
    print("=" * 70)


if __name__ == "__main__":
    main()


