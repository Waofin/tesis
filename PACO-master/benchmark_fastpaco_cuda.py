"""
Script de benchmark para comparar rendimiento CPU vs GPU en FastPACO.

Compara:
- FastPACO Serial (CPU)
- FastPACO Parallel (CPU, multiprocessing)
- FastPACO_CUDA (GPU)
"""

import numpy as np
import time
from pathlib import Path
import sys

# Agregar ruta de PACO
sys.path.insert(0, str(Path(__file__).parent))

from paco.processing.fastpaco import FastPACO
from paco.processing.fastpaco_cuda import FastPACO_CUDA, CUDA_AVAILABLE


def benchmark_computeStatistics():
    """Compara computeStatistics CPU vs GPU."""
    print("=" * 70)
    print("BENCHMARK: computeStatistics CPU vs GPU")
    print("=" * 70)
    
    # Crear datos de prueba
    image_size = (100, 100)  # Imagen pequeña para prueba rápida
    n_frames = 50
    psf_size = (10, 10)
    
    print(f"\nDatos de prueba:")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    
    # Generar datos sintéticos
    image_stack = np.random.randn(n_frames, image_size[0], image_size[1]).astype(np.float32)
    angles = np.linspace(0, 360, n_frames)
    psf = np.random.randn(psf_size[0], psf_size[1]).astype(np.float32)
    
    # Preparar coordenadas de píxeles (solo algunos para prueba)
    x, y = np.meshgrid(np.arange(10, image_size[0]-10), np.arange(10, image_size[1]-10))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"  Píxeles a procesar: {len(phi0s)}")
    
    # Benchmark CPU Serial
    print("\n" + "-" * 70)
    print("1. CPU Serial (FastPACO)")
    print("-" * 70)
    
    fp_cpu = FastPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf,
        psf_rad=4,
        patch_area=49
    )
    
    start = time.perf_counter()
    Cinv_cpu, m_cpu, h_cpu = fp_cpu.computeStatistics(phi0s)
    time_cpu = time.perf_counter() - start
    
    print(f"  Tiempo: {time_cpu:.2f} segundos ({time_cpu/60:.2f} minutos)")
    print(f"  Throughput: {len(phi0s)/time_cpu:.1f} píxeles/segundo")
    
    # Benchmark CPU Parallel (opcional, puede fallar)
    print("\n" + "-" * 70)
    print("2. CPU Parallel (FastPACO, 4 cores)")
    print("-" * 70)
    
    try:
        start = time.perf_counter()
        Cinv_cpu_par, m_cpu_par, h_cpu_par = fp_cpu.computeStatisticsParallel(phi0s, cpu=4)
        time_cpu_par = time.perf_counter() - start
        
        print(f"  Tiempo: {time_cpu_par:.2f} segundos ({time_cpu_par/60:.2f} minutos)")
        print(f"  Throughput: {len(phi0s)/time_cpu_par:.1f} píxeles/segundo")
        
        speedup_par = time_cpu / time_cpu_par
        print(f"  [SPEEDUP vs Serial] {speedup_par:.2f}x mas rapido")
        time_cpu_par_available = True
    except Exception as e:
        print(f"  [WARNING] CPU Parallel fallo: {e}")
        print(f"  Continuando con comparacion Serial vs GPU...")
        time_cpu_par_available = False
    
    # Benchmark GPU
    if not CUDA_AVAILABLE:
        print("\n" + "-" * 70)
        print("3. GPU (FastPACO_CUDA) - NO DISPONIBLE")
        print("-" * 70)
        print("  Instala CuPy con: pip install cupy-cuda12x")
        return
    
    print("\n" + "-" * 70)
    print("3. GPU (FastPACO_CUDA)")
    print("-" * 70)
    
    try:
        fp_gpu = FastPACO_CUDA(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=4,
            patch_area=49
        )
        
        # Calentar GPU
        _ = fp_gpu.computeStatistics_CUDA(phi0s[:10], batch_size=10)
        
        # Benchmark
        start = time.perf_counter()
        Cinv_gpu, m_gpu, h_gpu = fp_gpu.computeStatistics_CUDA(phi0s, batch_size=500)
        time_gpu = time.perf_counter() - start
        
        print(f"  Tiempo: {time_gpu:.2f} segundos ({time_gpu/60:.2f} minutos)")
        print(f"  Throughput: {len(phi0s)/time_gpu:.1f} píxeles/segundo")
        
        # Calcular speedups
        speedup_vs_serial = time_cpu / time_gpu
        print(f"\n  [SPEEDUP vs Serial] {speedup_vs_serial:.2f}x mas rapido")
        
        if time_cpu_par_available:
            speedup_vs_parallel = time_cpu_par / time_gpu
            print(f"  [SPEEDUP vs Parallel] {speedup_vs_parallel:.2f}x mas rapido")
        
        # Verificar precisión
        diff_m = np.abs(m_cpu - m_gpu).max()
        diff_Cinv = np.abs(Cinv_cpu - Cinv_gpu).max()
        
        print(f"\n  Verificacion de precision:")
        print(f"    Diferencia en m: {diff_m:.2e}")
        print(f"    Diferencia en Cinv: {diff_Cinv:.2e}")
        
        if diff_m < 1e-2 and diff_Cinv < 1e-1:
            print(f"    [OK] Resultados son similares (dentro de precision)")
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
    image_size = (50, 50)
    n_frames = 50
    psf_size = (10, 10)
    
    print(f"\nDatos de prueba:")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    
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
    print("1. CPU (FastPACO Serial)")
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
    
    # Benchmark GPU
    if not CUDA_AVAILABLE:
        print("\n" + "-" * 70)
        print("2. GPU (FastPACO_CUDA) - NO DISPONIBLE")
        print("-" * 70)
        return
    
    print("\n" + "-" * 70)
    print("2. GPU (FastPACO_CUDA)")
    print("-" * 70)
    
    try:
        fp_gpu = FastPACO_CUDA(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=4,
            patch_area=49
        )
        
        # Benchmark
        start = time.perf_counter()
        a_gpu, b_gpu = fp_gpu.PACOCalc(phi0s, cpu=1)
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
        
        if diff_a < 1e-2 and diff_b < 1e-2:
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
    print("BENCHMARK CUDA PARA FastPACO")
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
        print("  Instala con: pip install cupy-cuda12x")
    
    # Benchmark computeStatistics
    benchmark_computeStatistics()
    
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

