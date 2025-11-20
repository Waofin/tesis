"""
Benchmark FastPACO_CUDA_Optimized con dataset sintético grande.
"""

import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from paco.processing.fastpaco import FastPACO
from paco.processing.fastpaco_cuda_optimized import FastPACO_CUDA_Optimized, CUDA_AVAILABLE
from astropy.io import fits


def load_synthetic_dataset(data_dir: Path, name: str):
    """Cargar dataset sintético."""
    cube_file = data_dir / f"{name}_cube.fits"
    pa_file = data_dir / f"{name}_pa.fits"
    psf_file = data_dir / f"{name}_psf.fits"
    
    if not all(f.exists() for f in [cube_file, pa_file, psf_file]):
        return None
    
    with fits.open(cube_file) as hdul:
        cube = hdul[0].data.astype(np.float32)
    
    with fits.open(pa_file) as hdul:
        angles = hdul[0].data.astype(np.float32)
        if angles.ndim > 1:
            angles = angles.flatten()
    
    with fits.open(psf_file) as hdul:
        psf = hdul[0].data.astype(np.float32)
        psf = psf / np.sum(psf)  # Normalizar
    
    return {
        'cube': cube,
        'angles': angles,
        'psf': psf,
        'n_frames': cube.shape[0],
        'height': cube.shape[1],
        'width': cube.shape[2]
    }


def benchmark_synthetic_dataset(data_dir: Path, dataset_name: str):
    """Benchmark con dataset sintético."""
    print("=" * 70)
    print(f"BENCHMARK: {dataset_name.upper()}")
    print("=" * 70)
    
    # Cargar dataset
    dataset = load_synthetic_dataset(data_dir, dataset_name)
    if dataset is None:
        print(f"ERROR: No se pudo cargar {dataset_name}")
        return None
    
    image_stack = dataset['cube']
    angles = dataset['angles']
    psf = dataset['psf']
    
    print(f"\nDataset cargado:")
    print(f"  Image size: {image_stack.shape[1:]}")
    print(f"  N frames: {len(image_stack)}")
    print(f"  PSF size: {psf.shape}")
    
    # Configurar parámetros de PACO
    psf_rad = 4
    patch_size = 49
    
    # Preparar coordenadas de píxeles (todos los píxeles)
    h, w = image_stack.shape[1], image_stack.shape[2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"  Píxeles a procesar: {len(phi0s):,}")
    
    # Benchmark CPU
    print("\n" + "-" * 70)
    print("1. CPU (FastPACO Serial)")
    print("-" * 70)
    
    fp_cpu = FastPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf,
        psf_rad=psf_rad,
        patch_area=patch_size
    )
    
    start = time.perf_counter()
    a_cpu, b_cpu = fp_cpu.PACOCalc(phi0s, cpu=1)
    time_cpu = time.perf_counter() - start
    
    print(f"  Tiempo: {time_cpu:.2f} segundos ({time_cpu/60:.2f} minutos)")
    print(f"  Throughput: {len(phi0s)/time_cpu:.1f} píxeles/segundo")
    
    # Benchmark GPU Optimized
    if not CUDA_AVAILABLE:
        print("\n" + "-" * 70)
        print("2. GPU (FastPACO_CUDA_Optimized) - NO DISPONIBLE")
        print("-" * 70)
        return None
    
    print("\n" + "-" * 70)
    print("2. GPU (FastPACO_CUDA_Optimized)")
    print("-" * 70)
    
    try:
        fp_gpu = FastPACO_CUDA_Optimized(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=psf_rad,
            patch_area=patch_size
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
        
        # Verificar precisión
        valid_mask = ~(np.isnan(a_cpu) | np.isnan(a_gpu) | np.isnan(b_cpu) | np.isnan(b_gpu))
        if np.any(valid_mask):
            diff_a = np.abs(a_cpu[valid_mask] - a_gpu[valid_mask]).max()
            diff_b = np.abs(b_cpu[valid_mask] - b_gpu[valid_mask]).max()
            
            print(f"\n  Verificacion de precision:")
            print(f"    Diferencia en a: {diff_a:.2e}")
            print(f"    Diferencia en b: {diff_b:.2e}")
            
            if diff_a < 1e-1 and diff_b < 1e-1:
                print(f"    [OK] Resultados son similares")
            else:
                print(f"    [WARNING] Hay diferencias (puede ser normal por precision float32)")
        
        return {
            'dataset': dataset_name,
            'image_size': image_stack.shape[1:],
            'n_frames': len(image_stack),
            'n_pixels': len(phi0s),
            'time_cpu': time_cpu,
            'time_gpu': time_gpu,
            'speedup': speedup
        }
            
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Ejecutar benchmarks con datasets sintéticos grandes."""
    print("\n" + "=" * 70)
    print("BENCHMARK FastPACO_CUDA_Optimized CON DATOS SINTÉTICOS GRANDES")
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
        print("\n[ERROR] CuPy no esta disponible")
        return
    
    # Buscar directorio de datos sintéticos
    data_dir = Path("./synthetic_data")
    if not data_dir.exists():
        print(f"\n[ERROR] Directorio no existe: {data_dir}")
        print("  Ejecuta primero: python generate_synthetic_large.py")
        return
    
    # Listar datasets disponibles
    datasets = [
        'synthetic_medium_200x200',
        'synthetic_large_300x300',
        'synthetic_xlarge_500x500',
        'synthetic_xxlarge_800x800'
    ]
    
    available_datasets = []
    for ds_name in datasets:
        if (data_dir / f"{ds_name}_cube.fits").exists():
            available_datasets.append(ds_name)
            # Obtener tamaño
            dataset = load_synthetic_dataset(data_dir, ds_name)
            if dataset:
                size = dataset['n_frames'] * dataset['height'] * dataset['width']
                print(f"  [OK] {ds_name}: {dataset['height']}x{dataset['width']}, {dataset['n_frames']} frames (tamaño: {size:,})")
    
    if not available_datasets:
        print("\n[ERROR] No se encontraron datasets sintéticos")
        return
    
    # Probar con el más grande disponible
    largest = available_datasets[-1]  # El último es el más grande
    
    print(f"\n" + "=" * 70)
    print(f"PROBANDO CON DATASET: {largest.upper()}")
    print("=" * 70)
    
    result = benchmark_synthetic_dataset(data_dir, largest)
    
    # Resumen
    if result:
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print(f"\n{result['dataset']}:")
        print(f"  Tamaño: {result['image_size']}")
        print(f"  Frames: {result['n_frames']}")
        print(f"  Píxeles procesados: {result['n_pixels']:,}")
        print(f"  CPU: {result['time_cpu']:.2f}s ({result['time_cpu']/60:.2f} min)")
        print(f"  GPU: {result['time_gpu']:.2f}s ({result['time_gpu']/60:.2f} min)")
        print(f"  [SPEEDUP] {result['speedup']:.2f}x mas rapido")
        if result['speedup'] > 1.0:
            print(f"  [OK] GPU es mas rapida!")
        else:
            print(f"  [WARNING] GPU es mas lenta (overhead supera beneficio)")
    
    print("\n" + "=" * 70)
    print("Benchmark completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

