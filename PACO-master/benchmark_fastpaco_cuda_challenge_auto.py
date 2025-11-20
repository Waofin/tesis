"""
Benchmark FastPACO_CUDA con datasets reales del EIDC Challenge (versión automática).

Esta versión ejecuta automáticamente sin interacción del usuario.
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Agregar ruta de PACO
sys.path.insert(0, str(Path(__file__).parent))

from paco_real_data_benchmark import load_challenge_dataset, prepare_dataset_for_paco
from paco.processing.fastpaco import FastPACO
from paco.processing.fastpaco_cuda import FastPACO_CUDA, CUDA_AVAILABLE


def benchmark_challenge_dataset(data_dir: Path, instrument: str, dataset_id: int, 
                                crop_size=None, test_size=None):
    """
    Benchmark FastPACO CPU vs GPU con un dataset del challenge.
    """
    print("=" * 70)
    print(f"BENCHMARK: {instrument.upper()} Dataset {dataset_id}")
    print("=" * 70)
    
    # Cargar dataset
    dataset = load_challenge_dataset(data_dir, instrument, dataset_id)
    if dataset is None:
        print(f"ERROR: No se pudo cargar {instrument}_{dataset_id}")
        return None
    
    # Preparar datos para PACO
    paco_data = prepare_dataset_for_paco(dataset, crop_size=crop_size)
    
    image_stack = paco_data['image_stack']
    angles = paco_data['angles']
    psf = paco_data['psf_template']
    
    config = {
        'psf_rad': paco_data['psf_rad'],
        'patch_size': paco_data['patch_size']
    }
    
    print(f"\nConfiguración del benchmark:")
    print(f"  Instrumento: {instrument}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Image size: {image_stack.shape[1:]}")
    print(f"  N frames: {len(image_stack)}")
    print(f"  PSF radius: {config['psf_rad']}")
    print(f"  Patch size: {config['patch_size']}")
    
    # Si test_size está especificado, usar una región más pequeña
    if test_size is not None:
        h, w = image_stack.shape[1], image_stack.shape[2]
        test_h, test_w = test_size
        
        if test_h < h or test_w < w:
            center_h, center_w = h // 2, w // 2
            start_h = center_h - test_h // 2
            end_h = start_h + test_h
            start_w = center_w - test_w // 2
            end_w = start_w + test_w
            
            image_stack = image_stack[:, start_h:end_h, start_w:end_w]
            print(f"  [TEST] Usando región de prueba: {image_stack.shape[1:]}")
    
    # Preparar coordenadas de píxeles
    h, w = image_stack.shape[1], image_stack.shape[2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
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
        psf_rad=config['psf_rad'],
        patch_area=config['patch_size']
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
        return None
    
    print("\n" + "-" * 70)
    print("2. GPU (FastPACO_CUDA)")
    print("-" * 70)
    
    try:
        fp_gpu = FastPACO_CUDA(
            image_stack=image_stack,
            angles=angles,
            psf=psf,
            psf_rad=config['psf_rad'],
            patch_area=config['patch_size']
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
        
        if diff_a < 1e-1 and diff_b < 1e-1:
            print(f"    [OK] Resultados son similares (dentro de precision)")
        else:
            print(f"    [WARNING] Hay diferencias (puede ser normal por precision float32)")
        
        return {
            'instrument': instrument,
            'dataset_id': dataset_id,
            'image_size': image_stack.shape[1:],
            'n_frames': len(image_stack),
            'n_pixels': len(phi0s),
            'time_cpu': time_cpu,
            'time_gpu': time_gpu,
            'speedup': speedup,
            'diff_a': diff_a,
            'diff_b': diff_b
        }
            
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Ejecutar benchmarks automáticamente."""
    print("\n" + "=" * 70)
    print("BENCHMARK FastPACO_CUDA CON DATOS REALES DEL CHALLENGE")
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
    
    # Buscar directorio de datos
    possible_dirs = [
        Path("./data"),
        Path("../data"),
        Path("../../data"),
        Path("./challenge_data"),
        Path("../challenge_data"),
    ]
    
    # También buscar en directorios comunes
    for base in [Path("."), Path(".."), Path("../..")]:
        for pattern in ["*data*", "*challenge*", "*fits*"]:
            dirs = list(base.glob(pattern))
            possible_dirs.extend([d for d in dirs if d.is_dir()])
    
    data_dir = None
    for d in possible_dirs:
        if d.exists():
            # Verificar que tenga archivos FITS
            fits_files = list(d.glob("*.fits"))
            if fits_files:
                data_dir = d
                print(f"\n[OK] Directorio encontrado: {data_dir}")
                print(f"  Archivos FITS encontrados: {len(fits_files)}")
                break
    
    if data_dir is None:
        print("\n[ERROR] No se encontro directorio con archivos FITS")
        print("  Buscando en:")
        for d in possible_dirs[:5]:
            print(f"    - {d}")
        print("\n  Por favor, proporciona el directorio manualmente:")
        print("    python benchmark_fastpaco_cuda_challenge.py")
        return
    
    # Listar datasets disponibles
    instruments = ['sphere_irdis', 'nirc2', 'lmircam']
    available_datasets = []
    
    for inst in instruments:
        for ds_id in [1, 2, 3, 4, 5]:
            cube_file = data_dir / f"{inst}_cube_{ds_id}.fits"
            if cube_file.exists():
                available_datasets.append((inst, ds_id))
    
    if not available_datasets:
        print("\n[ERROR] No se encontraron datasets")
        return
    
    print(f"\nDatasets disponibles: {len(available_datasets)}")
    for inst, ds_id in available_datasets:
        dataset = load_challenge_dataset(data_dir, inst, ds_id)
        if dataset:
            size = dataset['n_frames'] * dataset['height'] * dataset['width']
            print(f"  - {inst} dataset {ds_id}: {dataset['height']}x{dataset['width']}, {dataset['n_frames']} frames (tamaño: {size:,})")
    
    # Encontrar el dataset más grande
    largest = None
    largest_size = 0
    
    for inst, ds_id in available_datasets:
        dataset = load_challenge_dataset(data_dir, inst, ds_id)
        if dataset:
            size = dataset['n_frames'] * dataset['height'] * dataset['width']
            if size > largest_size:
                largest_size = size
                largest = (inst, ds_id, dataset)
    
    if not largest:
        print("\n[ERROR] No se pudo determinar dataset mas grande")
        return
    
    inst, ds_id, dataset_info = largest
    print(f"\n" + "=" * 70)
    print(f"PROBANDO CON DATASET MAS GRANDE: {inst.upper()} Dataset {ds_id}")
    print("=" * 70)
    print(f"  Tamaño: {dataset_info['height']}x{dataset_info['width']}")
    print(f"  Frames: {dataset_info['n_frames']}")
    print(f"  Tamaño total: {largest_size:,} píxeles")
    
    # Para datasets muy grandes, usar región de prueba
    total_pixels = dataset_info['height'] * dataset_info['width']
    use_test_region = total_pixels > 10000  # Si más de 100x100
    
    if use_test_region:
        print(f"\n  [INFO] Dataset grande detectado, usando región de prueba 100x100")
        test_size = (100, 100)
    else:
        test_size = None
    
    # Ejecutar benchmark
    result = benchmark_challenge_dataset(
        data_dir, inst, ds_id,
        crop_size=None,
        test_size=test_size
    )
    
    # Resumen
    if result:
        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        print(f"\n{result['instrument']} Dataset {result['dataset_id']}:")
        print(f"  Tamaño: {result['image_size']}")
        print(f"  Frames: {result['n_frames']}")
        print(f"  Píxeles procesados: {result['n_pixels']:,}")
        print(f"  CPU: {result['time_cpu']:.2f}s ({result['time_cpu']/60:.2f} min)")
        print(f"  GPU: {result['time_gpu']:.2f}s ({result['time_gpu']/60:.2f} min)")
        print(f"  [SPEEDUP] {result['speedup']:.2f}x mas rapido")
        print(f"  Ahorro de tiempo: {result['time_cpu'] - result['time_gpu']:.2f}s ({(result['time_cpu'] - result['time_gpu'])/60:.2f} min)")
    
    print("\n" + "=" * 70)
    print("Benchmark completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

