"""
Benchmark FastPACO_CUDA con datasets reales del EIDC Challenge.

Compara rendimiento CPU vs GPU usando los datasets más grandes del challenge.
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
    
    Parameters
    ----------
    data_dir : Path
        Directorio con los archivos FITS
    instrument : str
        Nombre del instrumento
    dataset_id : int
        ID del dataset
    crop_size : tuple, optional
        Tamaño para recortar (height, width). Si None, usa tamaño completo
    test_size : tuple, optional
        Tamaño de prueba más pequeño (height, width) para pruebas rápidas
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
    psf = paco_data['psf_template']  # prepare_dataset_for_paco retorna 'psf_template'
    
    # Config extraído de paco_data
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
    
    # Si test_size está especificado, usar una región más pequeña para prueba rápida
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
        print("  Instala CuPy con: pip install cupy-cuda12x")
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
            'diff_b': diff_b,
            'a_cpu': a_cpu,
            'b_cpu': b_cpu,
            'a_gpu': a_gpu,
            'b_gpu': b_gpu
        }
            
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Ejecutar benchmarks con datasets del challenge."""
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
        print("\n[WARNING] CuPy no esta disponible")
        print("  Instala con: pip install cupy-cuda12x")
        return
    
    # Solicitar directorio de datos
    print("\n" + "-" * 70)
    print("CONFIGURACION")
    print("-" * 70)
    
    data_dir_input = input("Directorio con archivos FITS del challenge (o Enter para usar './data'): ").strip()
    if not data_dir_input:
        data_dir = Path("./data")
    else:
        data_dir = Path(data_dir_input)
    
    if not data_dir.exists():
        print(f"ERROR: Directorio no existe: {data_dir}")
        return
    
    # Listar datasets disponibles
    print(f"\nBuscando datasets en: {data_dir}")
    instruments = ['sphere_irdis', 'nirc2', 'lmircam']
    available_datasets = []
    
    for inst in instruments:
        for ds_id in [1, 2, 3, 4, 5]:
            cube_file = data_dir / f"{inst}_cube_{ds_id}.fits"
            if cube_file.exists():
                available_datasets.append((inst, ds_id))
                print(f"  [OK] {inst} dataset {ds_id}")
    
    if not available_datasets:
        print("  [ERROR] No se encontraron datasets")
        return
    
    # Seleccionar datasets para benchmark
    print("\n" + "-" * 70)
    print("SELECCION DE DATASETS")
    print("-" * 70)
    print("Opciones:")
    print("  1. Probar con dataset más grande disponible")
    print("  2. Probar con todos los datasets disponibles")
    print("  3. Seleccionar dataset específico")
    
    opcion = input("\nSelecciona opcion (1/2/3): ").strip()
    
    datasets_to_test = []
    
    if opcion == "1":
        # Encontrar el dataset más grande
        largest = None
        largest_size = 0
        
        for inst, ds_id in available_datasets:
            dataset = load_challenge_dataset(data_dir, inst, ds_id)
            if dataset:
                size = dataset['n_frames'] * dataset['height'] * dataset['width']
                if size > largest_size:
                    largest_size = size
                    largest = (inst, ds_id)
        
        if largest:
            datasets_to_test = [largest]
            print(f"\nDataset más grande: {largest[0]} dataset {largest[1]}")
    
    elif opcion == "2":
        datasets_to_test = available_datasets
        print(f"\nProbando con {len(datasets_to_test)} datasets")
    
    elif opcion == "3":
        print("\nDatasets disponibles:")
        for i, (inst, ds_id) in enumerate(available_datasets, 1):
            print(f"  {i}. {inst} dataset {ds_id}")
        
        seleccion = input("\nSelecciona numero: ").strip()
        try:
            idx = int(seleccion) - 1
            if 0 <= idx < len(available_datasets):
                datasets_to_test = [available_datasets[idx]]
            else:
                print("ERROR: Seleccion invalida")
                return
        except:
            print("ERROR: Seleccion invalida")
            return
    
    else:
        print("ERROR: Opcion invalida")
        return
    
    # Preguntar si usar tamaño completo o región de prueba
    print("\n" + "-" * 70)
    print("TAMAÑO DE PRUEBA")
    print("-" * 70)
    print("Para datasets grandes, puedes usar una región más pequeña para prueba rápida.")
    use_test_size = input("¿Usar región de prueba (100x100)? [y/N]: ").strip().lower() == 'y'
    
    test_size = (100, 100) if use_test_size else None
    
    # Ejecutar benchmarks
    results = []
    
    for inst, ds_id in datasets_to_test:
        print("\n" + "=" * 70)
        result = benchmark_challenge_dataset(
            data_dir, inst, ds_id, 
            crop_size=None,  # Usar tamaño completo
            test_size=test_size
        )
        
        if result:
            results.append(result)
    
    # Resumen final
    if results:
        print("\n" + "=" * 70)
        print("RESUMEN DE RESULTADOS")
        print("=" * 70)
        
        for r in results:
            print(f"\n{r['instrument']} Dataset {r['dataset_id']}:")
            print(f"  Tamaño: {r['image_size']}, Frames: {r['n_frames']}, Píxeles: {r['n_pixels']}")
            print(f"  CPU: {r['time_cpu']:.2f}s, GPU: {r['time_gpu']:.2f}s")
            print(f"  Speedup: {r['speedup']:.2f}x")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\n  Speedup promedio: {avg_speedup:.2f}x")
    
    print("\n" + "=" * 70)
    print("Benchmark completado")
    print("=" * 70)


if __name__ == "__main__":
    main()

