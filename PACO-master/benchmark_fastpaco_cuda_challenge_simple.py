"""
Benchmark FastPACO_CUDA con datasets reales del EIDC Challenge (versión simplificada).

Esta versión carga datasets directamente sin dependencias adicionales.
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

from paco.processing.fastpaco import FastPACO
from paco.processing.fastpaco_cuda import FastPACO_CUDA, CUDA_AVAILABLE
from paco.util.util import createCircularMask

# Para cargar FITS
try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False
    print("ERROR: astropy no disponible. Instalar con: pip install astropy")
    sys.exit(1)


def load_challenge_dataset_simple(data_dir: Path, instrument: str, dataset_id: int):
    """Carga un dataset del challenge de forma simplificada."""
    if not FITS_AVAILABLE:
        return None
    
    try:
        # Intentar primero con formato estándar: instrument_cube_id.fits
        cube_file = data_dir / f"{instrument}_cube_{dataset_id}.fits"
        pa_file = data_dir / f"{instrument}_pa_{dataset_id}.fits"
        psf_file = data_dir / f"{instrument}_psf_{dataset_id}.fits"
        
        # Si no existe, intentar formato alternativo: instrument_name_cube.fits
        if not cube_file.exists():
            # Buscar archivos con patrones alternativos
            cube_patterns = [
                f"{instrument}_*cube*.fits",
                f"*{instrument}*cube*.fits",
                f"{instrument}_cube.fits"
            ]
            
            for pattern in cube_patterns:
                matches = list(data_dir.glob(pattern))
                if matches:
                    cube_file = matches[0]
                    # Intentar encontrar pa y psf correspondientes
                    base_name = cube_file.stem.replace('_cube', '').replace('cube', '')
                    pa_file = data_dir / f"{base_name}_pa.fits"
                    if not pa_file.exists():
                        pa_file = data_dir / f"{base_name}*pa*.fits"
                        pa_matches = list(data_dir.glob(str(pa_file)))
                        if pa_matches:
                            pa_file = pa_matches[0]
                    
                    psf_file = data_dir / f"{base_name}_psf.fits"
                    if not psf_file.exists():
                        psf_file = data_dir / f"{base_name}*psf*.fits"
                        psf_matches = list(data_dir.glob(str(psf_file)))
                        if psf_matches:
                            psf_file = psf_matches[0]
                    break
        
        if not all(f.exists() for f in [cube_file, pa_file, psf_file]):
            return None
        
        # Cargar cube
        with fits.open(cube_file) as hdul:
            cube = hdul[0].data
            if cube is None:
                return None
            if len(cube.shape) == 4:
                cube = cube[0]
            elif len(cube.shape) != 3:
                return None
        
        # Cargar ángulos
        with fits.open(pa_file) as hdul:
            angles = hdul[0].data
            if angles is None:
                return None
            if angles.ndim > 1:
                angles = angles.flatten()
            if len(angles) != cube.shape[0]:
                min_len = min(len(angles), cube.shape[0])
                angles = angles[:min_len]
                cube = cube[:min_len]
        
        # Cargar PSF
        with fits.open(psf_file) as hdul:
            psf = hdul[0].data
            if psf is None:
                return None
            if len(psf.shape) == 3:
                psf = psf[0]
            elif len(psf.shape) != 2:
                return None
        
        # Normalizar PSF
        psf_sum = np.sum(psf)
        if psf_sum > 0:
            psf = psf / psf_sum
        
        # Convertir a float32 para compatibilidad
        cube = cube.astype(np.float32)
        angles = angles.astype(np.float32)
        psf = psf.astype(np.float32)
        
        return {
            'cube': cube,
            'angles': angles,
            'psf': psf,
            'n_frames': cube.shape[0],
            'height': cube.shape[1],
            'width': cube.shape[2]
        }
    except Exception as e:
        print(f"ERROR cargando {instrument}_{dataset_id}: {e}")
        return None


def benchmark_challenge_dataset(data_dir: Path, instrument: str, dataset_id: int, 
                                test_size=None):
    """Benchmark FastPACO CPU vs GPU con un dataset del challenge."""
    print("=" * 70)
    print(f"BENCHMARK: {instrument.upper()} Dataset {dataset_id}")
    print("=" * 70)
    
    # Cargar dataset
    dataset = load_challenge_dataset_simple(data_dir, instrument, dataset_id)
    if dataset is None:
        print(f"ERROR: No se pudo cargar {instrument}_{dataset_id}")
        return None
    
    image_stack = dataset['cube']
    angles = dataset['angles']
    psf = dataset['psf']
    
    print(f"\nDataset cargado:")
    print(f"  Image size: {image_stack.shape[1:]}")
    print(f"  N frames: {len(image_stack)}")
    print(f"  PSF size: {psf.shape}")
    
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
    
    # Configurar parámetros de PACO
    psf_rad = 4
    patch_size = 49
    patch_width = psf.shape[0]
    
    # Verificar patch_size
    patch_shape = (patch_width, patch_width)
    mask = createCircularMask(patch_shape, radius=psf_rad)
    actual_patch_size = len(mask[mask])
    if actual_patch_size != patch_size:
        patch_size = actual_patch_size
        print(f"  Ajustando patch_size a: {patch_size}")
    
    # Preparar coordenadas de píxeles
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
            psf_rad=psf_rad,
            patch_area=patch_size
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
        print(f"  Ahorro de tiempo: {time_cpu - time_gpu:.2f}s ({(time_cpu - time_gpu)/60:.2f} min)")
        
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
    
    # Obtener directorio de argumentos de línea de comandos
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Buscar directorio de datos automáticamente
        possible_dirs = [
            Path("E:/TESIS/subchallenge1"),  # Prioridad: datasets grandes
            Path("./data"),
            Path("../data"),
            Path("../../data"),
            Path("./challenge_data"),
            Path("../challenge_data"),
            Path("E:/TESIS/data"),
            Path("E:/TESIS/challenge_data"),
            Path("E:/TESIS/exoimaging_challenge_extras-master/exoimaging_challenge_extras-master"),
        ]
        
        data_dir = None
        for d in possible_dirs:
            if d.exists():
                fits_files = list(d.glob("*.fits"))
                if fits_files:
                    data_dir = d
                    print(f"\n[OK] Directorio encontrado: {data_dir}")
                    print(f"  Archivos FITS encontrados: {len(fits_files)}")
                    break
    
    if data_dir is None or not data_dir.exists():
        print("\n[ERROR] No se encontro directorio con archivos FITS")
        print("\n  Uso:")
        print("    python benchmark_fastpaco_cuda_challenge_simple.py <directorio>")
        print("\n  Ejemplo:")
        print("    python benchmark_fastpaco_cuda_challenge_simple.py E:/TESIS/exoimaging_challenge_extras-master/exoimaging_challenge_extras-master")
        return
    
    if not data_dir.exists():
        print(f"\n[ERROR] Directorio no existe: {data_dir}")
        return
    
    # Listar datasets disponibles
    instruments = ['sphere_irdis', 'nirc2', 'lmircam', 'naco', 'sphere']
    available_datasets = []
    
    print("\nBuscando datasets...")
    
    # Buscar con formato estándar
    for inst in instruments:
        for ds_id in [1, 2, 3, 4, 5]:
            cube_file = data_dir / f"{inst}_cube_{ds_id}.fits"
            if cube_file.exists():
                dataset = load_challenge_dataset_simple(data_dir, inst, ds_id)
                if dataset:
                    size = dataset['n_frames'] * dataset['height'] * dataset['width']
                    available_datasets.append((inst, ds_id, size, dataset))
                    print(f"  [OK] {inst} dataset {ds_id}: {dataset['height']}x{dataset['width']}, {dataset['n_frames']} frames (tamaño: {size:,})")
    
    # Buscar archivos con formato alternativo (naco_betapic, sphere_v471tau, etc.)
    cube_files = list(data_dir.glob("*cube*.fits"))
    for cube_file in cube_files:
        # Extraer nombre base
        base_name = cube_file.stem.replace('_cube', '').replace('cube', '')
        
        # Buscar pa y psf correspondientes
        pa_files = list(data_dir.glob(f"{base_name}*pa*.fits"))
        psf_files = list(data_dir.glob(f"{base_name}*psf*.fits"))
        
        if pa_files and psf_files:
            # Determinar instrumento del nombre
            if 'naco' in base_name.lower():
                inst = 'naco'
            elif 'sphere' in base_name.lower():
                inst = 'sphere'
            else:
                inst = base_name.split('_')[0] if '_' in base_name else base_name
            
            # Usar dataset_id=1 como default para estos casos
            ds_id = 1
            
            # Verificar que no esté ya en la lista
            if not any((i, d) == (inst, ds_id) for i, d, _, _ in available_datasets):
                # Intentar cargar
                try:
                    with fits.open(cube_file) as hdul:
                        cube = hdul[0].data
                        if cube is not None and len(cube.shape) >= 3:
                            if len(cube.shape) == 4:
                                cube = cube[0]
                            
                            with fits.open(pa_files[0]) as hdul_pa:
                                angles = hdul_pa[0].data
                                if angles is not None:
                                    if angles.ndim > 1:
                                        angles = angles.flatten()
                                    
                                    with fits.open(psf_files[0]) as hdul_psf:
                                        psf = hdul_psf[0].data
                                        if psf is not None:
                                            if len(psf.shape) == 3:
                                                psf = psf[0]
                                            
                                            # Normalizar
                                            psf_sum = np.sum(psf)
                                            if psf_sum > 0:
                                                psf = psf / psf_sum
                                            
                                            # Ajustar tamaños
                                            min_len = min(len(angles), cube.shape[0])
                                            angles = angles[:min_len]
                                            cube = cube[:min_len]
                                            
                                            dataset = {
                                                'cube': cube.astype(np.float32),
                                                'angles': angles.astype(np.float32),
                                                'psf': psf.astype(np.float32),
                                                'n_frames': cube.shape[0],
                                                'height': cube.shape[1],
                                                'width': cube.shape[2]
                                            }
                                            
                                            size = dataset['n_frames'] * dataset['height'] * dataset['width']
                                            available_datasets.append((inst, ds_id, size, dataset))
                                            print(f"  [OK] {inst} ({base_name}): {dataset['height']}x{dataset['width']}, {dataset['n_frames']} frames (tamaño: {size:,})")
                except Exception as e:
                    print(f"  [WARNING] Error cargando {cube_file.name}: {e}")
                    continue
    
    if not available_datasets:
        print("\n[ERROR] No se encontraron datasets validos")
        return
    
    # Encontrar el dataset más grande
    # Priorizar lmircam que son los más grandes (cientos de MB)
    lmircam_datasets = [d for d in available_datasets if 'lmircam' in d[0].lower()]
    if lmircam_datasets:
        largest = max(lmircam_datasets, key=lambda x: x[2])
        inst, ds_id, size, dataset_info = largest
        print(f"\n  [INFO] Usando dataset LMIRCAM (más grande disponible)")
    else:
        # Buscar el más grande en general
        largest = max(available_datasets, key=lambda x: x[2])
        inst, ds_id, size, dataset_info = largest
    
    print(f"\n" + "=" * 70)
    print(f"PROBANDO CON DATASET MAS GRANDE: {inst.upper()} Dataset {ds_id}")
    print("=" * 70)
    print(f"  Tamaño: {dataset_info['height']}x{dataset_info['width']}")
    print(f"  Frames: {dataset_info['n_frames']}")
    print(f"  Tamaño total: {size:,} píxeles")
    
    # Para datasets muy grandes, usar región de prueba
    # Deshabilitado temporalmente para probar con dataset completo
    total_pixels = dataset_info['height'] * dataset_info['width']
    use_test_region = False  # total_pixels > 50000  # Solo para datasets muy grandes (> 224x224)
    
    if use_test_region:
        print(f"\n  [INFO] Dataset muy grande detectado, usando región de prueba 200x200")
        test_size = (200, 200)
    else:
        test_size = None
        print(f"\n  [INFO] Usando dataset completo: {dataset_info['height']}x{dataset_info['width']}")
    
    # Ejecutar benchmark
    result = benchmark_challenge_dataset(
        data_dir, inst, ds_id,
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

