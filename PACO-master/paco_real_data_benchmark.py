"""
PACO Performance Benchmark con Datos Reales del EIDC Phase 1

Este script adapta el benchmark de rendimiento de PACO para trabajar con
los datos reales del Exoplanet Imaging Data Challenge Phase 1.

Referencias:
- EIDC Phase 1: https://exoplanet-imaging-challenge.github.io/datasets1/
- EIDC Phase 1 Submission: https://exoplanet-imaging-challenge.github.io/submission1/
- Starting Kit: https://exoplanet-imaging-challenge.github.io/startingkit/
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Importar funciones del benchmark original
from paco_performance_benchmark import (
    FastPACO_Loky,
    run_paco_serial,
    run_paco_multiprocessing,
    run_paco_loky,
    run_benchmark,
    analyze_amdahl,
    plot_benchmark_results,
    save_results,
    analyze_scalability,
    BENCHMARK_CONFIG,
    LOKY_AVAILABLE,
    IN_NOTEBOOK
)

# Importar PACO
import paco.processing.fastpaco as fastPACO
from paco.util.util import *

# Para cargar FITS
try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False
    print("WARNING: astropy no disponible. Instalar con: pip install astropy")


def load_challenge_dataset(data_dir: Path, instrument: str, dataset_id: int) -> Optional[Dict]:
    """
    Carga un dataset del EIDC Phase 1 desde archivos FITS.
    
    Parameters
    ----------
    data_dir : Path
        Directorio que contiene los archivos FITS
    instrument : str
        Nombre del instrumento ('sphere_irdis', 'nirc2', 'lmircam')
    dataset_id : int
        ID del dataset (1, 2, 3, etc.)
    
    Returns
    -------
    dict
        Diccionario con los datos cargados:
        - 'cube': array 3D (n_frames, height, width)
        - 'angles': array 1D de ángulos de paralaje
        - 'psf': array 2D del PSF
        - 'pxscale': float, pixel scale
        - 'instrument': str
        - 'dataset_id': int
    """
    if not FITS_AVAILABLE:
        print(f"ERROR: astropy no disponible para cargar {instrument}_{dataset_id}")
        return None
    
    try:
        # Construir nombres de archivos según la convención del challenge
        cube_file = data_dir / f"{instrument}_cube_{dataset_id}.fits"
        pa_file = data_dir / f"{instrument}_pa_{dataset_id}.fits"
        psf_file = data_dir / f"{instrument}_psf_{dataset_id}.fits"
        pxscale_file = data_dir / f"{instrument}_pxscale_{dataset_id}.fits"
        
        # Verificar que existan los archivos
        missing_files = []
        for f in [cube_file, pa_file, psf_file, pxscale_file]:
            if not f.exists():
                missing_files.append(f.name)
        
        if missing_files:
            print(f"ERROR: Archivos faltantes para {instrument}_{dataset_id}: {missing_files}")
            return None
        
        print(f"\nCargando {instrument} dataset {dataset_id}...")
        
        # Cargar cube (3D array: n_frames, height, width)
        with fits.open(cube_file) as hdul:
            cube = hdul[0].data
            if cube is None:
                print(f"  ERROR: {cube_file.name} está vacío")
                return None
            # Asegurar que sea 3D
            if len(cube.shape) == 4:
                print(f"  WARNING: Cube 4D detectado {cube.shape}, usando primer canal")
                cube = cube[0]
            elif len(cube.shape) != 3:
                print(f"  ERROR: Cube tiene forma inesperada: {cube.shape}")
                return None
        
        # Cargar ángulos de paralaje (1D array)
        with fits.open(pa_file) as hdul:
            angles = hdul[0].data
            if angles is None:
                print(f"  ERROR: {pa_file.name} está vacío")
                return None
            # Aplanar si es necesario
            if angles.ndim > 1:
                angles = angles.flatten()
            # Asegurar que tenga el mismo número de frames que el cube
            if len(angles) != cube.shape[0]:
                print(f"  WARNING: Número de ángulos ({len(angles)}) != número de frames ({cube.shape[0]})")
                # Truncar o ajustar según sea necesario
                min_len = min(len(angles), cube.shape[0])
                angles = angles[:min_len]
                cube = cube[:min_len]
        
        # Cargar PSF (2D array)
        with fits.open(psf_file) as hdul:
            psf = hdul[0].data
            if psf is None:
                print(f"  ERROR: {psf_file.name} está vacío")
                return None
            # Asegurar que sea 2D
            if len(psf.shape) == 3:
                print(f"  WARNING: PSF 3D detectado {psf.shape}, usando primer frame")
                psf = psf[0]
            elif len(psf.shape) != 2:
                print(f"  ERROR: PSF tiene forma inesperada: {psf.shape}")
                return None
        
        # Cargar pixel scale (float)
        with fits.open(pxscale_file) as hdul:
            pxscale_data = hdul[0].data
            if pxscale_data is None:
                print(f"  WARNING: {pxscale_file.name} está vacío, usando valor por defecto")
                pxscale = 1.0
            else:
                # Extraer el valor escalar
                if isinstance(pxscale_data, np.ndarray):
                    pxscale = float(pxscale_data.flatten()[0])
                else:
                    pxscale = float(pxscale_data)
        
        # Convertir a float64 si es necesario (para compatibilidad con PACO)
        cube = cube.astype(np.float64)
        angles = angles.astype(np.float64)
        psf = psf.astype(np.float64)
        
        # Normalizar PSF (suma = 1.0)
        psf_sum = np.sum(psf)
        if psf_sum > 0:
            psf = psf / psf_sum
        else:
            print(f"  WARNING: PSF tiene suma cero, usando PSF sin normalizar")
        
        n_frames, height, width = cube.shape
        
        print(f"  Cube: {cube.shape} (n_frames={n_frames}, size={height}x{width})")
        print(f"  Angles: {len(angles)} frames")
        print(f"  PSF: {psf.shape}")
        print(f"  Pixel scale: {pxscale} arcsec/pixel")
        print(f"  [OK] Dataset cargado correctamente")
        
        return {
            'cube': cube,
            'angles': angles,
            'psf': psf,
            'pxscale': pxscale,
            'instrument': instrument,
            'dataset_id': dataset_id,
            'n_frames': n_frames,
            'height': height,
            'width': width
        }
        
    except Exception as e:
        print(f"  ERROR cargando {instrument}_{dataset_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_dataset_for_paco(dataset: Dict, crop_size: Optional[Tuple[int, int]] = None) -> Dict:
    """
    Prepara un dataset para ser usado con PACO.
    
    Parameters
    ----------
    dataset : dict
        Dataset cargado con load_challenge_dataset
    crop_size : tuple, optional
        Si se proporciona, recorta el cube a este tamaño (height, width)
        centrado en el medio de la imagen
    
    Returns
    -------
    dict
        Diccionario con datos preparados para PACO
    """
    cube = dataset['cube']
    angles = dataset['angles']
    psf = dataset['psf']
    
    # Recortar si se especifica
    if crop_size is not None:
        crop_h, crop_w = crop_size
        h, w = cube.shape[1], cube.shape[2]
        
        if crop_h < h or crop_w < w:
            # Calcular centro
            center_h, center_w = h // 2, w // 2
            start_h = center_h - crop_h // 2
            end_h = start_h + crop_h
            start_w = center_w - crop_w // 2
            end_w = start_w + crop_w
            
            cube = cube[:, start_h:end_h, start_w:end_w]
            print(f"  Cube recortado a {cube.shape}")
    
    # Determinar parámetros de PACO basados en el PSF
    # IMPORTANTE: Según el README, patch_area debe ser 13, 49, o 113 (radios 3, 4, 5 respectivamente)
    # El valor por defecto es 49 (radio 4). Usaremos este valor fijo.
    from paco.util.util import createCircularMask
    
    # Usar patch_size fijo de 49 (radio 4) como recomienda el README
    # Esto es más consistente con el uso típico de PACO
    patch_size = 49  # Valor recomendado por el README (radio 4)
    psf_rad = 4  # Radio correspondiente a patch_size=49
    
    # El tamaño del patch que se extraerá es igual al tamaño del PSF
    patch_width = psf.shape[0]  # Esto será m_pwidth en el constructor
    
    # Verificar que el radio 4 funcione con el tamaño del PSF
    patch_shape = (patch_width, patch_width)
    test_mask = createCircularMask(patch_shape, radius=psf_rad)
    actual_patch_size = len(test_mask[test_mask])
    
    if actual_patch_size != patch_size:
        print(f"  WARNING: patch_size esperado={patch_size}, pero con psf_rad={psf_rad} y patch_width={patch_width} se obtiene {actual_patch_size}")
        print(f"  Usando patch_size calculado: {actual_patch_size}")
        patch_size = actual_patch_size
    else:
        print(f"  patch_size={patch_size} (radio {psf_rad}) configurado correctamente para patch_width={patch_width}")
    
    return {
        'image_stack': cube,
        'angles': angles,
        'psf_template': psf,
        'psf_rad': psf_rad,
        'patch_size': patch_size,
        'image_size': (cube.shape[1], cube.shape[2]),
        'n_frames': cube.shape[0],
        'instrument': dataset['instrument'],
        'dataset_id': dataset['dataset_id'],
        'pxscale': dataset['pxscale']
    }


def benchmark_real_dataset(data_dir: Path, instrument: str, dataset_id: int, 
                           config: Dict, crop_size: Optional[Tuple[int, int]] = None,
                           n_trials: int = 1) -> Dict:
    """
    Ejecuta el benchmark completo en un dataset real del challenge.
    
    Parameters
    ----------
    data_dir : Path
        Directorio con los archivos FITS
    instrument : str
        Nombre del instrumento
    dataset_id : int
        ID del dataset
    config : dict
        Configuración del benchmark (cpu_counts, etc.)
    crop_size : tuple, optional
        Tamaño para recortar el cube (height, width)
    n_trials : int
        Número de trials para cada configuración
    
    Returns
    -------
    dict
        Resultados del benchmark
    """
    print("\n" + "="*70)
    print(f"BENCHMARK CON DATOS REALES: {instrument.upper()} Dataset {dataset_id}")
    print("="*70)
    
    # Cargar dataset
    dataset = load_challenge_dataset(data_dir, instrument, dataset_id)
    if dataset is None:
        print(f"ERROR: No se pudo cargar {instrument}_{dataset_id}")
        return None
    
    # Preparar para PACO
    paco_data = prepare_dataset_for_paco(dataset, crop_size=crop_size)
    
    # Actualizar config con los valores reales del dataset
    config_real = config.copy()
    config_real['image_size'] = paco_data['image_size']
    config_real['psf_rad'] = paco_data['psf_rad']
    config_real['patch_size'] = paco_data['patch_size']
    config_real['n_frames'] = paco_data['n_frames']
    config_real['instrument'] = paco_data['instrument']
    config_real['dataset_id'] = paco_data['dataset_id']
    
    print(f"\nConfiguración del benchmark:")
    print(f"  Instrumento: {instrument}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Image size: {config_real['image_size']}")
    print(f"  N frames: {config_real['n_frames']}")
    print(f"  PSF radius: {config_real['psf_rad']}")
    print(f"  Patch size: {config_real['patch_size']}")
    print(f"  CPU counts: {config_real['cpu_counts']}")
    print(f"  N trials: {n_trials}")
    
    # Ejecutar benchmark (usar la función original pero con datos reales)
    # La función run_benchmark espera: config, test_data (tuple: image_stack, angles, psf_template)
    from paco_performance_benchmark import run_benchmark
    
    # Preparar test_data en el formato esperado (tupla)
    test_data = (
        paco_data['image_stack'],
        paco_data['angles'],
        paco_data['psf_template']
    )
    
    # Forzar que solo se ejecute serial y loky (no multiprocessing)
    # Esto es importante en Windows/notebooks donde multiprocessing puede tener problemas
    skip_multiprocessing = os.environ.get('SKIP_MULTIPROCESSING', '0') == '1' or IN_NOTEBOOK
    
    results = run_benchmark(
        config_real,
        test_data,
        n_trials=n_trials,
        skip_multiprocessing=skip_multiprocessing
    )
    
    # Agregar metadata del dataset
    results['dataset_info'] = {
        'instrument': instrument,
        'dataset_id': dataset_id,
        'pxscale': dataset['pxscale'],
        'original_size': (dataset['height'], dataset['width']),
        'final_size': config_real['image_size'],
        'crop_size': crop_size
    }
    
    return results


def benchmark_all_datasets(data_dir: Path, config: Dict, 
                          instruments: Optional[List[str]] = None,
                          dataset_ids: Optional[List[int]] = None,
                          crop_size: Optional[Tuple[int, int]] = None,
                          n_trials: int = 1) -> Dict:
    """
    Ejecuta el benchmark en todos los datasets disponibles.
    
    Parameters
    ----------
    data_dir : Path
        Directorio con los archivos FITS
    config : dict
        Configuración del benchmark
    instruments : list, optional
        Lista de instrumentos a procesar. Si None, usa todos disponibles
    dataset_ids : list, optional
        Lista de IDs de datasets a procesar. Si None, detecta automáticamente
    crop_size : tuple, optional
        Tamaño para recortar los cubes
    n_trials : int
        Número de trials
    
    Returns
    -------
    dict
        Resultados de todos los benchmarks
    """
    if instruments is None:
        # Detectar instrumentos disponibles
        instruments = []
        for f in data_dir.glob("*_cube_*.fits"):
            instrument = f.name.split('_cube_')[0]
            if instrument not in instruments:
                instruments.append(instrument)
    
    print(f"\nInstrumentos detectados: {instruments}")
    
    all_results = {}
    
    for instrument in instruments:
        # Detectar datasets disponibles para este instrumento
        if dataset_ids is None:
            dataset_ids = []
            for f in data_dir.glob(f"{instrument}_cube_*.fits"):
                # Extraer ID del nombre del archivo
                parts = f.stem.split('_')
                if len(parts) >= 3:
                    try:
                        ds_id = int(parts[-1])
                        if ds_id not in dataset_ids:
                            dataset_ids.append(ds_id)
                    except ValueError:
                        pass
            dataset_ids = sorted(dataset_ids)
        
        print(f"\nProcesando {instrument} (datasets: {dataset_ids})...")
        
        for ds_id in dataset_ids:
            key = f"{instrument}_{ds_id}"
            print(f"\n{'='*70}")
            print(f"Procesando: {key}")
            print(f"{'='*70}")
            
            results = benchmark_real_dataset(
                data_dir, instrument, ds_id, config,
                crop_size=crop_size, n_trials=n_trials
            )
            
            if results is not None:
                all_results[key] = results
            else:
                print(f"  [SKIP] {key} no se pudo procesar")
    
    return all_results


if __name__ == "__main__":
    # Ejemplo de uso
    data_dir = Path("../subchallenge1")
    
    # Configuración del benchmark
    config = BENCHMARK_CONFIG.copy()
    config['cpu_counts'] = [1, 2, 4, 8]  # Ajustar según disponibilidad
    
    # Opcional: recortar imágenes grandes para acelerar el benchmark
    # crop_size = (100, 100)  # Recortar a 100x100 píxeles centrado
    crop_size = None  # Usar tamaño completo
    
    # Ejecutar benchmark en un dataset específico
    print("Ejecutando benchmark en dataset de ejemplo...")
    results = benchmark_real_dataset(
        data_dir,
        instrument='sphere_irdis',
        dataset_id=1,
        config=config,
        crop_size=crop_size,
        n_trials=1
    )
    
    if results:
        # Analizar resultados
        from paco_performance_benchmark import analyze_amdahl, plot_benchmark_results, save_results
        
        analysis = analyze_amdahl(results)
        print("\nAnálisis completado")
        
        # Guardar resultados
        output_dir = Path("output/real_data_benchmark")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_path, csv_path = save_results(results, analysis, output_dir)
        print(f"\nResultados guardados:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        # Generar gráficos
        try:
            plot_benchmark_results(results, analysis, output_dir)
            print(f"  Gráficos guardados en: {output_dir}")
        except Exception as e:
            print(f"  ERROR generando gráficos: {e}")

