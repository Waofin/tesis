"""
Script completo de Benchmark y Profiling para PACO en VIP-master

Este script:
1. Detecta errores en la implementación de PACO
2. Realiza profiling completo del algoritmo
3. Compara tiempos de ejecución con diferentes datasets
4. Carga datasets desde GitHub
5. Genera reportes detallados

Autor: César Cerda - Universidad del Bío-Bío
"""

import numpy as np
import time
import sys
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Profiling
try:
    import cProfile
    import pstats
    from io import StringIO
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False
    print("[WARNING] cProfile no disponible, profiling limitado")

# Memory profiling
try:
    from memory_profiler import profile
    MEMORY_PROFILING_AVAILABLE = True
except ImportError:
    MEMORY_PROFILING_AVAILABLE = False
    print("[WARNING] memory_profiler no disponible, instalarlo con: pip install memory-profiler")

# Line profiling
try:
    from line_profiler import LineProfiler
    LINE_PROFILING_AVAILABLE = True
except ImportError:
    LINE_PROFILING_AVAILABLE = False
    print("[WARNING] line_profiler no disponible, instalarlo con: pip install line-profiler")


class PACOErrorDetector:
    """Detector de errores en la implementación de PACO"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.recommendations = []
    
    def check_sample_covariance(self):
        """Verifica el error crítico en sample_covariance()"""
        print("\n" + "="*70)
        print("ANÁLISIS DE sample_covariance()")
        print("="*70)
        
        # Cargar la función original
        try:
            sys.path.insert(0, str(Path("VIP-master/VIP-master/src")))
            from vip_hci.invprob.paco import sample_covariance
            
            # Crear datos de prueba
            T = 10  # Número de frames
            P = 5   # Número de píxeles en el patch
            r = np.random.randn(T, P)
            m = np.mean(r, axis=0)
            
            # Probar la función actual
            try:
                S_current = sample_covariance(r, m, T)
                print(f"[OK] sample_covariance() ejecutada sin errores")
                print(f"  Shape de S: {S_current.shape}")
                print(f"  Debería ser: ({P}, {P})")
                
                # Verificar si la implementación es correcta
                # La implementación correcta debería ser:
                # S = (1.0/T) * sum((p - m)^T @ (p - m) for p in r)
                r_centered = r - m[np.newaxis, :]
                S_correct = (1.0 / T) * np.dot(r_centered.T, r_centered)
                
                # Comparar
                if not np.allclose(S_current, S_correct, rtol=1e-5):
                    self.errors.append({
                        'type': 'ERROR CRÍTICO',
                        'function': 'sample_covariance()',
                        'description': 'La implementación actual usa np.cov() incorrectamente',
                        'location': 'VIP-master/VIP-master/src/vip_hci/invprob/paco.py:1304',
                        'impact': 'Alto - Resultados numéricos incorrectos',
                        'fix': 'Usar: S = (1.0/T) * np.dot((r - m).T, (r - m))'
                    })
                    print(f"[ERROR] La implementacion es incorrecta")
                    print(f"  Diferencia máxima: {np.max(np.abs(S_current - S_correct)):.6e}")
                    print(f"  La implementación correcta debería usar operaciones vectorizadas")
                else:
                    print(f"[OK] La implementacion parece correcta (aunque ineficiente)")
                    
            except Exception as e:
                self.errors.append({
                    'type': 'ERROR DE EJECUCIÓN',
                    'function': 'sample_covariance()',
                    'description': f'Error al ejecutar: {str(e)}',
                    'location': 'VIP-master/VIP-master/src/vip_hci/invprob/paco.py:1304'
                })
                print(f"[ERROR] Error ejecutando sample_covariance(): {e}")
                
        except ImportError as e:
            self.errors.append({
                'type': 'ERROR DE IMPORTACIÓN',
                'function': 'sample_covariance()',
                'description': f'No se puede importar VIP: {str(e)}',
                'location': 'VIP-master/VIP-master/src/vip_hci/invprob/paco.py'
            })
            print(f"[ERROR] No se puede importar VIP: {e}")
    
    def check_memory_allocation(self):
        """Verifica el problema de pre-asignación de memoria"""
        print("\n" + "="*70)
        print("ANÁLISIS DE PRE-ASIGNACIÓN DE MEMORIA")
        print("="*70)
        
        try:
            sys.path.insert(0, str(Path("VIP-master/VIP-master/src")))
            from vip_hci.invprob.paco import FastPACO
            
            # Leer el código fuente
            paco_file = Path("VIP-master/VIP-master/src/vip_hci/invprob/paco.py")
            if paco_file.exists():
                with open(paco_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Buscar compute_statistics
                if 'def compute_statistics' in content:
                    # Verificar si pre-asigna para todos los píxeles
                    if 'self.height' in content and 'self.width' in content:
                        # Buscar líneas problemáticas
                        lines = content.split('\n')
                        for i, line in enumerate(lines[850:870], start=851):
                            if 'Cinv = np.zeros' in line and 'self.height' in line and 'self.width' in line:
                                self.warnings.append({
                                    'type': 'PROBLEMA DE MEMORIA',
                                    'function': 'compute_statistics()',
                                    'description': 'Pre-asigna memoria para TODOS los píxeles (height x width)',
                                    'location': f'VIP-master/VIP-master/src/vip_hci/invprob/paco.py:{i}',
                                    'impact': 'Alto - Causa problemas de memoria con imágenes grandes',
                                    'fix': 'Pre-asignar solo para los píxeles en phi0s'
                                })
                                print(f"[WARNING] PROBLEMA detectado en linea {i}")
                                print(f"  {line.strip()}")
                                print(f"  Pre-asigna: (height, width, patch_area_pixels, patch_area_pixels)")
                                print(f"  Para una imagen 321x321 con patch_area_pixels=99481:")
                                print(f"  Memoria: 321 × 321 × 99481 × 99481 × 8 bytes = ~7.25 PiB")
                                print(f"  Esto es excesivo y causa errores de memoria")
                                
        except Exception as e:
            print(f"[WARNING] Error analizando memoria: {e}")
    
    def check_matrix_inversion(self):
        """Verifica la falta de regularización en inversión de matrices"""
        print("\n" + "="*70)
        print("ANÁLISIS DE INVERSIÓN DE MATRICES")
        print("="*70)
        
        try:
            paco_file = Path("VIP-master/VIP-master/src/vip_hci/invprob/paco.py")
            if paco_file.exists():
                with open(paco_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Buscar np.linalg.inv
                if 'np.linalg.inv' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines[1250:1253], start=1251):
                        if 'np.linalg.inv' in line:
                            self.warnings.append({
                                'type': 'PROBLEMA NUMÉRICO',
                                'function': 'compute_statistics_at_pixel()',
                                'description': 'Usa np.linalg.inv() sin regularización',
                                'location': f'VIP-master/VIP-master/src/vip_hci/invprob/paco.py:{i}',
                                'impact': 'Medio - Puede fallar con matrices mal condicionadas',
                                'fix': 'Usar scipy.linalg.inv() con regularización diagonal o np.linalg.pinv()'
                            })
                            print(f"[WARNING] PROBLEMA detectado en linea {i}")
                            print(f"  {line.strip()}")
                            print(f"  No hay regularización, puede fallar con matrices singulares")
                            
        except Exception as e:
            print(f"[WARNING] Error analizando inversion: {e}")
    
    def check_vectorization(self):
        """Verifica la falta de vectorización en sample_covariance"""
        print("\n" + "="*70)
        print("ANÁLISIS DE VECTORIZACIÓN")
        print("="*70)
        
        try:
            paco_file = Path("VIP-master/VIP-master/src/vip_hci/invprob/paco.py")
            if paco_file.exists():
                with open(paco_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Buscar list comprehension en sample_covariance
                lines = content.split('\n')
                for i, line in enumerate(lines[1303:1306], start=1304):
                    if 'np.sum([' in line and 'for p in r' in content:
                        self.recommendations.append({
                            'type': 'OPTIMIZACIÓN',
                            'function': 'sample_covariance()',
                            'description': 'Usa list comprehension en lugar de operaciones vectorizadas',
                            'location': f'VIP-master/VIP-master/src/vip_hci/invprob/paco.py:{i}',
                            'impact': 'Alto - Speedup estimado: ~4.9×',
                            'fix': 'Reemplazar con: S = (1.0/T) * np.dot((r - m).T, (r - m))'
                        })
                        print(f"[INFO] OPTIMIZACION recomendada en linea {i}")
                        print(f"  {line.strip()}")
                        print(f"  Usa list comprehension, debería usar operaciones vectorizadas")
                        print(f"  Speedup estimado: ~4.9×")
                        
        except Exception as e:
            print(f"[WARNING] Error analizando vectorizacion: {e}")
    
    def generate_report(self):
        """Genera un reporte completo de errores y recomendaciones"""
        print("\n" + "="*70)
        print("REPORTE COMPLETO DE ERRORES Y RECOMENDACIONES")
        print("="*70)
        
        if self.errors:
            print(f"\n[ERRORES] ERRORES ENCONTRADOS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"\n{i}. {error['type']} - {error['function']}")
                print(f"   Descripción: {error['description']}")
                print(f"   Ubicación: {error['location']}")
                if 'impact' in error:
                    print(f"   Impacto: {error['impact']}")
                if 'fix' in error:
                    print(f"   Solución: {error['fix']}")
        else:
            print("\n[OK] No se encontraron errores criticos")
        
        if self.warnings:
            print(f"\n[ADVERTENCIAS] ADVERTENCIAS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"\n{i}. {warning['type']} - {warning['function']}")
                print(f"   Descripción: {warning['description']}")
                print(f"   Ubicación: {warning['location']}")
                if 'impact' in warning:
                    print(f"   Impacto: {warning['impact']}")
                if 'fix' in warning:
                    print(f"   Solución: {warning['fix']}")
        else:
            print("\n[OK] No se encontraron advertencias")
        
        if self.recommendations:
            print(f"\n[RECOMENDACIONES] RECOMENDACIONES DE OPTIMIZACION ({len(self.recommendations)}):")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"\n{i}. {rec['type']} - {rec['function']}")
                print(f"   Descripción: {rec['description']}")
                print(f"   Ubicación: {rec['location']}")
                if 'impact' in rec:
                    print(f"   Impacto: {rec['impact']}")
                if 'fix' in rec:
                    print(f"   Solución: {rec['fix']}")
        else:
            print("\n[OK] No hay recomendaciones de optimizacion")
        
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'recommendations': self.recommendations
        }


class PACOProfiler:
    """Profiler completo para PACO"""
    
    def __init__(self):
        self.profile_results = {}
    
    def profile_function(self, func, *args, **kwargs):
        """Profila una función usando cProfile"""
        if not PROFILING_AVAILABLE:
            return None
        
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 funciones
        
        return {
            'result': result,
            'profile': s.getvalue()
        }
    
    def time_function(self, func, *args, **kwargs):
        """Mide el tiempo de ejecución de una función"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed


def load_dataset_from_github(repo_url, file_path, local_path=None, branch='main'):
    """
    Carga un dataset desde GitHub
    
    Parameters
    ----------
    repo_url : str
        URL del repositorio de GitHub (sin .git)
    file_path : str
        Ruta del archivo en el repositorio
    local_path : str, optional
        Ruta local donde guardar el archivo
    branch : str, optional
        Rama del repositorio (default: 'main')
    
    Returns
    -------
    data : np.ndarray
        Datos cargados
    """
    import urllib.request
    from astropy.io import fits
    
    if local_path is None:
        local_path = Path("datasets") / Path(file_path).name
    
    # Crear directorio si no existe
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Si el archivo ya existe localmente, usarlo
    if Path(local_path).exists():
        print(f"[INFO] Usando archivo local existente: {local_path}")
        try:
            if local_path.suffix == '.fits':
                data = fits.getdata(local_path)
                print(f"[OK] Datos cargados: shape={data.shape}")
                return data
            else:
                data = np.loadtxt(local_path)
                print(f"[OK] Datos cargados: shape={data.shape}")
                return data
        except Exception as e:
            print(f"[WARNING] Error leyendo archivo local, descargando de nuevo: {e}")
    
    # Convertir URL de GitHub a raw URL
    if 'github.com' in repo_url:
        # Remover .git si está presente
        repo_url = repo_url.replace('.git', '')
        # Convertir a raw URL
        if '/tree/' in repo_url:
            repo_url = repo_url.replace('/tree/', '/raw/')
        elif '/blob/' in repo_url:
            repo_url = repo_url.replace('/blob/', '/raw/')
        else:
            # Si no tiene /tree/ o /blob/, agregar /raw/branch/
            repo_url = repo_url.rstrip('/') + f'/raw/{branch}'
        
        file_url = f"{repo_url}/{file_path}"
    else:
        file_url = f"{repo_url}/{file_path}"
    
    print(f"Descargando desde: {file_url}")
    print(f"Guardando en: {local_path}")
    
    try:
        # Descargar archivo
        urllib.request.urlretrieve(file_url, local_path)
        print(f"[OK] Archivo descargado correctamente")
        
        # Cargar datos
        if local_path.suffix == '.fits':
            data = fits.getdata(local_path)
            print(f"[OK] Datos cargados: shape={data.shape}")
            return data
        else:
            data = np.loadtxt(local_path)
            print(f"[OK] Datos cargados: shape={data.shape}")
            return data
            
    except Exception as e:
        print(f"[ERROR] Error descargando/cargando datos: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_challenge_dataset(repo_url, instrument, dataset_id, local_dir=None, branch='main'):
    """
    Carga un dataset completo del Exoplanet Imaging Data Challenge desde GitHub
    
    Parameters
    ----------
    repo_url : str
        URL del repositorio de GitHub (ej: 'https://github.com/Waofin/tesis')
    instrument : str
        Instrumento ('sphere_irdis', 'nirc2', 'lmircam')
    dataset_id : int
        ID del dataset (1, 2, 3, etc.)
    local_dir : str, optional
        Directorio local donde guardar los archivos
    branch : str, optional
        Rama del repositorio (default: 'main')
    
    Returns
    -------
    cube : np.ndarray
        Cube de datos (time, y, x)
    angles : np.ndarray
        Ángulos de paralaje
    psf : np.ndarray
        PSF
    pixscale : float
        Pixel scale en arcsec/pixel
    """
    if local_dir is None:
        local_dir = Path("datasets") / "subchallenge1"
    else:
        local_dir = Path(local_dir)
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"CARGANDO DATASET DEL CHALLENGE")
    print("="*70)
    print(f"Instrumento: {instrument}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Repositorio: {repo_url}")
    print("="*70)
    
    # Nombres de archivos según la convención del challenge
    base_name = f"{instrument}_{dataset_id}"
    cube_file = f"{base_name}_cube.fits"
    pa_file = f"{base_name}_pa.fits"
    psf_file = f"{base_name}_psf.fits"
    pxscale_file = f"{base_name}_pxscale.fits"
    
    # Ruta en el repositorio
    repo_path = f"subchallenge1"
    
    # Cargar cube
    print(f"\n1. Cargando cube: {cube_file}")
    cube_path = local_dir / cube_file
    cube = load_dataset_from_github(
        repo_url, 
        f"{repo_path}/{cube_file}",
        cube_path,
        branch=branch
    )
    if cube is None:
        raise ValueError(f"No se pudo cargar el cube: {cube_file}")
    print(f"   [OK] Cube shape: {cube.shape}")
    
    # Cargar ángulos
    print(f"\n2. Cargando ángulos: {pa_file}")
    pa_path = local_dir / pa_file
    pa_data = load_dataset_from_github(
        repo_url,
        f"{repo_path}/{pa_file}",
        pa_path,
        branch=branch
    )
    if pa_data is None:
        raise ValueError(f"No se pudo cargar los ángulos: {pa_file}")
    
    # Los ángulos pueden venir como array 1D o 2D
    if pa_data.ndim == 1:
        angles = pa_data
    else:
        angles = pa_data.flatten()
    
    print(f"   [OK] Ángulos shape: {angles.shape}")
    
    # Cargar PSF
    print(f"\n3. Cargando PSF: {psf_file}")
    psf_path = local_dir / psf_file
    psf = load_dataset_from_github(
        repo_url,
        f"{repo_path}/{psf_file}",
        psf_path,
        branch=branch
    )
    if psf is None:
        raise ValueError(f"No se pudo cargar el PSF: {psf_file}")
    print(f"   [OK] PSF shape: {psf.shape}")
    
    # Cargar pixel scale
    print(f"\n4. Cargando pixel scale: {pxscale_file}")
    pxscale_path = local_dir / pxscale_file
    pixscale_data = load_dataset_from_github(
        repo_url,
        f"{repo_path}/{pxscale_file}",
        pxscale_path,
        branch=branch
    )
    if pixscale_data is None:
        print(f"   [WARNING] No se pudo cargar pixel scale, usando valor por defecto: 0.027")
        pixscale = 0.027
    else:
        # Pixel scale puede venir como array o float
        if isinstance(pixscale_data, np.ndarray):
            pixscale = float(pixscale_data.flatten()[0])
        else:
            pixscale = float(pixscale_data)
        print(f"   [OK] Pixel scale: {pixscale} arcsec/pixel")
    
    print("\n" + "="*70)
    print("[OK] DATASET COMPLETO CARGADO")
    print("="*70)
    print(f"  Cube: {cube.shape}")
    print(f"  Ángulos: {len(angles)} frames")
    print(f"  PSF: {psf.shape}")
    print(f"  Pixel scale: {pixscale} arcsec/pixel")
    print("="*70)
    
    return cube, angles, psf, pixscale


def list_available_datasets(repo_url, branch='main'):
    """
    Lista los datasets disponibles en subchallenge1
    
    Parameters
    ----------
    repo_url : str
        URL del repositorio de GitHub
    branch : str
        Rama del repositorio
    
    Returns
    -------
    datasets : list
        Lista de tuplas (instrument, dataset_id)
    """
    # Instrumentos conocidos del challenge
    instruments = ['sphere_irdis', 'nirc2', 'lmircam']
    dataset_ids = [1, 2, 3]
    
    available = []
    for instrument in instruments:
        for dataset_id in dataset_ids:
            base_name = f"{instrument}_{dataset_id}"
            cube_file = f"{base_name}_cube.fits"
            repo_path = f"subchallenge1/{cube_file}"
            
            # Intentar verificar si existe (esto requeriría hacer una petición HTTP)
            # Por ahora, retornamos todos los posibles
            available.append((instrument, dataset_id))
    
    return available


def benchmark_paco(cube, angles, psf, pixscale, fwhm_arcsec, phi0s=None, verbose=True):
    """
    Ejecuta benchmark completo de PACO
    
    Parameters
    ----------
    cube : np.ndarray
        Cube de datos (time, y, x)
    angles : np.ndarray
        Ángulos de derotación
    psf : np.ndarray
        PSF no saturada
    pixscale : float
        Pixel scale en arcsec/pixel
    fwhm_arcsec : float
        FWHM en arcseconds
    phi0s : np.ndarray, optional
        Coordenadas de píxeles a procesar
    verbose : bool
        Mostrar información detallada
    
    Returns
    -------
    results : dict
        Resultados del benchmark
    """
    results = {
        'success': False,
        'time': None,
        'n_pixels': 0,
        'memory_peak': None,
        'errors': []
    }
    
    try:
        sys.path.insert(0, str(Path("VIP-master/VIP-master/src")))
        from vip_hci.invprob.paco import FastPACO
        
        # Crear instancia
        if verbose:
            print(f"\nCreando instancia de FastPACO...")
            print(f"  Cube shape: {cube.shape}")
            print(f"  Ángulos: {len(angles)}")
            print(f"  PSF shape: {psf.shape}")
            print(f"  FWHM: {fwhm_arcsec} arcsec")
            print(f"  Pixel scale: {pixscale} arcsec/pixel")
        
        # Verificar tamaño del cube
        if cube.shape[1] > 100 or cube.shape[2] > 100:
            print(f"[WARNING] ADVERTENCIA: Cube grande ({cube.shape[1]}x{cube.shape[2]})")
            print(f"  VIP puede tener problemas de memoria")
            print(f"  Considera recortar el cube o usar menos píxeles")
            
            # Recortar automáticamente si es muy grande
            if cube.shape[1] > 200 or cube.shape[2] > 200:
                print(f"  Recortando cube a 100x100...")
                center_y, center_x = cube.shape[1] // 2, cube.shape[2] // 2
                half_size = 50
                cube = cube[:, 
                           center_y - half_size:center_y + half_size,
                           center_x - half_size:center_x + half_size]
                print(f"  Nuevo shape: {cube.shape}")
        
        vip_paco = FastPACO(
            cube=cube,
            angles=angles,
            psf=psf,
            fwhm=fwhm_arcsec,
            pixscale=pixscale,
            verbose=verbose
        )
        
        # Generar coordenadas si no se proporcionan
        if phi0s is None:
            img_size = min(cube.shape[1], cube.shape[2])
            center = img_size // 2
            test_radius = min(5, center - 5)
            
            y_coords, x_coords = np.meshgrid(
                np.arange(center - test_radius, center + test_radius),
                np.arange(center - test_radius, center + test_radius)
            )
            phi0s = np.column_stack((x_coords.flatten(), y_coords.flatten()))
        
        results['n_pixels'] = len(phi0s)
        
        if verbose:
            print(f"\nEjecutando PACOCalc con {len(phi0s)} píxeles...")
        
        # Medir tiempo
        profiler = PACOProfiler()
        start_time = time.time()
        
        try:
            a, b = vip_paco.PACOCalc(phi0s, use_subpixel_psf_astrometry=False, cpu=1)
            elapsed = time.time() - start_time
            
            results['success'] = True
            results['time'] = elapsed
            
            # Calcular SNR
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.divide(b, np.sqrt(a), out=np.zeros_like(b), where=(a > 0))
                snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
            
            results['snr_max'] = float(np.nanmax(snr))
            results['snr_mean'] = float(np.nanmean(snr))
            results['snr_std'] = float(np.nanstd(snr))
            
            if verbose:
                print(f"[OK] Ejecucion exitosa")
                print(f"  Tiempo: {elapsed:.2f} segundos")
                print(f"  SNR máximo: {results['snr_max']:.2f}")
                print(f"  SNR medio: {results['snr_mean']:.2f}")
                
        except MemoryError as e:
            results['errors'].append(f"Error de memoria: {str(e)}")
            print(f"[ERROR] Error de memoria: {e}")
            print(f"  Esto es un problema conocido de VIP")
            print(f"  VIP pre-asigna memoria para TODOS los píxeles")
            
        except Exception as e:
            results['errors'].append(f"Error de ejecución: {str(e)}")
            print(f"[ERROR] Error ejecutando PACO: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        results['errors'].append(f"Error de importación: {str(e)}")
        print(f"[ERROR] No se puede importar VIP: {e}")
        print(f"  Asegúrate de que VIP-master está en el path correcto")
        
    return results


def main():
    """Función principal"""
    print("="*70)
    print("BENCHMARK Y PROFILING COMPLETO DE PACO EN VIP-master")
    print("="*70)
    
    # 1. Detectar errores
    print("\n" + "="*70)
    print("PASO 1: DETECCIÓN DE ERRORES")
    print("="*70)
    
    detector = PACOErrorDetector()
    detector.check_sample_covariance()
    detector.check_memory_allocation()
    detector.check_matrix_inversion()
    detector.check_vectorization()
    report = detector.generate_report()
    
    # 2. Benchmark con datos sintéticos
    print("\n" + "="*70)
    print("PASO 2: BENCHMARK CON DATOS SINTÉTICOS")
    print("="*70)
    
    # Generar datos sintéticos pequeños para prueba rápida
    n_frames = 20
    img_size = 64
    cube = np.random.randn(n_frames, img_size, img_size) * 0.1
    angles = np.linspace(0, 90, n_frames)
    
    # PSF gaussiano
    psf_size = 21
    center = psf_size // 2
    y, x = np.ogrid[:psf_size, :psf_size]
    psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.0**2))
    psf = psf / np.sum(psf)
    
    pixscale = 0.027  # arcsec/pixel
    fwhm_arcsec = 4.0 * pixscale
    
    results_synthetic = benchmark_paco(
        cube, angles, psf, pixscale, fwhm_arcsec, 
        verbose=True
    )
    
    # 3. Cargar datos desde GitHub (subchallenge1)
    print("\n" + "="*70)
    print("PASO 3: CARGA DE DATOS DESDE GITHUB (subchallenge1)")
    print("="*70)
    
    repo_url = "https://github.com/Waofin/tesis"
    
    print(f"\nRepositorio: {repo_url}")
    print("Carpeta: subchallenge1")
    print("\nDatasets disponibles:")
    print("  - sphere_irdis: datasets 1, 2, 3")
    print("  - nirc2: datasets 1, 2, 3")
    print("  - lmircam: datasets 1, 2, 3")
    
    # Intentar cargar un dataset de ejemplo
    print("\n" + "="*70)
    print("CARGANDO DATASET DE EJEMPLO (sphere_irdis_1)")
    print("="*70)
    
    try:
        cube_challenge, angles_challenge, psf_challenge, pixscale_challenge = load_challenge_dataset(
            repo_url=repo_url,
            instrument='sphere_irdis',
            dataset_id=1,
            branch='main'
        )
        
        print("\n[OK] Dataset cargado correctamente")
        print(f"  Ejecutando benchmark con datos reales...")
        
        # Ejecutar benchmark con datos reales
        fwhm_arcsec = 4.0 * pixscale_challenge
        results_challenge = benchmark_paco(
            cube_challenge, 
            angles_challenge, 
            psf_challenge, 
            pixscale_challenge, 
            fwhm_arcsec,
            verbose=True
        )
        
        if results_challenge['success']:
            print(f"\n[OK] Benchmark con datos reales completado")
            print(f"  Tiempo: {results_challenge['time']:.2f}s")
            print(f"  Píxeles procesados: {results_challenge['n_pixels']}")
            print(f"  SNR máximo: {results_challenge['snr_max']:.2f}")
        else:
            print(f"\n[ERROR] Benchmark con datos reales falló")
            for error in results_challenge['errors']:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"\n[WARNING] No se pudo cargar dataset de ejemplo: {e}")
        print("  Esto puede ser porque:")
        print("  - El repositorio no es público")
        print("  - Los archivos no existen en esa ubicación")
        print("  - Problemas de conexión")
        print("\n  Para cargar manualmente, usa:")
        print("  cube, angles, psf, pixscale = load_challenge_dataset(")
        print("      repo_url='https://github.com/Waofin/tesis',")
        print("      instrument='sphere_irdis',")
        print("      dataset_id=1")
        print("  )")
    
    # 4. Resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    
    print(f"\n[OK] Analisis de errores completado")
    print(f"  - Errores encontrados: {len(report['errors'])}")
    print(f"  - Advertencias: {len(report['warnings'])}")
    print(f"  - Recomendaciones: {len(report['recommendations'])}")
    
    if results_synthetic['success']:
        print(f"\n[OK] Benchmark completado")
        print(f"  - Tiempo de ejecución: {results_synthetic['time']:.2f}s")
        print(f"  - Píxeles procesados: {results_synthetic['n_pixels']}")
        print(f"  - SNR máximo: {results_synthetic['snr_max']:.2f}")
    else:
        print(f"\n[ERROR] Benchmark fallo")
        for error in results_synthetic['errors']:
            print(f"  - {error}")
    
    print("\n" + "="*70)
    print("FIN DEL BENCHMARK")
    print("="*70)


if __name__ == "__main__":
    main()

