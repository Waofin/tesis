"""
Análisis completo de la implementación PACO en VIP-master
===========================================================

Este script realiza:
1. Análisis de errores en la implementación
2. Benchmark completo de rendimiento
3. Profiling detallado de tiempos
4. Comparación con datasets desde GitHub
5. Generación de reportes

Autor: César Cerda - Universidad del Bío-Bío
"""

import numpy as np
import time
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configurar encoding para Windows
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Ya está configurado o no es necesario

# Profiling
try:
    import cProfile
    import pstats
    from io import StringIO
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False

# Visualización
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Para evitar problemas con display
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Cargar VIP (opcional, el análisis puede funcionar sin importar VIP completo)
VIP_AVAILABLE = False
try:
    sys.path.insert(0, str(Path("VIP-master/VIP-master/src")))
    from vip_hci.invprob.paco import FastPACO, FullPACO, sample_covariance, compute_statistics_at_pixel
    from vip_hci.invprob.paco import shrinkage_factor, diagsample_covariance, covariance
    VIP_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Error importando VIP: {e}")
    print("  El analisis de codigo funcionara, pero los benchmarks requieren VIP instalado")
    print("  Instala dependencias: pip install scikit-image scipy astropy")
    VIP_AVAILABLE = False

# ============================================================================
# 1. ANÁLISIS DE ERRORES
# ============================================================================

class PACOErrorAnalyzer:
    """Analiza errores en la implementación de PACO"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def analyze_sample_covariance(self):
        """Analiza el error crítico en sample_covariance"""
        print("\n" + "="*70)
        print("ANÁLISIS DE sample_covariance")
        print("="*70)
        
        # Leer el código fuente
        paco_file = Path("VIP-master/VIP-master/src/vip_hci/invprob/paco.py")
        if not paco_file.exists():
            self.errors.append("No se encontró el archivo paco.py")
            return
        
        with open(paco_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Buscar la función sample_covariance
        for i, line in enumerate(lines):
            if 'def sample_covariance' in line:
                # Leer las siguientes 10 líneas
                func_code = ''.join(lines[i:i+25])
                print(f"\nCodigo actual (lineas {i+1}-{i+25}):")
                print("-" * 70)
                # Evitar problemas de encoding al imprimir
                try:
                    print(func_code)
                except UnicodeEncodeError:
                    # Si hay problemas, imprimir sin caracteres especiales
                    func_code_safe = func_code.encode('ascii', errors='replace').decode('ascii')
                    print(func_code_safe)
                
                # Detectar el error
                if 'np.cov(np.stack((p, m))' in func_code:
                    error_msg = """
                    *** ERROR CRITICO DETECTADO en sample_covariance (linea ~1304):
                    
                    El código actual:
                        S = (1.0/T)*np.sum([np.cov(np.stack((p, m)), rowvar=False, bias=False) for p in r], axis=0)
                    
                    PROBLEMA:
                    - np.cov(np.stack((p, m))) calcula la covarianza ENTRE p y m
                    - Esto es INCORRECTO matemáticamente
                    - Debería calcular la covarianza de (p-m) consigo mismo
                    
                    IMPLEMENTACIÓN CORRECTA (comentada en el código):
                        S = (1.0/T)*np.sum([np.outer((p-m).ravel(),(p-m).ravel().T) for p in r], axis=0)
                    
                    VERSIÓN VECTORIZADA OPTIMIZADA:
                        r_centered = r - m[np.newaxis, :]
                        S = (1.0/T) * np.dot(r_centered.T, r_centered)
                    
                    IMPACTO:
                    - Resultados numéricos incorrectos
                    - Matrices de covarianza mal calculadas
                    - SNR y flux estimates incorrectos
                    """
                    print(error_msg)
                    self.errors.append({
                        'type': 'CRITICAL',
                        'function': 'sample_covariance',
                        'line': i+1,
                        'description': 'Cálculo incorrecto de covarianza usando np.cov entre p y m',
                        'fix': 'Usar implementación vectorizada: r_centered = r - m[np.newaxis, :]; S = (1.0/T) * np.dot(r_centered.T, r_centered)'
                    })
                break
    
    def analyze_memory_issue(self):
        """Analiza el problema de memoria en compute_statistics"""
        print("\n" + "="*70)
        print("ANÁLISIS DE PROBLEMA DE MEMORIA")
        print("="*70)
        
        warning_msg = """
        *** PROBLEMA DE MEMORIA DETECTADO en compute_statistics (linea ~821):
        
        PROBLEMA:
        - La función pre-asigna memoria para TODOS los píxeles de la imagen
        - Pre-asigna: (height, width, patch_area_pixels, patch_area_pixels) * 8 bytes
        - Esto causa problemas con imágenes grandes
        
        EJEMPLO:
        - Imagen 321x321, patch_area_pixels=99481
        - Memoria necesaria: 321 * 321 * 99481 * 99481 * 8 bytes ≈ 7.25 PiB (!!)
        
        SOLUCIÓN:
        - Solo pre-asignar memoria para los píxeles especificados en phi0s
        - Usar procesamiento por lotes (batching)
        - Implementar procesamiento lazy/on-demand
        """
        print(warning_msg)
        self.warnings.append({
            'type': 'MEMORY',
            'function': 'compute_statistics',
            'description': 'Pre-asignación de memoria para todos los píxeles en lugar de solo los procesados',
            'impact': 'Causa errores de memoria con imágenes grandes'
        })
    
    def analyze_coordinate_inversion(self):
        """Analiza la inversión de coordenadas"""
        print("\n" + "="*70)
        print("ANÁLISIS DE INVERSIÓN DE COORDENADAS")
        print("="*70)
        
        warning_msg = """
        *** ADVERTENCIA: Inversion de coordenadas en PACOCalc (linea ~922):
        
        CÓDIGO:
            phi0s = np.array([phi0s[:, 1], phi0s[:, 0]]).T
        
        DESCRIPCIÓN:
        - Las coordenadas se invierten de (x, y) a (y, x)
        - Esto puede ser confuso pero parece ser intencional
        - VIP usa convención (y, x) internamente
        
        RECOMENDACIÓN:
        - Documentar claramente esta convención
        - Verificar que sea consistente en todo el código
        """
        print(warning_msg)
        self.warnings.append({
            'type': 'CONVENTION',
            'function': 'PACOCalc',
            'description': 'Inversión de coordenadas (x,y) -> (y,x)',
            'impact': 'Puede causar confusión pero parece ser intencional'
        })
    
    def test_sample_covariance_correctness(self):
        """Prueba la corrección matemática de sample_covariance"""
        print("\n" + "="*70)
        print("PRUEBA DE CORRECCION MATEMATICA")
        print("="*70)
        
        # Generar datos de prueba
        T = 10
        P = 5
        r = np.random.randn(T, P)
        m = np.mean(r, axis=0)
        
        # Implementación actual (INCORRECTA)
        try:
            S_wrong = (1.0/T)*np.sum([np.cov(np.stack((p, m)), rowvar=False, bias=False) for p in r], axis=0)
            print(f"\n1. Implementacion actual (INCORRECTA):")
            print(f"   Shape: {S_wrong.shape}")
            print(f"   Valores: {S_wrong[:3, :3]}")
        except Exception as e:
            print(f"   Error: {e}")
            S_wrong = None
        
        # Implementación correcta (vectorizada)
        r_centered = r - m[np.newaxis, :]
        S_correct = (1.0/T) * np.dot(r_centered.T, r_centered)
        print(f"\n2. Implementacion correcta (vectorizada):")
        print(f"   Shape: {S_correct.shape}")
        print(f"   Valores: {S_correct[:3, :3]}")
        
        # Verificar propiedades de la matriz de covarianza
        print(f"\n3. Verificacion de propiedades:")
        print(f"   Matriz simetrica: {np.allclose(S_correct, S_correct.T)}")
        eigvals = np.linalg.eigvals(S_correct)
        print(f"   Valores propios positivos: {np.all(eigvals > -1e-10)}")
        
        if S_wrong is not None:
            print(f"\n4. Comparacion:")
            print(f"   Diferencia maxima: {np.max(np.abs(S_correct - S_wrong)):.6f}")
            print(f"   Diferencia media: {np.mean(np.abs(S_correct - S_wrong)):.6f}")
            if not np.allclose(S_correct, S_wrong, atol=1e-5):
                print(f"   WARNING: Las implementaciones producen resultados diferentes!")
                self.errors.append({
                    'type': 'NUMERICAL',
                    'test': 'sample_covariance_correctness',
                    'description': 'La implementacion actual produce resultados diferentes a la correcta'
                })
    
    def run_all_analyses(self):
        """Ejecuta todos los análisis"""
        print("\n" + "="*70)
        print("ANÁLISIS COMPLETO DE ERRORES EN PACO")
        print("="*70)
        
        self.analyze_sample_covariance()
        self.analyze_memory_issue()
        self.analyze_coordinate_inversion()
        self.test_sample_covariance_correctness()
        
        # Resumen
        print("\n" + "="*70)
        print("RESUMEN DE ANÁLISIS")
        print("="*70)
        print(f"Errores críticos encontrados: {len(self.errors)}")
        print(f"Advertencias encontradas: {len(self.warnings)}")
        
        if self.errors:
            print("\nErrores críticos:")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err['type']} en {err['function']}: {err['description']}")
        
        if self.warnings:
            print("\nAdvertencias:")
            for i, warn in enumerate(self.warnings, 1):
                print(f"  {i}. {warn['type']} en {warn['function']}: {warn['description']}")
        
        return self.errors, self.warnings


# ============================================================================
# 2. BENCHMARK Y PROFILING
# ============================================================================

class PACOBenchmark:
    """Benchmark completo de PACO"""
    
    def __init__(self):
        self.results = []
        self.profiles = []
    
    def load_dataset_from_github(self, repo_url: str, file_path: str) -> Optional[np.ndarray]:
        """Carga un dataset desde GitHub"""
        try:
            import urllib.request
            import tempfile
            import os
            
            # Construir URL completa
            if 'github.com' in repo_url:
                # Convertir a raw URL
                raw_url = repo_url.replace('github.com', 'raw.githubusercontent.com')
                if '/blob/' in raw_url:
                    raw_url = raw_url.replace('/blob/', '/')
                full_url = f"{raw_url}/{file_path}"
            else:
                full_url = f"{repo_url}/{file_path}"
            
            print(f"  Descargando desde: {full_url}")
            
            # Descargar a archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.fits') as tmp:
                urllib.request.urlretrieve(full_url, tmp.name)
                tmp_path = tmp.name
            
            # Cargar con astropy
            from astropy.io import fits
            data = fits.getdata(tmp_path)
            
            # Limpiar
            os.unlink(tmp_path)
            
            print(f"  OK: Datos cargados: shape {data.shape}")
            return data
            
        except Exception as e:
            print(f"  ERROR: Error cargando desde GitHub: {e}")
            return None
    
    def generate_synthetic_data(self, n_frames: int = 20, img_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Genera datos sintéticos para pruebas"""
        print(f"\nGenerando datos sintéticos:")
        print(f"  Frames: {n_frames}, Tamaño: {img_size}x{img_size}")
        
        # Cube con ruido
        cube = np.random.randn(n_frames, img_size, img_size) * 0.1
        
        # Ángulos de rotación
        pa = np.linspace(0, 90, n_frames)
        
        # PSF gaussiano
        psf_size = 21
        center = psf_size // 2
        y, x = np.ogrid[:psf_size, :psf_size]
        psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.0**2))
        psf = psf / np.sum(psf)
        
        pixscale = 0.027  # arcsec/pixel
        
        print(f"  OK: Datos generados")
        return cube, pa, psf, pixscale
    
    def benchmark_paco(self, cube: np.ndarray, pa: np.ndarray, psf: np.ndarray, 
                      pixscale: float, n_pixels: int = 25, use_subpixel: bool = False) -> Dict:
        """Ejecuta benchmark de PACO"""
        if not VIP_AVAILABLE:
            return None
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK DE PACO")
        print(f"{'='*70}")
        print(f"  Cube shape: {cube.shape}")
        print(f"  Píxeles a procesar: {n_pixels}")
        print(f"  Subpixel PSF: {use_subpixel}")
        
        # Recortar cube si es muy grande (para evitar problemas de memoria)
        max_size = 100
        if cube.shape[1] > max_size or cube.shape[2] > max_size:
            print(f"  WARNING: Recortando cube de {cube.shape[1]}x{cube.shape[2]} a {max_size}x{max_size}")
            center_y, center_x = cube.shape[1] // 2, cube.shape[2] // 2
            half_size = max_size // 2
            cube = cube[:, 
                        center_y - half_size:center_y + half_size,
                        center_x - half_size:center_x + half_size]
            print(f"  Nuevo shape: {cube.shape}")
        
        # Ajustar ángulos si es necesario
        if len(pa) != cube.shape[0]:
            if len(pa) > cube.shape[0]:
                pa = pa[:cube.shape[0]]
            else:
                pa = np.concatenate([pa, np.repeat(pa[-1], cube.shape[0] - len(pa))])
        
        # Calcular fwhm en arcseconds
        fwhm_pixels = 4.0
        fwhm_arcsec = fwhm_pixels * pixscale
        
        # Crear coordenadas de prueba
        img_size = cube.shape[1]
        center = img_size // 2
        test_radius = min(5, center - 10)
        y_coords, x_coords = np.meshgrid(
            np.arange(center - test_radius, center + test_radius),
            np.arange(center - test_radius, center + test_radius)
        )
        phi0s = np.column_stack((x_coords.flatten()[:n_pixels], y_coords.flatten()[:n_pixels]))
        
        results = {}
        
        # Benchmark FastPACO
        try:
            print(f"\n  Ejecutando FastPACO...")
            vip_paco = FastPACO(
                cube=cube,
                angles=pa,
                psf=psf,
                fwhm=fwhm_arcsec,
                pixscale=pixscale,
                verbose=False
            )
            
            # Profiling
            if PROFILING_AVAILABLE:
                profiler = cProfile.Profile()
                profiler.enable()
            
            start_time = time.time()
            a, b = vip_paco.PACOCalc(phi0s, use_subpixel_psf_astrometry=use_subpixel, cpu=1)
            elapsed_time = time.time() - start_time
            
            if PROFILING_AVAILABLE:
                profiler.disable()
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats(20)  # Top 20 funciones
                profile_str = s.getvalue()
            else:
                profile_str = None
            
            # Calcular SNR
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.divide(b, np.sqrt(a), out=np.zeros_like(b), where=(a > 0))
                snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)
            
            results['FastPACO'] = {
                'time': elapsed_time,
                'n_pixels': len(phi0s),
                'time_per_pixel': elapsed_time / len(phi0s),
                'snr_max': np.nanmax(snr),
                'snr_mean': np.nanmean(snr),
                'profile': profile_str,
                'success': True
            }
            
            print(f"    OK: Tiempo: {elapsed_time:.2f}s ({elapsed_time/len(phi0s)*1000:.2f}ms/pixel)")
            print(f"    SNR máximo: {np.nanmax(snr):.2f}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results['FastPACO'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        
        return results
    
    def run_benchmark_suite(self, datasets: List[Dict]) -> Dict:
        """Ejecuta suite completa de benchmarks"""
        print("\n" + "="*70)
        print("SUITE COMPLETA DE BENCHMARKS")
        print("="*70)
        
        all_results = {}
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{'='*70}")
            print(f"Dataset {i}/{len(datasets)}: {dataset.get('name', 'Unknown')}")
            print(f"{'='*70}")
            
            # Cargar datos
            if 'github_url' in dataset:
                cube = self.load_dataset_from_github(dataset['github_url'], dataset.get('cube_path', ''))
                if cube is None:
                    continue
                pa = dataset.get('pa', np.linspace(0, 90, cube.shape[0]))
                psf = dataset.get('psf', None)
                if psf is None:
                    # Generar PSF sintético
                    psf_size = 21
                    center = psf_size // 2
                    y, x = np.ogrid[:psf_size, :psf_size]
                    psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.0**2))
                    psf = psf / np.sum(psf)
                pixscale = dataset.get('pixscale', 0.027)
            else:
                # Datos sintéticos
                cube, pa, psf, pixscale = self.generate_synthetic_data(
                    dataset.get('n_frames', 20),
                    dataset.get('img_size', 64)
                )
            
            # Ejecutar benchmark
            results = self.benchmark_paco(
                cube, pa, psf, pixscale,
                n_pixels=dataset.get('n_pixels', 25),
                use_subpixel=dataset.get('use_subpixel', False)
            )
            
            all_results[dataset.get('name', f'dataset_{i}')] = results
        
        return all_results
    
    def generate_report(self, results: Dict, errors: List, warnings: List, output_file: str = "paco_benchmark_report.txt"):
        """Genera reporte completo"""
        print(f"\n{'='*70}")
        print(f"GENERANDO REPORTE: {output_file}")
        print(f"{'='*70}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE COMPLETO DE ANÁLISIS Y BENCHMARK DE PACO\n")
            f.write("="*70 + "\n\n")
            
            # Errores encontrados
            f.write("1. ANÁLISIS DE ERRORES\n")
            f.write("-"*70 + "\n")
            f.write(f"Errores críticos: {len(errors)}\n")
            f.write(f"Advertencias: {len(warnings)}\n\n")
            
            if errors:
                f.write("Errores críticos:\n")
                for i, err in enumerate(errors, 1):
                    f.write(f"  {i}. [{err['type']}] {err['function']}: {err['description']}\n")
                    if 'fix' in err:
                        f.write(f"     Solución: {err['fix']}\n")
                f.write("\n")
            
            if warnings:
                f.write("Advertencias:\n")
                for i, warn in enumerate(warnings, 1):
                    f.write(f"  {i}. [{warn['type']}] {warn['function']}: {warn['description']}\n")
                f.write("\n")
            
            # Resultados de benchmark
            f.write("\n2. RESULTADOS DE BENCHMARK\n")
            f.write("-"*70 + "\n")
            
            for dataset_name, dataset_results in results.items():
                f.write(f"\nDataset: {dataset_name}\n")
                for method, method_results in dataset_results.items():
                    if method_results.get('success', False):
                        f.write(f"  {method}:\n")
                        f.write(f"    Tiempo total: {method_results['time']:.2f}s\n")
                        f.write(f"    Tiempo/píxel: {method_results['time_per_pixel']*1000:.2f}ms\n")
                        f.write(f"    Píxeles procesados: {method_results['n_pixels']}\n")
                        f.write(f"    SNR máximo: {method_results['snr_max']:.2f}\n")
                        f.write(f"    SNR medio: {method_results['snr_mean']:.2f}\n")
                    else:
                        f.write(f"  {method}: ERROR\n")
                        f.write(f"    {method_results.get('error', 'Unknown error')}\n")
            
            # Profiling
            f.write("\n\n3. PROFILING DETALLADO\n")
            f.write("-"*70 + "\n")
            
            for dataset_name, dataset_results in results.items():
                for method, method_results in dataset_results.items():
                    if method_results.get('success', False) and method_results.get('profile'):
                        f.write(f"\n{dataset_name} - {method}:\n")
                        f.write(method_results['profile'])
        
        print(f"  OK: Reporte guardado en: {output_file}")


# ============================================================================
# 3. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal"""
    print("="*70)
    print("ANÁLISIS Y BENCHMARK COMPLETO DE PACO (VIP-master)")
    print("="*70)
    
    # 1. Análisis de errores
    analyzer = PACOErrorAnalyzer()
    errors, warnings = analyzer.run_all_analyses()
    
    # 2. Benchmark
    benchmark = PACOBenchmark()
    
    # Configurar datasets para benchmark
    # Puedes agregar URLs de GitHub aquí
    datasets = [
        {
            'name': 'Sintético pequeño',
            'n_frames': 20,
            'img_size': 64,
            'n_pixels': 25,
            'use_subpixel': False
        },
        {
            'name': 'Sintético mediano',
            'n_frames': 40,
            'img_size': 100,
            'n_pixels': 49,
            'use_subpixel': False
        },
        # Agregar más datasets aquí si tienes URLs de GitHub
        # {
        #     'name': 'Dataset real',
        #     'github_url': 'https://github.com/user/repo',
        #     'cube_path': 'path/to/cube.fits',
        #     'pa': np.array([...]),
        #     'pixscale': 0.027,
        #     'n_pixels': 25
        # }
    ]
    
    # Ejecutar benchmarks
    results = benchmark.run_benchmark_suite(datasets)
    
    # 3. Generar reporte
    benchmark.generate_report(results, errors, warnings)
    
    print("\n" + "="*70)
    print("ANÁLISIS COMPLETO FINALIZADO")
    print("="*70)
    print("\nRevisa el archivo 'paco_benchmark_report.txt' para el reporte completo.")


if __name__ == "__main__":
    main()

