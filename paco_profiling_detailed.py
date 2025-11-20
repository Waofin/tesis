"""
Profiling detallado de PACO con análisis de tiempos por función
===============================================================

Este script realiza profiling línea por línea y análisis de tiempos
para identificar cuellos de botella específicos.
"""

import numpy as np
import time
import sys
from pathlib import Path
import traceback

# Profiling avanzado
try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    print("⚠ line_profiler no disponible. Instala con: pip install line_profiler")

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("⚠ memory_profiler no disponible. Instala con: pip install memory_profiler")

try:
    import cProfile
    import pstats
    from io import StringIO
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False

# Cargar VIP
try:
    sys.path.insert(0, str(Path("VIP-master/VIP-master/src")))
    from vip_hci.invprob.paco import FastPACO, sample_covariance, compute_statistics_at_pixel
    VIP_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Error importando VIP: {e}")
    VIP_AVAILABLE = False


class DetailedProfiler:
    """Profiling detallado de funciones específicas"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def profile_sample_covariance(self, n_iterations: int = 100):
        """Profiling detallado de sample_covariance"""
        print("\n" + "="*70)
        print("PROFILING DETALLADO: sample_covariance")
        print("="*70)
        
        # Generar datos de prueba
        T = 40  # Número de frames
        P = 49  # Píxeles en patch (7x7)
        r = np.random.randn(T, P)
        m = np.mean(r, axis=0)
        
        # Implementación actual (INCORRECTA pero para profiling)
        print("\n1. Implementación actual (con error):")
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            S = (1.0/T)*np.sum([np.cov(np.stack((p, m)), rowvar=False, bias=False) for p in r], axis=0)
            times.append(time.perf_counter() - start)
        
        print(f"   Tiempo promedio: {np.mean(times)*1000:.3f}ms")
        print(f"   Tiempo total: {np.sum(times)*1000:.3f}ms")
        print(f"   Desviación estándar: {np.std(times)*1000:.3f}ms")
        
        # Implementación correcta vectorizada
        print("\n2. Implementación correcta (vectorizada):")
        times_correct = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            r_centered = r - m[np.newaxis, :]
            S_correct = (1.0/T) * np.dot(r_centered.T, r_centered)
            times_correct.append(time.perf_counter() - start)
        
        print(f"   Tiempo promedio: {np.mean(times_correct)*1000:.3f}ms")
        print(f"   Tiempo total: {np.sum(times_correct)*1000:.3f}ms")
        print(f"   Desviación estándar: {np.std(times_correct)*1000:.3f}ms")
        
        speedup = np.mean(times) / np.mean(times_correct)
        print(f"\n   Speedup: {speedup:.2f}x más rápido")
        
        self.timings['sample_covariance'] = {
            'current': np.mean(times),
            'correct': np.mean(times_correct),
            'speedup': speedup
        }
    
    def profile_compute_statistics_at_pixel(self, n_iterations: int = 50):
        """Profiling de compute_statistics_at_pixel"""
        print("\n" + "="*70)
        print("PROFILING DETALLADO: compute_statistics_at_pixel")
        print("="*70)
        
        # Generar patch de prueba
        T = 40
        P = 49
        patch = np.random.randn(T, P)
        
        # Descomponer en pasos para medir tiempos
        times_breakdown = {
            'mean': [],
            'sample_cov': [],
            'shrinkage': [],
            'diag_cov': [],
            'covariance': [],
            'inverse': []
        }
        
        print("\nDescomposición de tiempos:")
        for _ in range(n_iterations):
            # Mean
            start = time.perf_counter()
            m = np.mean(patch, axis=0)
            times_breakdown['mean'].append(time.perf_counter() - start)
            
            # Sample covariance
            start = time.perf_counter()
            r_centered = patch - m[np.newaxis, :]
            S = (1.0/T) * np.dot(r_centered.T, r_centered)
            times_breakdown['sample_cov'].append(time.perf_counter() - start)
            
            # Shrinkage factor
            start = time.perf_counter()
            from vip_hci.invprob.paco import shrinkage_factor
            rho = shrinkage_factor(S, T)
            times_breakdown['shrinkage'].append(time.perf_counter() - start)
            
            # Diagonal covariance
            start = time.perf_counter()
            from vip_hci.invprob.paco import diagsample_covariance
            F = diagsample_covariance(S)
            times_breakdown['diag_cov'].append(time.perf_counter() - start)
            
            # Covariance
            start = time.perf_counter()
            from vip_hci.invprob.paco import covariance
            C = covariance(rho, S, F)
            times_breakdown['covariance'].append(time.perf_counter() - start)
            
            # Inverse
            start = time.perf_counter()
            Cinv = np.linalg.inv(C)
            times_breakdown['inverse'].append(time.perf_counter() - start)
        
        # Mostrar resultados
        total_time = 0
        for step, times in times_breakdown.items():
            avg_time = np.mean(times) * 1000  # ms
            total_time += avg_time
            print(f"  {step:15s}: {avg_time:8.3f}ms ({np.sum(times)*1000:8.3f}ms total)")
        
        print(f"\n  {'TOTAL':15s}: {total_time:8.3f}ms")
        
        # Porcentajes
        print("\nDistribución de tiempo:")
        for step, times in times_breakdown.items():
            pct = (np.mean(times) / total_time * 1000) * 100
            print(f"  {step:15s}: {pct:5.1f}%")
        
        self.timings['compute_statistics_at_pixel'] = {
            'breakdown': {k: np.mean(v) for k, v in times_breakdown.items()},
            'total': total_time / 1000
        }
    
    def profile_full_paco_workflow(self, cube_shape: Tuple[int, int, int] = (40, 64, 64), n_pixels: int = 25):
        """Profiling del workflow completo de PACO"""
        print("\n" + "="*70)
        print("PROFILING DETALLADO: Workflow completo FastPACO")
        print("="*70)
        
        if not VIP_AVAILABLE:
            print("  ✗ VIP no disponible")
            return
        
        # Generar datos sintéticos
        n_frames, height, width = cube_shape
        cube = np.random.randn(n_frames, height, width) * 0.1
        pa = np.linspace(0, 90, n_frames)
        
        # PSF
        psf_size = 21
        center = psf_size // 2
        y, x = np.ogrid[:psf_size, :psf_size]
        psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.0**2))
        psf = psf / np.sum(psf)
        
        pixscale = 0.027
        fwhm_arcsec = 4.0 * pixscale
        
        # Coordenadas
        center_y, center_x = height // 2, width // 2
        test_radius = min(5, center_y - 10)
        y_coords, x_coords = np.meshgrid(
            np.arange(center_y - test_radius, center_y + test_radius),
            np.arange(center_x - test_radius, center_x + test_radius)
        )
        phi0s = np.column_stack((x_coords.flatten()[:n_pixels], y_coords.flatten()[:n_pixels]))
        
        # Crear instancia
        vip_paco = FastPACO(
            cube=cube,
            angles=pa,
            psf=psf,
            fwhm=fwhm_arcsec,
            pixscale=pixscale,
            verbose=False
        )
        
        # Medir tiempos de cada paso
        timings = {}
        
        # 1. compute_statistics
        print("\n1. compute_statistics:")
        start = time.perf_counter()
        phi0s_inv = np.array([phi0s[:, 1], phi0s[:, 0]]).T
        Cinv, m, patches = vip_paco.compute_statistics(phi0s_inv)
        timings['compute_statistics'] = time.perf_counter() - start
        print(f"   Tiempo: {timings['compute_statistics']:.3f}s")
        
        # 2. Normalizar PSF
        print("\n2. Normalizar PSF:")
        start = time.perf_counter()
        from vip_hci.fm import normalize_psf
        normalised_psf = normalize_psf(
            vip_paco.psf,
            fwhm='fit',
            size=None,
            threshold=None,
            mask_core=None,
            model='airy',
            imlib='vip-fft',
            interpolation='lanczos4',
            force_odd=False,
            full_output=False,
            verbose=False,
            debug=False
        )
        from vip_hci.invprob.paco import create_boolean_circular_mask
        psf_mask = create_boolean_circular_mask(normalised_psf.shape, radius=vip_paco.fwhm)
        timings['normalize_psf'] = time.perf_counter() - start
        print(f"   Tiempo: {timings['normalize_psf']:.3f}s")
        
        # 3. Loop principal (PACOCalc sin compute_statistics)
        print("\n3. Loop principal (cálculo de a y b):")
        start = time.perf_counter()
        
        dim = vip_paco.width / 2
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        a = np.zeros(len(phi0s))
        b = np.zeros(len(phi0s))
        
        for i, p0 in enumerate(phi0s_inv):
            from vip_hci.invprob.paco import get_rotated_pixel_coords
            angles_px = get_rotated_pixel_coords(x, y, p0, vip_paco.angles)
            
            if (int(np.max(angles_px.flatten())) >= vip_paco.width or
                int(np.min(angles_px.flatten())) < 0):
                a[i] = np.nan
                b[i] = np.nan
                continue
            
            Cinlst = []
            mlst = []
            hlst = []
            patch_list = []
            for l, ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0]), int(ang[1])])
                mlst.append(m[int(ang[0]), int(ang[1])])
                from vip_hci.preproc.rescaling import frame_shift
                offax = normalised_psf[psf_mask]
                hlst.append(offax)
                patch_list.append(patches[int(ang[0]), int(ang[1]), l])
            
            a[i] = vip_paco.al(hlst, Cinlst)
            b[i] = vip_paco.bl(hlst, Cinlst, patch_list, mlst)
        
        timings['main_loop'] = time.perf_counter() - start
        print(f"   Tiempo: {timings['main_loop']:.3f}s")
        
        # Resumen
        total_time = sum(timings.values())
        print(f"\n{'='*70}")
        print("RESUMEN DE TIEMPOS:")
        print(f"{'='*70}")
        for step, t in timings.items():
            pct = (t / total_time) * 100
            print(f"  {step:20s}: {t:8.3f}s ({pct:5.1f}%)")
        print(f"  {'TOTAL':20s}: {total_time:8.3f}s")
        
        self.timings['full_workflow'] = timings
    
    def generate_profiling_report(self, output_file: str = "paco_profiling_report.txt"):
        """Genera reporte de profiling"""
        print(f"\n{'='*70}")
        print(f"GENERANDO REPORTE DE PROFILING: {output_file}")
        print(f"{'='*70}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("REPORTE DE PROFILING DETALLADO DE PACO\n")
            f.write("="*70 + "\n\n")
            
            # sample_covariance
            if 'sample_covariance' in self.timings:
                f.write("1. PROFILING: sample_covariance\n")
                f.write("-"*70 + "\n")
                t = self.timings['sample_covariance']
                f.write(f"Tiempo implementación actual: {t['current']*1000:.3f}ms\n")
                f.write(f"Tiempo implementación correcta: {t['correct']*1000:.3f}ms\n")
                f.write(f"Speedup: {t['speedup']:.2f}x\n\n")
            
            # compute_statistics_at_pixel
            if 'compute_statistics_at_pixel' in self.timings:
                f.write("2. PROFILING: compute_statistics_at_pixel\n")
                f.write("-"*70 + "\n")
                t = self.timings['compute_statistics_at_pixel']
                f.write("Descomposición de tiempos:\n")
                for step, time_ms in t['breakdown'].items():
                    f.write(f"  {step:15s}: {time_ms*1000:8.3f}ms\n")
                f.write(f"\nTiempo total: {t['total']*1000:.3f}ms\n\n")
            
            # Full workflow
            if 'full_workflow' in self.timings:
                f.write("3. PROFILING: Workflow completo\n")
                f.write("-"*70 + "\n")
                t = self.timings['full_workflow']
                total = sum(t.values())
                for step, time_s in t.items():
                    pct = (time_s / total) * 100
                    f.write(f"  {step:20s}: {time_s:8.3f}s ({pct:5.1f}%)\n")
                f.write(f"\n  {'TOTAL':20s}: {total:8.3f}s\n")
        
        print(f"  ✓ Reporte guardado en: {output_file}")


def main():
    """Función principal"""
    print("="*70)
    print("PROFILING DETALLADO DE PACO")
    print("="*70)
    
    profiler = DetailedProfiler()
    
    # 1. Profiling de sample_covariance
    profiler.profile_sample_covariance(n_iterations=100)
    
    # 2. Profiling de compute_statistics_at_pixel
    profiler.profile_compute_statistics_at_pixel(n_iterations=50)
    
    # 3. Profiling del workflow completo
    profiler.profile_full_paco_workflow(cube_shape=(40, 64, 64), n_pixels=25)
    
    # 4. Generar reporte
    profiler.generate_profiling_report()
    
    print("\n" + "="*70)
    print("PROFILING COMPLETO FINALIZADO")
    print("="*70)


if __name__ == "__main__":
    main()

