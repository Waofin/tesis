"""
PACO Performance Benchmark Script

Este script implementa un análisis de rendimiento riguroso del algoritmo PACO
comparando versiones serial, multiprocessing y loky/joblib.

Siguiendo la metodología científica del Exoplanet Imaging Data Challenge Phase 1.

Referencias:
- EIDC Phase 1: https://exoplanet-imaging-challenge.github.io/submission1/
- Flasseur et al. 2018: PACO algorithm
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import sys

# Importaciones de PACO
import paco.processing.fastpaco as fastPACO
import paco.processing.fullpaco as fullPACO
from paco.util.util import *

# Para paralelización con loky
try:
    from joblib import Parallel, delayed
    LOKY_AVAILABLE = True
except ImportError:
    LOKY_AVAILABLE = False
    print("WARNING: joblib/loky no disponible. Instalar con: pip install joblib")


def _pixel_calc_wrapper(patch):
    """
    Wrapper serializable para pixelCalc que puede ser usado con loky.
    Esta función importa pixelCalc dentro para asegurar que esté disponible en workers.
    """
    # Importar dentro de la función para que esté disponible en cada worker
    from paco.util.util import pixelCalc
    return pixelCalc(patch)


def _process_fullpaco_pixel(args):
    """
    Procesa un solo píxel en FullPACO para paralelización con loky.
    Esta función es serializable y puede ser usada con joblib.Parallel.
    
    Parameters
    ----------
    args : tuple
        (p0, image_stack, angles, psf, psf_rad, pwidth, x, y, dim)
    
    Returns
    -------
    tuple
        (a_val, b_val) para el píxel
    """
    p0, image_stack, angles, psf, psf_rad, pwidth, x, y, dim = args
    
    # Importar funciones necesarias dentro para que estén disponibles en workers
    from paco.util.util import (
        createCircularMask, getRotatedPixels, pixelCalc
    )
    import numpy as np
    
    try:
        # Crear máscara
        mask = createCircularMask(psf.shape, radius=psf_rad)
        psf_area = len(mask[mask])
        
        # Obtener píxeles rotados para este punto
        angles_px = getRotatedPixels(x, y, p0, angles)
        
        # Verificar límites básicos
        nx, ny = image_stack.shape[1], image_stack.shape[2]
        if (int(np.max(angles_px.flatten())) >= ny or 
            int(np.min(angles_px.flatten())) < 0):
            return (np.nan, np.nan)
        
        # Arrays para este píxel
        n_frames = len(angles_px)
        patch = np.zeros((n_frames, psf_area))
        m = np.zeros((n_frames, psf_area))
        Cinv = np.zeros((n_frames, psf_area, psf_area))
        h = np.zeros((n_frames, psf_area))
        
        # Implementar lógica de getPatch manualmente
        k = int(pwidth / 2)
        if pwidth % 2 != 0:
            k2 = k + 1
        else:
            k2 = k
        
        # Procesar cada frame/ángulo
        for l, ang in enumerate(angles_px):
            # Verificar límites del patch
            if (int(ang[0]) + k2 > nx or int(ang[0]) - k < 0 or 
                int(ang[1]) + k2 > ny or int(ang[1]) - k < 0):
                return (np.nan, np.nan)
            
            # Extraer patch (implementación manual de getPatch)
            patch_l = np.array([
                image_stack[i][int(ang[0])-k:int(ang[0])+k2, 
                               int(ang[1])-k:int(ang[1])+k2][mask] 
                for i in range(len(image_stack))
            ])
            
            patch[l] = patch_l
            m[l], Cinv[l] = pixelCalc(patch[l])
            h[l] = psf[mask]
        
        # Calcular a y b usando la misma lógica que al() y bl() de PACO
        # a = sum(h * Cinv * h) para cada frame
        # Implementación exacta de al()
        a_vals = [np.dot(h[l], np.dot(Cinv[l], h[l]).T) for l in range(n_frames)]
        a_val = np.sum(np.array(a_vals), axis=0)
        
        # b = sum(Cinv * h * (patch - m)) para cada frame
        # Implementación exacta de bl(): r_fl[i][i] es patch[i] (el patch en frame i)
        b_vals = [np.dot(np.dot(Cinv[l], h[l]).T, (patch[l] - m[l])) for l in range(n_frames)]
        b_val = np.sum(np.array(b_vals), axis=0)
        
        return (a_val, b_val)
        
    except Exception as e:
        # En caso de error, retornar NaN
        return (np.nan, np.nan)


def is_running_in_notebook():
    """
    Detecta si el código se está ejecutando en un notebook (Jupyter/IPython).
    En notebooks, multiprocessing puede tener problemas en Windows.
    """
    try:
        # Detectar Jupyter/IPython
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            # Verificar si es un notebook (no solo IPython interactivo)
            if 'IPKernelApp' in str(type(ipython)):
                return True
            # También verificar variables de entorno comunes de notebooks
            import os
            if 'JPY_PARENT_PID' in os.environ:
                return True
    except:
        pass
    
    # Verificar si está en un entorno interactivo que podría ser notebook
    try:
        import sys
        # En notebooks, __main__ suele tener un nombre específico
        if hasattr(sys, 'ps1'):
            # Modo interactivo
            return True
    except:
        pass
    
    return False


# Detectar si estamos en notebook
IN_NOTEBOOK = is_running_in_notebook()
if IN_NOTEBOOK:
    print("[INFO] Ejecutandose en notebook detectado. Usando solo Loky para evitar problemas de multiprocessing en Windows.")


class FastPACO_Loky(fastPACO.FastPACO):
    """
    Versión de FastPACO que usa loky/joblib para paralelización.
    Mantiene la misma funcionalidad pero con mejor manejo de workers.
    """
    
    def computeStatisticsLoky(self, phi0s, cpu=1, backend='loky', n_jobs=None):
        """
        Computa estadísticas usando loky/joblib para paralelización.
        
        Parameters
        ----------
        phi0s : array
            Array de posiciones de píxeles
        cpu : int
            Número de procesos (compatibilidad con versión original)
        backend : str
            Backend a usar ('loky', 'multiprocessing')
        n_jobs : int, optional
            Número de jobs. Si None, usa cpu.
        
        Returns
        -------
        Cinv : np.ndarray
            Matriz de covarianza inversa
        m : np.ndarray
            Media de los patches
        h : np.ndarray
            Template PSF
        """
        if not LOKY_AVAILABLE:
            raise ImportError("joblib/loky no está disponible")
        
        if n_jobs is None:
            n_jobs = cpu
        
        print(f"Precomputando estadísticas usando {n_jobs} procesos con backend {backend}...")
        
        npx = len(phi0s)
        # CRÍTICO: Crear máscara y actualizar m_psf_area ANTES de crear los arrays
        # IMPORTANTE: La máscara debe crearse usando el tamaño del patch que se extraerá (m_pwidth x m_pwidth)
        # NO usar self.m_psf.shape directamente, porque getPatch extrae un patch de tamaño m_pwidth x m_pwidth
        # y luego aplica la máscara a ese patch
        patch_shape = (self.m_pwidth, self.m_pwidth)  # Tamaño del patch que se extraerá
        mask = createCircularMask(patch_shape, radius=self.m_psf_rad)
        actual_psf_area = len(mask[mask])
        # CRÍTICO: Actualizar m_psf_area ANTES de crear los arrays para evitar errores de broadcasting
        if self.m_psf_area != actual_psf_area:
            self.m_psf_area = actual_psf_area
        
        # Template PSF (igual que en versión original)
        h = np.zeros((self.m_height, self.m_width, self.m_psf_area))
        for p0 in phi0s:
            h[p0[0]][p0[1]] = self.m_psf[mask]
        
        # Preparar argumentos para paralelización
        arglist = [self.getPatch(p0, self.m_pwidth, mask) for p0 in phi0s]
        
        # Usar joblib Parallel con loky
        # Usar _pixel_calc_wrapper que es serializable y importa pixelCalc en cada worker
        if n_jobs == 1:
            # Serial para n_jobs=1
            from paco.util.util import pixelCalc
            data = [pixelCalc(arg) for arg in arglist]
        else:
            # Paralelo con loky
            # Usar la función wrapper que es completamente serializable
            data = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
                delayed(_pixel_calc_wrapper)(arg) for arg in arglist
            )
        
        # Reorganizar resultados
        m_list, Cinv_list = [], []
        for d in data:
            m_list.append(d[0])
            Cinv_list.append(d[1])
        
        m = np.array(m_list).reshape((self.m_height, self.m_width, self.m_psf_area))
        Cinv = np.array(Cinv_list).reshape((self.m_height, self.m_width, 
                                            self.m_psf_area, self.m_psf_area))
        
        return Cinv, m, h
    
    def PACOCalc(self, phi0s, cpu=1, backend='loky', use_loky=True):
        """
        Versión modificada de PACOCalc que permite elegir entre loky y multiprocessing.
        """
        npx = len(phi0s)
        dim = self.m_width/2
        try:
            assert (dim).is_integer()
        except AssertionError:
            print("Image cannot be properly rotated.")
            sys.exit(1)
        dim = int(dim)
        
        a = np.zeros(npx)
        b = np.zeros(npx)
        
        # Elegir método de paralelización
        if cpu == 1:
            Cinv, m, h = self.computeStatistics(phi0s)
        elif use_loky and LOKY_AVAILABLE:
            Cinv, m, h = self.computeStatisticsLoky(phi0s, cpu=cpu, backend=backend)
        else:
            Cinv, m, h = self.computeStatisticsParallel(phi0s, cpu=cpu)
        
        # Resto del código igual que la versión original
        patch = np.zeros((self.m_nFrames, self.m_nFrames, self.m_psf_area))
        mask = createCircularMask(self.m_psf.shape, radius=self.m_psf_rad)
        
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        print("Running PACO...")
        for i, p0 in enumerate(phi0s):
            angles_px = getRotatedPixels(x, y, p0, self.m_angles)
            
            if(int(np.max(angles_px.flatten())) >= self.m_width or 
               int(np.min(angles_px.flatten())) < 0):
                a[i] = np.nan
                b[i] = np.nan
                continue
            
            Cinlst = []
            mlst = []
            hlst = []
            for l, ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0])][int(ang[1])])
                mlst.append(m[int(ang[0])][int(ang[1])])
                hlst.append(h[int(ang[0])][int(ang[1])])
                patch[l] = self.getPatch(ang, self.m_pwidth, mask)
            
            Cinv_arr = np.array(Cinlst)
            m_arr = np.array(mlst)
            hl = np.array(hlst)
            
            a[i] = self.al(hlst, Cinlst)
            b[i] = self.bl(hlst, Cinlst, patch, mlst)
        
        print("Done")
        return a, b


def generate_test_data(nFrames, image_size, angles_range, signal_strength=5.0, noise_level=1.0, seed=None):
    """
    Genera un stack de imágenes ADI sintético para benchmarking.
    Similar a los datasets del EIDC Phase 1.
    """
    if seed is not None:
        np.random.seed(seed)
    
    height, width = image_size
    angles = np.linspace(0, angles_range, nFrames)
    
    # Generar ruido de fondo
    images = [np.reshape(np.random.normal(0, noise_level, height * width), 
                         (height, width)) for _ in range(nFrames)]
    
    # Crear señal sintética (gaussiana)
    center = (height // 2, width // 2)
    X, Y = np.meshgrid(np.arange(-height//2, height//2), 
                       np.arange(-width//2, width//2))
    xx, yy = np.meshgrid(np.arange(-center[0], height - center[0]), 
                         np.arange(-center[1], width - center[1]))
    
    # Señal normalizada para mantener SNR constante independiente de nFrames
    signal = gaussian2d(xx, yy, signal_strength / np.sqrt(nFrames), 2.0)
    
    # Rotar ruido y señal según los ángulos
    rot_noise = np.array([rotateImage(images[j], angles[j]) for j in range(nFrames)])
    rot_signal = np.array([rotateImage(signal, angles[j]) for j in range(nFrames)])
    
    # Combinar
    image_stack = rot_noise + rot_signal
    
    # Crear template PSF normalizado
    psf_size = 9  # Tamaño típico del PSF
    xx_psf, yy_psf = np.meshgrid(np.arange(-psf_size//2, psf_size//2 + 1),
                                 np.arange(-psf_size//2, psf_size//2 + 1))
    psf_template = gaussian2d(xx_psf, yy_psf, 1.0, 2.0)
    psf_template = psf_template / np.sum(psf_template)  # Normalizar
    
    return image_stack, angles, psf_template


def run_paco_serial(image_stack, angles, psf_template, config):
    """Ejecuta PACO en modo serial."""
    # CRÍTICO: Pasar px_scale=1.0 explícitamente para evitar que el constructor
    # calcule m_psf_rad incorrectamente como int(psf_rad/px_scale)
    fp = fastPACO.FastPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf_template,
        psf_rad=config['psf_rad'],
        px_scale=1.0,  # IMPORTANTE: usar 1.0 para que m_psf_rad = psf_rad
        patch_area=config['patch_size']
    )
    
    # CRÍTICO: Asegurar que m_psf_area y m_psf_rad sean consistentes
    # El constructor calcula m_psf_rad = int(psf_rad/px_scale), así que con px_scale=1.0 debería ser correcto
    # Pero verificamos y corregimos por si acaso
    from paco.util.util import createCircularMask
    # Verificar que m_psf_rad sea el correcto
    expected_mask = createCircularMask(psf_template.shape, radius=config['psf_rad'])
    expected_area = len(expected_mask[expected_mask])
    
    # DEBUG: Verificar valores antes de corregir
    print(f"  DEBUG run_paco_serial: Antes de corregir - m_psf_rad={fp.m_psf_rad}, m_psf_area={fp.m_psf_area}, config['psf_rad']={config['psf_rad']}, config['patch_size']={config['patch_size']}")
    
    if fp.m_psf_rad != config['psf_rad']:
        fp.m_psf_rad = config['psf_rad']
    if fp.m_psf_area != config['patch_size']:
        fp.m_psf_area = config['patch_size']
    
    # Debug: verificar consistencia
    if expected_area != config['patch_size']:
        print(f"  WARNING: Inconsistencia detectada: expected_area={expected_area}, config['patch_size']={config['patch_size']}")
        # Usar el valor del config que es el correcto
        fp.m_psf_area = config['patch_size']
    
    # DEBUG: Verificar valores después de corregir
    print(f"  DEBUG run_paco_serial: Después de corregir - m_psf_rad={fp.m_psf_rad}, m_psf_area={fp.m_psf_area}")
    
    start = time.perf_counter()
    a, b = fp.PACO(cpu=1, model_params={"sigma": config['sigma']}, 
                   model_name=gaussian2dModel)
    end = time.perf_counter()
    
    return a, b, end - start


def run_paco_multiprocessing(image_stack, angles, psf_template, config, n_cpus):
    """Ejecuta PACO con multiprocessing."""
    fp = fastPACO.FastPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf_template,
        psf_rad=config['psf_rad'],
        patch_area=config['patch_size']
    )
    
    # IMPORTANTE: Asegurar que m_psf_area y m_psf_rad sean consistentes
    from paco.util.util import createCircularMask
    
    # CRÍTICO: Asegurar que m_psf_rad y m_psf_area sean correctos
    # Usar directamente los valores del config que son los correctos
    fp.m_psf_rad = config['psf_rad']
    fp.m_psf_area = config['patch_size']  # Usar el valor del config que es el correcto
    
    start = time.perf_counter()
    a, b = fp.PACO(cpu=n_cpus, model_params={"sigma": config['sigma']}, 
                   model_name=gaussian2dModel)
    end = time.perf_counter()
    
    return a, b, end - start


def run_paco_loky(image_stack, angles, psf_template, config, n_cpus, backend='loky'):
    """Ejecuta PACO con loky/joblib."""
    if not LOKY_AVAILABLE:
        raise ImportError("loky no disponible")
    
    # CRÍTICO: Pasar px_scale=1.0 explícitamente
    fp = FastPACO_Loky(
        image_stack=image_stack,
        angles=angles,
        psf=psf_template,
        psf_rad=config['psf_rad'],
        px_scale=1.0,  # IMPORTANTE: usar 1.0 para que m_psf_rad = psf_rad
        patch_area=config['patch_size']
    )
    
    # IMPORTANTE: Asegurar que m_psf_area y m_psf_rad sean consistentes
    fp.m_psf_rad = config['psf_rad']
    fp.m_psf_area = config['patch_size']  # Usar el valor del config que es el correcto
    
    # Preparar coordenadas de píxeles
    x, y = np.meshgrid(np.arange(0, config['image_size'][0]),
                        np.arange(0, config['image_size'][1]))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    start = time.perf_counter()
    a, b = fp.PACOCalc(phi0s, cpu=n_cpus, backend=backend, use_loky=True)
    end = time.perf_counter()
    
    # Reshape como en PACO original
    a = np.reshape(a, config['image_size'])
    b = np.reshape(b, config['image_size'])
    
    return a, b, end - start


def run_fullpaco_serial(image_stack, angles, psf_template, config):
    """Ejecuta FullPACO en modo serial."""
    fp = fullPACO.FullPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf_template,
        psf_rad=config['psf_rad'],
        patch_area=config['patch_size']
    )
    
    # Bugfixes para código original
    if not hasattr(fp, 'm_p_size'):
        mask = createCircularMask(psf_template.shape, radius=config['psf_rad'])
        fp.m_p_size = len(mask[mask])
    if not hasattr(fp, 'angles'):
        fp.angles = fp.m_angles
    
    # Preparar coordenadas de píxeles
    x, y = np.meshgrid(np.arange(0, config['image_size'][0]),
                        np.arange(0, config['image_size'][1]))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    start = time.perf_counter()
    a, b = fp.PACOCalc(phi0s, cpu=1)
    end = time.perf_counter()
    
    # Reshape como en PACO original
    a = np.reshape(a, config['image_size'])
    b = np.reshape(b, config['image_size'])
    
    return a, b, end - start


def run_fullpaco_loky(image_stack, angles, psf_template, config, n_cpus, backend='loky'):
    """Ejecuta FullPACO con loky/joblib."""
    if not LOKY_AVAILABLE:
        raise ImportError("loky no disponible")
    
    fp = fullPACO.FullPACO(
        image_stack=image_stack,
        angles=angles,
        psf=psf_template,
        psf_rad=config['psf_rad'],
        patch_area=config['patch_size']
    )
    
    # Bugfixes para código original
    if not hasattr(fp, 'm_p_size'):
        mask = createCircularMask(psf_template.shape, radius=config['psf_rad'])
        fp.m_p_size = len(mask[mask])
    if not hasattr(fp, 'angles'):
        fp.angles = fp.m_angles
    
    # Preparar coordenadas
    dim = config['image_size'][0] / 2
    x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
    
    # Preparar píxeles a procesar
    x_flat, y_flat = np.meshgrid(np.arange(0, config['image_size'][0]),
                                  np.arange(0, config['image_size'][1]))
    phi0s = np.column_stack((x_flat.flatten(), y_flat.flatten()))
    
    # Preparar argumentos para paralelización
    mask = createCircularMask(psf_template.shape, radius=config['psf_rad'])
    pwidth = psf_template.shape[0]
    
    args_list = [
        (p0, image_stack, angles, psf_template, config['psf_rad'], 
         pwidth, x, y, dim)
        for p0 in phi0s
    ]
    
    print(f"Ejecutando FullPACO con {n_cpus} procesos usando loky...")
    start = time.perf_counter()
    
    # Paralelizar con loky
    results = Parallel(n_jobs=n_cpus, backend=backend, verbose=0)(
        delayed(_process_fullpaco_pixel)(args) for args in args_list
    )
    
    end = time.perf_counter()
    
    # Desempaquetar resultados
    a = np.array([r[0] for r in results])
    b = np.array([r[1] for r in results])
    
    # Reshape
    a = np.reshape(a, config['image_size'])
    b = np.reshape(b, config['image_size'])
    
    return a, b, end - start


def amdahl_speedup(p, n):
    """
    Calcula el speedup teórico según la ley de Amdahl.
    
    Parameters
    ----------
    p : float
        Fracción paralelizable del código (0-1)
    n : int
        Número de procesadores
    
    Returns
    -------
    speedup : float
        Speedup teórico
    """
    return 1 / ((1 - p) + p / n)


def analyze_amdahl(results):
    """
    Analiza los resultados del benchmark y calcula métricas de Amdahl.
    """
    # Calcular tiempos promedio
    serial_times = [r['time'] for r in results['serial']]
    t_serial = np.mean(serial_times)
    t_serial_std = np.std(serial_times)
    
    analysis = {
        'serial': {
            'mean_time': t_serial,
            'std_time': t_serial_std,
            'trials': len(serial_times)
        },
        'multiprocessing': {},
        'loky': {},
        'fullpaco_serial': {},
        'fullpaco_loky': {},
        'speedup_analysis': {},
        'skip_multiprocessing': results.get('skip_multiprocessing', False),
        'in_notebook': results.get('in_notebook', False)
    }
    
    # Analizar FullPACO Serial
    if results.get('fullpaco_serial'):
        fullpaco_serial_times = [r['time'] for r in results['fullpaco_serial']]
        if fullpaco_serial_times:
            t_fp_serial = np.mean(fullpaco_serial_times)
            t_fp_serial_std = np.std(fullpaco_serial_times)
            analysis['fullpaco_serial'] = {
                'mean_time': t_fp_serial,
                'std_time': t_fp_serial_std,
                'trials': len(fullpaco_serial_times)
            }
    
    # Analizar FullPACO Loky
    if LOKY_AVAILABLE and results.get('fullpaco_loky'):
        for n_cpu in results['fullpaco_loky']:
            if results['fullpaco_loky'][n_cpu]:
                times = [r['time'] for r in results['fullpaco_loky'][n_cpu]]
                t_mean = np.mean(times)
                t_std = np.std(times)
                # Speedup relativo a FullPACO serial
                if analysis.get('fullpaco_serial') and analysis['fullpaco_serial'].get('mean_time'):
                    t_fp_serial = analysis['fullpaco_serial']['mean_time']
                    speedup = t_fp_serial / t_mean
                else:
                    # Si no hay FullPACO serial, usar FastPACO serial como referencia
                    speedup = t_serial / t_mean
                efficiency = speedup / n_cpu
                
                if speedup > 1:
                    p_estimated = (speedup - 1) / (speedup * (1 - 1/n_cpu))
                else:
                    p_estimated = 0
                
                analysis['fullpaco_loky'][n_cpu] = {
                    'mean_time': t_mean,
                    'std_time': t_std,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'p_estimated': p_estimated,
                    'trials': len(times)
                }
    
    # Analizar multiprocessing (solo si hay datos)
    if not results.get('skip_multiprocessing', False):
        for n_cpu in results['multiprocessing']:
            if results['multiprocessing'][n_cpu]:
                times = [r['time'] for r in results['multiprocessing'][n_cpu]]
                t_mean = np.mean(times)
                t_std = np.std(times)
                speedup = t_serial / t_mean
                efficiency = speedup / n_cpu
                
                # Estimar fracción paralelizable (p) resolviendo Amdahl
                if speedup > 1:
                    p_estimated = (speedup - 1) / (speedup * (1 - 1/n_cpu))
                else:
                    p_estimated = 0
                
                analysis['multiprocessing'][n_cpu] = {
                    'mean_time': t_mean,
                    'std_time': t_std,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'p_estimated': p_estimated,
                    'trials': len(times)
                }
    
    # Analizar loky
    if LOKY_AVAILABLE and results.get('loky'):
        for n_cpu in results['loky']:
            if results['loky'][n_cpu]:
                times = [r['time'] for r in results['loky'][n_cpu]]
                t_mean = np.mean(times)
                t_std = np.std(times)
                speedup = t_serial / t_mean
                efficiency = speedup / n_cpu
                
                if speedup > 1:
                    p_estimated = (speedup - 1) / (speedup * (1 - 1/n_cpu))
                else:
                    p_estimated = 0
                
                analysis['loky'][n_cpu] = {
                    'mean_time': t_mean,
                    'std_time': t_std,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'p_estimated': p_estimated,
                    'trials': len(times)
                }
    
    # Calcular p promedio para predicción teórica
    all_p = []
    for method in ['multiprocessing', 'loky', 'fullpaco_loky']:
        if method in analysis:
            for n_cpu in analysis.get(method, {}):
                if analysis[method][n_cpu]:  # Verificar que no esté vacío
                    p_val = analysis[method][n_cpu].get('p_estimated', 0)
                    if 0 < p_val <= 1:
                        all_p.append(p_val)
    
    if all_p:
        p_mean = np.mean(all_p)
        analysis['speedup_analysis'] = {
            'p_mean': p_mean,
            'p_std': np.std(all_p),
            'theoretical_speedup': {n: amdahl_speedup(p_mean, n) 
                                    for n in results['config']['cpu_counts']}
        }
    
    return analysis


def run_benchmark(config, test_data, n_trials=1, skip_multiprocessing=None):
    """
    Ejecuta benchmark completo comparando todas las versiones.
    
    Parameters
    ----------
    config : dict
        Configuracion del benchmark
    test_data : tuple
        Datos de prueba (image_stack, angles, psf_template)
    n_trials : int
        Numero de trials
    skip_multiprocessing : bool, optional
        Si True, salta multiprocessing. Si None, detecta automaticamente en notebooks.
    """
    image_stack, angles, psf_template = test_data
    
    # Detectar si debemos saltar multiprocessing (problemas en Windows con notebooks)
    skip_mp = IN_NOTEBOOK if skip_multiprocessing is None else skip_multiprocessing
    
    results = {
        'serial': [],
        'multiprocessing': {n_cpu: [] for n_cpu in config['cpu_counts']},
        'loky': {n_cpu: [] for n_cpu in config['cpu_counts']},
        'fullpaco_serial': [],
        'fullpaco_loky': {n_cpu: [] for n_cpu in config['cpu_counts']},
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'skip_multiprocessing': skip_mp,
        'in_notebook': IN_NOTEBOOK
    }
    
    print("=" * 60)
    print("INICIANDO BENCHMARK")
    print("=" * 60)
    
    if skip_mp:
        print("[INFO] Multiprocessing deshabilitado (notebook detectado)")
        print("       Usando solo version serial y Loky para evitar problemas en Windows")
    
    # Benchmark Serial
    print("\n1. Ejecutando version SERIAL...")
    for trial in range(n_trials):
        print(f"   Trial {trial + 1}/{n_trials}")
        try:
            a, b, elapsed = run_paco_serial(image_stack, angles, psf_template, config)
            results['serial'].append({
                'time': elapsed,
                'trial': trial + 1
            })
            print(f"     Tiempo: {elapsed:.2f} segundos")
        except Exception as e:
            print(f"     ERROR: {e}")
    
    # Benchmark Multiprocessing (solo si no estamos en notebook)
    if not skip_mp:
        print("\n2. Ejecutando version MULTIPROCESSING...")
        for n_cpu in config['cpu_counts']:
            if n_cpu == 1:
                continue  # Ya medido en serial
            print(f"   CPUs: {n_cpu}")
            for trial in range(n_trials):
                print(f"     Trial {trial + 1}/{n_trials}")
                try:
                    a, b, elapsed = run_paco_multiprocessing(
                        image_stack, angles, psf_template, config, n_cpu)
                    results['multiprocessing'][n_cpu].append({
                        'time': elapsed,
                        'trial': trial + 1
                    })
                    print(f"       Tiempo: {elapsed:.2f} segundos")
                except Exception as e:
                    print(f"       ERROR: {e}")
    else:
        print("\n2. MULTIPROCESSING saltado (notebook detectado o problemas en Windows)")
    
    # Benchmark Loky
    if LOKY_AVAILABLE:
        print("\n3. Ejecutando version LOKY...")
        for n_cpu in config['cpu_counts']:
            if n_cpu == 1:
                continue  # Ya medido en serial
            print(f"   CPUs: {n_cpu}")
            for trial in range(n_trials):
                print(f"     Trial {trial + 1}/{n_trials}")
                try:
                    a, b, elapsed = run_paco_loky(
                        image_stack, angles, psf_template, config, n_cpu)
                    results['loky'][n_cpu].append({
                        'time': elapsed,
                        'trial': trial + 1
                    })
                    print(f"       Tiempo: {elapsed:.2f} segundos")
                except Exception as e:
                    print(f"       ERROR: {e}")
    else:
        print("\n3. Loky no disponible, saltando...")
        if skip_mp:
            print("   ADVERTENCIA: Sin Loky ni Multiprocessing, solo se ejecuto version serial")
    
    # Benchmark FullPACO Serial
    # OPTIMIZACIÓN: Para imágenes grandes, usar una región más pequeña para FullPACO Serial
    # para reducir el tiempo de ejecución
    fullpaco_image_size = config.get('image_size', (100, 100))
    fullpaco_test_size = config.get('fullpaco_test_size', None)  # None = auto-detect
    
    # Si la imagen es grande (>50x50), usar una región más pequeña para FullPACO Serial
    if fullpaco_test_size is None:
        h, w = fullpaco_image_size
        if h * w > 2500:  # Más de 50x50 = 2500 píxeles
            # Usar región central de 50x50 para FullPACO Serial
            fullpaco_test_size = (50, 50)
            print("\n4. Ejecutando FullPACO SERIAL...")
            print(f"   OPTIMIZACIÓN: Imagen grande ({h}x{w} = {h*w} píxeles) detectada.")
            print(f"   Usando región de prueba {fullpaco_test_size} para FullPACO Serial (más rápido).")
            print("   FullPACO Loky usará el tamaño completo.")
        else:
            fullpaco_test_size = fullpaco_image_size
            print("\n4. Ejecutando FullPACO SERIAL...")
    else:
        print("\n4. Ejecutando FullPACO SERIAL...")
        print(f"   Usando tamaño de prueba: {fullpaco_test_size}")
    
    # Preparar datos para FullPACO Serial (recortar si es necesario)
    if fullpaco_test_size != fullpaco_image_size:
        # Recortar imagen al centro para FullPACO Serial
        h_full, w_full = image_stack.shape[1], image_stack.shape[2]
        h_test, w_test = fullpaco_test_size
        start_h = (h_full - h_test) // 2
        end_h = start_h + h_test
        start_w = (w_full - w_test) // 2
        end_w = start_w + w_test
        
        image_stack_fp = image_stack[:, start_h:end_h, start_w:end_w]
        config_fp = config.copy()
        config_fp['image_size'] = fullpaco_test_size
    else:
        image_stack_fp = image_stack
        config_fp = config
    
    for trial in range(n_trials):
        print(f"   Trial {trial + 1}/{n_trials}")
        try:
            a, b, elapsed = run_fullpaco_serial(image_stack_fp, angles, psf_template, config_fp)
            results['fullpaco_serial'].append({
                'time': elapsed,
                'trial': trial + 1,
                'test_size': fullpaco_test_size  # Guardar tamaño usado
            })
            print(f"     Tiempo: {elapsed:.2f} segundos ({elapsed/60:.1f} minutos)")
        except KeyboardInterrupt:
            print(f"\n     INTERRUMPIDO por el usuario en trial {trial + 1}")
            print("     Continuando con FullPACO Loky...")
            break
        except Exception as e:
            print(f"     ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Benchmark FullPACO Loky
    if LOKY_AVAILABLE:
        print("\n5. Ejecutando FullPACO LOKY...")
        for n_cpu in config['cpu_counts']:
            if n_cpu == 1:
                continue  # Ya medido en serial
            print(f"   CPUs: {n_cpu}")
            for trial in range(n_trials):
                print(f"     Trial {trial + 1}/{n_trials}")
                try:
                    a, b, elapsed = run_fullpaco_loky(
                        image_stack, angles, psf_template, config, n_cpu)
                    results['fullpaco_loky'][n_cpu].append({
                        'time': elapsed,
                        'trial': trial + 1
                    })
                    print(f"       Tiempo: {elapsed:.2f} segundos")
                except Exception as e:
                    print(f"       ERROR: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print("\n5. FullPACO Loky no disponible (joblib no instalado)")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETADO")
    print("=" * 60)
    
    return results


def plot_benchmark_results(results, analysis, save_path=None):
    """
    Genera visualizaciones científicas de los resultados del benchmark.
    Incluye comparación entre FastPACO y FullPACO, siguiendo estándares científicos.
    """
    # Crear figura más grande para acomodar más gráficos
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    skip_mp = analysis.get('skip_multiprocessing', False)
    
    # 1. Tiempos de ejecución - Comparación FastPACO vs FullPACO
    ax1 = fig.add_subplot(gs[0, 0])
    
    # FastPACO
    serial_time = analysis['serial']['mean_time']
    ax1.errorbar([1], [serial_time], yerr=[analysis['serial']['std_time']], 
                 marker='o', label='FastPACO Serial', capsize=5, capthick=2, 
                 color='blue', markersize=8, linewidth=2)
    
    if analysis.get('loky'):
        loky_cpus = sorted([n for n in analysis['loky'] if analysis['loky'][n]])
        if loky_cpus:
            loky_times = [analysis['loky'][n]['mean_time'] for n in loky_cpus]
            loky_stds = [analysis['loky'][n]['std_time'] for n in loky_cpus]
            ax1.errorbar(loky_cpus, loky_times, yerr=loky_stds, 
                         marker='^', label='FastPACO Loky', capsize=5, capthick=2,
                         color='lightblue', markersize=8, linewidth=2)
    
    # FullPACO
    if analysis.get('fullpaco_serial') and analysis['fullpaco_serial'].get('mean_time'):
        fp_serial_time = analysis['fullpaco_serial']['mean_time']
        fp_serial_std = analysis['fullpaco_serial'].get('std_time', 0)
        ax1.errorbar([1], [fp_serial_time], yerr=[fp_serial_std], 
                     marker='s', label='FullPACO Serial', capsize=5, capthick=2,
                     color='red', markersize=8, linewidth=2)
    
    if analysis.get('fullpaco_loky'):
        fp_loky_cpus = sorted([n for n in analysis['fullpaco_loky'] if analysis['fullpaco_loky'][n]])
        if fp_loky_cpus:
            fp_loky_times = [analysis['fullpaco_loky'][n]['mean_time'] for n in fp_loky_cpus]
            fp_loky_stds = [analysis['fullpaco_loky'][n]['std_time'] for n in fp_loky_cpus]
            ax1.errorbar(fp_loky_cpus, fp_loky_times, yerr=fp_loky_stds, 
                         marker='v', label='FullPACO Loky', capsize=5, capthick=2,
                         color='orange', markersize=8, linewidth=2)
    
    ax1.set_xlabel('Número de CPUs', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tiempo de Ejecución (segundos)', fontsize=11, fontweight='bold')
    ax1.set_title('Tiempos de Ejecución: FastPACO vs FullPACO', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0.5, max(8, max(loky_cpus if analysis.get('loky') else [1])) + 0.5])
    
    # 2. Speedup - Comparación FastPACO vs FullPACO
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Baseline
    ax2.plot([1], [1], 'ko', markersize=10, label='Baseline (1x)', zorder=5)
    
    # FastPACO Loky
    if analysis.get('loky'):
        loky_cpus = sorted([n for n in analysis['loky'] if analysis['loky'][n]])
        if loky_cpus:
            loky_speedups = [analysis['loky'][n]['speedup'] for n in loky_cpus]
            ax2.plot(loky_cpus, loky_speedups, '^-', label='FastPACO Loky', 
                    color='lightblue', linewidth=2, markersize=8, zorder=3)
    
    # FullPACO Loky
    if analysis.get('fullpaco_loky'):
        fp_loky_cpus = sorted([n for n in analysis['fullpaco_loky'] if analysis['fullpaco_loky'][n]])
        if fp_loky_cpus:
            fp_loky_speedups = [analysis['fullpaco_loky'][n]['speedup'] for n in fp_loky_cpus]
            ax2.plot(fp_loky_cpus, fp_loky_speedups, 'v-', label='FullPACO Loky',
                    color='orange', linewidth=2, markersize=8, zorder=3)
    
    # Línea de speedup perfecto
    all_cpus = set()
    if analysis.get('loky'):
        all_cpus.update([n for n in analysis['loky'] if analysis['loky'][n]])
    if analysis.get('fullpaco_loky'):
        all_cpus.update([n for n in analysis['fullpaco_loky'] if analysis['fullpaco_loky'][n]])
    if all_cpus:
        max_cpu = max(all_cpus)
        perfect_cpus = list(range(1, max_cpu + 1))
        ax2.plot(perfect_cpus, perfect_cpus, 'k--', alpha=0.5, linewidth=1.5,
                label='Speedup Perfecto', zorder=1)
    
    # Speedup teórico según Amdahl (si está disponible)
    if analysis.get('speedup_analysis', {}).get('p_mean'):
        p_mean = analysis['speedup_analysis']['p_mean']
        if all_cpus:
            theoretical_cpus = sorted(all_cpus)
            theoretical = [amdahl_speedup(p_mean, n) for n in theoretical_cpus]
            ax2.plot(theoretical_cpus, theoretical, 'r--', alpha=0.7, linewidth=2,
                     label=f'Amdahl Teórico (p={p_mean:.3f})', zorder=2)
    
    ax2.set_xlabel('Número de CPUs', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold')
    ax2.set_title('Speedup: FastPACO vs FullPACO', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0.5, max(all_cpus) + 0.5 if all_cpus else 8.5])
    
    # 3. Eficiencia - Comparación FastPACO vs FullPACO
    ax3 = fig.add_subplot(gs[0, 2])
    
    # FastPACO Loky
    if analysis.get('loky'):
        loky_cpus = sorted([n for n in analysis['loky'] if analysis['loky'][n]])
        if loky_cpus:
            loky_efficiency = [analysis['loky'][n]['efficiency'] for n in loky_cpus]
            ax3.plot(loky_cpus, loky_efficiency, '^-', label='FastPACO Loky',
                    color='lightblue', linewidth=2, markersize=8)
    
    # FullPACO Loky
    if analysis.get('fullpaco_loky'):
        fp_loky_cpus = sorted([n for n in analysis['fullpaco_loky'] if analysis['fullpaco_loky'][n]])
        if fp_loky_cpus:
            fp_loky_efficiency = [analysis['fullpaco_loky'][n]['efficiency'] for n in fp_loky_cpus]
            ax3.plot(fp_loky_cpus, fp_loky_efficiency, 'v-', label='FullPACO Loky',
                    color='orange', linewidth=2, markersize=8)
    
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Eficiencia Perfecta (100%)')
    ax3.set_xlabel('Número de CPUs', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Eficiencia (Speedup / CPUs)', fontsize=11, fontweight='bold')
    ax3.set_title('Eficiencia de Paralelización', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0, 1.2])
    
    # 4. Comparación de Speedup: FastPACO vs FullPACO (Barras)
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Recopilar datos para comparación
    comparison_data = []
    all_cpus_set = set()
    
    if analysis.get('loky'):
        for n_cpu in analysis['loky']:
            if analysis['loky'][n_cpu]:
                all_cpus_set.add(n_cpu)
    
    if analysis.get('fullpaco_loky'):
        for n_cpu in analysis['fullpaco_loky']:
            if analysis['fullpaco_loky'][n_cpu]:
                all_cpus_set.add(n_cpu)
    
    all_cpus_sorted = sorted(all_cpus_set)
    
    fastpaco_speedups = []
    fullpaco_speedups = []
    cpu_labels = []
    
    for n_cpu in all_cpus_sorted:
        fastpaco_val = None
        fullpaco_val = None
        
        if analysis.get('loky') and n_cpu in analysis['loky'] and analysis['loky'][n_cpu]:
            fastpaco_val = analysis['loky'][n_cpu]['speedup']
        
        if analysis.get('fullpaco_loky') and n_cpu in analysis['fullpaco_loky'] and analysis['fullpaco_loky'][n_cpu]:
            fullpaco_val = analysis['fullpaco_loky'][n_cpu]['speedup']
        
        if fastpaco_val is not None or fullpaco_val is not None:
            fastpaco_speedups.append(fastpaco_val if fastpaco_val is not None else 0)
            fullpaco_speedups.append(fullpaco_val if fullpaco_val is not None else 0)
            cpu_labels.append(f"{n_cpu} CPUs")
    
    if fastpaco_speedups or fullpaco_speedups:
        x = np.arange(len(cpu_labels))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, fastpaco_speedups, width, label='FastPACO Loky',
                       color='lightblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
        bars2 = ax4.bar(x + width/2, fullpaco_speedups, width, label='FullPACO Loky',
                       color='orange', alpha=0.8, edgecolor='darkred', linewidth=1.5)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax4.set_xlabel('Configuración', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Speedup', fontsize=11, fontweight='bold')
        ax4.set_title('Comparación de Speedup: FastPACO vs FullPACO', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cpu_labels, fontsize=10)
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 5. Comparación de Tiempos: FastPACO vs FullPACO
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Comparar tiempos seriales
    times_data = []
    labels_data = []
    colors_data = []
    
    if analysis.get('serial'):
        times_data.append(analysis['serial']['mean_time'])
        labels_data.append('FastPACO\nSerial')
        colors_data.append('blue')
    
    if analysis.get('fullpaco_serial') and analysis['fullpaco_serial'].get('mean_time'):
        times_data.append(analysis['fullpaco_serial']['mean_time'])
        labels_data.append('FullPACO\nSerial')
        colors_data.append('red')
    
    if times_data:
        bars = ax5.bar(range(len(times_data)), times_data, color=colors_data, alpha=0.7,
                      edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(times_data)))
        ax5.set_xticklabels(labels_data, fontsize=9, fontweight='bold')
        ax5.set_ylabel('Tiempo (segundos)', fontsize=11, fontweight='bold')
        ax5.set_title('Comparación Tiempos Seriales', fontsize=12, fontweight='bold')
        
        # Agregar valores
        for i, (bar, time) in enumerate(zip(bars, times_data)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 6. Resumen de Métricas (Tabla de texto)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Crear tabla de resumen
    summary_text = "RESUMEN DE RESULTADOS DEL BENCHMARK\n" + "="*80 + "\n\n"
    
    # FastPACO
    summary_text += "FASTPACO (Algorithm 2 - Precomputación de Estadísticas):\n"
    summary_text += f"  • Serial: {analysis['serial']['mean_time']:.2f} ± {analysis['serial']['std_time']:.2f} segundos\n"
    
    if analysis.get('loky'):
        for n_cpu in sorted([n for n in analysis['loky'] if analysis['loky'][n]]):
            data = analysis['loky'][n_cpu]
            summary_text += f"  • Loky {n_cpu} CPUs: {data['mean_time']:.2f} ± {data['std_time']:.2f} s "
            summary_text += f"(Speedup: {data['speedup']:.2f}x, Eficiencia: {data['efficiency']:.1%})\n"
    
    # FullPACO
    if analysis.get('fullpaco_serial') and analysis['fullpaco_serial'].get('mean_time'):
        summary_text += "\nFULLPACO (Algorithm 1 - Precisión Subpixel Completa):\n"
        fp_serial = analysis['fullpaco_serial']
        summary_text += f"  • Serial: {fp_serial['mean_time']:.2f} ± {fp_serial['std_time']:.2f} segundos\n"
        
        if analysis.get('fullpaco_loky'):
            for n_cpu in sorted([n for n in analysis['fullpaco_loky'] if analysis['fullpaco_loky'][n]]):
                data = analysis['fullpaco_loky'][n_cpu]
                summary_text += f"  • Loky {n_cpu} CPUs: {data['mean_time']:.2f} ± {data['std_time']:.2f} s "
                summary_text += f"(Speedup: {data['speedup']:.2f}x, Eficiencia: {data['efficiency']:.1%})\n"
    
    # Análisis de Amdahl
    if analysis.get('speedup_analysis', {}).get('p_mean'):
        p_mean = analysis['speedup_analysis']['p_mean']
        summary_text += f"\nAnálisis de Amdahl:\n"
        summary_text += f"  • Fracción paralelizable (p): {p_mean:.3f}\n"
        if analysis['speedup_analysis'].get('max_theoretical_speedup'):
            max_speedup = analysis['speedup_analysis']['max_theoretical_speedup']
            summary_text += f"  • Speedup máximo teórico: {max_speedup:.2f}x\n"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Benchmark de Rendimiento: FastPACO vs FullPACO con Paralelización Loky',
                fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Gráficos guardados en {save_path}")
    
    plt.show()


def save_results(results, analysis, config, output_dir):
    """
    Guarda los resultados del benchmark en varios formatos.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar resultados completos en JSON
    json_path = output_path / f"benchmark_results_{timestamp}.json"
    
    # Convertir numpy arrays a listas para JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    analysis_serializable = convert_to_serializable(analysis)
    
    with open(json_path, 'w') as f:
        json.dump({
            'results': results_serializable,
            'analysis': analysis_serializable,
            'config': config
        }, f, indent=2)
    
    print(f"Resultados guardados en JSON: {json_path}")
    
    # Crear tabla resumen en CSV
    summary_data = []
    
    # FastPACO Serial
    summary_data.append({
        'Algorithm': 'FastPACO',
        'Method': 'Serial',
        'CPUs': 1,
        'Mean_Time': analysis['serial']['mean_time'],
        'Std_Time': analysis['serial']['std_time'],
        'Speedup': 1.0,
        'Efficiency': 1.0,
        'P_Estimated': np.nan
    })
    
    # FastPACO Multiprocessing (solo si no fue saltado)
    if not analysis.get('skip_multiprocessing', False) and analysis.get('multiprocessing'):
        for n_cpu in sorted(analysis['multiprocessing'].keys()):
            if analysis['multiprocessing'][n_cpu]:
                data = analysis['multiprocessing'][n_cpu]
                summary_data.append({
                    'Algorithm': 'FastPACO',
                    'Method': 'Multiprocessing',
                    'CPUs': n_cpu,
                    'Mean_Time': data['mean_time'],
                    'Std_Time': data['std_time'],
                    'Speedup': data['speedup'],
                    'Efficiency': data['efficiency'],
                    'P_Estimated': data['p_estimated']
                })
    
    # FastPACO Loky
    if analysis.get('loky'):
        for n_cpu in sorted(analysis['loky'].keys()):
            if analysis['loky'][n_cpu]:
                data = analysis['loky'][n_cpu]
                summary_data.append({
                    'Algorithm': 'FastPACO',
                    'Method': 'Loky',
                    'CPUs': n_cpu,
                    'Mean_Time': data['mean_time'],
                    'Std_Time': data['std_time'],
                    'Speedup': data['speedup'],
                    'Efficiency': data['efficiency'],
                    'P_Estimated': data['p_estimated']
                })
    
    # FullPACO Serial
    if analysis.get('fullpaco_serial') and analysis['fullpaco_serial'].get('mean_time'):
        fp_serial = analysis['fullpaco_serial']
        summary_data.append({
            'Algorithm': 'FullPACO',
            'Method': 'Serial',
            'CPUs': 1,
            'Mean_Time': fp_serial['mean_time'],
            'Std_Time': fp_serial['std_time'],
            'Speedup': 1.0,
            'Efficiency': 1.0,
            'P_Estimated': np.nan
        })
    
    # FullPACO Loky
    if analysis.get('fullpaco_loky'):
        for n_cpu in sorted(analysis['fullpaco_loky'].keys()):
            if analysis['fullpaco_loky'][n_cpu]:
                data = analysis['fullpaco_loky'][n_cpu]
                summary_data.append({
                    'Algorithm': 'FullPACO',
                    'Method': 'Loky',
                    'CPUs': n_cpu,
                    'Mean_Time': data['mean_time'],
                    'Std_Time': data['std_time'],
                    'Speedup': data['speedup'],
                    'Efficiency': data['efficiency'],
                    'P_Estimated': data['p_estimated']
                })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = output_path / f"benchmark_summary_{timestamp}.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Resumen guardado en CSV: {csv_path}")
    
    return json_path, csv_path


def analyze_scalability(benchmark_results, analysis):
    """
    Analiza cómo escala el algoritmo con diferentes números de CPUs
    y compara multiprocessing vs loky.
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE ESCALABILIDAD")
    print("="*60)
    
    serial_time = analysis['serial']['mean_time']
    
    # Extraer datos de speedup
    skip_mp = analysis.get('skip_multiprocessing', False)
    
    mp_cpus = []
    mp_speedups = []
    mp_efficiencies = []
    if not skip_mp and analysis.get('multiprocessing'):
        mp_cpus = sorted([n for n in analysis['multiprocessing'] if analysis['multiprocessing'][n]])
        if mp_cpus:
            mp_speedups = [analysis['multiprocessing'][n]['speedup'] for n in mp_cpus]
            mp_efficiencies = [analysis['multiprocessing'][n]['efficiency'] for n in mp_cpus]
    
    loky_cpus = []
    loky_speedups = []
    loky_efficiencies = []
    if analysis.get('loky'):
        loky_cpus = sorted([n for n in analysis['loky'] if analysis['loky'][n]])
        if loky_cpus:
            loky_speedups = [analysis['loky'][n]['speedup'] for n in loky_cpus]
            loky_efficiencies = [analysis['loky'][n]['efficiency'] for n in loky_cpus]
    
    # FullPACO Loky
    fp_loky_cpus = []
    fp_loky_speedups = []
    fp_loky_efficiencies = []
    if analysis.get('fullpaco_loky'):
        fp_loky_cpus = sorted([n for n in analysis['fullpaco_loky'] if analysis['fullpaco_loky'][n]])
        if fp_loky_cpus:
            fp_loky_speedups = [analysis['fullpaco_loky'][n]['speedup'] for n in fp_loky_cpus]
            fp_loky_efficiencies = [analysis['fullpaco_loky'][n]['efficiency'] for n in fp_loky_cpus]
    
    # Análisis de escalabilidad
    print(f"\nFastPACO Serial (baseline): {serial_time:.2f} segundos")
    
    if analysis.get('fullpaco_serial') and analysis['fullpaco_serial'].get('mean_time'):
        fp_serial_time = analysis['fullpaco_serial']['mean_time']
        print(f"FullPACO Serial (baseline): {fp_serial_time:.2f} segundos")
    
    if skip_mp:
        print("\n[INFO] Multiprocessing fue saltado (notebook detectado)")
        print("       Mostrando solo resultados de Loky\n")
    
    print("\n" + "="*70)
    print("SPEEDUP POR NÚMERO DE CPUs")
    print("="*70)
    
    if skip_mp:
        print(f"\n{'CPUs':<8} {'FastPACO Loky':<20} {'FullPACO Loky':<20}")
        print("-" * 50)
        all_cpus_combined = sorted(set(loky_cpus + fp_loky_cpus))
        for n_cpu in all_cpus_combined:
            fastpaco_val = None
            fullpaco_val = None
            if n_cpu in loky_cpus:
                fastpaco_val = analysis['loky'][n_cpu]['speedup']
            if n_cpu in fp_loky_cpus:
                fullpaco_val = analysis['fullpaco_loky'][n_cpu]['speedup']
            
            fastpaco_str = f"{fastpaco_val:.2f}x" if fastpaco_val else "N/A"
            fullpaco_str = f"{fullpaco_val:.2f}x" if fullpaco_val else "N/A"
            print(f"{n_cpu:<8} {fastpaco_str:<20} {fullpaco_str:<20}")
    else:
        print(f"{'CPUs':<8} {'Multiprocessing':<20} {'Loky':<20} {'Mejora Loky':<15}")
        print("-" * 65)
        
        # Usar todos los CPUs disponibles (de mp o loky)
        all_cpus = sorted(set(mp_cpus + loky_cpus))
        for n_cpu in all_cpus:
            mp_speedup = None
            loky_speedup = None
            improvement = None
            
            if n_cpu in mp_cpus:
                mp_speedup = analysis['multiprocessing'][n_cpu]['speedup']
            
            if n_cpu in loky_cpus:
                loky_speedup = analysis['loky'][n_cpu]['speedup']
            
            if mp_speedup and loky_speedup:
                if loky_speedup > mp_speedup:
                    improvement = f"+{((loky_speedup/mp_speedup - 1) * 100):.1f}%"
                else:
                    improvement = f"{((loky_speedup/mp_speedup - 1) * 100):.1f}%"
            
            mp_str = f"{mp_speedup:.2f}x" if mp_speedup else "N/A"
            loky_str = f"{loky_speedup:.2f}x" if loky_speedup else "N/A"
            improvement_str = improvement if improvement else "N/A"
            
            print(f"{n_cpu:<8} {mp_str:<20} {loky_str:<20} {improvement_str:<15}")
    
    # Análisis de eficiencia
    print("\n" + "="*70)
    print("EFICIENCIA POR NÚMERO DE CPUs")
    print("="*70)
    
    if skip_mp:
        print(f"\n{'CPUs':<8} {'FastPACO Loky':<20} {'FullPACO Loky':<20}")
        print("-" * 50)
        all_cpus_combined = sorted(set(loky_cpus + fp_loky_cpus))
        for n_cpu in all_cpus_combined:
            fastpaco_eff = None
            fullpaco_eff = None
            if n_cpu in loky_cpus:
                fastpaco_eff = analysis['loky'][n_cpu]['efficiency']
            if n_cpu in fp_loky_cpus:
                fullpaco_eff = analysis['fullpaco_loky'][n_cpu]['efficiency']
            
            fastpaco_str = f"{fastpaco_eff:.1%}" if fastpaco_eff else "N/A"
            fullpaco_str = f"{fullpaco_eff:.1%}" if fullpaco_eff else "N/A"
            print(f"{n_cpu:<8} {fastpaco_str:<20} {fullpaco_str:<20}")
    else:
        print(f"{'CPUs':<8} {'Multiprocessing':<20} {'Loky':<20}")
        print("-" * 50)
        for n_cpu in all_cpus:
            mp_eff = None
            loky_eff = None
            
            if n_cpu in mp_cpus:
                mp_eff = analysis['multiprocessing'][n_cpu]['efficiency']
            if n_cpu in loky_cpus:
                loky_eff = analysis['loky'][n_cpu]['efficiency']
            
            mp_str = f"{mp_eff:.2%}" if mp_eff else "N/A"
            loky_str = f"{loky_eff:.2%}" if loky_eff else "N/A"
            print(f"{n_cpu:<8} {mp_str:<20} {loky_str:<20}")
    
    # Determinar punto de saturación
    print("\n" + "="*70)
    print("SPEEDUP MÁXIMO ALCANZADO")
    print("="*70)
    
    if skip_mp:
        if loky_speedups:
            loky_max_speedup = max(loky_speedups)
            loky_max_cpu = loky_cpus[loky_speedups.index(loky_max_speedup)]
            print(f"\n  FastPACO Loky: Speedup máximo {loky_max_speedup:.2f}x con {loky_max_cpu} CPUs")
        
        if fp_loky_speedups:
            fp_max_speedup = max(fp_loky_speedups)
            fp_max_cpu = fp_loky_cpus[fp_loky_speedups.index(fp_max_speedup)]
            print(f"  FullPACO Loky: Speedup máximo {fp_max_speedup:.2f}x con {fp_max_cpu} CPUs")
    else:
        if mp_speedups and len(mp_speedups) > 1:
            mp_max_speedup = max(mp_speedups)
            mp_max_cpu = mp_cpus[mp_speedups.index(mp_max_speedup)]
            print(f"  Multiprocessing: Speedup maximo {mp_max_speedup:.2f}x con {mp_max_cpu} CPUs")
        
        if loky_speedups:
            loky_max_speedup = max(loky_speedups)
            loky_max_cpu = loky_cpus[loky_speedups.index(loky_max_speedup)]
            print(f"  Loky: Speedup maximo {loky_max_speedup:.2f}x con {loky_max_cpu} CPUs")
            
            if mp_speedups and loky_max_speedup > mp_max_speedup:
                print(f"  [OK] Loky es {((loky_max_speedup/mp_max_speedup - 1) * 100):.1f}% mejor que Multiprocessing")
    
    # Predicción según Amdahl
    if analysis.get('speedup_analysis', {}).get('p_mean'):
        p_mean = analysis['speedup_analysis']['p_mean']
        max_theoretical_speedup = 1 / (1 - p_mean)
        print(f"\nLey de Amdahl:")
        print(f"  Fraccion paralelizable estimada (p): {p_mean:.3f}")
        print(f"  Speedup maximo teorico: {max_theoretical_speedup:.2f}x")
        print(f"  Esto significa que agregar mas CPUs despues de cierto punto")
        print(f"  no mejorara significativamente el rendimiento.")
    
    print("="*60)


# Configuración por defecto del benchmark (exportable)
BENCHMARK_CONFIG = {
    'nFrames': 50,
    'image_size': (100, 100),
    'angles_range': 60,
    'patch_size': 49,
    'psf_rad': 4,
    'sigma': 2.0,
    'n_trials': 1,
    'cpu_counts': [1, 2, 4, 8],
    'output_dir': 'output/performance_analysis/'
}


if __name__ == "__main__":
    # En Windows, necesitamos freeze_support cuando se ejecuta como script
    multiprocessing.freeze_support()
    
    print("=" * 60)
    print("PACO PERFORMANCE BENCHMARK")
    print("=" * 60)
    if IN_NOTEBOOK:
        print("[INFO] Ejecutandose en notebook - Multiprocessing sera saltado")
    print(f"\nConfiguracion:")
    for key, value in BENCHMARK_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Generar datos de prueba
    print("\nGenerando datos de prueba...")
    test_data = generate_test_data(
        BENCHMARK_CONFIG['nFrames'],
        BENCHMARK_CONFIG['image_size'],
        BENCHMARK_CONFIG['angles_range'],
        seed=42
    )
    image_stack, angles, psf_template = test_data
    print(f"  Stack shape: {image_stack.shape}")
    print(f"  Ángulos shape: {angles.shape}")
    print(f"  PSF template shape: {psf_template.shape}")
    
    # Ejecutar benchmark
    benchmark_results = run_benchmark(
        BENCHMARK_CONFIG, 
        test_data, 
        n_trials=BENCHMARK_CONFIG['n_trials']
    )
    
    # Analizar resultados
    print("\nAnalizando resultados...")
    analysis = analyze_amdahl(benchmark_results)
    
    # Visualizar
    print("\nGenerando visualizaciones...")
    plot_benchmark_results(
        benchmark_results, 
        analysis, 
        save_path=BENCHMARK_CONFIG['output_dir'] + 'benchmark_plots.png'
    )
    
    # Guardar resultados
    print("\nGuardando resultados...")
    json_path, csv_path = save_results(
        benchmark_results, 
        analysis, 
        BENCHMARK_CONFIG, 
        BENCHMARK_CONFIG['output_dir']
    )
    
    # Imprimir resumen
    print("\n" + "=" * 60)
    print("RESUMEN DEL BENCHMARK")
    print("=" * 60)
    print(f"\nTiempo serial promedio: {analysis['serial']['mean_time']:.2f} ± {analysis['serial']['std_time']:.2f} seg")
    
    if analysis.get('skip_multiprocessing'):
        print("\n[INFO] Multiprocessing fue saltado (notebook detectado)")
        print("       Solo se muestran resultados de Loky")
    
    if analysis.get('speedup_analysis', {}).get('p_mean'):
        print(f"\nFraccion paralelizable estimada (p): {analysis['speedup_analysis']['p_mean']:.3f} ± {analysis['speedup_analysis']['p_std']:.3f}")
        print("\nSpeedup teorico segun Amdahl:")
        for n, speedup in analysis['speedup_analysis']['theoretical_speedup'].items():
            print(f"  {n} CPUs: {speedup:.2f}x")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETADO")
    print("=" * 60)

