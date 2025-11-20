"""
FastPACO_CUDA con pre-extract de patches.

Esta versión pre-extrae y almacena patches durante computeStatistics,
eliminando las llamadas a getPatch en PACOCalc (que consumen 71% del tiempo).

OPTIMIZACIÓN: Usa multiprocessing para extraer patches en paralelo en CPU.
"""

import numpy as np
from .fastpaco_cuda import FastPACO_CUDA
from paco.util.util import getRotatedPixels, createCircularMask
from multiprocessing import Pool, cpu_count
import os

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


def _extract_patch_worker(args):
    """Worker function para extraer patches en paralelo."""
    p0, im_stack, pwidth, mask, height, width = args
    k = pwidth // 2
    x, y = int(p0[0]), int(p0[1])
    
    # Verificar límites
    if (x < k or x >= width - k or y < k or y >= height - k):
        return None, (x, y)
    
    # Extraer patch para todos los frames
    patch = np.array([im_stack[i][y-k:y+k+1, x-k:x+k+1][mask] 
                      for i in range(len(im_stack))], dtype=np.float32)
    
    return patch, (x, y)


class FastPACO_CUDA_PreExtract(FastPACO_CUDA):
    """
    Versión de FastPACO_CUDA que pre-extrae patches.
    
    Optimización clave:
    - Pre-extrae y almacena patches durante computeStatistics
    - Elimina llamadas a getPatch en PACOCalc (71% del tiempo)
    - Almacena patches en GPU para acceso directo
    - Usa multiprocessing para extraer patches en paralelo
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializar con almacenamiento de patches."""
        super().__init__(*args, **kwargs)
        # Almacenar patches pre-extraídos
        self.patches_gpu = None  # (H, W, T, psf_area) en GPU
        self.patches_cpu = None  # Backup en CPU si es necesario
        self._patches_ready = False
        self._patches_phi0s = None
        # Usar todos los cores disponibles para extracción de patches
        self.n_cpu_cores = max(1, cpu_count() - 1)  # Dejar 1 core libre
        print(f"[INFO] Usando {self.n_cpu_cores} cores de CPU para extracción de patches")
    
    def computeStatistics_CUDA(self, phi0s, batch_size=5000, gpu_batch_size=500):
        """
        Versión optimizada que pre-extrae patches usando multiprocessing.
        
        Además de calcular estadísticas, extrae y almacena todos los patches
        necesarios para PACOCalc, eliminando llamadas a getPatch.
        """
        print("Precomputing Statistics on GPU (with parallel patch pre-extraction)...")
        npx = len(phi0s)
        dim = int(self.m_width / 2)
        
        # Crear máscara
        patch_shape = (self.m_pwidth, self.m_pwidth)
        mask = createCircularMask(patch_shape, radius=self.m_psf_rad)
        mask_gpu = cp.asarray(mask, dtype=cp.bool_)
        actual_psf_area = len(mask[mask])
        
        if self.m_psf_area != actual_psf_area:
            self.m_psf_area = actual_psf_area
        
        # Arrays de salida para estadísticas
        h = np.zeros((self.m_height, self.m_width, self.m_psf_area))
        m = np.zeros((self.m_height, self.m_width, self.m_psf_area))
        Cinv = np.zeros((self.m_height, self.m_width, self.m_psf_area, self.m_psf_area))
        
        # NUEVO: Array para almacenar patches pre-extraídos
        # Shape: (H, W, T, psf_area) - patches para cada posición y frame
        patches_cpu = np.zeros((self.m_height, self.m_width, self.m_nFrames, self.m_psf_area), 
                              dtype=np.float32)
        
        # Transferir image stack a GPU una vez
        im_stack_gpu = cp.asarray(self.m_im_stack, dtype=cp.float32)
        psf_mask_gpu = cp.asarray(self.m_psf[mask], dtype=cp.float32)
        
        progress_interval = max(1, npx // 20)
        k = self.m_pwidth // 2
        
        # OPTIMIZACIÓN: Extraer patches en paralelo usando multiprocessing
        print("  Extrayendo patches en paralelo (CPU multiprocessing)...")
        import time
        start_extract = time.perf_counter()
        
        # Preparar argumentos para workers
        worker_args = [(p0, self.m_im_stack, self.m_pwidth, mask, 
                       self.m_height, self.m_width) for p0 in phi0s]
        
        # Extraer patches en paralelo
        with Pool(processes=self.n_cpu_cores) as pool:
            results = pool.map(_extract_patch_worker, worker_args, 
                              chunksize=max(1, len(worker_args) // (self.n_cpu_cores * 4)))
        
        time_extract = time.perf_counter() - start_extract
        print(f"  Patches extraídos en {time_extract:.2f}s usando {self.n_cpu_cores} cores")
        
        # Organizar patches y calcular estadísticas
        patches_dict = {}
        valid_coords = []
        
        for patch, (x, y) in results:
            if patch is not None:
                patches_dict[(x, y)] = patch
                valid_coords.append((x, y))
                # Almacenar patch pre-extraído
                for t in range(self.m_nFrames):
                    patches_cpu[y, x, t] = patch[t]
        
        print(f"  Patches válidos extraídos: {len(valid_coords)}/{npx}")
        
        # Procesar patches en batches para GPU
        patches_list = [patches_dict[coord] for coord in valid_coords]
        
        if not patches_list:
            print("  [WARNING] No se encontraron patches válidos")
            return Cinv, m, h
        
        # Procesar estadísticas en GPU
        print("  Calculando estadísticas en GPU...")
        n_patches = len(patches_list)
        
        for gpu_batch_start in range(0, n_patches, gpu_batch_size):
            gpu_batch_end = min(gpu_batch_start + gpu_batch_size, n_patches)
            gpu_batch_patches = patches_list[gpu_batch_start:gpu_batch_end]
            
            patches_array = np.array(gpu_batch_patches, dtype=np.float32)
            patches_gpu = cp.asarray(patches_array, dtype=cp.float32)
            
            batch_size_gpu = patches_gpu.shape[0]
            T = patches_gpu.shape[1]
            P = patches_gpu.shape[2]
            
            # Calcular medias
            m_batch_gpu = cp.mean(patches_gpu, axis=1)
            
            # Procesar con múltiples streams
            n_streams = min(4, batch_size_gpu)
            streams = [cp.cuda.Stream() for _ in range(n_streams)]
            chunk_size = (batch_size_gpu + n_streams - 1) // n_streams
            
            m_batch = []
            Cinv_batch = []
            
            for stream_idx, stream in enumerate(streams):
                with stream:
                    chunk_start = stream_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, batch_size_gpu)
                    
                    for i in range(chunk_start, chunk_end):
                        patch_gpu = patches_gpu[i]
                        m_pixel_gpu = m_batch_gpu[i]
                        
                        S_gpu = self._sampleCovariance_CUDA(patch_gpu, m_pixel_gpu, T)
                        rho = self._shrinkageFactor_CUDA(S_gpu, T)
                        F_gpu = cp.diag(cp.diag(S_gpu))
                        C_gpu = (1.0 - rho) * S_gpu + rho * F_gpu
                        
                        try:
                            Cinv_pixel_gpu = cp.linalg.inv(C_gpu)
                        except cp.linalg.LinAlgError:
                            Cinv_pixel_gpu = cp.linalg.pinv(C_gpu)
                        
                        m_batch.append(cp.asnumpy(m_pixel_gpu))
                        Cinv_batch.append(cp.asnumpy(Cinv_pixel_gpu))
            
            for stream in streams:
                stream.synchronize()
            
            # Guardar resultados
            for i, (x, y) in enumerate(valid_coords[gpu_batch_start:gpu_batch_end]):
                if 0 <= y < self.m_height and 0 <= x < self.m_width:
                    m[y, x] = m_batch[i]
                    Cinv[y, x] = Cinv_batch[i]
                    h[y, x] = self.m_psf[mask]
            
            if gpu_batch_end % progress_interval == 0:
                progress = (gpu_batch_end / npx) * 100
                print(f"  Progreso: {progress:.1f}% ({gpu_batch_end}/{npx} píxeles)", end='\r')
            
            cp.cuda.Stream.null.synchronize()
        
        print(f"\nDone (procesados {npx} píxeles)")
        
        # Transferir patches a GPU y almacenar
        print("Transferring pre-extracted patches to GPU...")
        self.patches_gpu = cp.asarray(patches_cpu, dtype=cp.float32)
        self.patches_cpu = patches_cpu  # Backup en CPU
        self._patches_ready = True
        self._patches_phi0s = np.array(phi0s, copy=True)
        
        print(f"  Patches almacenados: {patches_cpu.shape}")
        print(f"  Memoria GPU: {self.patches_gpu.nbytes / 1024**2:.1f} MB")
        
        return Cinv, m, h
    
    def al_CUDA(self, h_gpu, Cinv_gpu):
        """Versión GPU vectorizada de al()."""
        # h_gpu: (n_pixels, n_frames, psf_area)
        # Cinv_gpu: (n_pixels, n_frames, psf_area, psf_area)
        
        # Calcular h^T @ Cinv @ h para cada frame y pixel
        h_Cinv_h = cp.einsum('pfi,pfij,pfj->pf', h_gpu, Cinv_gpu, h_gpu)
        
        # Sumar sobre frames: (n_pixels,)
        a_gpu = cp.sum(h_Cinv_h, axis=1)
        
        return a_gpu
    
    def bl_CUDA(self, h_gpu, Cinv_gpu, patch_gpu, m_gpu):
        """Versión GPU vectorizada de bl()."""
        # Calcular (patch - m)
        diff_gpu = patch_gpu - m_gpu  # (n_pixels, n_frames, psf_area)
        
        # Calcular Cinv @ h
        Cinv_h_gpu = cp.einsum('pfij,pfj->pfi', Cinv_gpu, h_gpu)
        
        # Calcular (Cinv @ h)^T @ (patch - m)
        b_per_frame = cp.einsum('pfi,pfi->pf', Cinv_h_gpu, diff_gpu)
        
        # Sumar sobre frames: (n_pixels,)
        b_gpu = cp.sum(b_per_frame, axis=1)
        
        return b_gpu
    
    def PACOCalc_CUDA_PreExtract(self, phi0s, batch_size=10000):
        """
        Versión GPU optimizada que usa patches pre-extraídos.
        
        No llama a getPatch, usa patches almacenados en GPU.
        """
        npx = len(phi0s)
        dim = int(self.m_width / 2)
        
        # Asegurarse de que los patches pre-extraídos correspondan a estos phi0s
        need_recompute = (
            not self._patches_ready or
            self.patches_gpu is None or
            self._patches_phi0s is None or
            len(self._patches_phi0s) != len(phi0s) or
            np.any(self._patches_phi0s != phi0s)
        )
        
        if need_recompute:
            print("Precomputing statistics with patch pre-extraction...")
            Cinv, m, h = self.computeStatistics(phi0s)
        else:
            # Aun así obtener estadísticas (pueden depender de phi0s)
            Cinv, m, h = self.computeStatistics(phi0s)
        
        if not self._patches_ready or self.patches_gpu is None:
            raise RuntimeError("Patches no pre-extraídos. computeStatistics no almacenó patches.")
        
        # Transferir estadísticas a GPU
        print("Transferring statistics to GPU...")
        Cinv_gpu = cp.asarray(Cinv, dtype=cp.float32)  # (H, W, psf_area, psf_area)
        m_gpu = cp.asarray(m, dtype=cp.float32)  # (H, W, psf_area)
        h_gpu = cp.asarray(h, dtype=cp.float32)  # (H, W, psf_area)
        
        # Preparar meshgrid en CPU (para getRotatedPixels)
        x_cpu, y_cpu = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        # Arrays de salida en GPU
        a_gpu = cp.zeros(npx, dtype=cp.float32)
        b_gpu = cp.zeros(npx, dtype=cp.float32)
        
        print("Running PACO on GPU (using pre-extracted patches)...")
        
        progress_interval = max(1, npx // 20)
        
        # Procesar en batches grandes
        for batch_start in range(0, npx, batch_size):
            batch_end = min(batch_start + batch_size, npx)
            batch_phi0s = phi0s[batch_start:batch_end]
            batch_size_actual = len(batch_phi0s)
            
            # Arrays para almacenar datos del batch
            batch_patches_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area), 
                                        dtype=cp.float32)
            batch_Cinv_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area, self.m_psf_area), 
                                     dtype=cp.float32)
            batch_m_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area), dtype=cp.float32)
            batch_h_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area), dtype=cp.float32)
            
            # Procesar cada píxel del batch
            n_streams = min(8, batch_size_actual)
            streams = [cp.cuda.Stream() for _ in range(n_streams)]
            chunk_size = (batch_size_actual + n_streams - 1) // n_streams
            
            for stream_idx, stream in enumerate(streams):
                with stream:
                    chunk_start = stream_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, batch_size_actual)
                    
                    for i in range(chunk_start, chunk_end):
                        p0 = batch_phi0s[i]
                        p0_x, p0_y = int(p0[0]), int(p0[1])
                        
                        # Calcular ángulos rotados
                        angles_px = getRotatedPixels(x_cpu, y_cpu, p0, self.m_angles)
                        
                        # Verificar límites
                        if (np.max(angles_px.flatten()) >= self.m_width or 
                            np.min(angles_px.flatten()) < 0):
                            a_gpu[batch_start + i] = cp.nan
                            b_gpu[batch_start + i] = cp.nan
                            continue
                        
                        # OPTIMIZACIÓN: Usar patches pre-extraídos en lugar de getPatch
                        for l in range(self.m_nFrames):
                            ang = angles_px[l]
                            ang_x, ang_y = int(ang[0]), int(ang[1])
                            
                            if (0 <= ang_y < self.m_height and 0 <= ang_x < self.m_width):
                                # Obtener patch pre-extraído directamente de GPU
                                batch_patches_gpu[i, l] = self.patches_gpu[ang_y, ang_x, l]
                                
                                # Extraer estadísticas
                                batch_Cinv_gpu[i, l] = Cinv_gpu[ang_y, ang_x]
                                batch_m_gpu[i, l] = m_gpu[ang_y, ang_x]
                                batch_h_gpu[i, l] = h_gpu[ang_y, ang_x]
                            else:
                                batch_patches_gpu[i, l] = cp.nan
                                batch_Cinv_gpu[i, l] = cp.nan
                                batch_m_gpu[i, l] = cp.nan
                                batch_h_gpu[i, l] = cp.nan
            
            # Sincronizar streams
            for stream in streams:
                stream.synchronize()
            
            # Calcular a y b para todo el batch de una vez (vectorizado)
            a_batch = self.al_CUDA(batch_h_gpu, batch_Cinv_gpu)
            b_batch = self.bl_CUDA(batch_h_gpu, batch_Cinv_gpu, batch_patches_gpu, batch_m_gpu)
            
            # Guardar resultados
            a_gpu[batch_start:batch_end] = a_batch
            b_gpu[batch_start:batch_end] = b_batch
            
            # Progreso
            if batch_end % progress_interval == 0:
                progress = (batch_end / npx) * 100
                print(f"  Progreso: {progress:.1f}% ({batch_end}/{npx} píxeles)", end='\r')
        
        # Transferir resultados a CPU
        a = cp.asnumpy(a_gpu)
        b = cp.asnumpy(b_gpu)
        
        print(f"\nDone (procesados {npx} píxeles)")
        return a, b
    
    def PACOCalc(self, phi0s, cpu=1):
        """Sobrescribir PACOCalc para usar versión con pre-extract."""
        if cpu == 1 and CUDA_AVAILABLE:
            return self.PACOCalc_CUDA_PreExtract(phi0s, batch_size=10000)
        else:
            # Fallback a versión CPU
            return super().PACOCalc(phi0s, cpu=cpu)
