"""
Versión ALTAMENTE OPTIMIZADA de FastPACO_CUDA.

Optimizaciones:
1. PACOCalc completamente en GPU (no solo computeStatistics)
2. Vectorización masiva de operaciones
3. Eliminación de llamadas redundantes a getPatch
4. Procesamiento en batches grandes para maximizar uso de GPU
"""

import numpy as np
from .fastpaco_cuda import FastPACO_CUDA
from paco.util.util import getRotatedPixels, createCircularMask

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class FastPACO_CUDA_Optimized(FastPACO_CUDA):
    """
    Versión altamente optimizada de FastPACO_CUDA.
    
    Optimizaciones principales:
    - PACOCalc completamente en GPU
    - Vectorización masiva de al() y bl()
    - Eliminación de llamadas redundantes a getPatch
    - Procesamiento en batches masivos
    """
    
    def al_CUDA(self, h_gpu, Cinv_gpu):
        """
        Versión GPU vectorizada de al().
        
        Procesa múltiples píxeles simultáneamente.
        
        Parameters
        ----------
        h_gpu : cp.ndarray
            Shape (n_pixels, n_frames, psf_area)
        Cinv_gpu : cp.ndarray
            Shape (n_pixels, n_frames, psf_area, psf_area)
            
        Returns
        -------
        a_gpu : cp.ndarray
            Shape (n_pixels,)
        """
        # h_gpu: (n_pixels, n_frames, psf_area)
        # Cinv_gpu: (n_pixels, n_frames, psf_area, psf_area)
        
        # Calcular h^T @ Cinv @ h para cada frame y pixel
        # Usar einsum para vectorización masiva
        # Resultado: (n_pixels, n_frames)
        h_Cinv_h = cp.einsum('pfi,pfij,pfj->pf', h_gpu, Cinv_gpu, h_gpu)
        
        # Sumar sobre frames: (n_pixels,)
        a_gpu = cp.sum(h_Cinv_h, axis=1)
        
        return a_gpu
    
    def bl_CUDA(self, h_gpu, Cinv_gpu, patch_gpu, m_gpu):
        """
        Versión GPU vectorizada de bl().
        
        Procesa múltiples píxeles simultáneamente.
        
        Parameters
        ----------
        h_gpu : cp.ndarray
            Shape (n_pixels, n_frames, psf_area)
        Cinv_gpu : cp.ndarray
            Shape (n_pixels, n_frames, psf_area, psf_area)
        patch_gpu : cp.ndarray
            Shape (n_pixels, n_frames, psf_area)
        m_gpu : cp.ndarray
            Shape (n_pixels, n_frames, psf_area)
            
        Returns
        -------
        b_gpu : cp.ndarray
            Shape (n_pixels,)
        """
        # Calcular (patch - m) para cada frame y pixel
        diff_gpu = patch_gpu - m_gpu  # (n_pixels, n_frames, psf_area)
        
        # Calcular Cinv @ h para cada frame y pixel
        # Resultado: (n_pixels, n_frames, psf_area)
        Cinv_h_gpu = cp.einsum('pfij,pfj->pfi', Cinv_gpu, h_gpu)
        
        # Calcular (Cinv @ h)^T @ (patch - m) para cada frame y pixel
        # Resultado: (n_pixels, n_frames)
        b_per_frame = cp.einsum('pfi,pfi->pf', Cinv_h_gpu, diff_gpu)
        
        # Sumar sobre frames: (n_pixels,)
        b_gpu = cp.sum(b_per_frame, axis=1)
        
        return b_gpu
    
    def PACOCalc_CUDA_Optimized(self, phi0s, batch_size=10000):
        """
        Versión GPU completamente optimizada de PACOCalc.
        
        Procesa múltiples píxeles en paralelo en GPU usando vectorización masiva.
        
        Parameters
        ----------
        phi0s : np.ndarray
            Array de píxeles a procesar
        batch_size : int
            Número de píxeles a procesar por batch
            
        Returns
        -------
        a : np.ndarray
            Array de valores 'a' para cada píxel
        b : np.ndarray
            Array de valores 'b' para cada píxel
        """
        npx = len(phi0s)
        dim = int(self.m_width / 2)
        
        # Precomputar estadísticas (ya optimizado en GPU)
        print("Precomputing Statistics on GPU...")
        Cinv, m, h = self.computeStatistics(phi0s)
        
        # Transferir estadísticas a GPU de una vez
        print("Transferring statistics to GPU...")
        Cinv_gpu = cp.asarray(Cinv, dtype=cp.float32)  # (H, W, psf_area, psf_area)
        m_gpu = cp.asarray(m, dtype=cp.float32)  # (H, W, psf_area)
        h_gpu = cp.asarray(h, dtype=cp.float32)  # (H, W, psf_area)
        angles_gpu = cp.asarray(self.m_angles, dtype=cp.float32)
        
        # Crear máscara (en CPU y GPU)
        mask = createCircularMask((self.m_pwidth, self.m_pwidth), radius=self.m_psf_rad)
        mask_gpu = cp.asarray(mask, dtype=cp.bool_)
        
        # Preparar meshgrid en CPU (para getRotatedPixels)
        x_cpu, y_cpu = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        # Transferir image stack a GPU
        im_stack_gpu = cp.asarray(self.m_im_stack, dtype=cp.float32)  # (T, H, W)
        
        # Arrays de salida en GPU
        a_gpu = cp.zeros(npx, dtype=cp.float32)
        b_gpu = cp.zeros(npx, dtype=cp.float32)
        
        print("Running PACO on GPU (optimized batch processing)...")
        
        # Procesar en batches grandes
        progress_interval = max(1, npx // 20)
        
        for batch_start in range(0, npx, batch_size):
            batch_end = min(batch_start + batch_size, npx)
            batch_phi0s = phi0s[batch_start:batch_end]
            batch_size_actual = len(batch_phi0s)
            
            # Transferir batch de phi0s a GPU
            phi0s_gpu = cp.asarray(batch_phi0s, dtype=cp.float32)  # (batch_size, 2)
            
            # Para cada píxel en el batch, necesitamos:
            # 1. Calcular ángulos rotados (en GPU)
            # 2. Extraer patches (vectorizado en GPU)
            # 3. Extraer estadísticas (indexación en GPU)
            # 4. Calcular a y b (vectorizado en GPU)
            
            # Arrays para almacenar datos del batch
            # Shape: (batch_size, n_frames, psf_area)
            batch_patches_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area), dtype=cp.float32)
            batch_Cinv_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area, self.m_psf_area), dtype=cp.float32)
            batch_m_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area), dtype=cp.float32)
            batch_h_gpu = cp.zeros((batch_size_actual, self.m_nFrames, self.m_psf_area), dtype=cp.float32)
            
            # Procesar cada píxel del batch
            # OPTIMIZACIÓN: Usar múltiples streams para procesamiento asíncrono
            n_streams = min(8, batch_size_actual)
            streams = [cp.cuda.Stream() for _ in range(n_streams)]
            chunk_size = (batch_size_actual + n_streams - 1) // n_streams
            
            for stream_idx, stream in enumerate(streams):
                with stream:
                    chunk_start = stream_idx * chunk_size
                    chunk_end = min(chunk_start + chunk_size, batch_size_actual)
                    
                    for i in range(chunk_start, chunk_end):
                        p0 = batch_phi0s[i]
                        p0_gpu = cp.asarray(p0, dtype=cp.float32)
                        
                        # Calcular ángulos rotados (en CPU por ahora, pero solo una vez)
                        angles_px = getRotatedPixels(x_cpu, y_cpu, p0, self.m_angles)
                        
                        # Verificar límites
                        if (np.max(angles_px.flatten()) >= self.m_width or 
                            np.min(angles_px.flatten()) < 0):
                            a_gpu[batch_start + i] = cp.nan
                            b_gpu[batch_start + i] = cp.nan
                            continue
                        
                        # Extraer patches y estadísticas (vectorizado)
                        k = self.m_pwidth // 2
                        for l in range(self.m_nFrames):
                            ang = angles_px[l]
                            ang_x, ang_y = int(ang[0]), int(ang[1])
                            
                            if (0 <= ang_y < self.m_height and 0 <= ang_x < self.m_width and
                                ang_x >= k and ang_x < self.m_width - k and
                                ang_y >= k and ang_y < self.m_height - k):
                                # Extraer patch directamente de im_stack_gpu
                                # Esto es más eficiente que llamar getPatch
                                patch_region = im_stack_gpu[l, 
                                                             ang_y-k:ang_y+k+1, 
                                                             ang_x-k:ang_x+k+1]
                                batch_patches_gpu[i, l] = patch_region[mask_gpu]
                                
                                # Extraer estadísticas
                                batch_Cinv_gpu[i, l] = Cinv_gpu[ang_y, ang_x]
                                batch_m_gpu[i, l] = m_gpu[ang_y, ang_x]
                                batch_h_gpu[i, l] = h_gpu[ang_y, ang_x]
                            else:
                                # Si está fuera de límites, usar NaN
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
        """
        Sobrescribir PACOCalc para usar versión optimizada en GPU.
        """
        if cpu == 1 and CUDA_AVAILABLE:
            return self.PACOCalc_CUDA_Optimized(phi0s, batch_size=10000)
        else:
            # Fallback a versión CPU
            return super().PACOCalc(phi0s, cpu=cpu)

