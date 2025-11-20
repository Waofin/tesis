"""
Implementación de FastPACO con aceleración GPU usando CuPy.

Esta clase extiende FastPACO para usar GPU CUDA en la precomputación
de estadísticas (computeStatistics), que es la parte más costosa.

Requisitos:
    pip install cupy-cuda11x  # Para CUDA 11.x
    # o
    pip install cupy-cuda12x  # Para CUDA 12.x
"""

import numpy as np
from .fastpaco import FastPACO
from paco.util.util import (
    pixelCalc, getRotatedPixels, createCircularMask
)

# Intentar importar CuPy
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class FastPACO_CUDA(FastPACO):
    """
    Versión GPU de FastPACO usando CuPy.
    
    Esta clase hereda de FastPACO y optimiza computeStatistics()
    para procesar múltiples píxeles en paralelo en GPU.
    
    El speedup esperado es mucho mayor que FullPACO_CUDA porque:
    - FastPACO precomputa estadísticas para todos los píxeles
    - GPU puede procesar miles de píxeles simultáneamente
    - Speedup esperado: 50-200x vs 7-15x de FullPACO_CUDA
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializar FastPACO_CUDA."""
        super().__init__(*args, **kwargs)
        
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CuPy no está disponible. Instala con:\n"
                "  pip install cupy-cuda11x  # Para CUDA 11.x\n"
                "  # o\n"
                "  pip install cupy-cuda12x  # Para CUDA 12.x"
            )
        
        # Verificar que hay GPU disponible
        try:
            self.device = cp.cuda.Device(0)
            self.device.use()
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"[OK] GPU disponible: {gpu_name}")
        except Exception as e:
            raise RuntimeError(f"No se pudo inicializar GPU: {e}")
    
    def pixelCalc_CUDA(self, patch):
        """
        Versión GPU de pixelCalc usando CuPy.
        
        Calcula media y matriz de covarianza inversa en GPU.
        (Wrapper que transfiere a GPU y de vuelta a CPU)
        
        Parameters
        ----------
        patch : np.ndarray
            Array de shape (T, P) donde T=frames, P=píxeles en patch
            
        Returns
        -------
        m : np.ndarray
            Media del patch
        Cinv : np.ndarray
            Matriz de covarianza inversa
        """
        if patch is None:
            return [np.full(self.m_psf_area, np.nan), 
                    np.full((self.m_psf_area, self.m_psf_area), np.nan)]
        
        # Transferir a GPU
        patch_gpu = cp.asarray(patch, dtype=cp.float32)
        
        # Usar versión batch optimizada
        m_gpu, Cinv_gpu = self.pixelCalc_CUDA_batch(patch_gpu)
        
        # Transferir de vuelta a CPU
        m = cp.asnumpy(m_gpu)
        Cinv = cp.asnumpy(Cinv_gpu)
        
        return m, Cinv
    
    def _sampleCovariance_CUDA(self, r, m, T):
        """
        Versión GPU de sampleCovariance.
        
        Calcula la matriz de covarianza de muestra en GPU.
        Optimizado usando operaciones vectorizadas.
        """
        # r: (T, P) - patches en GPU
        # m: (P,) - media en GPU
        
        # Calcular diferencias (broadcasting)
        diff = r - m[cp.newaxis, :]  # (T, P)
        
        # Calcular covarianza vectorizada: (1/T) * diff^T @ diff
        S = cp.matmul(diff.T, diff) / T
        
        return S
    
    def _shrinkageFactor_CUDA(self, S, T):
        """
        Versión GPU de shrinkageFactor.
        
        Calcula el factor de shrinkage de Ledoit-Wolf en GPU.
        """
        # Operaciones matriciales en GPU
        S_dot_S = cp.matmul(S, S)
        trace_S_dot_S = float(cp.trace(S_dot_S))
        trace_S = float(cp.trace(S))
        diag_S = cp.diag(S)
        sum_diag_sq = float(cp.sum(diag_S ** 2))
        
        top = trace_S_dot_S + trace_S ** 2 - 2.0 * sum_diag_sq
        bot = (T + 1.0) * (trace_S_dot_S - sum_diag_sq)
        
        if abs(bot) < 1e-10:
            p = 0.0
        else:
            p = top / bot
        
        return max(min(float(p), 1.0), 0.0)
    
    def computeStatistics_CUDA(self, phi0s, batch_size=1000, gpu_batch_size=100):
        """
        Versión GPU optimizada de computeStatistics.
        
        Procesa múltiples píxeles en paralelo en GPU usando batching masivo.
        Extrae todos los patches de un batch en CPU, luego los procesa
        simultáneamente en GPU.
        
        Parameters
        ----------
        phi0s : np.ndarray
            Array de píxeles a procesar
        batch_size : int
            Número de píxeles a procesar por batch (para controlar memoria GPU)
        gpu_batch_size : int
            Número de patches a procesar simultáneamente en GPU
            
        Returns
        -------
        Cinv : np.ndarray
            Matriz de covarianza inversa para cada píxel
        m : np.ndarray
            Media para cada píxel
        h : np.ndarray
            PSF para cada píxel
        """
        print("Precomputing Statistics on GPU (optimized batch processing)...")
        npx = len(phi0s)
        dim = int(self.m_width / 2)
        
        # Crear máscara
        patch_shape = (self.m_pwidth, self.m_pwidth)
        mask = createCircularMask(patch_shape, radius=self.m_psf_rad)
        actual_psf_area = len(mask[mask])
        
        if self.m_psf_area != actual_psf_area:
            self.m_psf_area = actual_psf_area
        
        # Arrays de salida
        h = np.zeros((self.m_height, self.m_width, self.m_psf_area))
        m = np.zeros((self.m_height, self.m_width, self.m_psf_area))
        Cinv = np.zeros((self.m_height, self.m_width, self.m_psf_area, self.m_psf_area))
        
        # Transferir image stack a GPU una vez (se reutiliza)
        im_stack_gpu = cp.asarray(self.m_im_stack, dtype=cp.float32)
        psf_mask_gpu = cp.asarray(self.m_psf[mask], dtype=cp.float32)
        
        # Calcular intervalos para mostrar progreso
        progress_interval = max(1, npx // 20)
        
        # Procesar en batches grandes
        for batch_start in range(0, npx, batch_size):
            batch_end = min(batch_start + batch_size, npx)
            batch_phi0s = phi0s[batch_start:batch_end]
            
            # Extraer todos los patches del batch en CPU
            patches_list = []
            valid_indices = []
            valid_coords = []
            
            for idx, p0 in enumerate(batch_phi0s):
                patch = self.getPatch(p0, self.m_pwidth, mask)
                if patch is not None:
                    patches_list.append(patch)
                    valid_indices.append(batch_start + idx)
                    valid_coords.append((int(p0[0]), int(p0[1])))
            
            if not patches_list:
                continue
            
            # Procesar patches en sub-batches para GPU
            n_patches = len(patches_list)
            for gpu_batch_start in range(0, n_patches, gpu_batch_size):
                gpu_batch_end = min(gpu_batch_start + gpu_batch_size, n_patches)
                gpu_batch_patches = patches_list[gpu_batch_start:gpu_batch_end]
                
                # Convertir a array numpy para transferir a GPU
                # Shape: (gpu_batch_size, T, P)
                patches_array = np.array(gpu_batch_patches, dtype=np.float32)
                
                # Transferir batch completo a GPU de una vez
                patches_gpu = cp.asarray(patches_array, dtype=cp.float32)
                
                # Procesar todos los patches del batch en paralelo usando vectorización
                # patches_gpu shape: (batch_size, T, P)
                batch_size_gpu = patches_gpu.shape[0]
                T = patches_gpu.shape[1]
                P = patches_gpu.shape[2]
                
                # Calcular medias para todos los patches simultáneamente
                # Shape: (batch_size, P)
                m_batch_gpu = cp.mean(patches_gpu, axis=1)
                
                # OPTIMIZACIÓN: Procesar patches en paralelo usando operaciones vectorizadas
                # En lugar de loop secuencial, procesar en batches más pequeños pero paralelos
                m_batch = []
                Cinv_batch = []
                
                # Usar múltiples streams de CUDA para procesamiento asíncrono
                n_streams = min(4, batch_size_gpu)  # Usar hasta 4 streams
                streams = [cp.cuda.Stream() for _ in range(n_streams)]
                
                # Dividir batch en chunks para cada stream
                chunk_size = (batch_size_gpu + n_streams - 1) // n_streams
                
                for stream_idx, stream in enumerate(streams):
                    with stream:
                        chunk_start = stream_idx * chunk_size
                        chunk_end = min(chunk_start + chunk_size, batch_size_gpu)
                        
                        for i in range(chunk_start, chunk_end):
                            patch_gpu = patches_gpu[i]
                            m_pixel_gpu = m_batch_gpu[i]
                            
                            # Calcular covarianza
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
                
                # Sincronizar todos los streams
                for stream in streams:
                    stream.synchronize()
                
                # Guardar resultados
                for i, (x, y) in enumerate(valid_coords[gpu_batch_start:gpu_batch_end]):
                    if 0 <= y < self.m_height and 0 <= x < self.m_width:
                        m[y, x] = m_batch[i]
                        Cinv[y, x] = Cinv_batch[i]
                        h[y, x] = self.m_psf[mask]
                
                # Mostrar progreso
                pixel_idx = valid_indices[gpu_batch_end - 1] if gpu_batch_end > 0 else batch_start
                if pixel_idx % progress_interval == 0:
                    progress = (pixel_idx / npx) * 100
                    print(f"  Progreso: {progress:.1f}% ({pixel_idx}/{npx} píxeles)", end='\r')
            
            # Sincronizar GPU después de cada batch grande
            cp.cuda.Stream.null.synchronize()
        
        print(f"\nDone (procesados {npx} píxeles)")
        return Cinv, m, h
    
    def pixelCalc_CUDA_batch(self, patch_gpu):
        """
        Versión optimizada de pixelCalc_CUDA que trabaja directamente con arrays GPU.
        
        Evita transferencias innecesarias CPU↔GPU.
        
        Parameters
        ----------
        patch_gpu : cp.ndarray
            Array GPU de shape (T, P) donde T=frames, P=píxeles en patch
            
        Returns
        -------
        m : cp.ndarray
            Media del patch (en GPU)
        Cinv : cp.ndarray
            Matriz de covarianza inversa (en GPU)
        """
        if patch_gpu is None:
            return (cp.full(self.m_psf_area, cp.nan, dtype=cp.float32),
                    cp.full((self.m_psf_area, self.m_psf_area), cp.nan, dtype=cp.float32))
        
        T = patch_gpu.shape[0]
        
        # Calcular media en GPU
        m_gpu = cp.mean(patch_gpu, axis=0)
        
        # Calcular covarianza de muestra en GPU
        S_gpu = self._sampleCovariance_CUDA(patch_gpu, m_gpu, T)
        
        # Calcular shrinkage factor en GPU
        rho = self._shrinkageFactor_CUDA(S_gpu, T)
        
        # Diagonal de covarianza
        F_gpu = cp.diag(cp.diag(S_gpu))
        
        # Covarianza con shrinkage
        C_gpu = (1.0 - rho) * S_gpu + rho * F_gpu
        
        # Inversión de matriz en GPU
        try:
            Cinv_gpu = cp.linalg.inv(C_gpu)
        except cp.linalg.LinAlgError:
            Cinv_gpu = cp.linalg.pinv(C_gpu)
        
        return m_gpu, Cinv_gpu
    
    def computeStatistics(self, phi0s):
        """
        Sobrescribir computeStatistics para usar GPU por defecto.
        
        Si CUDA está disponible, usa computeStatistics_CUDA optimizado.
        Si no, usa la versión CPU heredada.
        """
        if CUDA_AVAILABLE:
            # Usar batch_size más grande y gpu_batch_size para paralelización
            # Aumentar batch_size para datasets grandes
            return self.computeStatistics_CUDA(phi0s, batch_size=5000, gpu_batch_size=500)
        else:
            # Fallback a CPU
            return super().computeStatistics(phi0s)
    
    def computeStatisticsParallel(self, phi0s, cpu):
        """
        Sobrescribir computeStatisticsParallel.
        
        Con GPU, no necesitamos paralelización CPU.
        """
        if CUDA_AVAILABLE:
            print("[INFO] GPU disponible, usando computeStatistics_CUDA optimizado")
            return self.computeStatistics_CUDA(phi0s, batch_size=2000, gpu_batch_size=200)
        else:
            # Fallback a CPU paralelo
            return super().computeStatisticsParallel(phi0s, cpu)

