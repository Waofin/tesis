"""
Implementación de FullPACO con aceleración GPU usando CuPy.

Esta clase extiende FullPACO para usar GPU CUDA en las operaciones
computacionalmente intensivas, especialmente pixelCalc().

Requisitos:
    pip install cupy-cuda11x  # Para CUDA 11.x
    # o
    pip install cupy-cuda12x  # Para CUDA 12.x
"""

import numpy as np
from .fullpaco import FullPACO
from paco.util.util import (
    pixelCalc, getRotatedPixels, createCircularMask,
    cartToPol, gridPolToCart
)

# Intentar importar CuPy
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None


class FullPACO_CUDA(FullPACO):
    """
    Versión GPU de FullPACO usando CuPy.
    
    Esta clase hereda de FullPACO y reemplaza las operaciones
    computacionalmente intensivas con versiones GPU.
    """
    
    def __init__(self, *args, **kwargs):
        """Inicializar FullPACO_CUDA."""
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
            return [np.full(49, np.nan), np.full((49, 49), np.nan)]
        
        # Transferir a GPU
        patch_gpu = cp.asarray(patch, dtype=cp.float32)
        T = patch_gpu.shape[0]
        size = patch_gpu.shape[1]
        
        # Calcular media en GPU
        m_gpu = cp.mean(patch_gpu, axis=0)
        
        # Calcular covarianza de muestra en GPU
        # Nota: La implementación original usa np.cov que es diferente
        # Aquí usamos una versión simplificada que es más eficiente en GPU
        S_gpu = self._sampleCovariance_CUDA(patch_gpu, m_gpu, T)
        
        # Calcular shrinkage factor en GPU
        rho = self._shrinkageFactor_CUDA(S_gpu, T)
        
        # Diagonal de covarianza
        F_gpu = cp.diag(cp.diag(S_gpu))
        
        # Covarianza con shrinkage
        C_gpu = (1.0 - rho) * S_gpu + rho * F_gpu
        
        # Inversión de matriz en GPU (MUY eficiente)
        try:
            Cinv_gpu = cp.linalg.inv(C_gpu)
        except cp.linalg.LinAlgError:
            # Fallback a pseudoinversa si la matriz es singular
            Cinv_gpu = cp.linalg.pinv(C_gpu)
        
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
        # Usar operaciones básicas que no requieren NVRTC
        diff = r - m[cp.newaxis, :]  # (T, P)
        
        # Calcular covarianza vectorizada: (1/T) * diff^T @ diff
        # Usar cp.matmul en lugar de cp.dot para mejor compatibilidad
        S = cp.matmul(diff.T, diff) / T
        
        return S
    
    def _shrinkageFactor_CUDA(self, S, T):
        """
        Versión GPU de shrinkageFactor.
        
        Calcula el factor de shrinkage de Ledoit-Wolf en GPU.
        """
        # Operaciones matriciales en GPU
        # Usar cp.matmul en lugar de cp.dot
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
    
    def PACOCalc_CUDA(self, phi0s, batch_size=1000, use_gpu_pixelcalc=True):
        """
        Versión GPU de PACOCalc procesando píxeles en batches.
        
        Esta función procesa píxeles en batches para optimizar
        el uso de GPU y reducir overhead de transferencia.
        
        Parameters
        ----------
        phi0s : np.ndarray
            Array de píxeles a procesar
        batch_size : int
            Número de píxeles a procesar por batch
        use_gpu_pixelcalc : bool
            Si True, usa pixelCalc_CUDA. Si False, usa pixelCalc CPU.
            
        Returns
        -------
        a : np.ndarray
            Array de valores 'a' para cada píxel
        b : np.ndarray
            Array de valores 'b' para cada píxel
        """
        npx = len(phi0s)
        dim = self.m_width / 2
        a = np.zeros(npx)
        b = np.zeros(npx)
        
        # Crear máscara
        mask = createCircularMask(self.m_psf.shape, radius=self.m_psf_rad)
        if not hasattr(self, 'm_p_size') or self.m_p_size != len(mask[mask]):
            self.m_p_size = len(mask[mask])
        
        # Setup coordenadas (usar coordenadas centradas en la imagen)
        x, y = np.meshgrid(np.arange(0, self.m_width), np.arange(0, self.m_height))
        
        print(f"Running PACO on GPU...")
        print(f"  Total pixels: {npx}")
        print(f"  Batch size: {batch_size}")
        print(f"  GPU pixelCalc: {use_gpu_pixelcalc}")
        
        # Calcular intervalos para mostrar progreso
        progress_interval = max(1, npx // 20)
        
        # Procesar en batches
        for batch_start in range(0, npx, batch_size):
            batch_end = min(batch_start + batch_size, npx)
            batch_phi0s = phi0s[batch_start:batch_end]
            
            # Procesar cada píxel en el batch
            for i, p0 in enumerate(batch_phi0s):
                idx = batch_start + i
                
                # Mostrar progreso
                if idx % progress_interval == 0:
                    progress = (idx / npx) * 100
                    print(f"  Progreso: {progress:.1f}% ({idx}/{npx} píxeles)", end='\r')
                
                # Obtener ángulos rotados
                angles_px = getRotatedPixels(x, y, p0, self.m_angles)
                
                # Arrays para este píxel
                h_list = []
                Cinv_list = []
                patch_list = []
                m_list = []
                
                # Procesar cada frame
                for l, ang in enumerate(angles_px):
                    # Extraer patch (esto se hace en CPU por ahora)
                    patch_l = self.getPatch(ang, self.m_pwidth, mask)
                    patch_list.append(patch_l)
                    
                    # Calcular estadísticas (en GPU o CPU según use_gpu_pixelcalc)
                    if use_gpu_pixelcalc:
                        m_l, Cinv_l = self.pixelCalc_CUDA(patch_l)
                    else:
                        m_l, Cinv_l = pixelCalc(patch_l)
                    
                    m_list.append(m_l)
                    Cinv_list.append(Cinv_l)
                    h_list.append(self.m_psf[mask])
                
                # Calcular a y b
                a[idx] = self.al(h_list, Cinv_list)
                b[idx] = self.bl(h_list, Cinv_list, patch_list, m_list)
            
            # Sincronizar GPU después de cada batch
            if use_gpu_pixelcalc:
                cp.cuda.Stream.null.synchronize()
        
        print(f"\nDone (procesados {npx} píxeles)")
        return a, b
    
    def PACOCalc(self, phi0s, cpu=1):
        """
        Sobrescribir PACOCalc para usar GPU por defecto.
        
        Si cpu=1, usa GPU. Si cpu>1, usa CPU con paralelización.
        """
        if cpu == 1:
            # Usar GPU
            return self.PACOCalc_CUDA(phi0s, batch_size=1000, use_gpu_pixelcalc=True)
        else:
            # Usar CPU con paralelización (heredado de FullPACO)
            return super().PACOCalc(phi0s, cpu=cpu)

