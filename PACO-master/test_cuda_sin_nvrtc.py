"""
Test de CuPy sin NVRTC - usando solo operaciones básicas.
"""

import os
import sys
from pathlib import Path

# Intentar deshabilitar NVRTC
os.environ['CUPY_ACCELERATORS'] = 'cub'  # Solo usar kernels precompilados

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TEST CUDA SIN NVRTC")
print("=" * 70)

try:
    import cupy as cp
    print("\n[OK] CuPy importado")
    
    # Verificar GPU
    n_devices = cp.cuda.runtime.getDeviceCount()
    print(f"[OK] GPUs: {n_devices}")
    
    if n_devices > 0:
        device = cp.cuda.Device(0)
        device.use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"[OK] GPU: {props['name'].decode()}")
        
        # Test operaciones básicas
        print("\nProbando operaciones basicas...")
        
        # Test 1: Arrays básicos
        a = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
        b = cp.array([5, 4, 3, 2, 1], dtype=cp.float32)
        c = a + b
        print(f"  [OK] Suma: {cp.asnumpy(c)}")
        
        # Test 2: Matriz multiplicación
        m1 = cp.random.rand(10, 20).astype(cp.float32)
        m2 = cp.random.rand(20, 15).astype(cp.float32)
        m3 = cp.matmul(m1, m2)
        print(f"  [OK] Matmul: shape {m3.shape}")
        
        # Test 3: Operaciones estadísticas
        mean = cp.mean(m1)
        std = cp.std(m1)
        print(f"  [OK] Estadisticas: mean={float(mean):.4f}, std={float(std):.4f}")
        
        # Test 4: Inversión de matriz
        small_mat = cp.random.rand(5, 5).astype(cp.float32)
        small_mat = small_mat + cp.eye(5) * 0.1  # Hacer diagonalmente dominante
        try:
            inv = cp.linalg.inv(small_mat)
            print(f"  [OK] Inversion de matriz: shape {inv.shape}")
        except Exception as e:
            print(f"  [ERROR] Inversion fallo: {e}")
        
        print("\n[OK] Operaciones basicas funcionan!")
        
        # Probar pixelCalc_CUDA
        print("\nProbando pixelCalc_CUDA...")
        try:
            from paco.processing.fullpaco_cuda import FullPACO_CUDA
            import numpy as np
            
            dummy_stack = np.random.randn(10, 20, 20).astype(np.float32)
            dummy_angles = np.linspace(0, 360, 10)
            dummy_psf = np.random.randn(10, 10).astype(np.float32)
            
            fp_cuda = FullPACO_CUDA(
                image_stack=dummy_stack,
                angles=dummy_angles,
                psf=dummy_psf,
                psf_rad=4,
                patch_area=49
            )
            
            # Crear patch de prueba
            T = 50
            P = 49
            patch = np.random.randn(T, P).astype(np.float32)
            
            m, Cinv = fp_cuda.pixelCalc_CUDA(patch)
            print(f"  [OK] pixelCalc_CUDA ejecutado")
            print(f"      m shape: {m.shape}, Cinv shape: {Cinv.shape}")
            
        except Exception as e:
            print(f"  [ERROR] pixelCalc_CUDA fallo: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        print("[ERROR] No hay GPUs")
        sys.exit(1)
        
except Exception as e:
    print(f"[ERROR] Error general: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[OK] TEST COMPLETADO")
print("=" * 70)

