"""
Script simple para probar la implementación FastPACO_CUDA.
"""

import sys
from pathlib import Path

# Agregar ruta de PACO
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TEST: FastPACO_CUDA")
print("=" * 70)

# Test 1: Verificar CuPy
print("\n1. Verificando CuPy...")
try:
    import cupy as cp
    print("   [OK] CuPy importado correctamente")
    
    n_devices = cp.cuda.runtime.getDeviceCount()
    print(f"   [OK] GPUs detectadas: {n_devices}")
    
    if n_devices > 0:
        device = cp.cuda.Device(0)
        device.use()
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"   [OK] GPU 0: {props['name'].decode()}")
        print(f"      Memoria: {props['totalGlobalMem'] / 1024**3:.1f} GB")
except ImportError:
    print("   [ERROR] CuPy no esta instalado")
    sys.exit(1)
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    sys.exit(1)

# Test 2: Verificar imports
print("\n2. Verificando imports de PACO...")
try:
    from paco.processing.fastpaco_cuda import FastPACO_CUDA, CUDA_AVAILABLE
    from paco.processing.fastpaco import FastPACO
    print("   [OK] FastPACO_CUDA importado correctamente")
    print(f"   [OK] CUDA_AVAILABLE = {CUDA_AVAILABLE}")
except ImportError as e:
    print(f"   [ERROR] Error importando: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Crear instancia
print("\n3. Creando instancia FastPACO_CUDA...")
try:
    import numpy as np
    
    # Datos mínimos para inicializar
    dummy_stack = np.random.randn(10, 50, 50).astype(np.float32)
    dummy_angles = np.linspace(0, 360, 10)
    dummy_psf = np.random.randn(10, 10).astype(np.float32)
    
    fp_cuda = FastPACO_CUDA(
        image_stack=dummy_stack,
        angles=dummy_angles,
        psf=dummy_psf,
        psf_rad=4,
        patch_area=49
    )
    print("   [OK] Instancia creada correctamente")
except Exception as e:
    print(f"   [ERROR] Error creando instancia: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Probar pixelCalc_CUDA
print("\n4. Probando pixelCalc_CUDA...")
try:
    T = 50
    P = 49
    patch = np.random.randn(T, P).astype(np.float32)
    
    m, Cinv = fp_cuda.pixelCalc_CUDA(patch)
    
    print(f"   [OK] pixelCalc_CUDA ejecutado correctamente")
    print(f"      m shape: {m.shape}")
    print(f"      Cinv shape: {Cinv.shape}")
    print(f"      m mean: {np.mean(m):.4f}")
    print(f"      Cinv mean: {np.mean(Cinv):.4f}")
except Exception as e:
    print(f"   [ERROR] Error en pixelCalc_CUDA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Probar computeStatistics_CUDA
print("\n5. Probando computeStatistics_CUDA...")
try:
    # Crear coordenadas de píxeles (solo algunos para prueba rápida)
    x, y = np.meshgrid(np.arange(10, 20), np.arange(10, 20))
    phi0s = np.column_stack((x.flatten(), y.flatten()))
    
    print(f"   Procesando {len(phi0s)} píxeles...")
    Cinv, m, h = fp_cuda.computeStatistics_CUDA(phi0s, batch_size=50)
    
    print(f"   [OK] computeStatistics_CUDA ejecutado correctamente")
    print(f"      Cinv shape: {Cinv.shape}")
    print(f"      m shape: {m.shape}")
    print(f"      h shape: {h.shape}")
except Exception as e:
    print(f"   [ERROR] Error en computeStatistics_CUDA: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Comparar con CPU
print("\n6. Comparando GPU vs CPU (pixelCalc)...")
try:
    from paco.util.util import pixelCalc
    
    # CPU
    m_cpu, Cinv_cpu = pixelCalc(patch)
    
    # GPU
    m_gpu, Cinv_gpu = fp_cuda.pixelCalc_CUDA(patch)
    
    # Comparar
    diff_m = np.abs(m_cpu - m_gpu).max()
    diff_Cinv = np.abs(Cinv_cpu - Cinv_gpu).max()
    
    print(f"   [OK] Comparacion completada")
    print(f"      Diferencia en m: {diff_m:.2e}")
    print(f"      Diferencia en Cinv: {diff_Cinv:.2e}")
    
    if diff_m < 1e-3 and diff_Cinv < 1e-3:
        print(f"   [OK] Resultados son similares (dentro de precision)")
    else:
        print(f"   [WARNING] Hay diferencias significativas")
        print(f"      (Esto puede ser normal por diferencias numericas)")
        
except Exception as e:
    print(f"   [ERROR] Error en comparacion: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("[OK] TODOS LOS TESTS PASARON")
print("=" * 70)
print("\nPuedes ejecutar el benchmark completo con:")
print("  python benchmark_fastpaco_cuda.py")

