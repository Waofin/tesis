"""
Script para verificar alternativas a CuPy si hay problemas de instalación.

Este script verifica qué opciones están disponibles para aceleración GPU.
"""

import sys

print("=" * 70)
print("VERIFICACIÓN DE OPCIONES PARA ACELERACIÓN GPU")
print("=" * 70)

# Verificar Python
print(f"\n1. Versión de Python: {sys.version}")
print(f"   Versión: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

if sys.version_info >= (3, 13):
    print("   ⚠️ ADVERTENCIA: Python 3.13 puede tener problemas de compatibilidad")
    print("      Recomendado: Usar Python 3.11 o 3.12 para CuPy")

# Verificar CuPy
print("\n2. Verificando CuPy...")
try:
    import cupy as cp
    n_devices = cp.cuda.runtime.getDeviceCount()
    print(f"   ✅ CuPy está instalado")
    print(f"   ✅ GPUs detectadas: {n_devices}")
    if n_devices > 0:
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"   ✅ GPU 0: {props['name'].decode()}")
    CUDA_AVAILABLE = True
except ImportError:
    print("   ❌ CuPy NO está instalado")
    print("      Instala con: pip install cupy-cuda11x")
    CUDA_AVAILABLE = False
except Exception as e:
    print(f"   ⚠️ CuPy instalado pero error: {e}")
    CUDA_AVAILABLE = False

# Verificar JAX
print("\n3. Verificando JAX...")
try:
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    print(f"   ✅ JAX está instalado")
    print(f"   ✅ Dispositivos: {[str(d) for d in devices]}")
    JAX_AVAILABLE = True
except ImportError:
    print("   ❌ JAX NO está instalado")
    print("      Instala con: pip install jax[cuda11_local]")
    JAX_AVAILABLE = False
except Exception as e:
    print(f"   ⚠️ JAX instalado pero error: {e}")
    JAX_AVAILABLE = False

# Verificar Numba CUDA
print("\n4. Verificando Numba CUDA...")
try:
    from numba import cuda
    print(f"   ✅ Numba CUDA está instalado")
    if cuda.is_available():
        print(f"   ✅ CUDA disponible en Numba")
        print(f"   ✅ GPUs detectadas: {len(cuda.gpus)}")
    else:
        print(f"   ⚠️ CUDA no disponible en Numba")
    NUMBA_AVAILABLE = True
except ImportError:
    print("   ❌ Numba NO está instalado")
    print("      Instala con: pip install numba")
    NUMBA_AVAILABLE = False
except Exception as e:
    print(f"   ⚠️ Numba instalado pero error: {e}")
    NUMBA_AVAILABLE = False

# Resumen
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

if CUDA_AVAILABLE:
    print("✅ CuPy está disponible - Puedes usar la implementación CUDA")
elif JAX_AVAILABLE:
    print("✅ JAX está disponible - Alternativa viable (requiere adaptar código)")
elif NUMBA_AVAILABLE:
    print("✅ Numba CUDA está disponible - Alternativa viable (requiere más trabajo)")
else:
    print("❌ Ninguna opción GPU disponible")
    print("\nOpciones:")
    print("  1. Instalar CuPy: pip install cupy-cuda11x")
    print("  2. Usar Python 3.11/3.12 si estás en Python 3.13")
    print("  3. Usar JAX: pip install jax[cuda11_local]")
    print("  4. Continuar con CPU (Loky paralelización)")

print("\n" + "=" * 70)


