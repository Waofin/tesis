"""
Generar datos sintéticos grandes para probar FastPACO_CUDA.

Crea datasets sintéticos de diferentes tamaños para validar el speedup de GPU.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from astropy.io import fits


def generate_synthetic_dataset(output_dir: Path, name: str, 
                              image_size: tuple, n_frames: int, 
                              psf_size: tuple = (25, 25)):
    """
    Genera un dataset sintético para PACO.
    
    Parameters
    ----------
    output_dir : Path
        Directorio donde guardar los archivos
    name : str
        Nombre del dataset (ej: "synthetic_large_500x500")
    image_size : tuple
        Tamaño de la imagen (height, width)
    n_frames : int
        Número de frames temporales
    psf_size : tuple
        Tamaño del PSF (height, width)
    """
    print(f"\nGenerando dataset: {name}")
    print(f"  Image size: {image_size}")
    print(f"  N frames: {n_frames}")
    print(f"  PSF size: {psf_size}")
    
    h, w = image_size
    psf_h, psf_w = psf_size
    
    # Generar cube sintético (ruido + señal débil)
    print("  Generando cube...")
    cube = np.random.randn(n_frames, h, w).astype(np.float32)
    # Agregar señal débil en el centro
    center_h, center_w = h // 2, w // 2
    signal_strength = 0.1
    for t in range(n_frames):
        cube[t, center_h-2:center_h+3, center_w-2:center_w+3] += signal_strength
    
    # Generar ángulos de paralaje
    print("  Generando ángulos...")
    angles = np.linspace(0, 360, n_frames).astype(np.float32)
    
    # Generar PSF (Gaussiana)
    print("  Generando PSF...")
    psf = np.zeros((psf_h, psf_w), dtype=np.float32)
    center_psf_h, center_psf_w = psf_h // 2, psf_w // 2
    sigma = min(psf_h, psf_w) / 6.0
    
    y, x = np.ogrid[:psf_h, :psf_w]
    dist_sq = (x - center_psf_w)**2 + (y - center_psf_h)**2
    psf = np.exp(-dist_sq / (2 * sigma**2))
    psf = psf / np.sum(psf)  # Normalizar
    
    # Generar pixel scale
    pxscale = 0.01  # arcsec/pixel
    
    # Guardar archivos FITS
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cube_file = output_dir / f"{name}_cube.fits"
    pa_file = output_dir / f"{name}_pa.fits"
    psf_file = output_dir / f"{name}_psf.fits"
    pxscale_file = output_dir / f"{name}_pxscale.fits"
    
    print(f"  Guardando archivos...")
    fits.writeto(cube_file, cube, overwrite=True)
    fits.writeto(pa_file, angles, overwrite=True)
    fits.writeto(psf_file, psf, overwrite=True)
    fits.writeto(pxscale_file, np.array([pxscale]), overwrite=True)
    
    total_pixels = h * w
    total_size = n_frames * h * w
    
    print(f"  [OK] Dataset generado:")
    print(f"    Total píxeles: {total_pixels:,}")
    print(f"    Tamaño total: {total_size:,}")
    print(f"    Archivos guardados en: {output_dir}")
    
    return {
        'cube': cube,
        'angles': angles,
        'psf': psf,
        'pxscale': pxscale,
        'total_pixels': total_pixels,
        'total_size': total_size
    }


def main():
    """Generar múltiples datasets sintéticos de diferentes tamaños."""
    print("=" * 70)
    print("GENERADOR DE DATOS SINTÉTICOS GRANDES")
    print("=" * 70)
    
    output_dir = Path("./synthetic_data")
    output_dir.mkdir(exist_ok=True)
    
    # Definir datasets a generar
    datasets = [
        {
            'name': 'synthetic_medium_200x200',
            'image_size': (200, 200),
            'n_frames': 100,
            'psf_size': (25, 25)
        },
        {
            'name': 'synthetic_large_300x300',
            'image_size': (300, 300),
            'n_frames': 150,
            'psf_size': (25, 25)
        },
        {
            'name': 'synthetic_xlarge_500x500',
            'image_size': (500, 500),
            'n_frames': 200,
            'psf_size': (25, 25)
        },
        {
            'name': 'synthetic_xxlarge_800x800',
            'image_size': (800, 800),
            'n_frames': 300,
            'psf_size': (25, 25)
        }
    ]
    
    results = []
    
    for config in datasets:
        try:
            result = generate_synthetic_dataset(
                output_dir,
                config['name'],
                config['image_size'],
                config['n_frames'],
                config['psf_size']
            )
            results.append({
                'name': config['name'],
                **result
            })
        except Exception as e:
            print(f"  [ERROR] Error generando {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Píxeles: {r['total_pixels']:,}")
        print(f"  Tamaño total: {r['total_size']:,}")
    
    print(f"\n[OK] Datasets generados en: {output_dir}")
    print("\nPara usar estos datasets:")
    print(f"  python benchmark_fastpaco_cuda_challenge_simple.py {output_dir}")


if __name__ == "__main__":
    main()

