"""
Cargador de datasets desde GitHub para PACO
============================================

Este script facilita la carga de datasets desde repositorios de GitHub
para usar en benchmarks y pruebas de PACO.
"""

import numpy as np
import urllib.request
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import json


class GitHubDatasetLoader:
    """Cargador de datasets desde GitHub"""
    
    def __init__(self, cache_dir: str = "datasets_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_file(self, url: str, filename: Optional[str] = None) -> Path:
        """Descarga un archivo desde una URL"""
        if filename is None:
            filename = url.split('/')[-1]
        
        cache_file = self.cache_dir / filename
        
        # Si ya existe en cache, usar ese
        if cache_file.exists():
            print(f"  ✓ Usando archivo en cache: {cache_file}")
            return cache_file
        
        print(f"  Descargando: {url}")
        try:
            urllib.request.urlretrieve(url, cache_file)
            print(f"  ✓ Descargado: {cache_file}")
            return cache_file
        except Exception as e:
            print(f"  ✗ Error descargando: {e}")
            raise
    
    def convert_github_url_to_raw(self, github_url: str) -> str:
        """Convierte URL de GitHub a raw URL"""
        if 'github.com' not in github_url:
            return github_url
        
        # Convertir a raw URL
        raw_url = github_url.replace('github.com', 'raw.githubusercontent.com')
        if '/blob/' in raw_url:
            raw_url = raw_url.replace('/blob/', '/')
        if '/tree/' in raw_url:
            raw_url = raw_url.replace('/tree/', '/')
        
        return raw_url
    
    def load_fits(self, url: str, filename: Optional[str] = None) -> np.ndarray:
        """Carga un archivo FITS desde URL"""
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("astropy no está instalado. Instala con: pip install astropy")
        
        raw_url = self.convert_github_url_to_raw(url)
        cache_file = self.download_file(raw_url, filename)
        
        data = fits.getdata(cache_file)
        return data
    
    def load_dataset(self, config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Carga un dataset completo desde configuración
        
        Parameters
        ----------
        config : dict
            Configuración del dataset con:
            - cube_url: URL del cube FITS
            - pa_url: URL de los ángulos (opcional)
            - psf_url: URL del PSF (opcional)
            - pixscale: pixel scale en arcsec/pixel
            - pa: array de ángulos (opcional, si no se proporciona pa_url)
        
        Returns
        -------
        cube : np.ndarray
            Cube de datos (time, y, x)
        pa : np.ndarray
            Ángulos de paralaje
        psf : np.ndarray
            PSF
        pixscale : float
            Pixel scale
        """
        print(f"\nCargando dataset: {config.get('name', 'Unknown')}")
        print("="*70)
        
        # Cargar cube
        if 'cube_url' not in config:
            raise ValueError("config debe contener 'cube_url'")
        
        print("1. Cargando cube...")
        cube = self.load_fits(config['cube_url'])
        print(f"   ✓ Cube shape: {cube.shape}")
        
        # Cargar ángulos
        print("\n2. Cargando ángulos...")
        if 'pa_url' in config:
            pa_data = self.load_fits(config['pa_url'])
            if pa_data.ndim > 1:
                pa = pa_data.flatten()
            else:
                pa = pa_data
            print(f"   ✓ Ángulos: {len(pa)} frames")
        elif 'pa' in config:
            pa = np.array(config['pa'])
            print(f"   ✓ Ángulos (proporcionados): {len(pa)} frames")
        else:
            # Generar ángulos sintéticos
            pa = np.linspace(0, 90, cube.shape[0])
            print(f"   ⚠ Ángulos generados sintéticamente: {len(pa)} frames")
        
        # Verificar que el número de ángulos coincida
        if len(pa) != cube.shape[0]:
            print(f"   ⚠ ADVERTENCIA: Número de ángulos ({len(pa)}) != frames ({cube.shape[0]})")
            if len(pa) > cube.shape[0]:
                pa = pa[:cube.shape[0]]
                print(f"   Ajustado a: {len(pa)}")
            else:
                pa = np.concatenate([pa, np.repeat(pa[-1], cube.shape[0] - len(pa))])
                print(f"   Extendido a: {len(pa)}")
        
        # Cargar PSF
        print("\n3. Cargando PSF...")
        if 'psf_url' in config:
            psf = self.load_fits(config['psf_url'])
            print(f"   ✓ PSF shape: {psf.shape}")
        elif 'psf' in config:
            psf = np.array(config['psf'])
            print(f"   ✓ PSF (proporcionado): {psf.shape}")
        else:
            # Generar PSF sintético
            psf_size = 21
            center = psf_size // 2
            y, x = np.ogrid[:psf_size, :psf_size]
            psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * 2.0**2))
            psf = psf / np.sum(psf)
            print(f"   ⚠ PSF generado sintéticamente: {psf.shape}")
        
        # Pixel scale
        pixscale = config.get('pixscale', 0.027)
        print(f"\n4. Pixel scale: {pixscale} arcsec/pixel")
        
        print("\n✓ Dataset cargado correctamente")
        
        return cube, pa, psf, pixscale
    
    def load_from_config_file(self, config_file: str) -> Dict[str, Tuple]:
        """Carga múltiples datasets desde un archivo de configuración JSON"""
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        datasets = {}
        for name, config in configs.items():
            try:
                cube, pa, psf, pixscale = self.load_dataset(config)
                datasets[name] = (cube, pa, psf, pixscale)
            except Exception as e:
                print(f"✗ Error cargando dataset '{name}': {e}")
                continue
        
        return datasets


# ============================================================================
# Ejemplos de uso
# ============================================================================

def example_load_single_dataset():
    """Ejemplo de carga de un dataset individual"""
    loader = GitHubDatasetLoader()
    
    # Ejemplo de configuración
    config = {
        'name': 'Dataset de ejemplo',
        'cube_url': 'https://github.com/user/repo/blob/main/data/cube.fits',
        'pa_url': 'https://github.com/user/repo/blob/main/data/angles.fits',
        'psf_url': 'https://github.com/user/repo/blob/main/data/psf.fits',
        'pixscale': 0.027
    }
    
    cube, pa, psf, pixscale = loader.load_dataset(config)
    return cube, pa, psf, pixscale


def example_load_from_config():
    """Ejemplo de carga desde archivo de configuración"""
    loader = GitHubDatasetLoader()
    
    # Crear archivo de configuración de ejemplo
    config = {
        'dataset1': {
            'name': 'Dataset 1',
            'cube_url': 'https://github.com/user/repo/blob/main/data/cube1.fits',
            'pa_url': 'https://github.com/user/repo/blob/main/data/angles1.fits',
            'pixscale': 0.027
        },
        'dataset2': {
            'name': 'Dataset 2',
            'cube_url': 'https://github.com/user/repo/blob/main/data/cube2.fits',
            'pa': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],  # Ángulos directos
            'pixscale': 0.020
        }
    }
    
    # Guardar configuración
    with open('datasets_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Cargar datasets
    datasets = loader.load_from_config_file('datasets_config.json')
    
    return datasets


if __name__ == "__main__":
    print("="*70)
    print("CARGADOR DE DATASETS DESDE GITHUB")
    print("="*70)
    print("\nEste script proporciona funciones para cargar datasets desde GitHub.")
    print("Ver ejemplos en el código para uso.")
    print("\nEjemplo de configuración JSON:")
    print("""
{
  "dataset_name": {
    "name": "Nombre del dataset",
    "cube_url": "https://github.com/user/repo/blob/main/data/cube.fits",
    "pa_url": "https://github.com/user/repo/blob/main/data/angles.fits",
    "psf_url": "https://github.com/user/repo/blob/main/data/psf.fits",
    "pixscale": 0.027
  }
}
    """)

