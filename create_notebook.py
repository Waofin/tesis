"""
Script para generar el notebook completo de benchmark y análisis de PACO
"""
import json

# Definir todas las celdas del notebook
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Benchmark y Análisis Completo de PACO en VIP-master\n",
            "\n",
            "Este notebook realiza:\n",
            "1. **Análisis de Errores**: Detecta problemas en la implementación de PACO\n",
            "2. **Carga de Datasets**: Descarga datasets desde GitHub (subchallenge1)\n",
            "3. **Benchmark Completo**: Mide tiempos de ejecución con diferentes datasets\n",
            "4. **Profiling**: Analiza el rendimiento del algoritmo\n",
            "5. **Visualizaciones**: Genera gráficos y reportes\n",
            "\n",
            "## Configuración para Colab\n",
            "\n",
            "1. **Seleccionar Runtime**: Runtime > Change runtime type > Hardware accelerator: GPU (opcional)\n",
            "2. **Ejecutar todas las celdas**: Runtime > Run all\n",
            "\n",
            "## Autor\n",
            "César Cerda - Universidad del Bío-Bío"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Instalación de Dependencias"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instalar dependencias necesarias\n",
            "!pip install -q astropy scipy matplotlib ipywidgets\n",
            "!pip install -q memory-profiler line-profiler\n",
            "\n",
            "print(\"✓ Dependencias instaladas\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Clonar Repositorios"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Clonar repositorios necesarios\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# URLs de los repositorios\n",
            "VIP_REPO_URL = \"https://github.com/vortex-exoplanet/VIP.git\"\n",
            "TESIS_REPO_URL = \"https://github.com/Waofin/tesis.git\"\n",
            "\n",
            "# Directorios de destino\n",
            "REPOS_DIR = Path(\"/content/repos\")\n",
            "VIP_DIR = REPOS_DIR / \"VIP\"\n",
            "TESIS_DIR = REPOS_DIR / \"tesis\"\n",
            "\n",
            "# Crear directorio de repositorios\n",
            "REPOS_DIR.mkdir(exist_ok=True)\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"CLONANDO REPOSITORIOS\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "# Clonar VIP si no existe\n",
            "if not VIP_DIR.exists():\n",
            "    print(f\"Clonando VIP desde {VIP_REPO_URL}...\")\n",
            "    !cd {REPOS_DIR} && git clone {VIP_REPO_URL}\n",
            "    print(\"✓ VIP clonado correctamente\")\n",
            "else:\n",
            "    print(\"✓ VIP ya existe\")\n",
            "\n",
            "# Clonar repositorio de tesis (para datasets)\n",
            "if not TESIS_DIR.exists():\n",
            "    print(f\"\\nClonando repositorio de tesis desde {TESIS_REPO_URL}...\")\n",
            "    !cd {REPOS_DIR} && git clone {TESIS_REPO_URL}\n",
            "    print(\"✓ Repositorio de tesis clonado correctamente\")\n",
            "else:\n",
            "    print(\"✓ Repositorio de tesis ya existe\")\n",
            "\n",
            "print(f\"\\n✓ Repositorios disponibles en: {REPOS_DIR}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Instalar VIP"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instalar VIP desde el repositorio clonado\n",
            "print(\"=\"*70)\n",
            "print(\"INSTALANDO VIP\")\n",
            "print(\"=\"*70)\n",
            "\n",
            "if VIP_DIR.exists():\n",
            "    print(f\"Instalando VIP desde {VIP_DIR}...\")\n",
            "    !cd {VIP_DIR} && pip install -q -e .\n",
            "    print(\"✓ VIP instalado correctamente\")\n",
            "else:\n",
            "    print(\"⚠ VIP no encontrado, instalando desde PyPI...\")\n",
            "    !pip install -q vip-hci\n",
            "    print(\"✓ VIP instalado desde PyPI\")\n",
            "\n",
            "# Verificar instalación\n",
            "import sys\n",
            "vip_src = VIP_DIR / \"src\"\n",
            "if vip_src.exists():\n",
            "    if str(vip_src) not in sys.path:\n",
            "        sys.path.insert(0, str(vip_src))\n",
            "    print(f\"✓ VIP agregado al path: {vip_src}\")\n",
            "\n",
            "try:\n",
            "    import vip_hci as vip\n",
            "    vip_version = vip.__version__ if hasattr(vip, '__version__') else 'desconocida'\n",
            "    print(f\"✓ VIP importado correctamente (versión: {vip_version})\")\n",
            "except ImportError as e:\n",
            "    print(f\"✗ Error importando VIP: {e}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Importar Librerías y Configuración"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Importar librerías necesarias\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import time\n",
            "import sys\n",
            "import traceback\n",
            "from pathlib import Path\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Profiling\n",
            "try:\n",
            "    import cProfile\n",
            "    import pstats\n",
            "    from io import StringIO\n",
            "    PROFILING_AVAILABLE = True\n",
            "except ImportError:\n",
            "    PROFILING_AVAILABLE = False\n",
            "\n",
            "# Detectar GPU\n",
            "try:\n",
            "    import torch\n",
            "    GPU_AVAILABLE = torch.cuda.is_available()\n",
            "    if GPU_AVAILABLE:\n",
            "        gpu_name = torch.cuda.get_device_name(0)\n",
            "        print(f\"✓ GPU disponible: {gpu_name}\")\n",
            "    else:\n",
            "        print(\"⚠ GPU no disponible, usando CPU\")\n",
            "except ImportError:\n",
            "    GPU_AVAILABLE = False\n",
            "    print(\"⚠ PyTorch no disponible\")\n",
            "\n",
            "print(\"=\"*70)\n",
            "print(\"CONFIGURACIÓN COMPLETA\")\n",
            "print(\"=\"*70)"
        ]
    }
]

# Continuar con más celdas... (el archivo sería muy largo, mejor lo hago en partes)
# Por ahora, guardo lo que tengo y luego agrego el resto

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Guardar notebook parcial
with open("PACO_VIP_Complete_Benchmark_Analysis_partial.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook parcial creado. Necesito agregar más celdas...")

