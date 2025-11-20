# PACO Module - Auto-patched for compatibility
from paco.processing.paco import PACO
from paco.processing.fullpaco import FullPACO  
from paco.processing.fastpaco import FastPACO

# Exportar solo las clases necesarias
__all__ = ['PACO', 'FullPACO', 'FastPACO']

# Versión del parcheo
__patched__ = True
