from pathlib import Path
lines=Path('tesisfinal.ipynb').read_text(encoding='utf-8').splitlines()
def find(needle):
    for idx,line in enumerate(lines,1):
        if needle in line:
            print(needle, idx)
            return idx
find("# Configurar origen de datasets")
find("def load_challenge_dataset_simple")
find("Datasets detectados:")
