from pathlib import Path
lines=Path('tesisfinal.ipynb').read_text(encoding='utf-8').splitlines()
def block(start,end):
    for ln in range(start,end+1):
        print(f"L{ln}:{lines[ln-1]}")
block(339,370)
block(420,520)
block(706,810)
