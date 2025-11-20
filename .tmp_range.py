from pathlib import Path
lines=Path('tesisfinal.ipynb').read_text(encoding='utf-8').splitlines()
start=680
end=760
for ln in range(start,end+1):
    print(f"L{ln}:{lines[ln-1]}")
