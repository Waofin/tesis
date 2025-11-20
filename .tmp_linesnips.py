from pathlib import Path
lines=Path('tesisfinal.ipynb').read_text(encoding='utf-8').splitlines()
def dump(start,end,label):
    print(f"-- {label} --")
    for ln in range(start,end+1):
        print(f"L{ln}:{lines[ln-1]}")

dump(339,370,'config')
dump(423,470,'load_part')
dump(811,860,'benchmark_part')
