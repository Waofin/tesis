import json
with open('tesisfinal.ipynb','r',encoding='utf-8') as f:
    nb=json.load(f)
cell=''.join(nb['cells'][11]['source'])
start=cell.index('result = subprocess.run(')
print(cell[start:start+200])
