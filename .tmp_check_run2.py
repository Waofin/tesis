import json
with open('tesisfinal.ipynb','r',encoding='utf-8') as f:
    nb=json.load(f)
cell=''.join(nb['cells'][11]['source'])
start=cell.index('def _run_lfs_cmd')
print(cell[start:start+400])
