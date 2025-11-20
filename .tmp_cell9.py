import json
with open('tesisfinal.ipynb','r',encoding='utf-8') as f:
    nb=json.load(f)
cell=nb['cells'][9]
for line in cell['source']:
    print(line,end='')
