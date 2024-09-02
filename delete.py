import json
import os
import sys

os.chdir(sys.path[0])
names = input('输入删除信息:')
names = list(map(str, names.split()))
with open('./knowledgebase.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for name in names:
            data.pop(name)
with open('./knowledgebase.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
print(f'{names} deleted')