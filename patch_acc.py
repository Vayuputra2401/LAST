import sys
import re

fname = r'c:\Users\pathi\OneDrive\Desktop\LAST\Paper\main.tex'
with open(fname, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if '85.6' in line and 'EfficientGCN-B0' not in line:
        line = line.replace('85.6', '85.85')
    new_lines.append(line)

with open(fname, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Updated main.tex")
