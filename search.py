import sys

fname = r'c:\Users\pathi\OneDrive\Desktop\LAST\Paper\main.tex'
with open(fname, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'Small' in line or 'small' in line or 'SFZ-S' in line:
            print(f"{i+1}: {line.strip()}")
        if '0.25' in line or '250' in line:
            print(f"{i+1} [250]: {line.strip()}")
