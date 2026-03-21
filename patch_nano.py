import sys
import re

fname = r'c:\Users\pathi\OneDrive\Desktop\LAST\Paper\main.tex'
with open(fname, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace "97\,K" with "101\,K"
text = text.replace('97\\,K', '101\\,K')

# Is Nano ever listed as "97 K" or something? No, it's always "97\,K"
# The table has 97\,K for K=3, A_ell. Wait, the ablation table for GCN partition count:
# '3 & \checkmark & 97\,K & \best{85.85} & +2.45* \\' -> '3 & \checkmark & 101\,K & \best{85.85} & +2.45* \\' 
# This correctly updates the ablation tables too.

with open(fname, 'w', encoding='utf-8') as f:
    f.write(text)
print("done")
