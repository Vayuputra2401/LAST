import re

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# For figure 1
text = text.replace(
    r'\resizebox{\columnwidth}{!}{\input{images/fig1_overview}}',
    r'\resizebox{0.90\columnwidth}{!}{\input{images/fig1_overview}}'
)

# For figure 4
text = text.replace(
    r'\resizebox{\columnwidth}{!}{\input{images/fig4_block}}',
    r'\resizebox{0.85\columnwidth}{!}{\input{images/fig4_block}}'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Scaled figures down.")
