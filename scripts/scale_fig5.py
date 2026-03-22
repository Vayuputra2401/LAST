import re

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# For figure 5
text = text.replace(
    r'\resizebox{\textwidth}{!}{\input{images/fig5_family}}',
    r'\resizebox{0.93\textwidth}{!}{\input{images/fig5_family}}'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Scaled figure 5 down.")
