import re

main_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(main_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Shrink fig 1
text = text.replace(r'\resizebox{0.87\columnwidth}{!}{\input{images/fig1_overview}}', r'\resizebox{0.80\columnwidth}{!}{\input{images/fig1_overview}}')
# Shrink fig 4
text = text.replace(r'\resizebox{0.82\columnwidth}{!}{\input{images/fig4_block}}', r'\resizebox{0.75\columnwidth}{!}{\input{images/fig4_block}}')
# Shrink fig 5
text = text.replace(r'\resizebox{0.93\textwidth}{!}{\input{images/fig5_family}}', r'\resizebox{0.85\textwidth}{!}{\input{images/fig5_family}}')
# Shrink fig 6 spacing
text = text.replace(r'\begin{figure}[t]', r'\begin{figure}[htpb]\vspace{-2mm}')
text = text.replace(r'\end{figure}', r'\vspace{-3mm}\end{figure}')
# Shrink tables spacing
text = text.replace(r'\begin{table}[htpb]', r'\begin{table}[htpb]\vspace{-2mm}')
text = text.replace(r'\end{table}', r'\vspace{-3mm}\end{table}')

with open(main_path, 'w', encoding='utf-8') as f:
    f.write(text)

fig6_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig6_scatter.tex'
with open(fig6_path, 'r', encoding='utf-8') as f:
    fig6 = f.read()

fig6 = fig6.replace('width=7.6cm, height=5.5cm', 'width=7.0cm, height=5.0cm')
with open(fig6_path, 'w', encoding='utf-8') as f:
    f.write(fig6)

print("Aggressively scaled components to fit 8 pages.")
