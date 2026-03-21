import re

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig6_scatter.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Change axis to log mode
text = text.replace('xmin=-0.3, xmax=11.2,', 'xmode=log, xmin=0.08, xmax=15,')
text = text.replace('xlabel={Parameters (M)},', 'xlabel={Parameters (M) - Log Scale},')
text = text.replace('xtick={0,2,4,6,8,10},', 'xtick={0.1, 0.3, 1, 3, 10}, xticklabels={0.1, 0.3, 1, 3, 10},')

# Update Large coordinates from 1.10 to 1.67
text = text.replace('(1.10,92.5)', '(1.67,92.5)')

# Move +KD label position for Small to not overlap with text
text = text.replace('at (axis cs:0.25,90.2) {+KD};', 'at (axis cs:0.35,90.2) {+KD};')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

# Also trim non-essential citations from main.tex
main_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(main_path, 'r', encoding='utf-8') as f:
    main_text = f.read()

main_text = main_text.replace('\\cite{batchnorm}', '')
main_text = main_text.replace('\\cite{sgd}', '')
main_text = main_text.replace('\\cite{hardswish}', '')
main_text = main_text.replace('\\cite{mixup}', '')
main_text = main_text.replace('\\cite{cutmix}', '')
main_text = main_text.replace('\\cite{droppath}', '')

with open(main_path, 'w', encoding='utf-8') as f:
    f.write(main_text)

# And remove them from references.bib
bib_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/references.bib'
with open(bib_path, 'r', encoding='utf-8') as f:
    bib = f.read()

def remove_entry(bib, key):
    return re.sub(r'@inproceedings\{' + key + r',.*?\n\}\n*', '', bib, flags=re.DOTALL)

for k in ['batchnorm', 'sgd', 'hardswish', 'mixup', 'cutmix', 'droppath']:
    bib = remove_entry(bib, k)

with open(bib_path, 'w', encoding='utf-8') as f:
    f.write(bib)

print("Updated fig6, main.tex and references.")
