import re

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace specific lines
content = content.replace(r'Large} (1.1\,M) with three-backbone', r'Large} (1.67\,M) with three-backbone')
content = content.replace(r'identical parameter budget (1.1\,M).', r'a competitive parameter budget (1.67\,M).')
content = content.replace(r'Large (1.1\,M) --- with Temporal', r'Large (1.67\,M) --- with Temporal')
content = content.replace(r'Large  & 32 & [48,96,192]   & [1,2,4] & 14 & 8  & 0.20 & 1.1\,M \\',
                          r'Large  & 32 & [48,96,192]   & [1,2,4] & 14 & 8  & 0.20 & 1.67\,M \\')
content = content.replace(r'\paragraph{Large (1.1\,M).}', r'\paragraph{Large (1.67\,M).}')
content = content.replace(r'regularisation at 1.1\,M params', r'regularisation at 1.67\,M params')
content = content.replace(r'Large & 1.10\,M & 4.40\,MB & 1.10\,MB', r'Large & 1.67\,M & 6.38\,MB & 1.67\,MB')
content = content.replace(r'\textbf{SFZ-Large}      & \textbf{1.10M} & \best{92.5}', r'\textbf{SFZ-Large}      & \textbf{1.67M} & \best{92.5}')
content = content.replace(r'by 0.4\,pp at the same parameter budget.', r'by 0.4\,pp at a comparable parameter scale.')
content = content.replace(r'SFZ-Large       & 1.10M & 4.20', r'SFZ-Large       & 1.67M & 4.71')
content = content.replace(r'\textbf{SFZ-Large}            & 1.10M & 92.5 & 84.1 \\', r'\textbf{SFZ-Large}            & 1.67M & 92.5 & 55.4 \\')

acc_m_old = r'''Even SFZ-Large, at identical parameter budget to B4, achieves a
slightly higher Acc/M (84.1 vs.\ 83.7) while using richer zero-parameter
priors.'''
acc_m_new = r'''SFZ-Large trades peak parameter efficiency for maximum capacity, pushing
the absolute accuracy frontier to 92.5\% at 1.67\,M parameters.'''
content = content.replace(acc_m_old, acc_m_new)

content = content.replace(r'its 1.1\,M parameters over', r'its 1.67\,M parameters over')
content = content.replace(r'\textbf{Mid-network (ours, B4-style)} & \textbf{1.10M}', r'\textbf{Mid-network (ours, B4-style)} & \textbf{1.67M}')
content = content.replace(r'at 0.83\,M fewer parameters', r'at 0.26\,M fewer parameters')
content = content.replace(r'only $11{\times}$ larger.', r'only $17{\times}$ larger.')
content = content.replace(r'surpassing EfficientGCN-B4 at identical budget.', r'surpassing EfficientGCN-B4.')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Update fig5_family.tex
fig5_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig5_family.tex'
with open(fig5_path, 'r', encoding='utf-8') as f:
    fig5 = f.read()
fig5 = fig5.replace(r'(c) SFZ-Large\ \ 1.1\,M', r'(c) SFZ-Large\ \ 1.67\,M')
with open(fig5_path, 'w', encoding='utf-8') as f:
    f.write(fig5)

# Update model_results_checklist.md
check_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/model_results_checklist.md'
with open(check_path, 'r', encoding='utf-8') as f:
    check = f.read()

check = check.replace('- [x] **Large Params:** 1.10M parameters', '- [x] **Large Params:** 1.67M parameters')
check = check.replace('- [x] **Large Memory (f32/int8):** 4.40 MB / 1.10 MB', '- [x] **Large Memory (f32/int8):** 6.38 MB / 1.67 MB')
check = check.replace('Large GFLOPs & Latency:** 4.20 GFLOPs', 'Large GFLOPs & Latency:** 4.71 GFLOPs')
check = check.replace('- [x] **Large Acc/M:** 84.1 *(Verified: 92.5/1.10 = 84.09)*', '- [x] **Large Acc/M:** 55.4 *(Verified: 92.5/1.67 = 55.38)*')
check = check.replace('- [x] **Mid-Network Fusion (Ours):** 92.5% (1.10M params) *(Verified: 1.93M - 1.10M = 0.83M lower params than late fusion as mentioned in text)*', '- [x] **Mid-Network Fusion (Ours):** 92.5% (1.67M params) *(Verified: 1.93M - 1.67M = 0.26M lower params than late fusion as mentioned in text)*')

with open(check_path, 'w', encoding='utf-8') as f:
    f.write(check)

print("Updates successfully written.")
