import sys

fname = r'c:\Users\pathi\OneDrive\Desktop\LAST\Paper\main.tex'
with open(fname, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace metrics for Small
# "Small (250\,K)" -> "Small (233\,K)"
text = text.replace('Small (250\,K)', 'Small (233\,K)')
# "Small** (250\,K)" or similar? No, just "250\,K". I'll replace any "250\,K" that happens near "Small".
# Let's just replace all "250\,K" and "0.25M" in context of SFZ-Small.
import re
text = re.sub(r'SFZ-Small(.*?)(0\.25M|250\\,K)', lambda m: 'SFZ-Small' + m.group(1) + ('233\\,K' if '250' in m.group(2) else '0.23M'), text)
text = re.sub(r'Small(.*?)(0\.25M|250\\,K)', lambda m: 'Small' + m.group(1) + ('233\\,K' if '250' in m.group(2) else '0.23M'), text)

# Table 2: SFZ-Small & 0.25M & \best{0.6}
text = text.replace('SFZ-Small       & 0.25M & \\best{0.6}', 'SFZ-Small       & 0.23M & \\best{0.67}')
text = text.replace('SFZ-Small        & 0.25M &', 'SFZ-Small        & 0.23M &')

# FLOPs and other tables: Let's find any remaining "250" or "0.25"
with open('debug_out.txt', 'w', encoding='utf-8') as debug:
    for line in text.split('\n'):
        if 'Small' in line or 'SFZ-S' in line:
            debug.write(line + '\n')

with open(fname, 'w', encoding='utf-8') as f:
    f.write(text)
print("done")
