import re

file_path = 'c:/Users/pathi/OneDrive/Desktop/LAST/Paper/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    # 1. Abstract
    "We present \\textbf{ShiftFuse-Zero}": "This work presents \\textbf{ShiftFuse-Zero}",
    
    # 2. Intro
    "We ask: \\textbf{how much spatial knowledge can we inject": "This work asks: \\textbf{how much spatial knowledge can be injected",
    
    # 3. Intro - "our core contribution" -> "the core contribution"
    "While our core contribution": "While the core contribution",
    "our primary innovation": "the primary innovation",
    
    # 4. Intro - "our Small variant" -> "the Small variant" 
    "Strikingly, our Small variant strictly beats": "Strikingly, the Small variant strictly beats",
    
    # 5. Related Work - KD
    "parameter range, as we do in this work.": "parameter range, as done in this work.",
    
    # 6. Methodology
    "We construct four complementary streams:": "This work constructs four complementary streams:",
    
    # 7. Block Ordering
    "We deliberately place Joint Embedding": "This work deliberately places Joint Embedding",
    "Consequently, we achieve superior": "Consequently, this work achieves superior",
    
    # 8. Training KD
    "We distil Large into Nano and Small~\cite{kd}:": "Large is distilled into Nano and Small~\cite{kd}:",
    
    # 9. Efficiency Frontier
    "we define the": "this work defines the",
    
    # 10. Conclusion
    "We presented \\textbf{ShiftFuse-Zero}": "This work presented \\textbf{ShiftFuse-Zero}",
    "We believe this paradigm": "It is believed that this paradigm"
}

for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new)
        print(f"Replaced: {old[:15]}... -> {new[:15]}...")
    else:
        print(f"WARNING: Could not find: {old}")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Replacement complete.")
