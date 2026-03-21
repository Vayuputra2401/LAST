#!/bin/bash
# Compile all standalone figure .tex files to PDF, then convert to PNG
# Run from the Paper/ directory on Lambda (or any machine with texlive + imagemagick)
# Usage: cd Paper && bash compile_figures.sh

set -e
cd "$(dirname "$0")/images"

FIGS="fig1_overview fig2_brasp fig3_sgpshift fig4_block fig5_family fig6_scatter fig9_tla fig10_curves"

echo "=== Compiling figures ==="
for f in $FIGS; do
  echo "  $f ..."
  pdflatex -interaction=nonstopmode "$f.tex" > /dev/null 2>&1
  # Convert PDF -> PNG at 300 DPI (requires imagemagick)
  convert -density 300 -quality 95 "$f.pdf" "$f.png"
  echo "  $f.png done"
done

# Clean up aux files
rm -f *.aux *.log

echo ""
echo "=== Done! Upload these PNGs to Overleaf Paper/images/ ==="
ls -lh *.png
