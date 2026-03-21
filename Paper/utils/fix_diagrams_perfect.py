fig1_content = r"""\documentclass[border=4pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,calc,decorations.pathreplacing}
\usepackage{xcolor}
\definecolor{bestcol}{HTML}{1565C0}
\definecolor{freegreen}{HTML}{2E7D32}
\definecolor{learnblue}{HTML}{1565C0}
\definecolor{fusred}{HTML}{C62828}
\newcommand{\best}[1]{{\color{bestcol}\textbf{#1}}}
\begin{document}
\begin{tikzpicture}[
  every node/.style={font=\scriptsize},
  box/.style={draw, rounded corners=3pt, minimum width=1.6cm,
              minimum height=0.55cm, align=center, fill=white},
  free/.style={box, fill=green!10, draw=freegreen!70},
  learn/.style={box, fill=blue!8, draw=learnblue!50},
  stem/.style={box, fill=orange!12, draw=orange!60},
  cls/.style={box, fill=gray!12, draw=gray!60},
  inp/.style={box, fill=yellow!12, draw=yellow!60!black, minimum width=1.4cm},
  arr/.style={-Stealth, thick, draw=gray!60},
  skip/.style={-Stealth, thick, draw=gray!40, rounded corners=5pt}
]

%% input streams
\node[inp] (xj)  at (0, 0)    {Joint};
\node[inp]  (xv)  at (0,-1.00) {Velocity};
\node[inp]  (xb)  at (0,-2.00) {Bone};
\node[inp]  (xbv) at (0,-3.00) {Bone-vel};

%% stem
\node[stem, minimum height=3.6cm, minimum width=1.5cm]
  (sfuse) at (2.1, -1.5) {Stream\\Fusion\\Concat};

\foreach \s in {xj,xv,xb,xbv}
  \draw[arr] (\s.east) -- (\s.east -| sfuse.west);

%% block internals (stacked, right of stem)
\node[free]  (brasp) at (5.0,  0.50)  {BRASP};
\node[free, below=0.50cm of brasp]  (sgp)   {SGPShift};
\node[learn, below=0.50cm of sgp]   (je)    {JointEmbed};
\node[learn, below=0.50cm of je]    (gcn)   {Graph Conv};
\node[learn, below=0.50cm of gcn]   (tcn)   {DS-TCN};
\node[free, below=0.50cm of tcn]    (dp)    {DropPath};
\node[learn, below=0.50cm of dp]    (res)   {$\oplus$ Skip};

% skip bypass
\draw[skip] (brasp.west) -- +(-0.6,0) |- (res.west);

\draw[arr] (sfuse.east) -- ++(0.5,0) |- (brasp.west);
\foreach \a/\b in {brasp/sgp, sgp/je, je/gcn, gcn/tcn, tcn/dp, dp/res}
  \draw[arr] (\a)--(\b);

\node[font=\tiny, text=learnblue!80] at (5.0, 1.0) {EfficientZero Block $\times N$};

%% TLA
\node[learn, fill=blue!15, below=0.70cm of res] (tla) {TLA (last)};
\draw[arr] (res)--(tla);

%% classifier
\node[cls, minimum height=2.8cm, minimum width=1.5cm]
  (pool) at (7.5, -4.50) {Gated\\Pool\\+ FC};
  
\draw[arr] (tla.east)  -- (tla.east -| pool.west);

%% output
\node[right=0.6cm of pool] (out) {\textbf{Logits}};
\draw[arr] (pool)--(out);

%% legend
\node[free,  minimum width=1.1cm] at (0,-7.50) {0-param};
\node[learn, minimum width=1.1cm] at (1.55,-7.50) {learnable};
\node[stem,  minimum width=1.1cm] at (3.1,-7.50) {fusion};

\end{tikzpicture}
\end{document}
"""

fig4_content = r"""\documentclass[border=4pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,calc,decorations.pathreplacing}
\usepackage{xcolor}
\definecolor{bestcol}{HTML}{1565C0}
\definecolor{freegreen}{HTML}{2E7D32}
\definecolor{learnblue}{HTML}{1565C0}
\definecolor{fusred}{HTML}{C62828}
\newcommand{\best}[1]{{\color{bestcol}\textbf{#1}}}
\begin{document}
\begin{tikzpicture}[
  every node/.style={font=\scriptsize},
  box/.style={draw, rounded corners=3pt, minimum width=3.5cm,
              minimum height=0.56cm, align=center},
  free/.style={box, fill=green!10, draw=freegreen!70},
  learn/.style={box, fill=blue!8,  draw=learnblue!50},
  hilit/.style={box, fill=blue!18, draw=learnblue!80},
  ann/.style={font=\tiny, text=gray!65, anchor=west},
  arr/.style={-Stealth, thick, draw=gray!55},
  skip/.style={-Stealth, thick, draw=gray!35, rounded corners=5pt}
]
\node[free]  (brasp) at (0,0)      {BRASP\ \ (Eq.\,1)};
\node[free, below=0.60cm of brasp] (sgp)   {SGPShift\ \ (Eq.\,2)};
\node[learn, below=0.60cm of sgp]  (je)    {Joint Embedding $\mathbf{b}_v$};
\node[learn, below=0.60cm of je]   (stca)  {STC-Attention ($V{\times}V$)};
\node[learn, below=0.60cm of stca] (gcn)   {Graph Conv $K{=}3 + \mathbf{A}_\ell$\ \ (Eq.\,3)};
\node[learn, below=0.60cm of gcn]  (tcn)   {DS-TCN\ (dil.\,1\,+\,dil.\,2)};
\node[free, below=0.60cm of tcn]   (dp)    {DropPath};
\node[learn, below=0.60cm of dp]   (res)   {Residual $\oplus$};
\node[hilit, below=0.60cm of res]  (tla)   {TLA\ \ (last block only)\ \ (Eq.\,4)};

%% skip
\draw[skip] (brasp.west) -- +(-0.6,0) |- (res.west);

\foreach \a/\b in {brasp/sgp,sgp/je,je/stca,stca/gcn,gcn/tcn,tcn/dp,dp/res,res/tla}
  \draw[arr] (\a)--(\b);

%% annotations right
\node[ann, right=0.35cm of brasp] {0 params, 0 FLOPs};
\node[ann, right=0.35cm of sgp]   {0 params, sparse avg};
\node[ann, right=0.35cm of je]    {$C_\text{in}{\times}V$ params};
\node[ann, right=0.35cm of gcn]   {$3C^2$ params};
\node[ann, right=0.35cm of tcn]   {$\approx C^2/2$ params};
\node[ann, right=0.35cm of dp]    {0 params};
\node[ann, right=0.35cm of tla]   {$O(T{\cdot}K)$ complexity};

\end{tikzpicture}
\end{document}
"""

fig5_content = r"""\documentclass[border=4pt]{standalone}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{shapes.geometric,arrows.meta,positioning,fit,calc,decorations.pathreplacing}
\usepackage{xcolor}
\definecolor{bestcol}{HTML}{1565C0}
\definecolor{freegreen}{HTML}{2E7D32}
\definecolor{learnblue}{HTML}{1565C0}
\definecolor{fusred}{HTML}{C62828}
\newcommand{\best}[1]{{\color{bestcol}\textbf{#1}}}
\begin{document}
\begin{tikzpicture}[
  every node/.style={font=\scriptsize},
  blk/.style ={draw, rounded corners=3pt, minimum width=2.1cm,
               minimum height=0.65cm, align=center,
               fill=blue!8, draw=learnblue!50},
  stem/.style={blk, fill=orange!12, draw=orange!55},
  fus/.style ={blk, fill=red!10,    draw=fusred!55},
  cls/.style ={blk, fill=gray!12,   draw=gray!55},
  tlabox/.style={blk, fill=blue!18, draw=learnblue!80},
  tit/.style={font=\small\bfseries},
  arr/.style={-Stealth, thick, draw=gray!55}
]

%% ─── (a) Nano ──────────────────────────────────────────────────────────────
\node[tit] (na_t) at (0, 0) {(a) SFZ-Nano\ \ 97\,K};
\node[stem]   (na_sf)  at (0,-0.85)  {StreamFusion\\$4{\to}C_s{=}24$};
\node[blk]    (na_s1)  at (0,-1.75)  {Stage 1 $\times$1\\$C{=}32$};
\node[blk]    (na_s2)  at (0,-2.65)  {Stage 2 $\times$1\\$C{=}64,s{=}2$};
\node[blk]    (na_s3)  at (0,-3.55)  {Stage 3 $\times$1\\$C{=}128,s{=}2$};
\node[tlabox] (na_tla) at (0,-4.45)  {TLA $K{=}8, r{=}16$};
\node[cls]    (na_cl)  at (0,-5.35)  {Pool-Gate + FC(60)};

\foreach \a/\b in {na_t/na_sf, na_sf/na_s1, na_s1/na_s2,
                   na_s2/na_s3, na_s3/na_tla, na_tla/na_cl}
  \draw[arr] (\a)--(\b);

%% ─── (b) Small ─────────────────────────────────────────────────────────────
\node[tit] (sm_t) at (5.0, 0) {(b) SFZ-Small\ \ 250\,K};

\node[stem] (sm_sfA) at (3.8,-1.00) {Stem A\\J+B};
\node[stem] (sm_sfB) at (6.2,-1.00) {Stem B\\V+BV};
\node[blk]  (sm_12a) at (3.8,-2.10) {S1-S2 $(1{+}2)$\\$C{=}32{\to}64$};
\node[blk]  (sm_12b) at (6.2,-2.10) {S1-S2 $(1{+}2)$\\$C{=}32{\to}64$};
\node[fus, minimum width=2.8cm]  (sm_csf) at (5.0,-3.30) {Cross-Stream\\Fusion (Eq.\,5)};
\node[tlabox](sm_s3a) at (3.8,-4.50){S3 $\times$1 + TLA\\$C{=}128$};
\node[tlabox](sm_s3b) at (6.2,-4.50){S3 $\times$1 + TLA\\$C{=}128$};
\node[cls, minimum width=2.8cm]  (sm_cl)  at (5.0,-5.60) {Weighted avg + FC(60)};

\draw[arr] (sm_t)--(sm_sfA);
\draw[arr] (sm_t)--(sm_sfB);
\draw[arr] (sm_sfA)--(sm_12a);
\draw[arr] (sm_sfB)--(sm_12b);
\draw[arr] (sm_12a.south) -- (sm_12a.south |- sm_csf.north);
\draw[arr] (sm_12b.south) -- (sm_12b.south |- sm_csf.north);
\draw[arr] (sm_csf.south) -- ++(0,-0.2) -| (sm_s3a.north);
\draw[arr] (sm_csf.south) -- ++(0,-0.2) -| (sm_s3b.north);
\draw[arr] (sm_s3a.south) -- (sm_s3a.south |- sm_cl.north);
\draw[arr] (sm_s3b.south) -- (sm_s3b.south |- sm_cl.north);

%% ─── (c) Large ─────────────────────────────────────────────────────────────
\node[tit] (lg_t) at (11.0, 0) {(c) SFZ-Large\ \ 1.67\,M};

\node[stem] (lg_sfJ) at ( 8.6,-0.85) {Stem J};
\node[stem] (lg_sfB) at (11.0,-0.85) {Stem B};
\node[stem] (lg_sfV) at (13.4,-0.85) {Stem V};
\node[blk]  (lg_12J) at ( 8.6,-1.95) {S1-S2 $(1{+}2)$\\$C{=}48{\to}96$};
\node[blk]  (lg_12B) at (11.0,-1.95) {S1-S2 $(1{+}2)$\\$C{=}48{\to}96$};
\node[blk]  (lg_12V) at (13.4,-1.95) {S1-S2 $(1{+}2)$\\$C{=}48{\to}96$};
\node[fus,minimum width=5.2cm] (lg_fc)
                     at (11.0,-3.20) {FusionConv $288{\to}192$\\BN + Hardswish};
\node[tlabox,minimum width=5.2cm]
             (lg_s3) at (11.0,-4.40) {Shared Stage 3 $\times$4 + TLA\\$C{=}192,\,K{=}14$};
\node[cls,minimum width=5.2cm]   (lg_cl) at (11.0,-5.50) {Pool-Gate + FC(60)};

\draw[arr] (lg_t)--(lg_sfJ);
\draw[arr] (lg_t)--(lg_sfB);
\draw[arr] (lg_t)--(lg_sfV);
\draw[arr] (lg_sfJ)--(lg_12J);
\draw[arr] (lg_sfB)--(lg_12B);
\draw[arr] (lg_sfV)--(lg_12V);
\draw[arr] (lg_12J.south) -- (lg_12J.south |- lg_fc.north);
\draw[arr] (lg_12B.south) -- (lg_12B.south |- lg_fc.north);
\draw[arr] (lg_12V.south) -- (lg_12V.south |- lg_fc.north);
\draw[arr] (lg_fc)--(lg_s3);
\draw[arr] (lg_s3)--(lg_cl);

\end{tikzpicture}
\end{document}
"""

with open('c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig1_overview.tex', 'w', encoding='utf-8') as f:
    f.write(fig1_content)
with open('c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig4_block.tex', 'w', encoding='utf-8') as f:
    f.write(fig4_content)
with open('c:/Users/pathi/OneDrive/Desktop/LAST/Paper/images/fig5_family.tex', 'w', encoding='utf-8') as f:
    f.write(fig5_content)

print("Diagrams securely fixed.")
